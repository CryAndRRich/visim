"""Early controversy prediction: from the first k comments of a thread, predict whether the thread ends
up controversial.

Label: controversy = mean of the z-scores of p_neg and reply_ratio over the comments of the first 48 hours;
the top 25% of threads are controversial. Means, standard deviations and the 75th percentile are taken on
the training articles only and applied to the test articles.

Articles: those with at least k comments in the first 48 hours. Split by publication time: the earliest 75%
train (5-fold CV inside for hyperparameters and the F1 threshold), the latest 25% test, scored once.

Features (first k comments by time):
    content     clickbait score, category, title length (words), publication hour
    structure   reply share, max depth, structural virality (reconstructed or two-level tree), distinct
                users, log hours from publication to the k-th comment, log mean gap between comments
    graph       share of frequent users, share of central users, mean PageRank and authority of the early
                commenters, from the user graphs rebuilt on the training articles only
    sentiment   p_neg, p_uncivil, polarization of the early comments (labels from corpus_labels.csv)
    text        TF-IDF of the early comments joined (ladder only, not a feature group)

Models: majority; single features as scores (hours to the k-th comment, early p_neg, early reply share;
sign chosen on train); logistic regression; random forest; HistGradientBoosting (3 seeds for the
stochastic ones, probabilities averaged, mean +- std over seeds reported); TF-IDF + LR; TF-IDF + features.
Metrics on test: ROC-AUC, PR-AUC, F1 of the controversial class with 95% bootstrap intervals; paired
bootstrap of the AUC difference against the main baseline (hours to the k-th comment).

Ablations (LR, RF, HGB): feature groups (each alone, all, all minus each), tree version, k = 5/10/20 on the
same articles (at least 20 comments), label definition (top 20/25/33%, p_neg only, reply_ratio only),
random split instead of time split. Interpretation: LR coefficients, permutation importance on test.

Usage (from the project root):
    python -m src.analysis.controversy                   # writes results/controversy/ (about 40 min on CPU)
    python -m src.analysis.controversy --skip-ablations  # model ladder only (about 3 min)
    python -m src.analysis.controversy --skip-ablations --label-after-k   # label without the early comments
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..classifiers.common import (CSV_ENCODING, N_BOOT, RESULTS_DIR, SEED, TASKS_DIR, f1_threshold, hardware,
                                  paired_bootstrap, run_info)
from ..features import reply_graph, user_graph
from ..features.structure import tree_metrics
from ..processing.snapshot import PROCESSED_DIR
from ..processing.trees import reconstructed_parents, two_level_parents
from .thread_sentiment import labelled_comments, sentiment_metrics

OUT = RESULTS_DIR / "controversy"
K = 10
TOP = 0.25
TRAIN_SHARE = 0.75
LATER_MIN = 5  # comments after the first k needed when the label leaves the early comments out
SEEDS = [42, 43, 44]
CATEGORIES = ["giai-tri", "kinh-doanh", "phap-luat", "the-gioi", "thoi-su"]
GROUPS = {
    "content": ["clickbait_score", "title_words", "hour"] + [f"cat_{c}" for c in CATEGORIES],
    "structure": ["reply_share", "max_depth", "structural_virality", "n_users", "log_hours_to_k", "log_mean_gap"],
    "graph": ["share_frequent", "share_central", "mean_pagerank", "mean_authority"],
    "sentiment": ["p_neg", "p_uncivil", "polarization"],
}
ALL = [f for g in GROUPS.values() for f in g]
BASELINES = {"hours_to_k": "log_hours_to_k", "early_p_neg": "p_neg", "early_reply_share": "reply_share"}


# ------------------------------------------------------------------------------------------- data

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Articles (with clickbait score) and labelled comments of the first 48 hours, sorted by time."""
    a = pd.read_parquet(PROCESSED_DIR / "articles.parquet",
                        columns=["article_id", "category", "title", "published_at", "comments_status"])
    a = a[a["comments_status"] == "done"].drop(columns="comments_status")
    a["article_id"] = a["article_id"].astype(str)
    cb = pd.read_csv(TASKS_DIR / "clickbait" / "corpus_labels.csv", dtype={"id": str}, encoding=CSV_ENCODING)
    a = a.merge(cb.rename(columns={"id": "article_id", "score": "clickbait_score"})[["article_id", "clickbait_score"]],
                on="article_id", how="left")
    a["published"] = pd.to_datetime(a["published_at"], utc=True)
    c = labelled_comments()
    c = c[c["within_48h"]].copy()
    c["article_id"] = c["article_id"].astype(str)
    c["created"] = pd.to_datetime(c["created_at"], utc=True)
    c = c.sort_values(["article_id", "created", "comment_id"]).reset_index(drop=True)
    return a, c


def thread_labels(c: pd.DataFrame, skip: int = 0) -> pd.DataFrame:
    """p_neg and reply_ratio over the 48-hour thread, leaving out the first `skip` comments."""
    if skip:
        c = c[c.groupby("article_id").cumcount() >= skip]
    g = c.groupby("article_id")
    return pd.DataFrame({"n_48h": g.size(), "full_p_neg": g["sentiment"].apply(lambda s: (s == "neg").mean()),
                         "full_reply_ratio": g["is_reply"].mean()}).reset_index()


def make_label(frame: pd.DataFrame, train: np.ndarray, definition: str = "combined", top: float = TOP) -> np.ndarray:
    """1 for the top `top` share of threads by the controversy score; statistics from train rows only."""
    cols = {"combined": ["full_p_neg", "full_reply_ratio"], "p_neg": ["full_p_neg"],
            "reply_ratio": ["full_reply_ratio"]}[definition]
    tr = frame.loc[train, cols]
    z = (frame[cols] - tr.mean()) / tr.std(ddof=0)
    score = z.mean(axis=1)
    cut = score[train].quantile(1 - top)
    return (score > cut).astype(int).to_numpy()


def early_features(a: pd.DataFrame, c: pd.DataFrame, k: int, min_comments: int = 0) -> pd.DataFrame:
    """Per-article features from the first k comments (all groups except graph, which depends on the split),
    for articles with at least max(k, min_comments) comments in the first 48 hours."""
    pub = a.set_index("article_id")["published"]
    rows = []
    for aid, g in c.groupby("article_id", sort=False):
        if len(g) < max(k, min_comments):
            continue
        e = g.iloc[:k]
        recs = e.to_dict("records")
        hours = (e["created"] - pub[aid]).dt.total_seconds().to_numpy() / 3600
        gaps = np.diff(hours)
        row = {"article_id": aid, "reply_share": float(e["is_reply"].mean()), "n_users": e["user_hash"].nunique(),
               "log_hours_to_k": float(np.log1p(max(hours[-1], 0))),
               "log_mean_gap": float(np.log1p(max(gaps.mean(), 0))) if len(gaps) else 0.0,
               "early_users": list(e["user_hash"].unique()), "early_text": " ".join(e["text"].astype(str)),
               **sentiment_metrics(e)}
        rec = tree_metrics(reconstructed_parents(recs)[0])
        two = tree_metrics(two_level_parents(recs))
        row.update({"max_depth_rec": rec["max_depth"], "structural_virality_rec": rec["structural_virality"],
                    "max_depth_2l": two["max_depth"], "structural_virality_2l": two["structural_virality"]})
        rows.append(row)
    f = pd.DataFrame(rows).merge(a, on="article_id", how="left")
    f["title_words"] = f["title"].fillna("").str.split().str.len()
    f["hour"] = f["published"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.hour
    for cat in CATEGORIES:
        f[f"cat_{cat}"] = (f["category"] == cat).astype(int)
    return f


def graph_features(f: pd.DataFrame, c: pd.DataFrame, train_ids: set) -> pd.DataFrame:
    """Frequent and central users and centralities from the graphs of the training articles only."""
    tc = c[c["article_id"].isin(train_ids)]
    users, _ = user_graph.user_table(tc)
    frequent = set(users.loc[users["is_frequent"], "user_hash"])
    g = reply_graph.build_graph(reply_graph.reply_edges(tc))
    pr = nx.pagerank(g, alpha=reply_graph.DAMPING, weight="weight")  # same settings as reply_graph.centrality,
    au = nx.hits(g, max_iter=1000, normalized=True)[1]               # without the slow betweenness
    cut = pd.Series(pr).quantile(1 - reply_graph.CENTRAL_TOP_SHARE)
    central = {u for u, v in pr.items() if v >= cut}
    g = pd.DataFrame({
        "share_frequent": [np.mean([u in frequent for u in us]) for us in f["early_users"]],
        "share_central": [np.mean([u in central for u in us]) for us in f["early_users"]],
        "mean_pagerank": [np.mean([pr.get(u, 0.0) for u in us]) for us in f["early_users"]],
        "mean_authority": [np.mean([au.get(u, 0.0) for u in us]) for us in f["early_users"]],
    }, index=f.index)
    return pd.concat([f.drop(columns=[x for x in g.columns if x in f.columns]), g], axis=1)


def time_split(f: pd.DataFrame) -> np.ndarray:
    order = f["published"].rank(method="first").to_numpy()
    return order <= int(round(TRAIN_SHARE * len(f)))


def random_split(f: pd.DataFrame, y_hint: np.ndarray) -> np.ndarray:
    idx = np.arange(len(f))
    tr, _ = train_test_split(idx, train_size=TRAIN_SHARE, random_state=SEED, stratify=y_hint)
    mask = np.zeros(len(f), bool)
    mask[tr] = True
    return mask


def prepare(a, c, k: int, split: str = "time", min_comments: int | None = None, tree: str = "rec",
            label: str = "combined", top: float = TOP, label_after_k: bool = False
            ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Feature frame, label vector and train mask for one configuration. With label_after_k the label uses
    only the comments after the first k (articles need at least k + LATER_MIN comments), so the label
    does not share comments with the features."""
    f = early_features(a, c, k, max(min_comments or 0, k + LATER_MIN if label_after_k else 0))
    f = f.merge(thread_labels(c, k if label_after_k else 0), on="article_id", how="left").reset_index(drop=True)
    f["max_depth"], f["structural_virality"] = f[f"max_depth_{tree}"], f[f"structural_virality_{tree}"]
    if split == "time":
        train = time_split(f)
    else:  # stratify the random split on the combined label computed on all rows
        hint = make_label(f, np.ones(len(f), bool), label, top)
        train = random_split(f, hint)
    f = graph_features(f, c, set(f.loc[train, "article_id"]))
    y = make_label(f, train, label, top)
    return f, y, train


# ------------------------------------------------------------------------------------------- models

def tabular(name: str, seed: int = SEED):
    if name == "lr":
        return (Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
                {"clf__C": [0.01, 0.1, 1, 10]})
    if name == "rf":
        return (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
                {"min_samples_leaf": [1, 5], "max_features": ["sqrt", 0.5]})
    if name == "hgb":
        return (HistGradientBoostingClassifier(random_state=seed),
                {"learning_rate": [0.03, 0.1], "max_depth": [3, None]})
    raise ValueError(name)


def cv() -> StratifiedKFold:
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def fit_predict(est, grid, x_tr, y_tr, x_te) -> tuple[np.ndarray, float, dict]:
    """Grid search by CV AUC on train; F1 threshold from out-of-fold train scores; test probabilities."""
    search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv(), n_jobs=1).fit(x_tr, y_tr)
    oof = cross_val_predict(clone(search.best_estimator_), x_tr, y_tr, cv=cv(), method="predict_proba")[:, 1]
    return search.best_estimator_.predict_proba(x_te)[:, 1], f1_threshold(y_tr, oof), {
        "params": search.best_params_, "cv_auc": round(float(search.best_score_), 4), "fitted": search.best_estimator_}


def evaluate(y, score, pred) -> dict:
    y, score, pred = np.asarray(y), np.asarray(score, float), np.asarray(pred)
    m = {"auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
         "f1": f1_score(y, pred, zero_division=0), "predicted_positive_share": pred.mean()}
    rng = np.random.default_rng(SEED)
    boot = {"auc": [], "pr_auc": [], "f1": []}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        boot["auc"].append(roc_auc_score(y[i], score[i]))
        boot["pr_auc"].append(average_precision_score(y[i], score[i]))
        boot["f1"].append(f1_score(y[i], pred[i], zero_division=0))
    out = {k: round(float(v), 3) for k, v in m.items()}
    for k, v in boot.items():
        out[f"{k}_ci95"] = [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return out


def run_tabular(name, f, y, train, cols) -> tuple[dict, np.ndarray, object]:
    """Seeds averaged for stochastic models; returns metrics, test scores and the seed-42 model."""
    x_tr, x_te = f.loc[train, cols].to_numpy(float), f.loc[~train, cols].to_numpy(float)
    seeds = [SEED] if name == "lr" else SEEDS
    scores, thr, per_seed, info0 = [], [], [], None
    for s in seeds:
        est, grid = tabular(name, s)
        sc, t, info = fit_predict(est, grid, x_tr, y[train], x_te)
        scores.append(sc)
        thr.append(t)
        per_seed.append(evaluate(y[~train], sc, (sc >= t).astype(int)))
        info0 = info0 or info
    score, t = np.mean(scores, axis=0), float(np.mean(thr))
    m = evaluate(y[~train], score, (score >= t).astype(int))
    m.update({"threshold": round(t, 4), "cv_auc": info0["cv_auc"], "params": json.dumps(info0["params"])})
    if len(seeds) > 1:
        for k in ("auc", "pr_auc", "f1"):
            v = [p[k] for p in per_seed]
            m[f"{k}_seed_mean_std"] = f"{np.mean(v):.3f} +- {np.std(v):.3f}"
    return m, score, info0["fitted"]


def run_text(f, y, train, with_features: bool) -> tuple[dict, np.ndarray]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
    tr_text, te_text = f.loc[train, "early_text"], f.loc[~train, "early_text"]
    x_tr, x_te = vec.fit_transform(tr_text), vec.transform(te_text)
    if with_features:
        sc = StandardScaler().fit(f.loc[train, ALL].to_numpy(float))
        x_tr = hstack([x_tr, csr_matrix(sc.transform(f.loc[train, ALL].to_numpy(float)))]).tocsr()
        x_te = hstack([x_te, csr_matrix(sc.transform(f.loc[~train, ALL].to_numpy(float)))]).tocsr()
    est, grid = LogisticRegression(max_iter=5000), {"C": [0.1, 1, 10]}
    score, t, info = fit_predict(est, grid, x_tr, y[train], x_te)
    m = evaluate(y[~train], score, (score >= t).astype(int))
    m.update({"threshold": round(t, 4), "cv_auc": info["cv_auc"], "params": json.dumps(info["params"])})
    return m, score


def run_baseline(feature, f, y, train) -> tuple[dict, np.ndarray]:
    """A single feature used as the score; its sign and F1 threshold are chosen on train."""
    tr = f.loc[train, feature].to_numpy(float)
    sign = 1.0 if roc_auc_score(y[train], tr) >= 0.5 else -1.0
    score = sign * f.loc[~train, feature].to_numpy(float)
    t = f1_threshold(y[train], sign * tr)
    m = evaluate(y[~train], score, (score >= t).astype(int))
    m.update({"sign": int(sign)})
    return m, score


# ------------------------------------------------------------------------------------------- experiments

def ladder(f, y, train) -> tuple[pd.DataFrame, dict, dict]:
    rows, scores, fitted = [], {}, {}
    prev = float(y[train].mean())
    maj = evaluate(y[~train], np.full((~train).sum(), prev), np.zeros((~train).sum(), int))
    rows.append({"level": "baseline", "model": "majority", **maj})
    scores["majority"] = np.full((~train).sum(), prev)
    for name, feat in BASELINES.items():
        m, s = run_baseline(feat, f, y, train)
        rows.append({"level": "baseline", "model": name, **m})
        scores[name] = s
    for name in ("lr", "rf", "hgb"):
        t0 = time.time()
        m, s, est = run_tabular(name, f, y, train, ALL)
        rows.append({"level": "classical", "model": name, **m, "seconds": round(time.time() - t0, 1)})
        scores[name], fitted[name] = s, est
    for name, wf in (("tfidf_lr", False), ("tfidf_plus_features_lr", True)):
        m, s = run_text(f, y, train, wf)
        rows.append({"level": "text", "model": name, **m})
        scores[name] = s
    return pd.DataFrame(rows), scores, fitted


def group_ablation(f, y, train) -> pd.DataFrame:
    configs = {f"only_{g}": cols for g, cols in GROUPS.items()}
    configs["all"] = ALL
    configs.update({f"all_minus_{g}": [x for x in ALL if x not in cols] for g, cols in GROUPS.items()})
    rows = []
    for cfg, cols in configs.items():
        for name in ("lr", "rf", "hgb"):
            m, _, _ = run_tabular(name, f, y, train, cols)
            rows.append({"features": cfg, "model": name, **m})
    return pd.DataFrame(rows)


def design_ablation(a, c) -> pd.DataFrame:
    rows = []

    def add(kind, value, **kw):
        f, y, train = prepare(a, c, **kw)
        for name in ("lr", "rf", "hgb"):
            m, _, _ = run_tabular(name, f, y, train, ALL)
            rows.append({"ablation": kind, "value": value, "model": name, "n_train": int(train.sum()),
                         "n_test": int((~train).sum()), "test_positive_share": round(float(y[~train].mean()), 3),
                         **m})

    for tree in ("rec", "2l"):
        add("tree", tree, k=K, tree=tree)
    for k in (5, 10, 20):
        add("k (articles with >= 20 comments)", k, k=k, min_comments=20)
    for top in (0.20, 0.25, 0.33):
        add("label top share", top, k=K, top=top)
    for definition in ("p_neg", "reply_ratio"):
        add("label definition", definition, k=K, label=definition)
    for split in ("time", "random"):
        add("split", split, k=K, split=split)
    return pd.DataFrame(rows)


def interpretation(f, y, train, fitted) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr = fitted["lr"]
    coef = pd.DataFrame({"feature": ALL, "coef_standardized": lr.named_steps["clf"].coef_[0]})
    coef = coef.reindex(coef["coef_standardized"].abs().sort_values(ascending=False).index)
    rows = []
    for name in ("lr", "hgb"):
        pi = permutation_importance(fitted[name], f.loc[~train, ALL].to_numpy(float), y[~train],
                                    scoring="roc_auc", n_repeats=30, random_state=SEED)
        rows += [{"model": name, "feature": ft, "auc_drop_mean": pi.importances_mean[i],
                  "auc_drop_std": pi.importances_std[i]} for i, ft in enumerate(ALL)]
    imp = pd.DataFrame(rows).sort_values(["model", "auc_drop_mean"], ascending=[True, False])
    return coef, imp


def run(skip_ablations: bool = False, label_after_k: bool = False) -> dict:
    t0 = time.time()
    out = OUT / "label_after_k" if label_after_k else OUT
    a, c = load()
    f, y, train = prepare(a, c, k=K, label_after_k=label_after_k)
    tables = {}
    tables["ladder.csv"], scores, fitted = ladder(f, y, train)
    yt = y[~train]
    best = tables["ladder.csv"].query("level != 'baseline'").sort_values("cv_auc", ascending=False).iloc[0]["model"]
    paired = [{"a": m, "b": "hours_to_k", **paired_bootstrap(yt, scores[m], scores["hours_to_k"], roc_auc_score)}
              for m in scores if m not in ("majority", "hours_to_k")]
    tables["paired_auc.csv"] = pd.DataFrame(paired)
    tables["lr_coefficients.csv"], tables["permutation_importance.csv"] = interpretation(f, y, train, fitted)
    if not skip_ablations:
        tables["ablation_groups.csv"] = group_ablation(f, y, train)
        tables["ablation_design.csv"] = design_ablation(a, c)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.round(4).to_csv(out / name, index=False, encoding=CSV_ENCODING)
    pd.DataFrame({"article_id": f.loc[~train, "article_id"], "y": yt,
                  **{m: np.round(s, 4) for m, s in scores.items()}}).to_csv(out / "test_scores.csv", index=False,
                                                                               encoding=CSV_ENCODING)
    summary = {"label_after_k": label_after_k,
        **run_info(), "hardware": hardware(), "k": K, "top_share": TOP,
        "articles": int(len(f)), "train": int(train.sum()), "test": int((~train).sum()),
        "train_period": [str(f.loc[train, "published"].min()), str(f.loc[train, "published"].max())],
        "test_period": [str(f.loc[~train, "published"].min()), str(f.loc[~train, "published"].max())],
        "train_positive_share": round(float(y[train].mean()), 3), "test_positive_share": round(float(yt.mean()), 3),
        "model_chosen_by_train_cv_auc": best,
        "ladder": tables["ladder.csv"][["model", "auc", "auc_ci95", "pr_auc", "f1", "cv_auc"]].to_dict("records"),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Early controversy prediction")
    p.add_argument("--skip-ablations", action="store_true", help="only the model ladder and interpretation")
    p.add_argument("--label-after-k", action="store_true",
                   help="label from the comments after the first k only (writes results/controversy/label_after_k/)")
    a = p.parse_args(argv)
    print(json.dumps(run(a.skip_ablations, a.label_after_k), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
