"""Clickbait model ladder on the dev set.

Levels run locally: baseline (majority class, keyword rules) and classical (TF-IDF + Logistic
Regression, Linear SVM, Complement Naive Bayes). PhoBERT and Qwen run on Kaggle
(notebooks/clickbait_*.ipynb) and report on the same dev items.

Data: data/tasks/clickbait/ (common.py). Training: train.csv (ViClickbait-2025, 3,414 headlines from 8
news sites). Evaluation: dev.csv (labelled VnExpress headlines). Models trained only on ViClickbait are
scored on all of dev; a model that also trains on dev is scored out of fold.

Ablations (classical level, best model family):
    input     title only vs title + lead paragraph (sapo)
    training  ViClickbait (all sources) vs its VnExpress part only vs ViClickbait + VnExpress dev
              (scored out of fold: 5-fold CV over dev, the scored fold never in training)

Usage (from the project root):
    python -m src.classifiers.clickbait ladder         # local levels (about 10 minutes on CPU)
    python -m src.classifiers.clickbait kaggle-bundle  # input files for the Kaggle notebooks
    python -m src.classifiers.clickbait collect        # merge Kaggle outputs into one table
    python -m src.classifiers.clickbait review --round N --config dev_scores_<config>.csv  # items to check
    python -m src.classifiers.clickbait apply-labels --round N  # write notes/clickbait_review/round_N.csv into dev
    python -m src.classifiers.clickbait extras [--family tfidf_lr]  # extra baselines and ablations (a few minutes)
    python -m src.classifiers.clickbait final --config tfidf_lr [--segment] [--balanced] [--threshold 0.5|cv|prior]
    python -m src.classifiers.clickbait final --from-file results/clickbait/kaggle/corpus_scores_<config>.csv
Output: results/clickbait/ (metrics.json, ladder.csv, ablation.csv, ladder_full.csv, and one
dev_scores_<config>.csv per configuration with columns id, pred, score). Kaggle outputs use the same
format and go to results/clickbait/kaggle/.

extras (results/clickbait/extras/: metrics.json, extras.csv, dev_scores_<config>.csv): stratified random
guess, handcrafted features + LR, decision threshold (0.5 / F1-optimal on out-of-fold ViClickbait scores /
prior shift to the VnExpress part of ViClickbait), word segmentation (pyvi), class_weight, learning curve
over the dev training folds, dev-only training. Wall-clock seconds and hardware are recorded.

final: labels for all of corpus.csv in data/tasks/clickbait/corpus_labels.csv (id, label, score), retrained on
ViClickbait + all of dev; dev headlines (matched to corpus rows by exact title) keep their hand-corrected
label. Run log in results/clickbait/final.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .common import CSV_ENCODING, RESULTS_DIR, ROOT, SEED, binary_metrics, bootstrap_ci, load_corpus, load_dev, load_train, run_info, write_results


# Hand-written keyword rules for curiosity gaps and sensational wording, not tuned on dev.
KEYWORDS = [
    r"\?", r"!", r"\bsốc\b", r"bất ngờ", r"\blý do\b", r"\bvì sao\b", r"\btại sao\b", r"\bđiều gì\b",
    r"\bđiều này\b", r"\bngười này\b", r"\bbí mật\b", r"\bsự thật\b", r"\bchấn động\b", r"\brúng động\b",
    r"\bxôn xao\b", r"\bgây bão\b", r"\bai ngờ\b", r"\bhé lộ\b", r"\bngỡ ngàng\b", r"\bchoáng\b",
    r"\brùng mình\b", r"\bbật khóc\b", r"không thể tin", r"\bbí ẩn\b", r"\bcực hot\b", r"\bkhủng\b",
    r"\bnóng bỏng\b", r"\bác mộng\b", r"\btan hoang\b", r"\bđiều bạn cần biết\b",
]
KEYWORD_RE = [re.compile(k, re.IGNORECASE) for k in KEYWORDS]


def keyword_score(texts: pd.Series) -> np.ndarray:
    return np.array([sum(bool(r.search(t)) for r in KEYWORD_RE) for t in texts], dtype=float)


def load_viclickbait() -> pd.DataFrame:
    df = load_train("clickbait")
    df["y"] = df["label"].astype(int)
    return df


def text_of(df: pd.DataFrame, use_sapo: bool) -> pd.Series:
    title = df["title"].fillna("").astype(str)
    return (title + " . " + df["sapo"].fillna("").astype(str)) if use_sapo else title


def features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


MODELS = {
    "tfidf_lr": (LogisticRegression(max_iter=2000, random_state=SEED), {"clf__C": [0.3, 1, 3, 10]}),
    "tfidf_svm": (LinearSVC(random_state=SEED), {"clf__C": [0.03, 0.1, 0.3, 1]}),
    "tfidf_nb": (ComplementNB(), {"clf__alpha": [0.1, 0.3, 1.0]}),
}


def fit(name: str, x: pd.Series, y: np.ndarray):
    """Grid search by 5-fold CV on the training data only (F1 of the clickbait class)."""
    clf, grid = MODELS[name]
    pipe = Pipeline([("feat", features()), ("clf", clone(clf))])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=1)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def score_and_pred(model, x: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(x)[:, 1]
        return s, (s >= 0.5).astype(int)
    s = model.decision_function(x)
    return s, (s >= 0).astype(int)


def evaluate(y, score, pred) -> dict:
    m = binary_metrics(y, score, pred)
    m.update(bootstrap_ci(y, score, pred))
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}


def out_of_fold_with_dev(name: str, params: dict, train: pd.DataFrame, dev: pd.DataFrame, use_sapo: bool):
    """ViClickbait + 4/5 of dev for training, the remaining fifth scored; repeated over 5 folds."""
    clf, _ = MODELS[name]
    score, pred = np.zeros(len(dev)), np.zeros(len(dev), dtype=int)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in folds.split(dev, dev["y"]):
        x = pd.concat([text_of(train, use_sapo), text_of(dev.iloc[tr], use_sapo)])
        y = np.concatenate([train["y"].to_numpy(), dev["y"].to_numpy()[tr]])
        pipe = Pipeline([("feat", features()), ("clf", clone(clf))])
        pipe.set_params(**params).fit(x, y)
        score[te], pred[te] = score_and_pred(pipe, text_of(dev.iloc[te], use_sapo))
    return score, pred


def run() -> dict:
    vc = load_viclickbait()
    dev = load_dev("clickbait")
    dev["y"] = dev["label"].astype(int)
    y = dev["y"].to_numpy()
    titles = load_corpus("clickbait")

    rows, per_config, fitted = [], {}, {}

    def add(level, model, train_name, use_sapo, score, pred, extra=None):
        r = {"level": level, "model": model, "train": train_name, "input": "title+sapo" if use_sapo else "title"}
        r.update(evaluate(y, score, pred))
        r.update(extra or {})
        rows.append(r)
        name = f"{model}_{train_name}_{r['input']}".replace("+", "_plus_").replace("(oof)", "_oof").replace("-", "")
        per_config[f"dev_scores_{name}.csv"] = pd.DataFrame({"id": dev["id"], "pred": pred, "score": np.round(score, 4)})

    # Baselines
    add("baseline", "majority", "-", False, np.zeros(len(dev)), np.zeros(len(dev), dtype=int))
    kw = keyword_score(dev["title"])
    add("baseline", "keywords", "-", False, kw, (kw >= 1).astype(int))

    # Classical, trained on all of ViClickbait, title only (main setting) and title + sapo
    for use_sapo in (False, True):
        for name in MODELS:
            model, params, cv_f1 = fit(name, text_of(vc, use_sapo), vc["y"].to_numpy())
            s, p = score_and_pred(model, text_of(dev, use_sapo))
            add("classical", name, "viclickbait", use_sapo, s, p,
                {"params": params, "viclickbait_cv_f1": round(cv_f1, 3)})
            fitted[(name, use_sapo)] = (model, params)
    ladder = pd.DataFrame(rows)

    # Training-data ablation with the best classical family (by dev PR-AUC, title only)
    main = ladder[(ladder.level == "classical") & (ladder.input == "title")]
    best = main.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    for use_sapo in (False, True):
        model, params = fitted[(best, use_sapo)]
        vne = vc[vc["source"] == "VnExpress"]
        m2, p2, cv2 = fit(best, text_of(vne, use_sapo), vne["y"].to_numpy())
        s, p = score_and_pred(m2, text_of(dev, use_sapo))
        add("ablation", best, "viclickbait_vnexpress_only", use_sapo, s, p, {"params": p2, "viclickbait_cv_f1": round(cv2, 3)})
        s, p = out_of_fold_with_dev(best, params, vc, dev, use_sapo)
        add("ablation", best, "viclickbait+vnexpress_dev(oof)", use_sapo, s, p, {"params": params})

    table = pd.DataFrame(rows)
    ablation = table[(table.model == best) & (table.level.isin(["classical", "ablation"]))]

    # Share of all 1,244 collected headlines predicted clickbait (sanity check against the public model's 84%)
    model, _ = fitted[(best, False)]
    _, p_all = score_and_pred(model, text_of(titles, False))
    share = {"model": best, "train": "viclickbait", "input": "title", "share": round(float(p_all.mean()), 3),
             "n_articles": int(len(titles))}

    payload = {
        **run_info(),
        "task": "clickbait", "eval_set": "data/tasks/clickbait/dev.csv",
        "dev_positive_share": round(float(y.mean()), 3), "dev_n": int(len(y)),
        "viclickbait_n": int(len(vc)), "viclickbait_positive_share": round(float(vc["y"].mean()), 3),
        "best_classical_by_dev_pr_auc": best, "corpus_predicted_share": share,
        "results": table.to_dict(orient="records"),
    }
    write_results("clickbait", payload, {"ladder.csv": table[table.level != "ablation"],
                                             "ablation.csv": ablation, **per_config})
    return payload


KAGGLE_DIR = ROOT / "data" / "kaggle" / "clickbait"
RESULT_DIR = RESULTS_DIR / "clickbait"


def kaggle_bundle() -> None:
    """Files to upload as a private Kaggle dataset: ViClickbait-2025, the dev headlines with their labels and
    all corpus headlines (to label with a Kaggle model). Headlines and lead paragraphs are public article
    text; no comments."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    vc = load_viclickbait()
    vc[["id", "title", "sapo", "source", "y"]].to_csv(KAGGLE_DIR / "viclickbait.csv", index=False)
    dev = load_dev("clickbait")
    dev["y"] = dev["label"].astype(int)
    dev[["id", "title", "sapo", "y"]].to_csv(KAGGLE_DIR / "dev_headlines.csv", index=False)
    corpus = load_corpus("clickbait")
    corpus[["id", "title", "sapo"]].to_csv(KAGGLE_DIR / "corpus_headlines.csv", index=False)
    print(f"wrote {len(vc)} + {len(dev)} + {len(corpus)} rows to {KAGGLE_DIR}")


def kaggle_level(config: str) -> str:
    if config.startswith("phobert") or "bert" in config:
        return "deep_learning"
    if config.startswith("qwen"):
        return "llm"
    return "other"


def _read_dev_scores(path, y_by_id: dict) -> pd.DataFrame:
    k = pd.read_csv(path, dtype={"id": str})
    if len(k) != len(y_by_id) or set(k["id"]) != set(y_by_id):
        raise ValueError(f"{path.name}: item ids do not match the dev set")
    k["y"] = k["id"].map(y_by_id)
    return k


def collect() -> pd.DataFrame:
    """Score every Kaggle output on dev (current labels) with the same metrics, and join with the local
    ladder and the extras table (each only if present).

    Kaggle files (results/clickbait/kaggle/):
        dev_scores_<config>.csv              one row each; level deep_learning (phobert*), llm (qwen*)
        dev_scores_<config>_tuned.csv        optional threshold-tuned variant (threshold chosen on training
                                             data in the notebook); scored like any other file
        dev_scores_<config>_seed<k>.csv      optional per-seed scores: give per-config mean and std of F1 and
                                             PR-AUC re-scored on the current labels (not separate rows)
        phobert_runs.json                    per-seed metrics from the notebook; used for mean and std only
                                             when per-seed score files are absent (labels at Kaggle run time)
    Ensembles: for each phobert dev score file and the local TF-IDF + LR score file (trained on ViClickbait,
    title), the mean of the two probabilities (pred at 0.5) and a rank average (score = mean of the two
    normalised ranks, pred taken from the probability mean)."""
    dev = load_dev("clickbait")
    y_by_id = dict(zip(dev["id"], dev["label"].astype(int)))
    kdir = RESULT_DIR / "kaggle"
    files = sorted(kdir.glob("dev_scores_*.csv")) if kdir.is_dir() else []
    seed_re = re.compile(r"^(?P<config>.+)_seed(?P<seed>\d+)$")
    rows, scores, seed_files = [], {}, {}
    for f in files:
        config = f.stem[len("dev_scores_"):]
        m = seed_re.match(config)
        if m:
            seed_files.setdefault(m["config"], []).append((int(m["seed"]), f))
            continue
        k = _read_dev_scores(f, y_by_id)
        scores[config] = k.set_index("id")
        note = "threshold tuned on training data (see notebook)" if config.endswith("_tuned") else ""
        rows.append({"level": kaggle_level(config), "model": config, "train": "see notebook", "input": "title",
                     **evaluate(k["y"].to_numpy(), k["score"].to_numpy(), k["pred"].to_numpy()), "note": note})

    # Mean and std over seeds
    seed_stats = {}
    for config, fs in seed_files.items():
        f1s, aps = [], []
        for _, f in sorted(fs):
            k = _read_dev_scores(f, y_by_id)
            mm = binary_metrics(k["y"], k["score"], k["pred"])
            f1s.append(mm["f1"]); aps.append(mm["pr_auc"])
        seed_stats[config] = {"seeds": len(fs), "f1_seed_mean": round(float(np.mean(f1s)), 3),
                              "f1_seed_std": round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 3),
                              "pr_auc_seed_mean": round(float(np.mean(aps)), 3),
                              "pr_auc_seed_std": round(float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0, 3),
                              "seed_source": "per-seed score files, current dev labels"}
    runs_path = kdir / "phobert_runs.json"
    if runs_path.exists():
        runs = pd.DataFrame(json.loads(runs_path.read_text(encoding="utf-8")).get("runs", []))
        if len(runs):
            for config, g in runs.groupby("config"):
                if config in seed_stats:
                    continue
                seed_stats[config] = {"seeds": len(g), "f1_seed_mean": round(float(g["f1"].mean()), 3),
                                      "f1_seed_std": round(float(g["f1"].std(ddof=1)) if len(g) > 1 else 0.0, 3),
                                      "pr_auc_seed_mean": round(float(g["pr_auc"].mean()), 3),
                                      "pr_auc_seed_std": round(float(g["pr_auc"].std(ddof=1)) if len(g) > 1 else 0.0, 3),
                                      "seed_source": "phobert_runs.json (dev labels at Kaggle run time, not re-scored)"}
    for r in rows:
        r.update(seed_stats.get(r["model"], {}))
    for config, st in seed_stats.items():
        if config not in scores:
            rows.append({"level": kaggle_level(config), "model": config, "train": "see notebook", "input": "title",
                         **st, "note": "seed summary only (no mean-score file)"})

    # Ensembles of TF-IDF + LR (ViClickbait, title) with each PhoBERT configuration
    tfidf_path = next((p for p in [RESULT_DIR / "dev_scores_tfidf_lr_viclickbait_title.csv",
                                   RESULT_DIR / "extras" / "dev_scores_tfidf_lr_thr_0.5.csv"] if p.exists()), None)
    if tfidf_path is not None:
        t = _read_dev_scores(tfidf_path, y_by_id).set_index("id")
        for config, k in scores.items():
            if not config.startswith("phobert") or config.endswith("_tuned"):
                continue
            k = k.loc[t.index]
            y = t["y"].to_numpy()
            mean = (t["score"].to_numpy() + k["score"].to_numpy()) / 2
            pred = (mean >= 0.5).astype(int)
            rank = (pd.Series(t["score"].to_numpy()).rank(pct=True).to_numpy()
                    + pd.Series(k["score"].to_numpy()).rank(pct=True).to_numpy()) / 2
            for variant, sc in (("mean", mean), ("rank", rank)):
                rows.append({"level": "ensemble", "model": f"tfidf_lr+{config}({variant})", "train": "viclickbait",
                             "input": "title", **evaluate(y, sc, pred),
                             "note": f"{tfidf_path.name} + dev_scores_{config}.csv; pred = probability mean >= 0.5"})

    parts = []
    ladder_path = RESULT_DIR / "ladder.csv"
    if ladder_path.exists():
        parts.append(pd.read_csv(ladder_path, encoding="utf-8-sig"))
    else:
        print(f"note: {ladder_path} not found; run the ladder action to include the local levels")
    extras_path = RESULT_DIR / "extras" / "extras.csv"
    if extras_path.exists():
        parts.append(pd.read_csv(extras_path, encoding="utf-8-sig"))
    if rows:
        parts.append(pd.DataFrame(rows))
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["level", "model", "train", "input", "f1", "f1_ci95", "pr_auc", "pr_auc_ci95", "predicted_positive_share"])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULT_DIR / "ladder_full.csv", index=False, encoding="utf-8-sig")
    return full


# ---------------------------------------------------------------------------------------------------------
# Extra baselines and ablations (TASK.md section 5, "Khuyến khích thử thêm")

EXTRAS_DIR = RESULT_DIR / "extras"
CORPUS_LABELS_PATH = ROOT / "data" / "tasks" / "clickbait" / "corpus_labels.csv"
QUOTES = "'\"\u2018\u2019\u201c\u201d"


def hardware() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}


def segment(texts: pd.Series) -> pd.Series:
    """Vietnamese word segmentation with pyvi (multi-syllable words joined by "_")."""
    from pyvi import ViTokenizer
    return texts.map(ViTokenizer.tokenize)


def handcrafted(texts) -> np.ndarray:
    """Length in characters and words, has ? and !, digit count, quote marks, colon, share of capitalised
    words, number of keyword rules matched."""
    texts = pd.Series(list(texts), dtype=str)
    rows = []
    for t in texts:
        words = t.split()
        rows.append([len(t), len(words), "?" in t, "!" in t, sum(c.isdigit() for c in t),
                     sum(t.count(q) for q in QUOTES), ":" in t,
                     (sum(w[:1].isupper() for w in words) / len(words)) if words else 0.0])
    return np.column_stack([np.array(rows, dtype=float), keyword_score(texts)])


def make_pipe(family: str, balanced: bool = False) -> Pipeline:
    clf = clone(MODELS[family][0])
    if balanced:
        if "class_weight" not in clf.get_params():
            raise ValueError(f"{family} does not support class_weight")
        clf.set_params(class_weight="balanced")
    return Pipeline([("feat", features()), ("clf", clf)])


def grid_fit(pipe: Pipeline, grid: dict, x, y):
    """Same search as fit(): 5-fold CV on the training data only, F1 of the clickbait class."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=1)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def raw_score(model, x) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)


def default_threshold(model) -> float:
    return 0.5 if hasattr(model, "predict_proba") else 0.0


def oof_train_scores(pipe: Pipeline, x, y) -> np.ndarray:
    """Out-of-fold scores on the training data (5-fold), for choosing a threshold inside training."""
    method = "predict_proba" if hasattr(pipe, "predict_proba") else "decision_function"
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    s = cross_val_predict(pipe, x, y, cv=cv, method=method, n_jobs=1)
    return s[:, 1] if s.ndim == 2 else s


def f1_threshold(y, s) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    return float(t[int(np.argmax(f1))])


def prior_shift(p: np.ndarray, src: float, tgt: float) -> np.ndarray:
    """Rescale probabilities from training prior src to target prior tgt (odds adjustment as in Saerens et
    al. 2002, with the target prior given rather than estimated by EM)."""
    a = (tgt / src) * p
    b = ((1 - tgt) / (1 - src)) * (1 - p)
    return a / (a + b)


def extras(family: str = "tfidf_lr") -> dict:
    vc = load_viclickbait()
    dev = load_dev("clickbait")
    dev["y"] = dev["label"].astype(int)
    y = dev["y"].to_numpy()
    yv = vc["y"].to_numpy()
    xv, xd = text_of(vc, False), text_of(dev, False)
    rate = float(yv.mean())
    vne_rate = float(vc.loc[vc["source"] == "VnExpress", "y"].mean())
    grid = MODELS[family][1]
    rows, per_config = [], {}

    def add(level, group, config, variant, score, pred, seconds, extra=None, save=True):
        extra = dict(extra or {})
        r = {"level": level, "group": group, "model": config, "variant": variant,
             "train": extra.pop("train", "viclickbait"), "input": "title"}
        r.update(evaluate(y, score, pred))
        r["seconds"] = round(float(seconds), 1)
        r.update(extra)
        rows.append(r)
        if save:
            per_config[f"dev_scores_{config}.csv"] = pd.DataFrame({"id": dev["id"], "pred": np.asarray(pred, int),
                                                                   "score": np.round(score, 4)})
        print(f"  {config:28s} f1={r['f1']:.3f} pr_auc={r['pr_auc']:.3f} share={r['predicted_positive_share']:.3f}"
              f" ({r['seconds']}s)", flush=True)

    # a. Baselines
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    s = rng.random(len(dev))
    add("baseline", "baseline", "random_stratified", f"pred = U(0,1) < {rate:.3f} (ViClickbait rate)",
        s, (s < rate).astype(int), time.time() - t0, {"train": "-"})

    t0 = time.time()
    hand = Pipeline([("feat", FunctionTransformer(handcrafted)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    m, params, cv_f1 = grid_fit(hand, {"clf__C": [0.01, 0.1, 1, 10]}, xv, yv)
    s, p = score_and_pred(m, xd)
    add("baseline", "baseline", "handcrafted_lr", "9 handcrafted features + StandardScaler + LR", s, p,
        time.time() - t0, {"params": params, "viclickbait_cv_f1": round(cv_f1, 3)})

    # b. Decision threshold (one fit, three rules)
    t0 = time.time()
    base, bparams, bcv = grid_fit(make_pipe(family), grid, xv, yv)
    fit_sec = time.time() - t0
    s0 = raw_score(base, xd)
    thr0 = default_threshold(base)
    add("ablation", "threshold", f"{family}_thr_0.5", "threshold 0.5 (0 for SVM margin)", s0, (s0 >= thr0).astype(int),
        fit_sec, {"params": bparams, "viclickbait_cv_f1": round(bcv, 3), "threshold": thr0})
    t0 = time.time()
    oof = oof_train_scores(clone(base), xv, yv)
    thr = f1_threshold(yv, oof)
    add("ablation", "threshold", f"{family}_thr_cv", "F1-optimal threshold on out-of-fold ViClickbait scores",
        s0, (s0 >= thr).astype(int), fit_sec + time.time() - t0, {"params": bparams, "threshold": round(thr, 4)})
    if hasattr(base, "predict_proba"):
        sp = prior_shift(s0, rate, vne_rate)
        add("ablation", "threshold", f"{family}_thr_prior",
            f"prior shift {rate:.3f} -> {vne_rate:.3f} (VnExpress part of ViClickbait), threshold 0.5",
            sp, (sp >= 0.5).astype(int), fit_sec,
            {"params": bparams, "threshold": 0.5, "equivalent_raw_threshold": round(float(
                prior_shift(np.array([0.5]), vne_rate, rate)[0]), 4)})

    # c. Word segmentation
    t0 = time.time()
    xv_seg, xd_seg = segment(xv), segment(xd)
    mseg, pseg, cvseg = grid_fit(make_pipe(family), grid, xv_seg, yv)
    s = raw_score(mseg, xd_seg)
    add("ablation", "segmentation", f"{family}_seg", "pyvi word segmentation", s,
        (s >= default_threshold(mseg)).astype(int), time.time() - t0, {"params": pseg, "viclickbait_cv_f1": round(cvseg, 3)})
    add("ablation", "segmentation", f"{family}_thr_0.5", "no segmentation (same model as threshold 0.5)", s0,
        (s0 >= thr0).astype(int), fit_sec, {"params": bparams}, save=False)

    # d. Class balancing
    if "class_weight" in MODELS[family][0].get_params():
        t0 = time.time()
        mbal, pbal, cvbal = grid_fit(make_pipe(family, balanced=True), grid, xv, yv)
        s = raw_score(mbal, xd)
        add("ablation", "class_weight", f"{family}_balanced", 'class_weight="balanced"', s,
            (s >= default_threshold(mbal)).astype(int), time.time() - t0, {"params": pbal, "viclickbait_cv_f1": round(cvbal, 3)})
        add("ablation", "class_weight", f"{family}_thr_0.5", "class_weight=None (same model as threshold 0.5)", s0,
            (s0 >= thr0).astype(int), fit_sec, {"params": bparams}, save=False)

    # e. Learning curve: ViClickbait + a share of the dev training folds, scored out of fold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev, y))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t0 = time.time()
        if frac == 0:
            score, pred = s0, (s0 >= thr0).astype(int)
            secs = fit_sec
        else:
            score, pred = np.zeros(len(dev)), np.zeros(len(dev), dtype=int)
            n_used = []
            for tr, te in folds:
                if frac < 1:
                    tr, _ = train_test_split(tr, train_size=frac, stratify=y[tr], random_state=SEED)
                n_used.append(len(tr))
                pipe = make_pipe(family).set_params(**bparams)
                pipe.fit(pd.concat([xv, xd.iloc[tr]]), np.concatenate([yv, y[tr]]))
                score[te] = raw_score(pipe, xd.iloc[te])
                pred[te] = (score[te] >= default_threshold(pipe)).astype(int)
            secs = time.time() - t0
        add("ablation", "learning_curve", f"{family}_lc_{int(frac * 100):03d}",
            f"ViClickbait + {int(frac * 100)}% of dev training folds (oof)", score, pred, secs,
            {"params": bparams, "train": f"viclickbait+{int(frac * 100)}%dev(oof)",
             "dev_train_items_per_fold": 0 if frac == 0 else int(np.mean(n_used))})

    # f. dev only, out of fold, hyperparameters by inner CV in each training fold
    t0 = time.time()
    score, pred = np.zeros(len(dev)), np.zeros(len(dev), dtype=int)
    fold_params = []
    for tr, te in folds:
        m, fp, _ = grid_fit(make_pipe(family), grid, xd.iloc[tr], y[tr])
        score[te] = raw_score(m, xd.iloc[te])
        pred[te] = (score[te] >= default_threshold(m)).astype(int)
        fold_params.append(fp)
    add("ablation", "training_data", f"{family}_devonly_oof", "dev only (oof, inner 5-fold CV for C)", score, pred,
        time.time() - t0, {"params": fold_params, "train": "dev(oof)"})

    table = pd.DataFrame(rows)
    payload = {
        **run_info(), "task": "clickbait", "eval_set": "data/tasks/clickbait/dev.csv", "family": family,
        "hardware": hardware(), "dev_n": int(len(y)), "dev_positive_share": round(float(y.mean()), 3),
        "viclickbait_positive_share": round(rate, 3), "viclickbait_vnexpress_positive_share": round(vne_rate, 3),
        "notes": ["seconds: wall-clock for the configuration; rows that reuse the threshold-0.5 model include its fit time",
                  "threshold rows share one model: PR-AUC identical for 0.5 and cv; the prior shift is monotonic so "
                  "ranking (PR-AUC) is unchanged up to ties"],
        "results": table.to_dict(orient="records"),
    }
    write_results("clickbait/extras", payload, {"extras.csv": table, **per_config})
    return payload


# ---------------------------------------------------------------------------------------------------------
# Final labels for the whole corpus (TASK.md section 4, "Bước cuối")

def _dev_rows_in_corpus(corpus: pd.DataFrame, dev: pd.DataFrame) -> pd.Series:
    """Map each dev headline to its corpus row by exact title; returns dev label indexed by corpus id."""
    matched = corpus[corpus["title"].isin(set(dev["title"]))]
    dup = matched["title"][matched["title"].duplicated(keep=False)]
    if len(dup):
        raise ValueError(f"{dup.nunique()} dev titles occur more than once in corpus.csv: {sorted(set(dup))[:5]}")
    if dev["title"].duplicated().any():
        raise ValueError("duplicate titles in dev.csv")
    missing = set(dev["title"]) - set(matched["title"])
    if missing:
        raise ValueError(f"{len(missing)} dev titles not found in corpus.csv: {sorted(missing)[:5]}")
    label_by_title = dict(zip(dev["title"], dev["label"]))
    return pd.Series(matched["title"].map(label_by_title).to_numpy(), index=matched["id"].to_numpy())


def final(config: str | None = None, from_file: str | None = None, use_segment: bool = False,
          balanced: bool = False, threshold: str = "0.5") -> dict:
    t0 = time.time()
    corpus = load_corpus("clickbait")
    dev = load_dev("clickbait")
    info = {"threshold_rule": threshold}
    if from_file:
        k = pd.read_csv(from_file, dtype={"id": str})
        if not {"id", "score", "pred"} <= set(k.columns):
            raise ValueError(f"{from_file}: needs columns id, score, pred")
        if len(k) != len(corpus) or k["id"].duplicated().any() or set(k["id"]) != set(corpus["id"]):
            raise ValueError(f"{from_file}: ids do not match corpus.csv exactly")
        k = k.set_index("id").loc[corpus["id"]]
        score, pred = k["score"].to_numpy(float), k["pred"].astype(int).to_numpy()
        info.update({"config": os.path.basename(from_file), "source_file": str(from_file), "threshold_rule": "from file"})
    else:
        if config not in MODELS:
            raise ValueError(f"--config must be one of {list(MODELS)} (or use --from-file)")
        vc = load_viclickbait()
        x = pd.concat([text_of(vc, False), text_of(dev, False)], ignore_index=True)
        y = np.concatenate([vc["y"].to_numpy(), dev["label"].astype(int).to_numpy()])
        xc = text_of(corpus, False)
        if use_segment:
            x, xc = segment(x), segment(xc)
        model, params, cv_f1 = grid_fit(make_pipe(config, balanced), MODELS[config][1], x, y)
        score = raw_score(model, xc)
        thr = default_threshold(model)
        if threshold == "cv":
            thr = f1_threshold(y, oof_train_scores(clone(model), x, y))
        elif threshold == "prior":
            if not hasattr(model, "predict_proba"):
                raise ValueError("--threshold prior needs probabilities (not tfidf_svm)")
            vne_rate = float(vc.loc[vc["source"] == "VnExpress", "y"].mean())
            score = prior_shift(score, float(y.mean()), vne_rate)
            info.update({"prior_from": round(float(y.mean()), 4), "prior_to": round(vne_rate, 4)})
        pred = (score >= thr).astype(int)
        info.update({"config": config, "segment": use_segment, "balanced": balanced, "params": params,
                     "train_cv_f1": round(cv_f1, 3), "threshold": round(float(thr), 4), "train_n": int(len(y)),
                     "score": "probability" if hasattr(model, "predict_proba") else "SVM decision function"})

    out = pd.DataFrame({"id": corpus["id"].to_numpy(), "label": pred.astype(int).astype(str), "score": np.round(score, 4)})
    dev_labels = _dev_rows_in_corpus(corpus, dev)
    is_dev = out["id"].isin(dev_labels.index).to_numpy()
    agree = float((out.loc[is_dev, "label"].to_numpy() == out.loc[is_dev, "id"].map(dev_labels).to_numpy()).mean())
    out.loc[is_dev, "label"] = out.loc[is_dev, "id"].map(dev_labels).to_numpy()
    out.to_csv(CORPUS_LABELS_PATH, index=False, encoding=CSV_ENCODING)
    info.update({
        "n": int(len(out)), "n_dev_matched": int(is_dev.sum()),
        "model_positive_share_all": round(float(pred.mean()), 4),
        "label_positive_share_all": round(float((out["label"] == "1").mean()), 4),
        "model_positive_share_non_dev": round(float(pred[~is_dev].mean()), 4),
        "model_positive_share_dev_rows": round(float(pred[is_dev].mean()), 4),
        "dev_label_positive_share": round(float((dev["label"] == "1").mean()), 4),
        "model_agrees_with_dev_label": round(agree, 4),
        "seconds": round(time.time() - t0, 1), "hardware": hardware(), "output": str(CORPUS_LABELS_PATH.relative_to(ROOT)),
    })
    payload = {**run_info(), "task": "clickbait", **info}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "final.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


REVIEW_DIR = ROOT / "notes" / "clickbait_review"
DEV_PATH = ROOT / "data" / "tasks" / "clickbait" / "dev.csv"


def checked_ids() -> set:
    """Ids already reviewed in earlier rounds (notes/clickbait_review/round_<n>.csv)."""
    ids = set()
    for f in REVIEW_DIR.glob("round_*.csv"):
        if re.fullmatch(r"round_\d+", f.stem):
            ids |= set(pd.read_csv(f, dtype=str, encoding=CSV_ENCODING, keep_default_na=False)["id"])
    return ids


def review(round_no: int, config: str, n: int = 10) -> pd.DataFrame:
    """Pick items to check by hand (TASK.md section 4): the n items the model is most sure are clickbait,
    the n it is most sure are not, and the n closest to the threshold that gives a predicted positive share
    equal to the label share. Items checked in earlier rounds are skipped; the three groups are disjoint.
    Writes round_<n>_pick.csv (id, group, score) and round_<n>_blind.csv (id, title, sapo; no label, no
    group, shuffled) for labelling without seeing the old label or the model's opinion."""
    dev = load_dev("clickbait")
    sc = pd.read_csv(RESULT_DIR / config, dtype={"id": str})
    score = pd.Series(sc["score"].to_numpy(float), index=sc["id"])
    s = score.drop(list(checked_ids() & set(score.index)))
    t = s.quantile(1 - (dev["label"] == "1").mean())
    top = s.nlargest(n).index
    bottom = s.drop(top).nsmallest(n).index
    rest = s.drop(top.union(bottom))
    near = (rest - t).abs().nsmallest(n).index
    pick = pd.concat([pd.DataFrame({"id": top, "group": "sure_1"}), pd.DataFrame({"id": bottom, "group": "sure_0"}),
                      pd.DataFrame({"id": near, "group": "uncertain"})], ignore_index=True)
    pick["score"] = pick["id"].map(score).round(4)
    pick["threshold"] = round(float(t), 4)
    pick["config"] = config
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pick.to_csv(REVIEW_DIR / f"round_{round_no}_pick.csv", index=False, encoding=CSV_ENCODING)
    blind = dev.set_index("id").loc[pick["id"], ["title", "sapo"]].reset_index()
    blind = blind.sample(frac=1, random_state=SEED + round_no)
    blind.to_csv(REVIEW_DIR / f"round_{round_no}_blind.csv", index=False, encoding=CSV_ENCODING)
    return pick


def apply_labels(round_no: int) -> dict:
    """Write the decided labels of round_<n>.csv (id, group, old, new, reason) into dev.csv, changing only
    the label column (pandas round-trips the file byte for byte)."""
    rev = pd.read_csv(REVIEW_DIR / f"round_{round_no}.csv", dtype=str, encoding=CSV_ENCODING, keep_default_na=False)
    dev = load_dev("clickbait")
    before = dev.copy()
    idx = dev.set_index("id").index
    new = dict(zip(rev["id"], rev["new"]))
    if not set(new) <= set(idx) or not set(new.values()) <= {"0", "1"}:
        raise ValueError("unknown id or label in review file")
    old_now = dict(zip(dev["id"], dev["label"]))
    for i, o in zip(rev["id"], rev["old"]):
        if old_now[i] != o:
            raise ValueError(f"{i}: review says old label {o}, dev.csv has {old_now[i]}")
    dev["label"] = [new.get(i, l) for i, l in zip(dev["id"], dev["label"])]
    assert (dev.drop(columns="label") == before.drop(columns="label")).all().all()
    dev.to_csv(DEV_PATH, index=False, encoding=CSV_ENCODING)
    changed = rev[rev["old"] != rev["new"]]
    return {"checked": len(rev), "changed": len(changed),
            "0->1": int(((changed.old == "0") & (changed.new == "1")).sum()),
            "1->0": int(((changed.old == "1") & (changed.new == "0")).sum()),
            "dev_positive": int((dev["label"] == "1").sum())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Clickbait model ladder on the dev set")
    parser.add_argument("action", nargs="?", default="ladder",
                        choices=["ladder", "kaggle-bundle", "collect", "review", "apply-labels", "extras", "final"])
    parser.add_argument("--round", type=int, help="review / apply-labels: round number")
    parser.add_argument("--config", help="review: dev_scores_<config>.csv in results/clickbait/ used to pick items; "
                                         "final: model family (tfidf_lr, tfidf_svm, tfidf_nb)")
    parser.add_argument("--family", default="tfidf_lr", choices=list(MODELS), help="extras: model family")
    parser.add_argument("--from-file", help="final: corpus_scores_<config>.csv (id, score, pred) from Kaggle")
    parser.add_argument("--segment", action="store_true", help="final: pyvi word segmentation before TF-IDF")
    parser.add_argument("--balanced", action="store_true", help='final: class_weight="balanced"')
    parser.add_argument("--threshold", default="0.5", choices=["0.5", "cv", "prior"],
                        help="final: 0.5 (0 for SVM), F1-optimal on out-of-fold training scores, or prior shift "
                             "to the VnExpress part of ViClickbait")
    args = parser.parse_args(argv)
    if args.action == "extras":
        t0 = time.time()
        payload = extras(args.family)
        cols = ["group", "model", "f1", "f1_ci95", "precision", "recall", "pr_auc", "pr_auc_ci95",
                "predicted_positive_share", "seconds"]
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(pd.DataFrame(payload["results"])[cols].to_string(index=False))
        print(f"hardware: {payload['hardware']}; total {time.time() - t0:.0f}s")
        return 0
    if args.action == "final":
        if not args.config and not args.from_file:
            parser.error("final needs --config or --from-file")
        payload = final(args.config, args.from_file, args.segment, args.balanced, args.threshold)
        print(json.dumps({k: v for k, v in payload.items() if k != "hardware"}, indent=2, ensure_ascii=False))
        return 0
    if args.action == "review":
        print(review(args.round, args.config).to_string(index=False))
        return 0
    if args.action == "apply-labels":
        print(apply_labels(args.round))
        return 0
    if args.action == "kaggle-bundle":
        kaggle_bundle()
        return 0
    if args.action == "collect":
        full = collect()
        print(full[["level", "model", "train", "input", "f1", "f1_ci95", "pr_auc", "pr_auc_ci95",
                    "predicted_positive_share"]].to_string(index=False))
        return 0
    payload = run()
    cols = ["level", "model", "train", "input", "f1", "f1_ci95", "precision", "recall", "pr_auc", "pr_auc_ci95",
            "predicted_positive_share"]
    print(pd.DataFrame(payload["results"])[cols].to_string(index=False))
    print("corpus predicted share:", payload["corpus_predicted_share"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
