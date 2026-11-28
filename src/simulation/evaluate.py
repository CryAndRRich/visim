"""Evaluation of the simulated threads against the real ones (reader simulation).

Sources per simulated article: the real thread, the simulated threads of conditions A, B, C, and two baselines
without an LLM: `retrieval` (the real thread of an earlier article of the same category, data/simulation/
retrieval.csv) and `category_mean` (per metric, the mean over earlier articles of the same category with at
least 15 comments, outside the simulated set; all such articles of the category when fewer than 5 are earlier).

Thread metrics, computed the same way for every source:
    structure   reply_ratio, max_depth, structural_virality (real threads: recovered tree)
    content     mean_length (words), self_bleu_4 (mean BLEU-4 of each comment against the others of the thread,
                smoothed as in Texygen; high = repetitive wording), semantic_mean_cosine (mean pairwise cosine of
                multilingual E5 embeddings, from src.simulation.gpu_eval, when its files exist)
    sentiment   p_neg, p_uncivil (labels of the final models, the same for real and simulated comments:
                data/simulation/generated/real_*_scores.csv and <run>/*_scores.csv), polarization (variance of
                P(pos) - P(neg) over the comments)

Tests (written to results/simulation_eval/):
    distribution.csv   real vs each source over the 100 articles: Mann-Whitney U, Kolmogorov-Smirnov, Cliff's delta
                       (source vs real), BH within each test type
    conditions.csv     per-article absolute error |source - real|; Wilcoxon signed-rank between sources (A vs B,
                       A vs C, A vs C within clickbait and other articles, each condition vs each baseline)
    spearman.csv       per metric and source, Spearman correlation with the real values over the 100 articles
    rewrite.csv        headline rewrite scenario (when <run>/../rewrite is labelled): rewritten vs original
                       headline per article for A and C, Wilcoxon signed-rank, direction expected from the
                       clickbait-effect analysis
    discriminator.csv  real vs simulated comments, split by article (5-fold GroupKFold): style features + LR and
                       TF-IDF + LR; AUC with a 95% interval from an article-level bootstrap. Retrieval comments
                       (real comments of other articles) are the control: their AUC should be near 0.5
    discriminator_features.csv  standardized style coefficients and the strongest word n-grams of each side
    thread_metrics.csv one row per article and source

    python -m src.simulation.evaluate [--run data/simulation/generated/main]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..classifiers.common import CSV_ENCODING, N_BOOT, RESULTS_DIR, SEED, run_info
from ..features.structure import tree_metrics
from ..processing.snapshot import PROCESSED_DIR
from ..processing.trees import ROOT, reconstructed_parents
from .prompts import SIM_DIR, text_key

OUT = RESULTS_DIR / "simulation_eval"
GENERATED = SIM_DIR / "generated"
CONDITIONS = ["A", "B", "C"]
BASELINES = ["retrieval", "category_mean"]
METRICS = ["reply_ratio", "max_depth", "structural_virality", "mean_length", "self_bleu_4", "semantic_mean_cosine",
           "p_neg", "p_uncivil", "polarization"]   # semantic_mean_cosine once src.simulation.gpu_eval has run
MIN_COMMENTS, MIN_EARLIER = 15, 5
Q = 0.05
MENTION = re.compile(r"@USER_[0-9a-f]+\s*:?", re.IGNORECASE)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
ABBREVIATIONS = {"ko", "k", "kg", "khg", "dc", "đc", "vs", "j", "r", "cx", "ng", "mn", "bt", "nc", "ns", "trc",
                 "ak", "z", "v", "ok", "oke", "mk", "mik", "e", "a", "ae", "ad"}


# ---------------------------------------------------------------- thread metrics

def self_bleu(texts: list[str], max_n: int = 4) -> float:
    """Mean BLEU-max_n of each comment against all other comments of the thread (Texygen's Self-BLEU, with
    smoothing method 1: a zero n-gram match counts 0.1). Comments without words are skipped."""
    toks = [list(text_key(t)) for t in texts]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return math.nan
    lens = np.array([len(t) for t in toks])
    counts = [[Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1)) for n in range(1, max_n + 1)] for t in toks]
    best = []                              # per n: n-gram -> (top count, its comment, second count)
    for n in range(max_n):
        top: dict = {}
        for j, doc in enumerate(counts):
            for g, c in doc[n].items():
                c1, o1, c2 = top.get(g, (0, -1, 0))
                top[g] = (c, j, c1) if c > c1 else (c1, o1, max(c2, c))
        best.append(top)
    scores = []
    for i, doc in enumerate(counts):
        log_p = 0.0
        for n in range(max_n):
            total = sum(doc[n].values())
            clipped = sum(min(c, best[n][g][0] if best[n][g][1] != i else best[n][g][2]) for g, c in doc[n].items())
            log_p += math.log((clipped if clipped else 0.1) / max(total, 1)) / max_n
        others = np.delete(lens, i)
        ref = others[np.lexsort((others, np.abs(others - lens[i])))[0]]   # closest length, shorter on ties
        bp = 1.0 if lens[i] > ref else math.exp(1 - ref / lens[i])
        scores.append(bp * math.exp(log_p))
    return float(np.mean(scores))


def thread_metrics(texts: list[str], parents: dict, p_pos: np.ndarray, p_neg: np.ndarray, sentiment: np.ndarray,
                   uncivil: np.ndarray) -> dict:
    tree = tree_metrics(parents)
    return {"size": len(texts), "reply_ratio": sum(p != ROOT for p in parents.values()) / len(texts),
            "max_depth": tree["max_depth"], "structural_virality": tree["structural_virality"],
            "mean_length": float(np.mean([len(str(t).split()) for t in texts])), "self_bleu_4": self_bleu(texts),
            "p_neg": float(np.mean(sentiment == "neg")), "p_uncivil": float(np.mean(uncivil)),
            "polarization": float(np.var(np.asarray(p_pos) - np.asarray(p_neg)))}


def read_scores(path: Path, keys: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype={k: str for k in keys}, encoding=CSV_ENCODING)


def real_comments() -> pd.DataFrame:
    c = pd.read_parquet(PROCESSED_DIR / "comments.parquet")
    c["key"] = c["comment_id"].astype(str)
    s = read_scores(GENERATED / "real_sentiment_scores.csv", ["id"]).rename(columns={"id": "key", "label": "sentiment"})
    u = read_scores(GENERATED / "real_incivility_scores.csv", ["id"]).rename(
        columns={"id": "key", "label": "uncivil", "score": "p_uncivil"})
    c = c.merge(s, on="key", how="left").merge(u, on="key", how="left")
    if c["sentiment"].isna().any() or c["uncivil"].isna().any():
        raise ValueError("real comments without model scores; run src.simulation.label first")
    c["article_id"] = c["article_id"].astype(str)
    return c


def simulated_comments(run: Path) -> pd.DataFrame:
    rows = []
    with open(run / "threads.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            rows += [{"article_id": r["article_id"], "condition": r["condition"], **c} for c in r["comments"]]
    keys = ["article_id", "condition", "id"]
    s = read_scores(run / "sentiment_scores.csv", keys).rename(columns={"label": "sentiment"})
    u = read_scores(run / "incivility_scores.csv", keys).rename(columns={"label": "uncivil", "score": "p_uncivil"})
    out = pd.DataFrame(rows).merge(s, on=keys, how="left").merge(u, on=keys, how="left")
    if out["sentiment"].isna().any() or out["uncivil"].isna().any():
        raise ValueError(f"simulated comments without labels; run src.simulation.label on {run}")
    return out


def real_table(comments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aid, g in comments.groupby("article_id"):
        parents, _ = reconstructed_parents(g.to_dict("records"))
        rows.append({"article_id": aid, **thread_metrics(
            g["text"].tolist(), parents, g["p_pos"].to_numpy(), g["p_neg"].to_numpy(), g["sentiment"].to_numpy(),
            g["uncivil"].to_numpy())})
    out = pd.DataFrame(rows)
    path = GENERATED / "real_semantic.csv"
    if path.exists():
        out = out.merge(read_scores(path, ["article_id"]), on="article_id", how="left")
    return out


def simulated_table(sim: pd.DataFrame, run: Path) -> pd.DataFrame:
    rows = []
    for (aid, cond), g in sim.groupby(["article_id", "condition"]):
        parents = {c: (ROOT if p is None or (isinstance(p, float) and math.isnan(p)) else p)
                   for c, p in zip(g["id"], g["parent"])}
        rows.append({"article_id": aid, "source": cond, **thread_metrics(
            g["text"].tolist(), parents, g["p_pos"].to_numpy(), g["p_neg"].to_numpy(), g["sentiment"].to_numpy(),
            g["uncivil"].to_numpy())})
    out = pd.DataFrame(rows)
    path = run / "semantic.csv"
    if path.exists():
        sem = read_scores(path, ["article_id", "condition"]).rename(columns={"condition": "source"})
        out = out.merge(sem, on=["article_id", "source"], how="left")
    return out


def metric_names(table: pd.DataFrame) -> list[str]:
    return [m for m in METRICS if m in table and table[m].notna().any()]


def metric_table(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen = pd.read_csv(SIM_DIR / "articles.csv", dtype={"article_id": str}, encoding=CSV_ENCODING)
    retrieval = pd.read_csv(SIM_DIR / "retrieval.csv", dtype={"article_id": str, "baseline_article_id": str},
                            encoding=CSV_ENCODING)
    arts = pd.read_parquet(PROCESSED_DIR / "articles.parquet", columns=["article_id", "category", "published_at"])
    arts["article_id"] = arts["article_id"].astype(str)
    arts["published"] = pd.to_datetime(arts["published_at"], utc=True)
    real_c = real_comments()
    sim_c = simulated_comments(run)
    real_all = real_table(real_c).merge(arts, on="article_id")
    ids = set(chosen["article_id"])
    parts = [real_all[real_all["article_id"].isin(ids)].assign(source="real"), simulated_table(sim_c, run)]
    by_id = real_all.set_index("article_id")
    parts.append(retrieval[["article_id", "baseline_article_id"]].merge(
        real_all.rename(columns={"article_id": "baseline_article_id"}), on="baseline_article_id")
        .drop(columns="baseline_article_id").assign(source="retrieval"))
    outside = real_all[~real_all["article_id"].isin(ids) & (real_all["size"] >= MIN_COMMENTS)]
    means = []
    for a in chosen.itertuples():
        same = outside[outside["category"] == a.category]
        earlier = same[same["published"] < by_id.loc[a.article_id, "published"]]
        use = earlier if len(earlier) >= MIN_EARLIER else same
        means.append({"article_id": a.article_id, "source": "category_mean", "n_used": len(use),
                      **use[metric_names(real_all)].mean().to_dict()})
    parts.append(pd.DataFrame(means))
    table = pd.concat(parts, ignore_index=True)[["article_id", "source", "size", *metric_names(real_all), "n_used"]]
    table = table.merge(chosen[["article_id", "category", "clickbait"]], on="article_id")
    return table, real_c[real_c["article_id"].isin(ids)], sim_c, retrieval


# ---------------------------------------------------------------- tests

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def wide(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    return table.pivot(index="article_id", columns="source", values=metric)


def multipletests(p, alpha, method):
    from statsmodels.stats.multitest import multipletests as mt   # only needed for the tests, not on the server
    return mt(p, alpha=alpha, method=method)


def distribution_tests(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in metric_names(table):
        w = wide(table, m)
        for src in CONDITIONS + ["retrieval"]:
            d = w[["real", src]].dropna()
            x, y = d[src].to_numpy(), d["real"].to_numpy()
            rows.append({"metric": m, "source": src, "n": len(d), "median_real": np.median(y),
                         "median_source": np.median(x), "cliffs_delta": cliffs_delta(x, y),
                         "p_mannwhitney": mannwhitneyu(x, y, alternative="two-sided").pvalue,
                         "p_ks": ks_2samp(x, y).pvalue})
    out = pd.DataFrame(rows)
    for col in ("p_mannwhitney", "p_ks"):
        out[col + "_bh"] = multipletests(out[col], alpha=Q, method="fdr_bh")[1]
    return out


def condition_tests(table: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank on the per-article absolute errors of two sources."""
    pairs = [("A", "B", "all"), ("A", "C", "all"), ("A", "C", "clickbait"), ("A", "C", "other")]
    pairs += [(c, b, "all") for c in CONDITIONS for b in BASELINES]
    clickbait = table.drop_duplicates("article_id").set_index("article_id")["clickbait"]
    rows = []
    for m in metric_names(table):
        w = wide(table, m)
        err = w.drop(columns="real").sub(w["real"], axis=0).abs()
        for a, b, group in pairs:
            d = err[[a, b]].dropna()
            if group != "all":
                d = d[clickbait.reindex(d.index).eq(1 if group == "clickbait" else 0)]
            diff = d[a] - d[b]
            try:
                p = wilcoxon(d[a], d[b]).pvalue if (diff != 0).any() else 1.0
            except ValueError:
                p = math.nan
            rows.append({"metric": m, "source_a": a, "source_b": b, "articles": group, "n": len(d),
                         "mae_a": d[a].mean(), "mae_b": d[b].mean(), "median_error_a": d[a].median(),
                         "median_error_b": d[b].median(), "a_better_share": float((diff < 0).mean()), "p": p})
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out["p_bh"] = np.nan
    out.loc[ok, "p_bh"] = multipletests(out.loc[ok, "p"], alpha=Q, method="fdr_bh")[1]
    return out


def spearman_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in metric_names(table):
        w = wide(table, m)
        for src in CONDITIONS + BASELINES:
            d = w[["real", src]].dropna()
            rho, p = spearmanr(d[src], d["real"]) if d[src].nunique() > 1 else (math.nan, math.nan)
            rows.append({"metric": m, "source": src, "n": len(d), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out["p_bh"] = np.nan
    out.loc[ok, "p_bh"] = multipletests(out.loc[ok, "p"], alpha=Q, method="fdr_bh")[1]
    return out


# ---------------------------------------------------------------- real vs simulated discriminator

def style_features(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for t in texts.astype(str):
        words = t.split()
        n = max(len(words), 1)
        letters = [ch for ch in t if ch.isalpha()]
        toks = text_key(t)
        rows.append({
            "log_words": math.log1p(len(words)), "mean_word_length": np.mean([len(w) for w in words]) if words else 0,
            "exclamations": t.count("!"), "questions": t.count("?"), "ellipsis": int("..." in t or "…" in t),
            "emoji": len(EMOJI.findall(t)), "mention": int(bool(MENTION.search(t))),
            "digits_share": sum(ch.isdigit() for ch in t) / max(len(t), 1),
            "upper_share": sum(ch.isupper() for ch in letters) / max(len(letters), 1),
            "no_diacritic_share": sum(w.isascii() for w in toks) / max(len(toks), 1),
            "abbreviation_share": sum(w in ABBREVIATIONS for w in toks) / max(len(toks), 1),
            "starts_upper": int(bool(t[:1].isupper())), "ends_punctuation": int(t.rstrip()[-1:] in ".!?…"),
            "sentences": len(re.findall(r"[.!?…]+", t)), "commas": t.count(",") / n,
        })
    return pd.DataFrame(rows)


def tfidf_model():
    return make_pipeline(
        FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                      ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True))]),
        LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=SEED))


def style_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def article_bootstrap_auc(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    index = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array).tolist()
    aucs = []
    for _ in range(N_BOOT):
        take = np.concatenate([index[k] for k in rng.integers(0, len(index), len(index))])
        if len(np.unique(y[take])) == 2:
            aucs.append(roc_auc_score(y[take], score[take]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def discriminator_sets(run: Path) -> dict[str, pd.DataFrame]:
    """Per opponent (A, B, C, retrieval): the real comments of the simulated articles (y = 0) and the
    opponent's comments (y = 1), keyed by the simulated article (for the split by article). The retrieval
    opponent is the real thread of the retrieval-baseline article."""
    chosen = set(pd.read_csv(SIM_DIR / "articles.csv", dtype={"article_id": str}, encoding=CSV_ENCODING)["article_id"])
    retrieval = pd.read_csv(SIM_DIR / "retrieval.csv", dtype={"article_id": str, "baseline_article_id": str},
                            encoding=CSV_ENCODING)
    c = pd.read_parquet(PROCESSED_DIR / "comments.parquet", columns=["article_id", "text"])
    c["article_id"] = c["article_id"].astype(str)
    real = c[c["article_id"].isin(chosen)].assign(y=0)
    sides = {}
    with open(run / "threads.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            sides.setdefault(r["condition"], []).extend((r["article_id"], x["text"]) for x in r["comments"])
    sides = {k: pd.DataFrame(v, columns=["article_id", "text"]) for k, v in sorted(sides.items())}
    sides["retrieval"] = retrieval[["article_id", "baseline_article_id"]].merge(
        c.rename(columns={"article_id": "baseline_article_id"}), on="baseline_article_id")[["article_id", "text"]]
    out = {}
    for name, other in sides.items():
        d = pd.concat([real, other.assign(y=1)], ignore_index=True)
        out[name] = d[d["article_id"].isin(set(other["article_id"]))].reset_index(drop=True)
    return out


def discriminator(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, features = [], []
    for name, d in discriminator_sets(run).items():
        text = d["text"].astype(str).map(lambda t: MENTION.sub(" @user ", t))
        y, groups = d["y"].to_numpy(), d["article_id"].to_numpy()
        inputs = {"style_lr": (style_model, style_features(d["text"])), "tfidf_lr": (tfidf_model, text)}
        for model_name, (make, x) in inputs.items():
            score = np.zeros(len(d))
            for fit, test in GroupKFold(5).split(x, y, groups):
                m = make().fit(x.iloc[fit], y[fit])
                score[test] = m.predict_proba(x.iloc[test])[:, 1]
            lo, hi = article_bootstrap_auc(y, score, groups)
            rows.append({"versus": name, "model": model_name, "comments_real": int((y == 0).sum()),
                         "comments_other": int((y == 1).sum()), "auc": roc_auc_score(y, score), "ci_low": lo,
                         "ci_high": hi})
            if name in CONDITIONS:
                m = make().fit(x, y)
                lr = m[-1]
                if model_name == "style_lr":
                    names = list(x.columns)
                else:
                    names = list(m[0].transformer_list[0][1].get_feature_names_out())
                coef = lr.coef_[0][:len(names)]
                f = pd.DataFrame({"versus": name, "model": model_name, "feature": names, "coefficient": coef})
                if model_name == "tfidf_lr":   # word n-grams only; the strongest of each side
                    f = pd.concat([f.nlargest(15, "coefficient"), f.nsmallest(15, "coefficient")])
                features.append(f)
    return pd.DataFrame(rows), pd.concat(features, ignore_index=True)


def rewrite_tests(table: pd.DataFrame, rewrite: Path) -> pd.DataFrame:
    """Headline rewrite scenario: per article, the thread simulated from the rewritten (neutral) headline vs the
    one from the original headline, for conditions A and C; Wilcoxon signed-rank and the share of articles
    that moved. `expected` is the direction a neutral headline should move a metric if simulated readers
    reacted like real ones: the opposite sign of the clickbait-minus-other Cliff's delta of the clickbait-effect
    analysis (results/clickbait_effect/tests.csv, all articles; none of those differences was significant)."""
    new = simulated_table(simulated_comments(rewrite), rewrite)
    rq1 = pd.read_csv(RESULTS_DIR / "clickbait_effect" / "tests.csv")
    rq1 = rq1[rq1["scope"] == "all"].set_index("metric")["cliffs_delta"]
    rows = []
    for cond in sorted(new["source"].unique()):
        n = new[new["source"] == cond].set_index("article_id")
        o = table[table["source"] == cond].set_index("article_id").reindex(n.index)
        for m in metric_names(n):
            d = pd.DataFrame({"original": o[m], "rewritten": n[m]}).dropna()
            diff = d["rewritten"] - d["original"]
            p = wilcoxon(d["rewritten"], d["original"]).pvalue if (diff != 0).any() else 1.0
            expected = -np.sign(rq1[m]) if m in rq1 else np.nan
            rows.append({"condition": cond, "metric": m, "n": len(d), "mean_original": d["original"].mean(),
                         "mean_rewritten": d["rewritten"].mean(), "median_change": diff.median(),
                         "share_up": float((diff > 0).mean()), "share_down": float((diff < 0).mean()),
                         "p": p, "expected_direction": expected,
                         "moved_as_expected": bool(np.sign(diff.median()) == expected) if expected == expected
                         else np.nan})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], alpha=Q, method="fdr_bh")[1]
    return out


def gpu_discriminator(path: Path) -> pd.DataFrame:
    """AUC and article-bootstrap interval of the out-of-fold scores written by src.simulation.gpu_eval."""
    s = pd.read_csv(path, dtype={"article_id": str}, encoding=CSV_ENCODING)
    rows = []
    for (name, model), d in s.groupby(["versus", "model"], sort=False):
        y, score = d["y"].to_numpy(), d["score"].to_numpy()
        lo, hi = article_bootstrap_auc(y, score, d["article_id"].to_numpy())
        rows.append({"versus": name, "model": model, "comments_real": int((y == 0).sum()),
                     "comments_other": int((y == 1).sum()), "auc": roc_auc_score(y, score), "ci_low": lo,
                     "ci_high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def run(run_dir: Path) -> dict:
    table = metric_table(run_dir)[0]
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {"thread_metrics.csv": table, "distribution.csv": distribution_tests(table),
               "conditions.csv": condition_tests(table), "spearman.csv": spearman_table(table)}
    outputs["discriminator.csv"], outputs["discriminator_features.csv"] = discriminator(run_dir)
    phobert = run_dir / "phobert_discriminator.csv"
    if phobert.exists():
        outputs["discriminator.csv"] = pd.concat([outputs["discriminator.csv"], gpu_discriminator(phobert)],
                                                 ignore_index=True)
    rewrite = run_dir.parent / "rewrite"
    if (rewrite / "sentiment_scores.csv").exists() and (rewrite / "incivility_scores.csv").exists():
        outputs["rewrite.csv"] = rewrite_tests(table, rewrite)
    for name, df in outputs.items():
        df.round(4).to_csv(OUT / name, index=False, encoding=CSV_ENCODING)
    means = table.groupby("source")[metric_names(table)].mean().round(4)
    summary = {**run_info(), "run": run_dir.name, "articles": int(table["article_id"].nunique()),
               "mean_by_source": means.to_dict("index"),
               "discriminator_auc": outputs["discriminator.csv"].round(3).to_dict("records")}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate simulated threads against the real ones")
    p.add_argument("--run", default=str(GENERATED / "main"))
    a = p.parse_args(argv)
    summary = run(Path(a.run))
    print(pd.DataFrame(summary["mean_by_source"]).T.to_string())
    print(pd.DataFrame(summary["discriminator_auc"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
