"""Clickbait effect: do threads under clickbait headlines differ from threads under other headlines?

Articles are split by the clickbait label of data/tasks/clickbait/corpus_labels.csv. For every thread
metric (structure from threads.parquet, reconstructed tree; sentiment from thread_sentiment.parquet):
Mann-Whitney U (two-sided) and Cliff's delta (P(clickbait > other) - P(clickbait < other)), with
Benjamini-Hochberg correction (q = 0.05) over the metrics of one analysis. Done on all articles and per
category (per-category tests are also corrected together, column p_bh_categories). Only articles whose comments were fetched are used; metrics undefined for empty threads
(ratios, depth, sentiment) are computed on threads with at least one comment.

Robustness checks:
    regression    OLS of each metric (log1p for size) on clickbait + category + publication hour +
                  log article length, HC3 standard errors
    threshold     clickbait = score >= 0.3 / 0.5 / 0.7, and Spearman between the score and each metric
    dev labels    only the 750 articles of the clickbait dev set, with their dev labels
    tree version  structural metrics on the two-level tree instead of the reconstructed tree

Usage (from the project root, after `python -m src.analysis.thread_sentiment`):
    python -m src.analysis.clickbait_effect     # writes results/clickbait_effect/
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from ..classifiers.common import CSV_ENCODING, RESULTS_DIR, TASKS_DIR, load_dev, run_info
from ..processing.snapshot import PROCESSED_DIR

OUT = RESULTS_DIR / "clickbait_effect"
STRUCTURE = {"size": "size", "reply_ratio": "reply_ratio", "max_depth": "max_depth_rec",
             "structural_virality": "structural_virality_rec"}
STRUCTURE_2L = {"max_depth": "max_depth_2l", "structural_virality": "structural_virality_2l"}
SENTIMENT = ["p_neg", "p_uncivil", "polarization"]
METRICS = list(STRUCTURE) + SENTIMENT
THRESHOLDS = [0.3, 0.5, 0.7]
Q = 0.05


def article_table() -> pd.DataFrame:
    """One row per article with fetched comments: clickbait label/score, metadata and thread metrics."""
    a = pd.read_parquet(PROCESSED_DIR / "articles.parquet",
                        columns=["article_id", "category", "published_at", "word_count", "comments_status"])
    a = a[a["comments_status"] == "done"].drop(columns="comments_status")
    t = pd.read_parquet(PROCESSED_DIR / "threads.parquet")
    s = pd.read_parquet(PROCESSED_DIR / "thread_sentiment.parquet")
    cb = pd.read_csv(TASKS_DIR / "clickbait" / "corpus_labels.csv", dtype={"id": str}, encoding=CSV_ENCODING)
    cb = cb.rename(columns={"id": "article_id", "label": "clickbait", "score": "clickbait_score"})
    a["article_id"] = a["article_id"].astype(str)
    t["article_id"] = t["article_id"].astype(str)
    s["article_id"] = s["article_id"].astype(str)
    df = (a.merge(t.drop(columns=["category", "published_at"]), on="article_id", how="left")
           .merge(s, on="article_id", how="left").merge(cb, on="article_id", how="left"))
    if df["clickbait"].isna().any():
        raise ValueError("articles without a clickbait label")
    df["hour"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_convert("Asia/Ho_Chi_Minh").dt.hour
    for k, col in STRUCTURE.items():
        df[k] = df[col]
    empty = df["size"] == 0
    undefined = [m for m in METRICS if m != "size"] + [c for c in STRUCTURE.values() if c != "size"]
    df.loc[empty, sorted(set(undefined) | set(STRUCTURE_2L.values()))] = np.nan
    return df


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) - P(X < Y), from the Mann-Whitney U statistic."""
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def compare(df: pd.DataFrame, group: pd.Series, metrics: dict, scope: str) -> pd.DataFrame:
    """Mann-Whitney U and Cliff's delta for each metric; BH correction within this call."""
    rows = []
    for name, col in metrics.items():
        d = df[[col]].assign(g=group.to_numpy()).dropna()
        x, y = d.loc[d["g"] == 1, col].to_numpy(), d.loc[d["g"] == 0, col].to_numpy()
        row = {"scope": scope, "metric": name, "n_clickbait": len(x), "n_other": len(y),
               "median_clickbait": np.median(x) if len(x) else np.nan,
               "median_other": np.median(y) if len(y) else np.nan,
               "mean_clickbait": x.mean() if len(x) else np.nan, "mean_other": y.mean() if len(y) else np.nan}
        if len(x) >= 2 and len(y) >= 2:
            row["p"] = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            row["cliffs_delta"] = cliffs_delta(x, y)
        rows.append(row)
    out = pd.DataFrame(rows)
    ok = out["p"].notna() if "p" in out else pd.Series(False, index=out.index)
    out["p_bh"] = np.nan
    out["significant"] = False
    if ok.any():
        rej, p_bh, _, _ = multipletests(out.loc[ok, "p"], alpha=Q, method="fdr_bh")
        out.loc[ok, "p_bh"], out.loc[ok, "significant"] = p_bh, rej
    return out


def main_tests(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {m: m for m in METRICS}
    parts = [compare(df, df["clickbait"], metrics, "all")]
    for cat, g in df.groupby("category"):
        parts.append(compare(g, g["clickbait"], metrics, f"category={cat}"))
    out = pd.concat(parts, ignore_index=True)
    # p_bh corrects within one scope; p_bh_categories corrects over all per-category tests together
    cat = (out["scope"] != "all") & out["p"].notna()
    out["p_bh_categories"] = np.nan
    out.loc[cat, "p_bh_categories"] = multipletests(out.loc[cat, "p"], alpha=Q, method="fdr_bh")[1]
    return out


def regression(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df.assign(log_words=np.log1p(df["word_count"].fillna(0)))
    for m in METRICS:
        y = np.log1p(d[m]) if m == "size" else d[m]
        dd = d.assign(y=y).dropna(subset=["y"])
        fit = smf.ols("y ~ clickbait + C(category) + C(hour) + log_words", data=dd).fit(cov_type="HC3")
        lo, hi = fit.conf_int().loc["clickbait"]
        rows.append({"metric": m, "outcome": "log1p(size)" if m == "size" else m, "n": int(fit.nobs),
                     "coef_clickbait": fit.params["clickbait"], "ci95_low": lo, "ci95_high": hi,
                     "p": fit.pvalues["clickbait"]})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], alpha=Q, method="fdr_bh")[1]
    out["significant"] = out["p_bh"] < Q
    return out


def thresholds(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {m: m for m in METRICS}
    parts = [compare(df, (df["clickbait_score"] >= t).astype(int), metrics, f"score>={t}") for t in THRESHOLDS]
    out = pd.concat(parts, ignore_index=True)
    rho = []
    for m in METRICS:
        d = df[["clickbait_score", m]].dropna()
        r = spearmanr(d["clickbait_score"], d[m])
        rho.append({"metric": m, "spearman_rho": float(r.correlation), "p": float(r.pvalue), "n": len(d)})
    rho = pd.DataFrame(rho)
    rho["p_bh"] = multipletests(rho["p"], alpha=Q, method="fdr_bh")[1]
    return out, rho


def dev_labels(df: pd.DataFrame) -> pd.DataFrame:
    dev = load_dev("clickbait")
    titles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", columns=["article_id", "title"])
    titles["article_id"] = titles["article_id"].astype(str)
    lab = dev[["title", "label"]].merge(titles, on="title", how="inner")
    d = df.merge(lab[["article_id", "label"]], on="article_id", how="inner")
    return compare(d, d["label"].astype(int), {m: m for m in METRICS}, "dev labels")


def tree_version(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([compare(df, df["clickbait"], {k: v for k, v in STRUCTURE.items() if k in STRUCTURE_2L},
                              "reconstructed tree"),
                      compare(df, df["clickbait"], STRUCTURE_2L, "two-level tree")], ignore_index=True)


def run() -> dict:
    df = article_table()
    tables = {"tests.csv": main_tests(df), "regression.csv": regression(df)}
    tables["threshold_tests.csv"], tables["spearman.csv"] = thresholds(df)
    tables["dev_label_tests.csv"] = dev_labels(df)
    tables["tree_version.csv"] = tree_version(df)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.round(4).to_csv(OUT / name, index=False, encoding=CSV_ENCODING)
    overall = tables["tests.csv"].query("scope == 'all'")
    summary = {
        **run_info(), "articles": int(len(df)), "clickbait": int(df["clickbait"].sum()),
        "threads_with_comments": int((df["size"] > 0).sum()),
        "clickbait_with_comments": int(((df["size"] > 0) & (df["clickbait"] == 1)).sum()),
        "overall": overall[["metric", "median_clickbait", "median_other", "cliffs_delta", "p", "p_bh",
                            "significant"]].round(4).to_dict("records"),
        "regression": tables["regression.csv"][["metric", "coef_clickbait", "ci95_low", "ci95_high", "p_bh"]]
                      .round(4).to_dict("records"),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    return summary


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Compare threads under clickbait and other headlines").parse_args(argv)
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
