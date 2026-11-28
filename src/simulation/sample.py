"""Articles, style examples and non-LLM baselines for the reader simulation.

    articles.csv        the simulated articles: published on or after CUTOFF, at least MIN_COMMENTS comments,
                        every clickbait article that qualifies plus non-clickbait articles drawn evenly
                        across categories (seed 42), N_ARTICLES in total
    examples.csv        style examples for condition B: per category, N_EXAMPLES comments from articles
                        published before CUTOFF (the earlier period), one comment per article, 5 to 60 words,
                        no mentions. A category with fewer than 30 articles before CUTOFF (the-gioi, collected only from 03/10,
                        and thoi-su) takes its examples from its earliest
                        articles outside the simulated set (column before_cutoff = False)
    retrieval.csv       retrieval baseline: for each simulated article, the real thread of an earlier article of
                        the same category, outside the simulated set, with the closest number of comments
    category_means.csv  category-mean baseline: per simulated article, the mean of each thread metric over
                        earlier articles of the same category with at least MIN_COMMENTS comments, outside the set
                        (all such articles of the category when fewer than MIN_EARLIER are earlier; earlier_only)
                        The first simulated article of a category may have no retrieval baseline.

Files hold ids and numbers only (texts stay in data/processed/). They are written to data/simulation/.

Usage (from the project root, after `python -m src.analysis.thread_sentiment`):
    python -m src.simulation.sample
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from ..classifiers.common import CSV_ENCODING, ROOT, SEED, TASKS_DIR, run_info
from ..processing.snapshot import PROCESSED_DIR

OUT = ROOT / "data" / "simulation"
CUTOFF = "2026-09-30"          # Vietnam date; earlier articles form the period that examples come from
MIN_COMMENTS = 15
N_ARTICLES = 100
N_EXAMPLES = 10
EXAMPLE_WORDS = (5, 60)
MIN_EARLIER = 5   # category-mean baseline needs at least this many earlier articles
METRICS = ["size", "reply_ratio", "max_depth_rec", "structural_virality_rec", "p_neg", "p_uncivil", "polarization"]


def article_table() -> pd.DataFrame:
    a = pd.read_parquet(PROCESSED_DIR / "articles.parquet", columns=["article_id", "category", "published_at",
                                                                     "comments_status"])
    a = a[a["comments_status"] == "done"].drop(columns="comments_status")
    t = pd.read_parquet(PROCESSED_DIR / "threads.parquet", columns=["article_id", *METRICS[:4]])
    s = pd.read_parquet(PROCESSED_DIR / "thread_sentiment.parquet", columns=["article_id", *METRICS[4:]])
    cb = pd.read_csv(TASKS_DIR / "clickbait" / "corpus_labels.csv", dtype={"id": str}, encoding=CSV_ENCODING)
    for d in (a, t, s):
        d["article_id"] = d["article_id"].astype(str)
    df = (a.merge(t, on="article_id").merge(s, on="article_id", how="left")
           .merge(cb.rename(columns={"id": "article_id", "label": "clickbait", "score": "clickbait_score"}),
                  on="article_id"))
    df["published"] = pd.to_datetime(df["published_at"], utc=True)
    df["date_vn"] = df["published"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.strftime("%Y-%m-%d")
    return df.sort_values("published").reset_index(drop=True)


def select(df: pd.DataFrame) -> pd.DataFrame:
    pool = df[(df["date_vn"] >= CUTOFF) & (df["size"] >= MIN_COMMENTS)]
    cb = pool[pool["clickbait"] == 1]
    rest = pool[pool["clickbait"] == 0]
    n_rest = N_ARTICLES - len(cb)
    cats = sorted(rest["category"].unique())
    quota = {c: n_rest // len(cats) + (i < n_rest % len(cats)) for i, c in enumerate(cats)}
    rng = np.random.default_rng(SEED)
    picked = [g.iloc[np.sort(rng.choice(len(g), quota[c], replace=False))] for c, g in rest.groupby("category")]
    out = pd.concat([cb, *picked]).sort_values("published")
    return out[["article_id", "category", "published_at", "clickbait", "clickbait_score", "size"]].reset_index(drop=True)


def examples(df: pd.DataFrame, chosen: set) -> pd.DataFrame:
    comments = pd.read_parquet(PROCESSED_DIR / "comments.parquet", columns=["comment_id", "article_id", "text"])
    comments["article_id"] = comments["article_id"].astype(str)
    outside = df[~df["article_id"].isin(chosen)]
    early = outside[outside["date_vn"] < CUTOFF]
    # a category with too few articles before CUTOFF (the-gioi was only collected from 03/10) takes its
    # examples from its earliest articles outside the simulated set instead
    short = [cat for cat in df["category"].unique() if early["category"].eq(cat).sum() < 3 * N_EXAMPLES]
    fallback = pd.concat([outside[outside["category"] == cat].head(3 * N_EXAMPLES) for cat in short])
    early = pd.concat([early[~early["category"].isin(short)], fallback])[["article_id", "category"]]
    c = comments.merge(early, on="article_id")
    words = c["text"].str.split().str.len()
    c = c[(words >= EXAMPLE_WORDS[0]) & (words <= EXAMPLE_WORDS[1]) & ~c["text"].str.contains("@USER_")]
    rng = np.random.default_rng(SEED)
    rows = []
    for cat, g in c.groupby("category"):
        per_article = g.groupby("article_id").sample(1, random_state=SEED)   # one comment per article
        take = per_article.iloc[np.sort(rng.choice(len(per_article), N_EXAMPLES, replace=False))]
        rows.append(take[["category", "article_id", "comment_id"]].assign(before_cutoff=cat not in short))
    return pd.concat(rows, ignore_index=True)


def baselines(df: pd.DataFrame, chosen_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = set(chosen_df["article_id"])
    outside = df[~df["article_id"].isin(chosen)]
    retrieval, means = [], []
    for r in chosen_df.itertuples():
        pub = df.loc[df["article_id"] == r.article_id, "published"].iloc[0]
        earlier = outside[(outside["category"] == r.category) & (outside["published"] < pub)]
        if len(earlier):
            gap = (earlier["size"] - r.size).abs()
            best = earlier.assign(gap=gap).sort_values(["gap", "published"], ascending=[True, False]).iloc[0]
            retrieval.append({"article_id": r.article_id, "baseline_article_id": best["article_id"],
                              "size": r.size, "baseline_size": int(best["size"])})
        big = earlier[earlier["size"] >= MIN_COMMENTS]
        fallback = len(big) < MIN_EARLIER       # first articles of a category: use all its other articles
        if fallback:
            big = outside[(outside["category"] == r.category) & (outside["size"] >= MIN_COMMENTS)]
        means.append({"article_id": r.article_id, "n_used": len(big), "earlier_only": not fallback,
                      **big[METRICS].mean().round(4).to_dict()})
    return pd.DataFrame(retrieval), pd.DataFrame(means)


def run() -> dict:
    df = article_table()
    chosen = select(df)
    ex = examples(df, set(chosen["article_id"]))
    retrieval, means = baselines(df, chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in {"articles.csv": chosen, "examples.csv": ex, "retrieval.csv": retrieval,
                    "category_means.csv": means}.items():
        t.to_csv(OUT / name, index=False, encoding=CSV_ENCODING)
    summary = {
        **run_info(), "cutoff": CUTOFF, "min_comments": MIN_COMMENTS,
        "articles": len(chosen), "clickbait": int(chosen["clickbait"].sum()),
        "by_category": chosen.groupby("category")["clickbait"].agg(["size", "sum"]).rename(
            columns={"size": "articles", "sum": "clickbait"}).to_dict("index"),
        "comments_to_generate": int(chosen["size"].sum()),
        "size_median_max": [float(chosen["size"].median()), int(chosen["size"].max())],
        "examples": len(ex), "retrieval_found": len(retrieval),
        "retrieval_size_gap_median": float((retrieval["size"] - retrieval["baseline_size"]).abs().median()),
        "examples_from_earliest_articles_instead_of_before_cutoff": sorted(ex.loc[~ex["before_cutoff"], "category"].unique()),
        "retrieval_missing": sorted(set(chosen["article_id"]) - set(retrieval["article_id"])),
        "category_mean_not_earlier_only": int((~means["earlier_only"]).sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Select articles, examples and baselines for the simulation").parse_args(argv)
    print(json.dumps(run(), indent=2, default=int))
    return 0


if __name__ == "__main__":
    sys.exit(main())
