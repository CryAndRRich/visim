"""Sentiment and incivility metrics of each comment thread.

Comment labels come from data/tasks/<task>/corpus_labels.csv (final model of each task; dev items keep
their dev label). For a set of comments:
    p_neg          share of comments labelled negative
    p_uncivil      share of comments labelled uncivil
    polarization   variance of s_i over the comments, with s_i = +1 (pos), 0 (neu), -1 (neg).
                   The definition with s_i = P(pos) - P(neg) needs per-comment probabilities, which the
                   corpus label files do not keep; the hard-label version is used instead.

Usage (from the project root):
    python -m src.analysis.thread_sentiment   # writes data/processed/thread_sentiment.parquet
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from ..classifiers.common import CSV_ENCODING, TASKS_DIR
from ..processing.snapshot import PROCESSED_DIR

POLARITY = {"neg": -1, "neu": 0, "pos": 1}


def comment_labels() -> pd.DataFrame:
    """comment_id, sentiment, polarity, uncivil (0/1)."""
    read = lambda t: pd.read_csv(TASKS_DIR / t / "corpus_labels.csv", dtype=str, encoding=CSV_ENCODING,
                                 keep_default_na=False)[["id", "label"]]
    s = read("sentiment").rename(columns={"id": "comment_id", "label": "sentiment"})
    u = read("incivility").rename(columns={"id": "comment_id", "label": "uncivil"})
    out = s.merge(u, on="comment_id", how="outer", validate="1:1")
    if out[["sentiment", "uncivil"]].isna().any().any():
        raise ValueError("sentiment and incivility corpus labels cover different comments")
    out["polarity"] = out["sentiment"].map(POLARITY)
    out["uncivil"] = out["uncivil"].astype(int)
    return out


def labelled_comments(comments: pd.DataFrame | None = None) -> pd.DataFrame:
    """comments.parquet joined with the comment labels."""
    if comments is None:
        comments = pd.read_parquet(PROCESSED_DIR / "comments.parquet")
    labels = comment_labels().rename(columns={"comment_id": "_key"})
    out = comments.assign(_key=comments["comment_id"].astype(str)).merge(labels, on="_key", how="left")
    out = out.drop(columns="_key")  # comment_id and parent_id keep their original type for the tree code
    if out["sentiment"].isna().any():
        raise ValueError(f"{int(out['sentiment'].isna().sum())} comments without labels")
    return out


def sentiment_metrics(group: pd.DataFrame) -> dict:
    n = len(group)
    if n == 0:
        return {"p_neg": np.nan, "p_uncivil": np.nan, "polarization": np.nan}
    return {"p_neg": float((group["sentiment"] == "neg").mean()),
            "p_uncivil": float(group["uncivil"].mean()),
            "polarization": float(group["polarity"].var(ddof=0))}


def build(within_48h: bool = False) -> pd.DataFrame:
    """One row per article with comments fetched. Metrics use all comments, like threads.parquet
    (96% of them were posted within 48 hours); within_48h=True keeps only those."""
    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", columns=["article_id", "comments_status"])
    c = labelled_comments()
    if within_48h:
        c = c[c["within_48h"]]
    groups = dict(tuple(c.groupby("article_id")))
    done = articles.loc[articles["comments_status"] == "done", "article_id"]
    return pd.DataFrame([{"article_id": a, "n_labelled": len(groups.get(a, c.iloc[:0])),
                          **sentiment_metrics(groups.get(a, c.iloc[:0]))} for a in done])


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Sentiment and incivility metrics per thread").parse_args(argv)
    t = build()
    t.to_parquet(PROCESSED_DIR / "thread_sentiment.parquet", index=False)
    summary = {"threads": int(len(t)), "threads_with_comments": int((t["n_labelled"] > 0).sum()),
               **{k: {"mean": round(float(t[k].mean()), 4), "median": round(float(t[k].median()), 4)}
                  for k in ["p_neg", "p_uncivil", "polarization"]}}
    (PROCESSED_DIR / "thread_sentiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
