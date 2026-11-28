"""Structural metrics of comment trees and the thread table.

Usage (from the project root):
    python -m src.features.structure     # writes data/processed/threads.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import pandas as pd

from ..processing.snapshot import PROCESSED_DIR
from ..processing.trees import ROOT, reconstructed_parents, two_level_parents


def tree_metrics(parents: dict) -> dict:
    """Metrics of a tree given as comment_id -> parent (ROOT = the article).

    depth: top-level comments have depth 1.
    structural_virality: mean shortest-path distance over all pairs of nodes of the undirected
    tree including the article node, i.e. Wiener index / C(n+1, 2) (Goel et al., 2016).
    """
    n = len(parents)
    if n == 0:
        return {"max_depth": 0, "mean_depth": math.nan, "structural_virality": math.nan}
    depth: dict = {ROOT: 0}

    for start in parents:                # walk up to a node of known depth, then fill the chain
        chain, node = [], start
        while node not in depth:
            chain.append(node)
            node = parents[node]
            if len(chain) > n:
                raise ValueError("cycle in reply tree")
        d = depth[node]
        for v in reversed(chain):
            d += 1
            depth[v] = d
    size = {node: 1 for node in parents}
    for node in sorted(parents, key=lambda v: depth[v], reverse=True):
        p = parents[node]
        if p != ROOT:
            size[p] += size[node]
    total = n + 1
    wiener = sum(s * (total - s) for s in size.values())
    depths = [depth[v] for v in parents]
    return {
        "max_depth": max(depths),
        "mean_depth": sum(depths) / n,
        "structural_virality": 2 * wiener / (total * (total - 1)),
    }


def thread_row(comments: list) -> dict:
    n = len(comments)
    n_replies = sum(1 for c in comments if c["is_reply"])
    row = {
        "size": n,
        "n_top": n - n_replies,
        "n_replies": n_replies,
        "reply_ratio": n_replies / n if n else math.nan,
        "n_users": len({c["user_hash"] for c in comments}),
    }
    for suffix, parents in (("2l", two_level_parents(comments)), ("rec", None)):
        if parents is None:
            parents, info = reconstructed_parents(comments)
            row.update({f"rec_{k}": v for k, v in info.items()})
        for k, v in tree_metrics(parents).items():
            row[f"{k}_{suffix}"] = v
    return row


def build_threads(articles: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    by_article = {aid: g.to_dict("records") for aid, g in comments.groupby("article_id")}
    rows = []
    for a in articles.itertuples():
        row = {"article_id": a.article_id, "category": a.category, "published_at": a.published_at}
        row.update(thread_row(by_article.get(a.article_id, [])))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Structural metrics")
    parser.parse_args(argv)
    d = PROCESSED_DIR
    articles = pd.read_parquet(d / "articles.parquet", columns=["article_id", "category", "published_at"])
    comments = pd.read_parquet(d / "comments.parquet")
    threads = build_threads(articles, comments)
    threads.to_parquet(d / "threads.parquet", index=False)
    rec_cols = [c for c in threads.columns if c.startswith("rec_")]
    summary = {
        "threads": len(threads),
        "reconstruction": {c[4:]: int(threads[c].sum()) for c in rec_cols},
        "median": threads[["size", "reply_ratio", "max_depth_2l", "max_depth_rec",
                           "structural_virality_2l", "structural_virality_rec"]].median().round(3).to_dict(),
        "max_depth_rec_distribution": threads["max_depth_rec"].value_counts().sort_index().to_dict(),
    }
    (d / "structure_summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=int))
    return 0


if __name__ == "__main__":
    sys.exit(main())
