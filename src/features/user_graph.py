"""User-article graph, user projection and Louvain communities.

- Bipartite graph: a user is linked to every article they commented on.
- Projection: two users are linked if they commented on at least one common article; the edge
  weight is the number of common articles. Only users active in at least 2 articles are kept,
  since a user seen in a single article only forms a clique with that article's commenters.
- Louvain (networkx, fixed seed) on the weighted projection.
- Frequent commenters: users in the top FREQUENT_TOP_SHARE of the article-degree distribution.
  For early controversy prediction this set must be recomputed on the training period only, to
  avoid leakage.

Usage (from the project root):
    python -m src.features.user_graph
"""
from __future__ import annotations

import argparse
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from ..processing.snapshot import PROCESSED_DIR

FREQUENT_TOP_SHARE = 0.05
LOUVAIN_SEED = 42
MIN_ARTICLES_FOR_PROJECTION = 2


def user_table(comments: pd.DataFrame, top_share: float = FREQUENT_TOP_SHARE) -> tuple[pd.DataFrame, int]:
    """One row per user: n_articles (bipartite degree), n_comments, is_frequent."""
    users = comments.groupby("user_hash").agg(n_articles=("article_id", "nunique"),
                                              n_comments=("comment_id", "size")).reset_index()
    threshold = int(np.ceil(users["n_articles"].quantile(1 - top_share)))
    users["is_frequent"] = users["n_articles"] >= threshold
    return users, threshold


def projection(comments: pd.DataFrame, min_articles: int = MIN_ARTICLES_FOR_PROJECTION) -> nx.Graph:
    pairs = comments[["user_hash", "article_id"]].drop_duplicates()
    deg = pairs.groupby("user_hash").size()
    pairs = pairs[pairs["user_hash"].map(deg) >= min_articles]
    u_codes, u_index = pd.factorize(pairs["user_hash"])
    a_codes, _ = pd.factorize(pairs["article_id"])
    b = sparse.csr_matrix((np.ones(len(pairs)), (u_codes, a_codes)))
    shared = sparse.triu(b @ b.T, k=1).tocoo()
    g = nx.Graph()
    g.add_nodes_from(u_index)
    g.add_weighted_edges_from(zip(u_index[shared.row], u_index[shared.col], shared.data.astype(int)))
    return g


def communities(g: nx.Graph, seed: int = LOUVAIN_SEED) -> tuple[dict, float]:
    parts = nx.community.louvain_communities(g, weight="weight", seed=seed)
    parts = sorted(parts, key=len, reverse=True)
    membership = {u: i for i, part in enumerate(parts) for u in part}
    return membership, nx.community.modularity(g, parts, weight="weight")


def describe_communities(comments: pd.DataFrame, articles: pd.DataFrame, membership: dict, top: int = 10) -> list:
    cat = articles.set_index("article_id")["category"]
    c = comments[comments["user_hash"].isin(membership.keys())].copy()
    c["community"] = c["user_hash"].map(membership)
    c["category"] = c["article_id"].map(cat)
    out = []
    for cid, g in c.groupby("community"):
        if cid >= top:
            break
        shares = g["category"].value_counts(normalize=True)
        out.append({"community": int(cid), "users": int(g["user_hash"].nunique()), "comments": int(len(g)),
                    "articles": int(g["article_id"].nunique()),
                    "category_share": {k: round(float(v), 3) for k, v in shares.items()}})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="User-article graph and communities")
    parser.parse_args(argv)
    d = PROCESSED_DIR
    comments = pd.read_parquet(d / "comments.parquet", columns=["comment_id", "article_id", "user_hash"])
    articles = pd.read_parquet(d / "articles.parquet", columns=["article_id", "category"])

    users, threshold = user_table(comments)
    g = projection(comments)
    membership, modularity = communities(g)
    users["community"] = users["user_hash"].map(membership).astype("Int64")
    users.to_parquet(d / "users.parquet", index=False)

    frequent = users[users["is_frequent"]]
    sizes = pd.Series(membership).value_counts()
    summary = {
        "bipartite": {"users": int(len(users)), "articles": int(comments["article_id"].nunique()),
                      "edges": int(comments[["user_hash", "article_id"]].drop_duplicates().shape[0]),
                      "user_degree_quantiles": users["n_articles"].quantile([.5, .9, .95, .99]).to_dict(),
                      "user_degree_max": int(users["n_articles"].max())},
        "frequent": {"rule": f"top {FREQUENT_TOP_SHARE:.0%} by number of articles", "min_articles": threshold,
                     "users": int(len(frequent)),
                     "comment_share": round(float(frequent["n_comments"].sum() / users["n_comments"].sum()), 4)},
        "projection": {"min_articles": MIN_ARTICLES_FOR_PROJECTION, "nodes": g.number_of_nodes(),
                       "edges": g.number_of_edges(), "density": round(nx.density(g), 5)},
        "louvain": {"seed": LOUVAIN_SEED, "communities": int(len(sizes)), "modularity": round(modularity, 4),
                    "largest_sizes": sizes.head(10).astype(int).tolist(),
                    "communities_with_at_least_20_users": int((sizes >= 20).sum())},
        "top_communities": describe_communities(comments, articles, membership),
    }
    (d / "graph_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
