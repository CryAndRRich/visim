"""Directed user reply graph: centrality and comparison of community detection methods.

- Edge u -> v, weight = number of replies, when a comment of u answers a comment of v in the
  reconstructed tree. Replies to oneself are dropped; top-level comments add no edge.
- Centrality (link analysis): weighted in/out degree, PageRank (d = 0.85), HITS (authority: answered
  by good hubs; hub: answers good authorities), betweenness on the undirected graph.
- Communities on the undirected weighted graph: Louvain, label propagation, greedy modularity
  (Clauset-Newman-Moore), compared by modularity and pairwise NMI. Girvan-Newman is run only on the
  core subgraph (top CORE_SIZE users by degree), since it recomputes betweenness after every removal.
- For early controversy prediction the graph and the "central user" set must be built from
  training-period articles only:
  call reply_edges on that subset.

Usage (from the project root):
    python -m src.features.reply_graph
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import networkx as nx
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

from ..processing.snapshot import PROCESSED_DIR
from ..processing.trees import ROOT, reconstructed_parents

SEED = 42
DAMPING = 0.85
CORE_SIZE = 150
CENTRAL_TOP_SHARE = 0.05


def reply_edges(comments: pd.DataFrame) -> pd.DataFrame:
    """One row per (source, target) user pair with the number of replies from source to target."""
    rows = []
    for _, g in comments.groupby("article_id"):
        recs = g.to_dict("records")
        user = {r["comment_id"]: r["user_hash"] for r in recs}
        parents, _ = reconstructed_parents(recs)
        for child, parent in parents.items():
            if parent != ROOT and parent in user and user[child] != user[parent]:
                rows.append((user[child], user[parent]))
    edges = pd.DataFrame(rows, columns=["source", "target"])
    return edges.groupby(["source", "target"]).size().rename("weight").reset_index()


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges[["source", "target", "weight"]].itertuples(index=False))
    return g


def undirected(g: nx.DiGraph) -> nx.Graph:
    """u-v weight = replies in both directions."""
    ug = nx.Graph()
    ug.add_nodes_from(g)
    for u, v, w in g.edges(data="weight"):
        prev = ug.get_edge_data(u, v, {"weight": 0})["weight"]
        ug.add_edge(u, v, weight=prev + w)
    return ug


def centrality(g: nx.DiGraph) -> pd.DataFrame:
    hubs, auths = nx.hits(g, max_iter=1000, normalized=True)
    between = nx.betweenness_centrality(undirected(g), weight=None)
    return pd.DataFrame({
        "replies_received": dict(g.in_degree(weight="weight")),
        "replies_sent": dict(g.out_degree(weight="weight")),
        "users_answered_by": dict(g.in_degree()),
        "pagerank": nx.pagerank(g, alpha=DAMPING, weight="weight"),
        "hub": hubs, "authority": auths, "betweenness": between,
    }).rename_axis("user_hash").reset_index()


def _induced(g: nx.Graph, nodes) -> nx.Graph:
    """Induced subgraph with nodes and edges in sorted order: subgraph views iterate over Python sets,
    whose order changes between runs, and Louvain and label propagation depend on node order."""
    nodes = set(nodes)
    h = nx.Graph()
    h.add_nodes_from(sorted(nodes))
    h.add_weighted_edges_from(sorted((min(u, v), max(u, v), w) for u, v, w in g.edges(nodes, data="weight")
                                     if u in nodes and v in nodes))
    return h


def _largest_component(g: nx.Graph) -> nx.Graph:
    return _induced(g, max(nx.connected_components(g), key=lambda c: (len(c), min(c))))


def _membership(parts) -> dict:
    parts = sorted(parts, key=lambda p: (-len(p), min(p)))
    return {u: i for i, part in enumerate(parts) for u in part}


def compare_communities(ug: nx.Graph, seed: int = SEED) -> tuple[dict, dict]:
    """Partitions of the largest connected component by three methods; modularity and pairwise NMI."""
    lcc = _largest_component(ug)
    parts = {
        "louvain": nx.community.louvain_communities(lcc, weight="weight", seed=seed),
        "label_propagation": list(nx.community.asyn_lpa_communities(lcc, weight="weight", seed=seed)),
        "greedy_modularity": nx.community.greedy_modularity_communities(lcc, weight="weight"),
    }
    nodes = list(lcc)
    labels = {m: [_membership(p)[u] for u in nodes] for m, p in parts.items()}
    summary = {"lcc_nodes": lcc.number_of_nodes(), "lcc_edges": lcc.number_of_edges(), "methods": {}, "nmi": {}}
    for m, p in parts.items():
        sizes = sorted((len(x) for x in p), reverse=True)
        summary["methods"][m] = {"communities": len(p), "modularity": round(nx.community.modularity(lcc, p), 4),
                                 "largest_sizes": sizes[:10], "with_at_least_20_users": sum(s >= 20 for s in sizes)}
    for a, b in itertools.combinations(parts, 2):
        summary["nmi"][f"{a}-{b}"] = round(float(normalized_mutual_info_score(labels[a], labels[b])), 4)
    return _membership(parts["louvain"]), summary


def girvan_newman_core(ug: nx.Graph, size: int = CORE_SIZE) -> dict:
    """Girvan-Newman on the largest component of the top-`size` users by degree, run down to single
    nodes; keep the level with the highest modularity and compare it with Louvain on the same subgraph."""
    top = sorted(ug.degree(weight="weight"), key=lambda x: (-x[1], x[0]))[:size]
    core = _largest_component(_induced(ug, [u for u, _ in top]))
    best, best_q = None, -1.0
    for level, parts in enumerate(nx.community.girvan_newman(core)):
        q = nx.community.modularity(core, parts)    # unweighted, like the edge removal itself
        if q > best_q:
            best, best_q, best_level = parts, q, level + 1
    louvain = nx.community.louvain_communities(core, seed=SEED)
    nodes = list(core)
    gn_m, lv_m = _membership(best), _membership(louvain)
    return {"core_nodes": core.number_of_nodes(), "core_edges": core.number_of_edges(),
            "girvan_newman": {"best_level": best_level, "communities": len(best), "modularity": round(best_q, 4)},
            "louvain_unweighted": {"communities": len(louvain),
                                   "modularity": round(nx.community.modularity(core, louvain), 4)},
            "nmi": round(float(normalized_mutual_info_score([gn_m[u] for u in nodes], [lv_m[u] for u in nodes])), 4)}


def describe(g: nx.DiGraph, cent: pd.DataFrame, users: pd.DataFrame) -> dict:
    """Overlap between central users (top CENTRAL_TOP_SHARE by PageRank) and frequent commenters."""
    k = max(1, int(round(len(cent) * CENTRAL_TOP_SHARE)))
    central = set(cent.nlargest(k, "pagerank")["user_hash"])
    frequent = set(users.loc[users["is_frequent"], "user_hash"])
    merged = cent.merge(users[["user_hash", "n_comments"]], on="user_hash")
    wcc = max(nx.weakly_connected_components(g), key=len)
    return {
        "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
        "replies": int(sum(w for _, _, w in g.edges(data="weight"))),
        "reciprocity": round(nx.reciprocity(g), 4),
        "largest_wcc_share": round(len(wcc) / g.number_of_nodes(), 4),
        "degree_assortativity_in_in": round(nx.degree_assortativity_coefficient(g, x="in", y="in"), 4),
        "central_users": {"rule": f"top {CENTRAL_TOP_SHARE:.0%} by PageRank", "users": k,
                          "also_frequent": len(central & frequent),
                          "jaccard_with_frequent": round(len(central & frequent) / len(central | frequent), 4)},
        "spearman": {f"{a}~{b}": round(float(spearmanr(merged[a], merged[b])[0]), 4)
                     for a, b in [("pagerank", "n_comments"), ("pagerank", "replies_received"),
                                  ("hub", "replies_sent"), ("authority", "replies_received")]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Directed reply graph between users")
    parser.parse_args(argv)
    d = PROCESSED_DIR
    comments = pd.read_parquet(d / "comments.parquet")
    users = pd.read_parquet(d / "users.parquet")

    g = build_graph(reply_edges(comments))
    ug = undirected(g)
    cent = centrality(g)
    membership, comm = compare_communities(ug)
    cent["community"] = cent["user_hash"].map(membership).astype("Int64")
    cent.to_parquet(d / "reply_centrality.parquet", index=False)

    summary = {"graph": describe(g, cent, users), "communities": comm,
               "girvan_newman_core": girvan_newman_core(ug), "seed": SEED, "damping": DAMPING}
    (d / "reply_graph_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
