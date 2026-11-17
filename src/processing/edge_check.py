"""Sample reconstructed reply edges for manual checking.

    python -m src.processing.edge_check sample   # writes edge_check_sample.csv (ids only)
    python -m src.processing.edge_check show N   # prints edge N of the sample for reading
    python -m src.processing.edge_check summary  # summarises verdicts in edge_check.csv

Only reattached edges are sampled (edges where the reconstructed parent differs from the API
parent). A verdict file `edge_check.csv` (sample_no, child_id, parent_id, verdict, note) is filled by
the reviewer; verdict is one of: correct, wrong, unclear. Notes must not quote comment text.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .snapshot import PROCESSED_DIR
from .trees import reconstructed_parents, two_level_parents

SEED = 42
N_SAMPLE = 50


def reattached_edges(comments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aid, g in comments.groupby("article_id"):
        recs = g.to_dict("records")
        two, (rec, _) = two_level_parents(recs), reconstructed_parents(recs)
        for child, parent in rec.items():
            if parent != two[child]:
                rows.append({"article_id": aid, "child_id": child, "parent_id": parent, "top_id": two[child]})
    return pd.DataFrame(rows)


def candidates(comments: pd.DataFrame, child_id: int) -> pd.DataFrame:
    """Earlier comments of the mentioned user in the same top-level thread (the chosen one is the last)."""
    child = comments.loc[comments["comment_id"] == child_id].iloc[0]
    top = child["parent_id"]
    thread = comments[(comments["article_id"] == child["article_id"]) &
                      ((comments["parent_id"] == top) | (comments["comment_id"] == top))]
    key = (child["created_at"], child_id)
    is_earlier = [(t, i) < key for t, i in zip(thread["created_at"], thread["comment_id"])]
    earlier = thread[(thread["user_hash"] == child["mention_user_hash"]) & pd.Series(is_earlier, index=thread.index)]
    return earlier.sort_values(["created_at", "comment_id"])


def show(comments: pd.DataFrame, edge: pd.Series, width: int = 400) -> None:
    by_id = comments.set_index("comment_id")

    def line(tag, cid):
        c = by_id.loc[cid]
        print(f"[{tag}] id={cid} user={c['user_hash']} t={c['created_at'][11:19]}: {c['text'][:width]}")

    line("TOP", edge["top_id"])
    for cid in candidates(comments, edge["child_id"])["comment_id"]:
        line("CHOSEN" if cid == edge["parent_id"] else "OTHER ", cid)
    line("CHILD", edge["child_id"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["sample", "show", "summary"])
    parser.add_argument("n", nargs="?", type=int)
    args = parser.parse_args(argv)
    d = PROCESSED_DIR
    if args.action == "summary":
        v = pd.read_csv(d / "edge_check.csv")
        print(v["verdict"].value_counts().to_string())
        judged = v[v["verdict"] != "unclear"]
        print(f"correct among judged: {(judged['verdict'] == 'correct').mean():.3f} ({len(judged)} judged)")
        return 0
    comments = pd.read_parquet(d / "comments.parquet")
    if args.action == "sample":
        edges = reattached_edges(comments)
        sample = edges.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
        sample["n_candidates"] = [len(candidates(comments, cid)) for cid in sample["child_id"]]
        sample.insert(0, "sample_no", range(1, len(sample) + 1))
        sample.to_csv(d / "edge_check_sample.csv", index=False)
        print(f"{len(edges)} reattached edges; sampled {len(sample)}; "
              f"with more than one candidate: {(sample['n_candidates'] > 1).sum()}")
        return 0
    sample = pd.read_csv(d / "edge_check_sample.csv")
    show(comments, sample.iloc[args.n - 1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
