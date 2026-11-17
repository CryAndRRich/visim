"""Reply trees of one article's comment thread.

Two versions are built for every article:
- two-level: the parent given by the API. VnExpress stores every reply under its top-level
  comment, so depth is at most 2.
- reconstructed: a reply that mentions a user (`mention_user_hash`) is attached to that user's
  latest earlier comment in the same top-level thread (the top-level comment itself or one of
  its replies). Without a usable mention the API parent is kept.

The article is a virtual root node `ROOT`; top-level comments hang under it. A reconstructed
parent always comes earlier in the thread than its child, so neither version has a cycle.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

ROOT = 0


def _time_key(c: dict) -> tuple:
    return (c["created_at"], c["comment_id"])


def _thread_roots(comments: list) -> tuple[dict, int]:
    """Map comment_id -> id of its top-level comment. Replies whose parent is missing
    (for example a deleted top-level comment) are treated as top-level. Returns (map, n_orphans)."""
    ids = {c["comment_id"] for c in comments}
    roots, orphans = {}, 0
    for c in comments:
        if c["is_reply"] and c["parent_id"] in ids and c["parent_id"] != c["comment_id"]:
            roots[c["comment_id"]] = c["parent_id"]
        else:
            if c["is_reply"]:
                orphans += 1
            roots[c["comment_id"]] = c["comment_id"]
    return roots, orphans


def two_level_parents(comments: Iterable[dict]) -> dict:
    """comment_id -> parent comment_id (ROOT for top-level comments)."""
    comments = list(comments)
    roots, _ = _thread_roots(comments)
    return {cid: (ROOT if root == cid else root) for cid, root in roots.items()}


def reconstructed_parents(comments: Iterable[dict]) -> tuple[dict, dict]:
    """comment_id -> parent comment_id after attaching mention replies to the mentioned comment.

    Returns (parents, info). `info` counts:
      mention_replies   replies carrying a mention of another user
      reattached        replies moved below a reply (edge differs from the two-level tree)
      to_top_author     mention points at the top-level author: parent unchanged
      unresolved        mentioned user has no earlier comment in the thread: parent unchanged
      self_mentions     reply mentions its own author: ignored
      orphans           replies whose top-level comment is missing (treated as top-level)
    """
    comments = list(comments)
    roots, orphans = _thread_roots(comments)
    parents = {cid: (ROOT if root == cid else root) for cid, root in roots.items()}
    info = {"mention_replies": 0, "reattached": 0, "to_top_author": 0, "unresolved": 0,
            "self_mentions": 0, "orphans": orphans}

    threads = defaultdict(list)
    for c in comments:
        threads[roots[c["comment_id"]]].append(c)
    for root_id, members in threads.items():
        top = [c for c in members if c["comment_id"] == root_id]
        replies = sorted((c for c in members if c["comment_id"] != root_id), key=_time_key)
        latest_by_user: dict = {}
        for c in top + replies:          # top-level comment first, then replies in time order
            target = c.get("mention_user_hash")
            if c["comment_id"] != root_id and target:
                if target == c["user_hash"]:
                    info["self_mentions"] += 1
                else:
                    info["mention_replies"] += 1
                    found = latest_by_user.get(target)
                    if found is None:
                        info["unresolved"] += 1
                    elif found == root_id:
                        info["to_top_author"] += 1
                    else:
                        parents[c["comment_id"]] = found
                        info["reattached"] += 1
            latest_by_user[c["user_hash"]] = c["comment_id"]
    return parents, info
