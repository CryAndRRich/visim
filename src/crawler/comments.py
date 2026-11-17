"""Comment collection through the (unofficial) VnExpress comment API.

The API returns two levels only:
top-level comments (index/get) and their replies (index/getreplay). Deeper
reply-to relations are recovered later from the mention hashes stored here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..processing.anonymize import UserHasher, clean_comment_html
from . import config
from .http_client import FetchError, PoliteSession

log = logging.getLogger(__name__)


class CommentApiError(FetchError):
    pass


def _check(payload: dict, what: str) -> dict:
    if not isinstance(payload, dict) or payload.get("error") not in (0, "0"):
        raise CommentApiError(f"{what}: API error {payload.get('error') if isinstance(payload, dict) else payload!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise CommentApiError(f"{what}: missing data object")
    return data


def fetch_top_level(session: PoliteSession, article_id: int, object_type: int) -> tuple:
    """Return (items, total, totalitem) for all top-level comments of an article."""
    items: list = []
    seen = set()
    total = totalitem = 0
    offset = 0
    for _ in range(config.MAX_COMMENT_PAGES):
        params = {
            "offset": offset, "limit": config.COMMENT_PAGE_LIMIT, "frommobile": 0, "sort": "like",
            "is_onload": 1, "objectid": article_id, "objecttype": object_type, "siteid": config.SITE_ID,
        }
        data = _check(session.get_json(config.COMMENT_API, params), f"comments of {article_id}")
        total = int(data.get("total") or 0)
        totalitem = int(data.get("totalitem") or 0)
        page = data.get("items") or []
        new = [it for it in page if it.get("comment_id") not in seen]
        for it in new:
            seen.add(it.get("comment_id"))
        items.extend(new)
        offset += len(page)
        if not page or not new or offset >= total:
            break
    else:
        log.warning("Article %s: stopped after %d comment pages", article_id, config.MAX_COMMENT_PAGES)
    return items, total, totalitem


def fetch_replies(session: PoliteSession, article_id: int, object_type: int,
                  comment_id: int, expected: int) -> list:
    items: list = []
    seen = set()
    offset = 0
    for _ in range(config.MAX_REPLY_PAGES):
        params = {
            "siteid": config.SITE_ID, "objectid": article_id, "objecttype": object_type,
            "id": comment_id, "limit": config.REPLY_PAGE_LIMIT, "offset": offset, "sort_by": "like",
        }
        data = _check(session.get_json(config.REPLY_API, params), f"replies of {comment_id}")
        page = data.get("items") or []
        new = [it for it in page if it.get("comment_id") not in seen]
        for it in new:
            seen.add(it.get("comment_id"))
        items.extend(new)
        offset += len(page)
        if not page or not new or offset >= expected:
            break
    return items


def fetch_thread(session: PoliteSession, article_id: int, object_type: int) -> tuple:
    """Return (raw_items, info) with all comments and replies of one article."""
    tops, total, totalitem = fetch_top_level(session, article_id, object_type)
    replies: list = []
    for c in tops:
        expected = int((c.get("replys") or {}).get("total") or 0)
        inline = (c.get("replys") or {}).get("items") or []
        if expected == 0:
            continue
        if len(inline) >= expected:
            replies.extend(inline)
        else:
            replies.extend(fetch_replies(session, article_id, object_type, c["comment_id"], expected))
    # De-duplicate across top-level and reply lists
    by_id: dict = {}
    for it in tops + replies:
        if it.get("comment_id") is not None:
            by_id[it["comment_id"]] = it
    items = list(by_id.values())
    info = {"api_total": total, "api_totalitem": totalitem, "n_collected": len(items)}
    if totalitem and len(items) < config.COMMENT_COUNT_TOLERANCE * totalitem:
        log.warning("Article %s: collected %d of %d comments reported by the API", article_id, len(items), totalitem)
    return items, info


def transform_comment(raw: dict, hasher: UserHasher, crawled_at: str) -> dict:
    """Map one raw API item to the anonymised comment record.

    full_name and every other personal field of the raw item are dropped here.
    """
    text, mentions = clean_comment_html(raw.get("content") or "", hasher)
    created = raw.get("creation_time")
    created_at = (datetime.fromtimestamp(int(created), tz=timezone.utc).isoformat()
                  if str(created or "").isdigit() else None)
    comment_id = int(raw["comment_id"])
    parent_id = int(raw.get("parent_id") or comment_id)
    return {
        "comment_id": comment_id,
        "parent_id": parent_id,
        "article_id": int(raw.get("article_id") or 0),
        "is_reply": parent_id != comment_id,
        "user_hash": hasher(raw.get("userid")),
        "text": text,
        "created_at": created_at,
        "likes": int(raw.get("userlike") or 0),
        "mention_user_hash": mentions[0] if mentions else None,
        "crawled_at": crawled_at,
    }
