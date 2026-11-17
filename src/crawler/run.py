"""Command line entry point.

Usage (from the project root):
    /usr/bin/python3 -m src.crawler.run discover          # find and store new articles
    /usr/bin/python3 -m src.crawler.run comments          # comments of articles older than 48h
    /usr/bin/python3 -m src.crawler.run daily             # discover + comments (used by the scheduler)
    /usr/bin/python3 -m src.crawler.run stats             # print collection statistics

    --pages N        scan category pages 1..N (older pages give articles already past 48h);
                     in 'daily' mode, comments are collected after each page round
    --max-minutes M  time budget: after M minutes no new article or comment thread is started,
                     the one in progress is finished

Exit codes: 0 ok, 1 unexpected error, 2 blocked by the server (HTTP 403/429), 3 another run is active.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..processing.anonymize import UserHasher, load_or_create_key
from . import config
from .articles import ParseError, category_url, parse_article, parse_category_page
from .comments import fetch_thread, transform_comment
from .http_client import CrawlBlockedError, FetchError, PoliteSession
from .storage import Store, utc_now

log = logging.getLogger("crawler")


def setup_logging(level: str) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = config.LOG_DIR / f"crawler_{datetime.now().strftime('%Y-%m-%d')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


class TimeBudget:
    """Deadline for a run; checked before starting each article or comment thread."""

    def __init__(self, minutes: float = 0, clock=time.monotonic):
        self.clock = clock
        self.end = clock() + minutes * 60 if minutes > 0 else None
        self._logged = False

    def expired(self) -> bool:
        if self.end is None or self.clock() < self.end:
            return False
        if not self._logged:
            log.info("Time budget reached: no new work is started")
            self._logged = True
        return True


def discover(session: PoliteSession, store: Store, categories: list, max_per_category: int = 0,
             page: int = 1, budget: TimeBudget = None, exhausted: set = None) -> dict:
    """Fetch new articles listed on page `page` of each category.

    `exhausted` collects categories whose page redirected elsewhere (end of pagination).
    """
    budget = budget or TimeBudget()
    exhausted = exhausted if exhausted is not None else set()
    summary = {"listed": 0, "new": 0, "failed": 0, "skipped_failed": 0}
    known = store.known_urls()
    for slug in categories:
        if slug in exhausted or budget.expired():
            continue
        url_page = category_url(slug, page)
        try:
            resp = session.get(url_page)
        except FetchError as exc:
            log.error("Category %s page %d: %s", slug, page, exc)
            continue
        if resp.url.rstrip("/") != url_page:
            log.info("Category %s page %d redirects to %s; stop paging", slug, page, resp.url)
            exhausted.add(slug)
            continue
        urls = parse_category_page(resp.text)
        summary["listed"] += len(urls)
        todo = [u for u in urls if u not in known]
        if max_per_category:
            todo = todo[:max_per_category]
        log.info("Category %s page %d: %d listed, %d new", slug, page, len(urls), len(todo))
        for url in todo:
            if budget.expired():
                break
            if store.failed_attempts(url) >= config.MAX_ARTICLE_FETCH_ATTEMPTS:
                summary["skipped_failed"] += 1
                continue
            try:
                html = session.get(url).text
                article = parse_article(html, url, slug, utc_now().replace(microsecond=0).isoformat())
            except (FetchError, ParseError) as exc:
                log.warning("Article %s: %s", url, exc)
                store.record_failed_url(url, slug, str(exc))
                summary["failed"] += 1
                continue
            if store.add_article(article):
                known.add(url)
                summary["new"] += 1
    return summary


def collect_comments(session: PoliteSession, store: Store, hasher: UserHasher,
                     min_age_hours: float, limit: int = 0, budget: TimeBudget = None) -> dict:
    budget = budget or TimeBudget()
    summary = {"articles": 0, "comments": 0, "failed": 0}
    due = store.articles_due(utc_now(), min_age_hours, config.MAX_COMMENT_FETCH_ATTEMPTS, limit or None)
    log.info("%d articles due for comment collection", len(due))
    for row in due:
        if budget.expired():
            break
        article_id, object_type = row["article_id"], row["object_type"]
        try:
            raw_items, info = fetch_thread(session, article_id, object_type)
            fetched_at = utc_now().replace(microsecond=0).isoformat()
            comments = [transform_comment(it, hasher, fetched_at) for it in raw_items]
        except FetchError as exc:
            log.warning("Comments of %s: %s", article_id, exc)
            store.mark_comments_failed(article_id, str(exc))
            summary["failed"] += 1
            continue
        n = store.save_comments(article_id, comments, info, fetched_at)
        summary["articles"] += 1
        summary["comments"] += n
        log.info("Article %s: %d comments (API totalitem %s)", article_id, n, info["api_totalitem"])
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VnExpress crawler (IT4868 project)")
    parser.add_argument("command", choices=["discover", "comments", "daily", "stats"])
    parser.add_argument("--categories", nargs="+", default=list(config.CATEGORIES),
                        choices=list(config.CATEGORIES))
    parser.add_argument("--max-per-category", type=int, default=0,
                        help="limit new articles per category in this run (0 = no limit)")
    parser.add_argument("--min-age-hours", type=float, default=config.COMMENT_MIN_AGE_HOURS,
                        help="collect comments only for articles at least this old")
    parser.add_argument("--limit", type=int, default=0, help="max articles for comment collection (0 = all)")
    parser.add_argument("--pages", type=int, default=1, help="scan category pages 1..N (default 1)")
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="stop starting new work after this many minutes (0 = no limit)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.pages < 1:
        parser.error("--pages must be >= 1")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "stats":
        store = Store(config.DATA_DIR)
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        store.close()
        return 0

    setup_logging(args.log_level)
    lock_file = open(config.DATA_DIR / ".crawler.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("Another crawler run is active; exiting")
        return 3

    store = Store(config.DATA_DIR)
    session = PoliteSession()
    started = datetime.now(timezone.utc)
    budget = TimeBudget(args.max_minutes)
    log.info("Run '%s' started (pages %d, time budget %s min, data dir %s, user agent %r)", args.command,
             args.pages, args.max_minutes or "none", config.DATA_DIR, session.user_agent)
    try:
        hasher = None
        if args.command in ("comments", "daily"):
            hasher = UserHasher(load_or_create_key(config.HASH_KEY_FILE))
        if args.command == "comments":
            log.info("Comment summary: %s",
                     collect_comments(session, store, hasher, args.min_age_hours, args.limit, budget))
        else:
            exhausted: set = set()
            for page in range(1, args.pages + 1):
                if budget.expired() or len(exhausted) == len(args.categories):
                    break
                log.info("Discovery page %d summary: %s", page,
                         discover(session, store, args.categories, args.max_per_category, page, budget, exhausted))
                if args.command == "daily":
                    log.info("Comment summary after page %d: %s", page,
                             collect_comments(session, store, hasher, args.min_age_hours, args.limit, budget))
    except CrawlBlockedError as exc:
        log.error("%s", exc)
        return 2
    except Exception:
        log.exception("Unexpected error")
        return 1
    finally:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        log.info("Run finished in %.0fs with %d HTTP requests", elapsed, session.request_count)
        store.close()
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
