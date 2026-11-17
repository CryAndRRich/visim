"""Storage: articles.jsonl (append), comments/<article_id>.jsonl (atomic), state.sqlite3.

The SQLite file is the source of truth for crawl state (what has been seen,
what still needs comments). JSONL files hold the content.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .articles import Article

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    article_id          INTEGER PRIMARY KEY,
    url                 TEXT NOT NULL UNIQUE,
    category            TEXT NOT NULL,
    published_at        TEXT NOT NULL,
    object_type         INTEGER,
    comment_enabled     INTEGER NOT NULL,
    discovered_at       TEXT NOT NULL,
    comments_status     TEXT NOT NULL DEFAULT 'pending'
                        CHECK (comments_status IN ('pending', 'done', 'failed', 'no_comment_box')),
    comments_fetched_at TEXT,
    n_comments          INTEGER,
    api_total           INTEGER,
    api_totalitem       INTEGER,
    fail_count          INTEGER NOT NULL DEFAULT 0,
    last_error          TEXT
);
CREATE TABLE IF NOT EXISTS failed_urls (
    url          TEXT PRIMARY KEY,
    category     TEXT NOT NULL,
    attempts     INTEGER NOT NULL,
    last_error   TEXT,
    last_attempt TEXT NOT NULL
);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.comments_dir = self.data_dir / "comments"
        self.articles_path = self.data_dir / "articles.jsonl"
        self.comments_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.data_dir / "state.sqlite3"))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # -- discovery -------------------------------------------------------------
    def known_urls(self) -> set:
        return {r["url"] for r in self.db.execute("SELECT url FROM articles")}

    def failed_attempts(self, url: str) -> int:
        row = self.db.execute("SELECT attempts FROM failed_urls WHERE url = ?", (url,)).fetchone()
        return row["attempts"] if row else 0

    def record_failed_url(self, url: str, category: str, error: str) -> None:
        self.db.execute(
            """INSERT INTO failed_urls (url, category, attempts, last_error, last_attempt)
               VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(url) DO UPDATE SET attempts = attempts + 1,
                   last_error = excluded.last_error, last_attempt = excluded.last_attempt""",
            (url, category, error[:500], utc_now().isoformat()),
        )
        self.db.commit()

    def add_article(self, article: Article) -> bool:
        """Persist a new article. Returns False if it was already known."""
        exists = self.db.execute("SELECT 1 FROM articles WHERE article_id = ? OR url = ?",
                                 (article.article_id, article.url)).fetchone()
        if exists:
            return False
        with open(self.articles_path, "a", encoding="utf-8") as f:
            f.write(article.to_json() + "\n")
        self.db.execute(
            """INSERT INTO articles (article_id, url, category, published_at, object_type,
                   comment_enabled, discovered_at, comments_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (article.article_id, article.url, article.category, article.published_at, article.object_type,
             int(article.comment_enabled), article.crawled_at,
             "pending" if article.comment_enabled else "no_comment_box"),
        )
        self.db.execute("DELETE FROM failed_urls WHERE url = ?", (article.url,))
        self.db.commit()
        return True

    # -- comments --------------------------------------------------------------
    def articles_due(self, now: datetime, min_age_hours: float, max_attempts: int,
                     limit: Optional[int] = None) -> list:
        # Same format as published_at (UTC, no microseconds) so string comparison is valid
        cutoff = (now - timedelta(hours=min_age_hours)).astimezone(timezone.utc).replace(microsecond=0).isoformat()
        sql = """SELECT article_id, object_type, published_at, fail_count FROM articles
                 WHERE comment_enabled = 1
                   AND (comments_status = 'pending' OR (comments_status = 'failed' AND fail_count < ?))
                   AND published_at <= ?
                 ORDER BY published_at"""
        params: list = [max_attempts, cutoff]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return list(self.db.execute(sql, params))

    def save_comments(self, article_id: int, comments: Iterable[dict], info: dict, fetched_at: str) -> int:
        comments = list(comments)
        target = self.comments_dir / f"{article_id}.jsonl"
        fd, tmp = tempfile.mkstemp(dir=str(self.comments_dir), prefix=f".{article_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for c in comments:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self.db.execute(
            """UPDATE articles SET comments_status = 'done', comments_fetched_at = ?, n_comments = ?,
                   api_total = ?, api_totalitem = ?, last_error = NULL WHERE article_id = ?""",
            (fetched_at, len(comments), info.get("api_total"), info.get("api_totalitem"), article_id),
        )
        self.db.commit()
        return len(comments)

    def mark_comments_failed(self, article_id: int, error: str) -> None:
        self.db.execute(
            """UPDATE articles SET comments_status = 'failed', fail_count = fail_count + 1, last_error = ?
               WHERE article_id = ?""",
            (error[:500], article_id),
        )
        self.db.commit()

    # -- reading -----------------------------------------------------------------
    def iter_articles(self) -> Iterator[dict]:
        """Articles from articles.jsonl, de-duplicated by id (last line wins)."""
        if not self.articles_path.exists():
            return iter(())
        latest: dict = {}
        with open(self.articles_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    latest[rec["article_id"]] = rec
        return iter(latest.values())

    def iter_comments(self) -> Iterator[dict]:
        for path in sorted(self.comments_dir.glob("*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def stats(self) -> dict:
        q = self.db.execute
        by_status = {r[0]: r[1] for r in q("SELECT comments_status, COUNT(*) FROM articles GROUP BY 1")}
        by_category = {r[0]: r[1] for r in q("SELECT category, COUNT(*) FROM articles GROUP BY 1")}
        row = q("SELECT COUNT(*), COALESCE(SUM(n_comments), 0) FROM articles WHERE comments_status = 'done'").fetchone()
        eligible = q("""SELECT COUNT(*) FROM articles WHERE comments_status = 'done' AND n_comments >= 15""").fetchone()[0]
        return {
            "articles": sum(by_status.values()),
            "by_status": by_status,
            "by_category": by_category,
            "articles_with_comments_fetched": row[0],
            "comments": row[1],
            "articles_with_at_least_15_comments": eligible,
            "failed_urls": q("SELECT COUNT(*) FROM failed_urls").fetchone()[0],
        }
