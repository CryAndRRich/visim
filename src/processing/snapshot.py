"""Freeze the crawled data into data/processed/.

The processed data contains every article whose comments were fetched (`comments_status = 'done'`)
plus the articles without a comment box, as they were when the data was frozen (13/10/2026: 1,244
articles, 20,468 comments). Collection has stopped and the raw crawl output is no longer kept; this
module documents how the processed files were made from the crawler output (data/raw/).

Usage (from the project root):
    python -m src.processing.snapshot

Output in data/processed/:
    articles.parquet   one row per article
    comments.parquet   one row per comment, with `within_48h`
    manifest.json      creation time, counts, checksum of the article id list
    stats.json         descriptive statistics (no text)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..crawler import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTICLE_COLUMNS = ["article_id", "url", "category", "category_path", "title", "sapo", "body", "published_at",
                   "article_type", "word_count", "comment_enabled", "internal_links", "tags", "crawled_at"]


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    db = sqlite3.connect(str(data_dir / "state.sqlite3"))
    state = pd.read_sql_query(
        "SELECT article_id, comments_status, n_comments, api_totalitem, comments_fetched_at FROM articles "
        "WHERE comments_status IN ('done', 'no_comment_box')", db)
    excluded = dict(db.execute("SELECT comments_status, COUNT(*) FROM articles "
                               "WHERE comments_status NOT IN ('done', 'no_comment_box') GROUP BY 1").fetchall())
    db.close()
    keep = set(state["article_id"])

    records = {}
    with open(data_dir / "articles.jsonl", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a["article_id"] in keep and a["article_id"] not in records:
                records[a["article_id"]] = {k: a.get(k) for k in ARTICLE_COLUMNS}
    articles = pd.DataFrame(list(records.values()), columns=ARTICLE_COLUMNS)
    articles = articles.merge(state, on="article_id", how="inner").sort_values("article_id").reset_index(drop=True)

    rows = []
    for aid in articles.loc[articles["comments_status"] == "done", "article_id"]:
        path = data_dir / "comments" / f"{aid}.jsonl"
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    comments = pd.DataFrame(rows)
    return articles, comments, excluded


def add_comment_age(articles: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    pub = articles.set_index("article_id")["published_at"].map(datetime.fromisoformat)
    created = comments["created_at"].map(datetime.fromisoformat)
    age = created - comments["article_id"].map(pub)
    comments = comments.copy()
    comments["hours_after_publish"] = age.map(lambda d: round(d.total_seconds() / 3600, 3))
    comments["within_48h"] = age <= timedelta(hours=48)
    return comments


def describe(articles: pd.DataFrame, comments: pd.DataFrame) -> dict:
    done = articles[articles["comments_status"] == "done"]
    per_article = comments.groupby("article_id").size().reindex(done["article_id"], fill_value=0)
    cat = done.assign(n=per_article.values).groupby("category")["n"]
    users = comments.groupby("user_hash")["article_id"].nunique()
    per_user = comments.groupby("user_hash").size().sort_values(ascending=False)
    top1 = max(1, len(per_user) // 100)
    return {
        "articles": int(len(articles)),
        "articles_with_comments_fetched": int(len(done)),
        "articles_without_comment_box": int((articles["comments_status"] == "no_comment_box").sum()),
        "comments": int(len(comments)),
        "replies": int(comments["is_reply"].sum()),
        "replies_with_mention": int((comments["is_reply"] & comments["mention_user_hash"].notna()).sum()),
        "users": int(comments["user_hash"].nunique()),
        "users_in_at_least_2_articles": int((users >= 2).sum()),
        "users_in_at_least_5_articles": int((users >= 5).sum()),
        "comment_share_of_top_1pct_users": round(float(per_user.iloc[:top1].sum() / len(comments)), 4),
        "published_from": str(articles["published_at"].min()),
        "published_to": str(articles["published_at"].max()),
        "comments_per_article": {
            "median": float(per_article.median()), "mean": round(float(per_article.mean()), 2),
            "max": int(per_article.max()), "zero": int((per_article == 0).sum()),
            "at_least_10": int((per_article >= 10).sum()), "at_least_15": int((per_article >= 15).sum()),
        },
        "comments_within_48h_share": round(float(comments["within_48h"].mean()), 4),
        "comment_count_vs_api_totalitem": [int(done["n_comments"].sum()), int(done["api_totalitem"].sum())],
        "by_category": {
            k: {"articles": int(g.size), "comments": int(g.sum()), "median": float(g.median()),
                "at_least_15": int((g >= 15).sum())} for k, g in cat
        },
        "articles_by_publish_date": {k: int(v) for k, v in
                                     articles["published_at"].str[:10].value_counts().sort_index().items()},
    }


def create_snapshot(data_dir: Path = None, force: bool = False) -> dict:
    data_dir = data_dir or config.DATA_DIR
    out = PROCESSED_DIR
    if (out / "articles.parquet").exists() and not force:
        raise FileExistsError(f"{out} already holds the frozen data; use --force to overwrite")
    articles, comments, excluded = load_raw(data_dir)
    comments = add_comment_age(articles, comments)
    out.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(out / "articles.parquet", index=False)
    comments.to_parquet(out / "comments.parquet", index=False)
    stats = describe(articles, comments)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = ",".join(str(i) for i in articles["article_id"])
    manifest = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "created_by": "src/processing/snapshot.py",
        "articles": stats["articles"],
        "comments": stats["comments"],
        "article_ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
        "excluded_articles_by_status": excluded,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the crawled data into data/processed/")
    parser.add_argument("--force", action="store_true", help="overwrite the frozen data")
    args = parser.parse_args(argv)
    try:
        manifest = create_snapshot(force=args.force)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
