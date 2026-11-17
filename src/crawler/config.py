"""Crawler configuration."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Target site -----------------------------------------------------------
BASE_URL = "https://vnexpress.net"
# Category slug -> display name
CATEGORIES = {
    "thoi-su": "Thời sự",
    "kinh-doanh": "Kinh doanh",
    "the-gioi": "Thế giới",
    "phap-luat": "Pháp luật",
    "giai-tri": "Giải trí",
}

# Unofficial comment endpoints (not documented by VnExpress; may change)
COMMENT_API = "https://usi-saas.vnexpress.net/index/get"
REPLY_API = "https://usi-saas.vnexpress.net/index/getreplay"
SITE_ID = 1000000
COMMENT_PAGE_LIMIT = 24   # page size used by the site itself
REPLY_PAGE_LIMIT = 50
# Safety caps so a malformed response can never cause an endless loop
MAX_COMMENT_PAGES = 200
MAX_REPLY_PAGES = 50

# --- Politeness ------------------------------------------------------------
MIN_INTERVAL_SECONDS = 1.5          # between two requests to the same host
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3                     # for timeouts and HTTP 5xx
RETRY_BACKOFF_SECONDS = (5, 15, 45)

_contact = os.environ.get("CRAWLER_CONTACT", "").strip()
USER_AGENT = (
    "IT4868-research-crawler/0.1 (academic course project, non-commercial"
    + (f"; contact: {_contact}" if _contact else "")
    + ")"
)

# --- Scheduling ------------------------------------------------------------
COMMENT_MIN_AGE_HOURS = 48          # comments are collected once, 48h after publication
MAX_ARTICLE_FETCH_ATTEMPTS = 3
MAX_COMMENT_FETCH_ATTEMPTS = 3
# Warn when we collect clearly fewer comments than the API reports
COMMENT_COUNT_TOLERANCE = 0.95

# --- Paths -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CRAWLER_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
LOG_DIR = Path(os.environ.get("CRAWLER_LOG_DIR", PROJECT_ROOT / "logs"))
HASH_KEY_FILE = Path(os.environ.get("CRAWLER_HASH_KEY_FILE", PROJECT_ROOT / "secrets" / "hash_key"))
