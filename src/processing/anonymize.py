"""Pseudonymisation and cleaning of comment text.

User ids on VnExpress are small integers, so a plain SHA-256 of the id could be
reversed by brute force. Ids are therefore hashed with HMAC-SHA256 under a
secret project key that lives outside the repository (secrets/hash_key).
The same key must be used for the whole project, otherwise hashes of the same
user will not match across runs.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import os
import re
import secrets
from pathlib import Path

from bs4 import BeautifulSoup

HASH_LENGTH = 12

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Vietnamese phone numbers: 0xxxxxxxxx or +84xxxxxxxxx, digits optionally
# separated by spaces, dots or dashes. A bare "84" prefix is deliberately not
# accepted: it would also match money amounts such as "84.000.000.000".
PHONE_RE = re.compile(r"(?<![\d.,])(?:\+84|0)(?:[\s.\-]?\d){8,10}(?![\d.,]?\d)")
WHITESPACE_RE = re.compile(r"\s+")


def load_or_create_key(path: Path) -> bytes:
    """Return the project hash key, creating it (mode 0600) on first use."""
    path = Path(path)
    if path.exists():
        key = path.read_text(encoding="ascii").strip()
        if len(key) < 32:
            raise ValueError(f"Hash key in {path} is too short")
        return key.encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(key + "\n")
    return key.encode("ascii")


class UserHasher:
    """Maps raw user ids to stable pseudonyms."""

    def __init__(self, key: bytes):
        if not key:
            raise ValueError("Empty hash key")
        self._key = key

    def __call__(self, user_id) -> str:
        if user_id is None or str(user_id).strip() == "":
            return ""
        digest = hmac.new(self._key, str(user_id).strip().encode("utf-8"), hashlib.sha256)
        return digest.hexdigest()[:HASH_LENGTH]


def _is_mention_span(tag) -> bool:
    classes = tag.get("class") or []
    return tag.name == "span" and tag.has_attr("data-userid") and "reply_name" in classes


def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)


def clean_comment_html(raw_html: str, hasher: UserHasher) -> tuple[str, list[str]]:
    """Convert comment HTML to anonymised plain text.

    Returns (text, mention_hashes). Mention spans such as
    <span class="reply_name myuser" data-userid="123">@Name</span>
    are replaced by "@USER_<hash>" so that the display name is dropped.
    """
    if not raw_html:
        return "", []
    soup = BeautifulSoup(raw_html, "lxml")
    mentions: list[str] = []
    for span in soup.find_all(_is_mention_span):
        user_hash = hasher(span["data-userid"])
        if user_hash:
            mentions.append(user_hash)
        span.replace_with(f"@USER_{user_hash}" if user_hash else "@USER")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = html.unescape(soup.get_text(" "))
    text = WHITESPACE_RE.sub(" ", text).strip()
    return mask_pii(text), mentions
