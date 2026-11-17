"""Article discovery (category pages) and article page parsing."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup

from . import config

ARTICLE_URL_RE = re.compile(r"^https://vnexpress\.net/[a-z0-9-]+-(\d{7,8})\.html$")
COMMENT_CONFIG_RE = re.compile(r"data-component-input='(\{\"type\":\"comment\"[^']*)'")


class ParseError(ValueError):
    pass


@dataclass
class Article:
    article_id: int
    url: str
    category: str                 # slug of the category page it was found on
    category_path: str            # e.g. "VnExpress,Thời sự,Chính trị"
    title: str
    sapo: str
    body: str
    published_at: str             # ISO 8601, UTC
    article_type: str             # "text", "video", "photo", ... (meta its_type)
    word_count: Optional[int]
    comment_enabled: bool
    object_type: Optional[int]    # objecttype for the comment API, None if no comment box
    internal_links: list = field(default_factory=list)  # ids of linked VnExpress articles
    tags: str = ""
    crawled_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def article_id_from_url(url: str) -> Optional[int]:
    m = ARTICLE_URL_RE.match(url.split("?")[0].split("#")[0])
    return int(m.group(1)) if m else None


def parse_category_page(html: str) -> list:
    """Return article URLs listed on a category page, in page order, without duplicates."""
    soup = BeautifulSoup(html, "lxml")
    urls: list = []
    seen = set()
    for a in soup.select("h3.title-news a[href], h2.title-news a[href]"):
        url = a["href"].split("?")[0].split("#")[0]
        if ARTICLE_URL_RE.match(url) and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _meta(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name}) \
        or soup.find("meta", attrs={"itemprop": name})
    return (tag.get("content") or "").strip() if tag else ""


def _parse_published(soup: BeautifulSoup) -> str:
    raw = _meta(soup, "pubdate")
    if raw:
        try:
            return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    unix = _meta(soup, "its_publication")
    if unix.isdigit():
        return datetime.fromtimestamp(int(unix), tz=timezone.utc).isoformat()
    raise ParseError("No publication time found")


def parse_comment_config(html: str) -> Optional[dict]:
    m = COMMENT_CONFIG_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def parse_article(html: str, url: str, category: str, crawled_at: str) -> Article:
    soup = BeautifulSoup(html, "lxml")
    meta_id = _meta(soup, "tt_article_id")
    article_id = int(meta_id) if meta_id.isdigit() else article_id_from_url(url)
    if article_id is None:
        raise ParseError(f"No article id for {url}")

    title_el = soup.select_one("h1.title-detail")
    title = title_el.get_text(" ", strip=True) if title_el else _meta(soup, "its_title")
    if not title:
        raise ParseError(f"No title for {url}")
    sapo_el = soup.select_one("p.description")
    sapo = sapo_el.get_text(" ", strip=True) if sapo_el else ""

    body_el = soup.select_one("article.fck_detail")
    paragraphs = body_el.select("p.Normal") if body_el else []
    body = "\n".join(p.get_text(" ", strip=True) for p in paragraphs if p.get_text(strip=True))

    internal_links: list = []
    if body_el:
        for a in body_el.select("a[href]"):
            linked = article_id_from_url(a["href"])
            if linked and linked != article_id and linked not in internal_links:
                internal_links.append(linked)

    cfg = parse_comment_config(html)
    object_type = None
    if cfg and str(cfg.get("article_type", "")).isdigit():
        object_type = int(cfg["article_type"])

    wc = _meta(soup, "its_wordcount")
    return Article(
        article_id=article_id,
        url=url,
        category=category,
        category_path=_meta(soup, "tt_list_folder_name"),
        title=title,
        sapo=sapo,
        body=body,
        published_at=_parse_published(soup),
        article_type=_meta(soup, "its_type") or "unknown",
        word_count=int(wc) if wc.isdigit() else None,
        comment_enabled=object_type is not None,
        object_type=object_type,
        internal_links=internal_links,
        tags=_meta(soup, "article:tag"),
        crawled_at=crawled_at,
    )


def category_url(slug: str, page: int = 1) -> str:
    """Page 1 is the category front page; page n > 1 is '<slug>-p<n>' (works up to about p10-p50)."""
    if slug not in config.CATEGORIES:
        raise ValueError(f"Unknown category {slug!r}")
    if page < 1:
        raise ValueError(f"Invalid page {page}")
    return f"{config.BASE_URL}/{slug}" if page == 1 else f"{config.BASE_URL}/{slug}-p{page}"
