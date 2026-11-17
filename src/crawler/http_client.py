"""Polite HTTP client: per-host rate limit, robots.txt, retries, hard stop on 403/429."""
from __future__ import annotations

import logging
import time
import urllib.robotparser
from typing import Callable, Optional
from urllib.parse import urlsplit

import requests

from . import config

log = logging.getLogger(__name__)


class CrawlBlockedError(RuntimeError):
    """The server refused us (HTTP 403/429). The whole run must stop."""


class FetchError(RuntimeError):
    """A single request failed permanently (after retries, or 4xx other than 403/429)."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class DisallowedByRobotsError(FetchError):
    pass


class PoliteSession:
    def __init__(
        self,
        user_agent: str = config.USER_AGENT,
        min_interval: float = config.MIN_INTERVAL_SECONDS,
        timeout: float = config.REQUEST_TIMEOUT_SECONDS,
        max_retries: int = config.MAX_RETRIES,
        backoff: tuple = config.RETRY_BACKOFF_SECONDS,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        respect_robots: bool = True,
    ):
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "vi,en;q=0.8"})
        self._sleep = sleep
        self._clock = clock
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self.respect_robots = respect_robots
        self.request_count = 0

    # -- rate limiting -------------------------------------------------------
    def _wait_for_slot(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            remaining = self.min_interval - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request[host] = self._clock()

    # -- robots.txt ------------------------------------------------------------
    def _robots_for(self, scheme: str, host: str) -> Optional[urllib.robotparser.RobotFileParser]:
        """None means 'everything allowed'. Fetched once per host."""
        if host in self._robots:
            return self._robots[host]
        robots_url = f"{scheme}://{host}/robots.txt"
        parser: Optional[urllib.robotparser.RobotFileParser]
        try:
            resp = self._raw_get(robots_url, host, params=None)
            if resp.status_code >= 500:
                # Standard practice: server error on robots.txt means "do not crawl".
                parser = urllib.robotparser.RobotFileParser()
                parser.disallow_all = True
            elif resp.status_code >= 400 or resp.status_code == 204 or not resp.text.strip():
                parser = None
            else:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(resp.text.splitlines())
        except requests.RequestException as exc:
            log.warning("Could not fetch %s (%s); treating host as disallowed", robots_url, exc)
            parser = urllib.robotparser.RobotFileParser()
            parser.disallow_all = True
        self._robots[host] = parser
        return parser

    def _check_robots(self, url: str) -> None:
        if not self.respect_robots:
            return
        parts = urlsplit(url)
        parser = self._robots_for(parts.scheme, parts.netloc)
        if parser is not None and not parser.can_fetch(self.user_agent, url):
            raise DisallowedByRobotsError(f"robots.txt disallows {url}")

    # -- requests --------------------------------------------------------------
    def _raw_get(self, url: str, host: str, params: Optional[dict]) -> requests.Response:
        self._wait_for_slot(host)
        self.request_count += 1
        return self.session.get(url, params=params, timeout=self.timeout)

    def get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        self._check_robots(url)
        host = urlsplit(url).netloc
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._raw_get(url, host, params)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code in (403, 429):
                    raise CrawlBlockedError(f"HTTP {resp.status_code} from {url}; stopping the run")
                if resp.status_code < 400:
                    return resp
                if resp.status_code < 500:
                    raise FetchError(f"HTTP {resp.status_code} for {url}", status=resp.status_code)
                last_error = f"HTTP {resp.status_code}"
            if attempt < self.max_retries:
                delay = self.backoff[min(attempt, len(self.backoff) - 1)]
                log.warning("%s for %s; retry %d/%d in %ss", last_error, url, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
        raise FetchError(f"Giving up on {url} after {self.max_retries + 1} attempts: {last_error}")

    def get_json(self, url: str, params: Optional[dict] = None) -> dict:
        resp = self.get(url, params=params)
        try:
            return resp.json()
        except ValueError as exc:
            raise FetchError(f"Invalid JSON from {url}: {exc}") from exc
