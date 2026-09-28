"""Fetches the source pages, splits them into passages and stores them.

Usage: python -m metjou_backend.ingest [sources.yaml]

Only pages listed in sources.yaml are fetched, robots.txt is respected and
there is a pause between requests to the same site. Unchanged pages are
skipped, so running it again is cheap.
"""

import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
import yaml

from metjou_backend import db
from metjou_backend.chunking import chunk_text
from metjou_backend.config import get_settings
from metjou_backend.embeddings import Embedder

log = logging.getLogger("ingest")

USER_AGENT = "MetJouBot/0.1 (safety app assistant; respects robots.txt)"
DELAY_SECONDS = 1.0
_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


@dataclass
class Source:
    organisation: str
    urls: list[str] = field(default_factory=list)
    sitemap: str | None = None
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    max_pages: int = 50


@dataclass(frozen=True)
class Page:
    title: str
    text: str


def load_sources(path: Path) -> list[Source]:
    data = yaml.safe_load(path.read_text())
    return [Source(**s) for s in data["sources"]]


def sitemap_urls(xml: str) -> list[str]:
    return _LOC.findall(xml)


def select_urls(source: Source, candidates: list[str]) -> list[str]:
    """Explicit URLs first, then sitemap URLs that match include and not exclude."""
    include = [re.compile(p) for p in source.include]
    exclude = [re.compile(p) for p in source.exclude]
    chosen = list(dict.fromkeys(source.urls))
    for url in candidates:
        if len(chosen) >= source.max_pages:
            break
        if url in chosen:
            continue
        if any(p.search(url) for p in include) and not any(p.search(url) for p in exclude):
            chosen.append(url)
    return chosen[: max(source.max_pages, len(source.urls))]


def extract_page(html: str, url: str) -> Page | None:
    """Main text of a page without menus and footers, or None if there is none."""
    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False, favor_precision=True
    )
    if not text:
        return None
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta and meta.title else urlsplit(url).path).strip()
    return Page(title=title, text=text)


def content_hash(page: Page) -> str:
    return hashlib.sha256(f"{page.title}\n{page.text}".encode()).hexdigest()


class Fetcher:
    """Polite HTTP client: robots.txt per site and a delay between requests."""

    def __init__(self, client: httpx.Client, delay: float = DELAY_SECONDS) -> None:
        self._client = client
        self._delay = delay
        self._robots: dict[str, RobotFileParser] = {}
        self._last: dict[str, float] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        site = f"{parts.scheme}://{parts.netloc}"
        if site not in self._robots:
            parser = RobotFileParser()
            response = self._client.get(f"{site}/robots.txt")
            # No robots.txt means everything is allowed; an error page means nothing is.
            if response.status_code == 404:
                parser.parse([])
            elif response.is_success:
                parser.parse(response.text.splitlines())
            else:
                parser.parse(["User-agent: *", "Disallow: /"])
            self._robots[site] = parser
        return self._robots[site].can_fetch(USER_AGENT, url)

    def get(self, url: str) -> str | None:
        if not self.allowed(url):
            log.warning("robots.txt disallows %s", url)
            return None
        host = urlsplit(url).netloc
        wait = self._last.get(host, 0) + self._delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()
        response = self._client.get(url)
        if not response.is_success:
            log.warning("%s returned %s", url, response.status_code)
            return None
        return response.text


def run(sources_path: Path) -> None:
    settings = get_settings()
    conn = db.connect(settings.database_url)
    embedder = Embedder(settings)
    keep: set[str] = set()
    stats = {"new": 0, "unchanged": 0, "failed": 0}

    client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30)
    with client:
        fetcher = Fetcher(client)
        for source in load_sources(sources_path):
            candidates: list[str] = []
            if source.sitemap and (xml := fetcher.get(source.sitemap)):
                candidates = sitemap_urls(xml)
            for url in select_urls(source, candidates):
                keep.add(url)
                html = fetcher.get(url)
                page = extract_page(html, url) if html else None
                if page is None:
                    stats["failed"] += 1
                    continue
                digest = content_hash(page)
                if db.document_hash(conn, url) == digest:
                    stats["unchanged"] += 1
                    continue
                passages = chunk_text(page.text)
                # The title gives short passages their context in the vector.
                vectors = embedder.embed_passages([f"{page.title}. {p}" for p in passages])
                db.replace_document(
                    conn,
                    url=url,
                    title=page.title,
                    organisation=source.organisation,
                    content_hash=digest,
                    passages=passages,
                    vectors=vectors,
                )
                stats["new"] += 1
                log.info("%s: %d passages", url, len(passages))

    removed = db.delete_documents_except(conn, keep)
    log.info("done: %s, removed %d", stats, removed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(Path(sys.argv[1] if len(sys.argv) > 1 else "sources.yaml"))
