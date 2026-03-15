"""URL crawler — fetches web pages, extracts text, follows links.

Uses only stdlib (html.parser) and httpx (transitive dep from mcp).
No BeautifulSoup or other heavy dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

# Tags whose content should be skipped entirely
_SKIP_TAGS = {"script", "style", "nav", "footer", "noscript", "svg", "iframe"}

# Tags that produce a line break in extracted text
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "br", "tr", "dt", "dd", "blockquote",
    "pre", "hr", "figcaption",
}

RATE_LIMIT = 0.3  # seconds between requests


@dataclass
class CrawledPage:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


@dataclass
class CrawlStats:
    pages_crawled: int = 0
    pages_skipped: int = 0
    total_chars: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  HTML text extraction
# --------------------------------------------------------------------------- #


class _TextExtractor(HTMLParser):
    """Extracts readable text from HTML, skipping scripts/styles/nav."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_code = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in ("pre", "code"):
            self._in_code = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a":
            pass  # handled by handle_data

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in ("pre", "code"):
            self._in_code = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._skip_depth > 0:
            return
        if self._in_code:
            self.parts.append(data)
        else:
            self.parts.append(data)

    def get_title(self) -> str:
        return " ".join("".join(self.title_parts).split()).strip()

    def get_text(self) -> str:
        text = "".join(self.parts)
        # Collapse excessive whitespace but preserve code blocks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def extract_text(html: str) -> tuple[str, str]:
    """Extract (title, text) from HTML content."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_title(), parser.get_text()


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract and resolve all href links from HTML."""
    links: list[str] = []
    for match in re.finditer(r'href=["\']([^"\'#]+)', html):
        raw = match.group(1).strip()
        if raw.startswith(("javascript:", "mailto:", "tel:", "data:")):
            continue
        absolute = urljoin(base_url, raw)
        links.append(absolute)
    return links


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication — strip fragment and trailing slash."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


# --------------------------------------------------------------------------- #
#  Crawler
# --------------------------------------------------------------------------- #


async def crawl(
    start_url: str,
    max_pages: int = 50,
    max_depth: int = 3,
    same_domain: bool = True,
) -> tuple[list[CrawledPage], CrawlStats]:
    """Crawl a URL and its linked pages, extracting text content.

    Returns (pages, stats). Only follows links within the same domain
    by default. Rate-limited to be polite.
    """
    parsed_start = urlparse(start_url)
    domain = parsed_start.netloc
    stats = CrawlStats()

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(_normalize_url(start_url), 0)]
    pages: list[CrawledPage] = []

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "Helios/1.0 (context indexer)"},
    ) as client:
        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)
            normalized = _normalize_url(url)
            if normalized in visited:
                continue
            visited.add(normalized)

            try:
                response = await client.get(url)
            except Exception as e:
                stats.errors.append(f"{url}: {e}")
                stats.pages_skipped += 1
                continue

            if response.status_code != 200:
                stats.pages_skipped += 1
                continue

            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                stats.pages_skipped += 1
                continue

            html = response.text
            title, text = extract_text(html)

            if len(text) < 50:
                stats.pages_skipped += 1
                continue

            page_links: list[str] = []
            if depth < max_depth:
                for link in extract_links(html, url):
                    link_normalized = _normalize_url(link)
                    link_parsed = urlparse(link_normalized)

                    if same_domain and link_parsed.netloc != domain:
                        continue
                    if link_normalized in visited:
                        continue

                    page_links.append(link_normalized)
                    queue.append((link_normalized, depth + 1))

            pages.append(CrawledPage(
                url=url,
                title=title or urlparse(url).path,
                text=text,
                links=page_links,
            ))
            stats.pages_crawled += 1
            stats.total_chars += len(text)

            # Rate limit
            await asyncio.sleep(RATE_LIMIT)

    return pages, stats


async def crawl_and_index(
    start_url: str,
    source_id: str,
    store,  # ContentStore — avoid circular import
    max_pages: int = 50,
    max_depth: int = 3,
) -> CrawlStats:
    """Crawl a URL and store all pages as documents."""
    pages, stats = await crawl(start_url, max_pages, max_depth)

    indexed_urls: set[str] = set()
    for page in pages:
        path_part = urlparse(page.url).path or "/"
        await store.upsert_document(
            source_id=source_id,
            path=page.url,
            relative_path=path_part,
            content=f"# {page.title}\n\n{page.text}",
            language="markdown",
            size=len(page.text),
        )
        indexed_urls.add(page.url)

    await store.remove_stale_documents(source_id, indexed_urls)
    await store.update_source_stats(source_id)

    return stats
