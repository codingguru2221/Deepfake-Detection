from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetRecord:
    source: str
    title: str
    url: str
    summary: str | None = None
    published_at: str | None = None
    tags: list[str] | None = None
    license: str | None = None


def _to_utc_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    except ValueError:
        return value


def _http_get_json(url: str, timeout: float) -> dict | list | None:
    try:
        req = Request(url, headers={"User-Agent": "DeepfakeDatasetCrawler/1.0"})
        with urlopen(req, timeout=timeout) as response:  # nosec B310
            if response.status != 200:
                return None
            payload = response.read()
            if not payload:
                return None
            return json.loads(payload.decode("utf-8", errors="replace"))
    except (URLError, OSError, json.JSONDecodeError):
        return None


def _http_get_text(url: str, timeout: float) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": "DeepfakeDatasetCrawler/1.0"})
        with urlopen(req, timeout=timeout) as response:  # nosec B310
            if response.status != 200:
                return None
            return response.read().decode("utf-8", errors="replace")
    except (URLError, OSError):
        return None


def fetch_figshare(max_items: int, timeout: float, search_query: str) -> list[DatasetRecord]:
    params = urlencode(
        {
            "search_for": search_query,
            "item_type": 3,  # 3 == dataset
            "page": 1,
            "page_size": max_items,
            "order_direction": "desc",
            "order": "published_date",
        }
    )
    url = f"https://api.figshare.com/v2/articles?{params}"
    payload = _http_get_json(url, timeout)
    if not isinstance(payload, list):
        return []

    records: list[DatasetRecord] = []
    for item in payload[:max_items]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        link = str(item.get("url_public_html", "")).strip()
        if not title or not link:
            continue
        records.append(
            DatasetRecord(
                source="figshare",
                title=title,
                url=link,
                summary=str(item.get("description", "")).strip() or None,
                published_at=_to_utc_iso(str(item.get("published_date", "")).strip() or None),
                tags=None,
                license=None,
            )
        )
    return records


def fetch_zenodo(max_items: int, timeout: float, search_query: str) -> list[DatasetRecord]:
    params = urlencode({"q": search_query, "size": max_items, "sort": "mostrecent"})
    url = f"https://zenodo.org/api/records?{params}"
    payload = _http_get_json(url, timeout)
    if not isinstance(payload, dict):
        return []
    hits = payload.get("hits", {})
    items = hits.get("hits", []) if isinstance(hits, dict) else []
    if not isinstance(items, list):
        return []

    records: list[DatasetRecord] = []
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        resource_type = metadata.get("resource_type", {})
        if isinstance(resource_type, dict) and resource_type.get("type") != "dataset":
            continue

        title = str(metadata.get("title", "")).strip()
        links = item.get("links", {})
        link = ""
        if isinstance(links, dict):
            link = str(links.get("html", "")).strip()
        if not title or not link:
            continue

        descriptions = metadata.get("description", "")
        tags = metadata.get("keywords", [])
        license_info = metadata.get("license", {})
        records.append(
            DatasetRecord(
                source="zenodo",
                title=title,
                url=link,
                summary=str(descriptions).strip() or None,
                published_at=_to_utc_iso(str(metadata.get("publication_date", "")).strip() or None),
                tags=[str(tag) for tag in tags] if isinstance(tags, list) else None,
                license=str(license_info.get("id", "")).strip() if isinstance(license_info, dict) else None,
            )
        )
    return records


def fetch_arxiv(max_items: int, timeout: float, search_query: str) -> list[DatasetRecord]:
    query = quote_plus(search_query)
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{query}&start=0&max_results={max_items}&sortBy=submittedDate&sortOrder=descending"
    )
    text = _http_get_text(url, timeout)
    if not text:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    records: list[DatasetRecord] = []
    for entry in root.findall("atom:entry", ns)[:max_items]:
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        if not title or not link:
            continue
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        records.append(
            DatasetRecord(
                source="arxiv",
                title=title,
                url=link,
                summary=summary or None,
                published_at=_to_utc_iso(published or None),
                tags=["research"],
                license=None,
            )
        )
    return records


def fetch_kaggle(max_items: int, timeout: float, search_query: str) -> list[DatasetRecord]:
    query = quote_plus(search_query)
    url = (
        "https://www.kaggle.com/datasets?"
        f"search={query}&sortBy=hottest&size=all&filetype=all&license=all"
    )
    html = _http_get_text(url, timeout)
    if not html:
        return []

    pattern = re.compile(r'href="/datasets/([^"#?]+)"')
    records: list[DatasetRecord] = []
    seen: set[str] = set()
    for match in pattern.finditer(html):
        slug = match.group(1).strip("/")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        records.append(
            DatasetRecord(
                source="kaggle",
                title=slug.replace("/", " / "),
                url=f"https://www.kaggle.com/datasets/{slug}",
                summary=None,
                published_at=None,
                tags=None,
                license=None,
            )
        )
        if len(records) >= max_items:
            break
    return records


class DatasetCrawler:
    def __init__(
        self,
        output_file: Path,
        max_items: int = 60,
        timeout_seconds: float = 8.0,
        refresh_hours: int = 24,
        search_query: str = "deepfake dataset",
        stop_event: threading.Event | None = None,
    ) -> None:
        self.output_file = output_file
        self.max_items = max(1, max_items)
        self.timeout_seconds = max(2.0, timeout_seconds)
        self.refresh_hours = max(1, refresh_hours)
        self.search_query = search_query
        self._lock = threading.Lock()
        self.stop_event = stop_event

    def should_refresh(self) -> bool:
        if not self.output_file.exists():
            return True
        modified = datetime.fromtimestamp(self.output_file.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(tz=timezone.utc) - modified
        return age > timedelta(hours=self.refresh_hours)

    def crawl_once(self) -> int:
        source_limit = max(1, self.max_items // 4)
        sources: list[tuple[str, Callable[[int, float, str], list[DatasetRecord]]]] = [
            ("kaggle", fetch_kaggle),
            ("figshare", fetch_figshare),
            ("zenodo", fetch_zenodo),
            ("arxiv", fetch_arxiv),
        ]

        all_records: list[DatasetRecord] = []
        dedupe: set[tuple[str, str]] = set()
        for source_name, fetcher in sources:
            if self.stop_event and self.stop_event.is_set():
                break
            try:
                records = fetcher(source_limit, self.timeout_seconds, self.search_query)
            except Exception as exc:
                logger.warning("Dataset crawler source failed: %s (%s)", source_name, exc)
                continue
            for record in records:
                if self.stop_event and self.stop_event.is_set():
                    break
                key = (record.source, record.url)
                if key in dedupe:
                    continue
                dedupe.add(key)
                all_records.append(record)
                if len(all_records) >= self.max_items:
                    break
            if len(all_records) >= self.max_items:
                break

        self._write_catalog(all_records[: self.max_items])
        return len(all_records)

    def _write_catalog(self, records: list[DatasetRecord]) -> None:
        payload = {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "query": self.search_query,
            "count": len(records),
            "items": [asdict(r) for r in records],
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.output_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
