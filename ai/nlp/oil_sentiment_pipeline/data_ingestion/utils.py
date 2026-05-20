"""Utilitaires partagés : normalisation, filtrage, déduplication."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional

from oil_sentiment_pipeline.data_ingestion.config import OIL_KEYWORDS
from oil_sentiment_pipeline.data_ingestion.schema import IngestRecord

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def is_oil_related(text: str, keywords: Optional[List[str]] = None) -> bool:
    lower = (text or "").lower()
    kws = keywords if keywords is not None else OIL_KEYWORDS
    return any(kw in lower for kw in kws)


def parse_iso_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value.replace("+00:00", "Z"), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def normalize_record(
    raw: Dict,
    *,
    min_text_length: int = 20,
    default_source: str = "unknown",
) -> Optional[IngestRecord]:
    """Valide et normalise un record brut. Retourne None si invalide."""
    text = strip_html(str(raw.get("text", "") or ""))
    if len(text) < min_text_length:
        return None

    date_raw = str(raw.get("date", "") or "").strip()
    dt = parse_iso_date(date_raw)
    date_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else now_iso()

    source = str(raw.get("source", "") or default_source).strip() or default_source
    url = str(raw.get("url", "") or "").strip()
    lang = str(raw.get("lang", "") or "").strip()

    record: IngestRecord = {
        "text": text[:8000],
        "date": date_str,
        "source": source,
    }
    if url:
        record["url"] = url
    if lang:
        record["lang"] = lang
    return record


def normalize_batch(
    records: Iterable[Dict],
    *,
    min_text_length: int = 20,
) -> List[IngestRecord]:
    out: List[IngestRecord] = []
    skipped = 0
    for raw in records:
        rec = normalize_record(raw, min_text_length=min_text_length)
        if rec:
            out.append(rec)
        else:
            skipped += 1
    if skipped:
        logger.debug("%d records ignorés (texte trop court ou vide).", skipped)
    return out


def deduplicate_records(
    records: List[IngestRecord],
    key_fn: Optional[Callable[[IngestRecord], str]] = None,
) -> List[IngestRecord]:
    if key_fn is None:

        def key_fn(rec: IngestRecord) -> str:
            base = rec["text"][:120].lower().strip()
            return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

    seen = set()
    out: List[IngestRecord] = []
    for rec in records:
        key = key_fn(rec)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def filter_by_date_range(
    records: List[IngestRecord],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[IngestRecord]:
    if not start_date and not end_date:
        return records

    start_dt = parse_iso_date(start_date) if start_date else None
    end_dt = parse_iso_date(end_date) if end_date else None
    if end_dt and end_dt.hour == 0 and end_dt.minute == 0:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    filtered: List[IngestRecord] = []
    for rec in records:
        dt = parse_iso_date(rec["date"])
        if dt is None:
            continue
        if start_dt and dt < start_dt:
            continue
        if end_dt and dt > end_dt:
            continue
        filtered.append(rec)
    return filtered
