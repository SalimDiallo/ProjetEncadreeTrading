"""Lecture / écriture CSV des records bruts."""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from oil_sentiment_pipeline.data_ingestion.schema import CSV_FIELDNAMES, IngestRecord

logger = logging.getLogger(__name__)


def save_records_csv(records: List[IngestRecord], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in CSV_FIELDNAMES}
            writer.writerow(row)
    logger.info("CSV sauvegardé : %s (%d lignes)", filepath, len(records))


def load_records_csv(filepath: str) -> List[Dict]:
    with open(filepath, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def timestamp_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
