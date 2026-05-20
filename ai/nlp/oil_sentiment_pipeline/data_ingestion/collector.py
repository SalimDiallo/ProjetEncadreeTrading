"""
collector.py
------------
Orchestrateur du module data_ingestion.

Usage :
    from oil_sentiment_pipeline.data_ingestion.collector import collect_all
    records = collect_all(save_csv=True, start_date="2024-01-01", end_date="2024-12-31")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from oil_sentiment_pipeline.data_ingestion.config import IngestionConfig
from oil_sentiment_pipeline.data_ingestion.io import save_records_csv, timestamp_suffix
from oil_sentiment_pipeline.data_ingestion.schema import VALID_SOURCES, IngestRecord
from oil_sentiment_pipeline.data_ingestion.utils import (
    deduplicate_records,
    filter_by_date_range,
    normalize_batch,
)
from oil_sentiment_pipeline.paths import RAW_DIR

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = str(RAW_DIR)


@dataclass
class SourceStats:
    source: str
    fetched: int = 0
    kept: int = 0
    mode: str = "live"  # live | mock | error
    error: Optional[str] = None


@dataclass
class CollectReport:
    records: List[IngestRecord] = field(default_factory=list)
    by_source: Dict[str, SourceStats] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.records)


def _fetch_source(name: str, cfg: IngestionConfig) -> tuple[List[Dict], str]:
    """Appelle le collecteur d'une source. Retourne (records bruts, mode)."""
    if name == "news":
        from oil_sentiment_pipeline.data_ingestion.news_scraper import fetch_all_news

        return fetch_all_news(
            max_per_source=cfg.max_per_source,
            filter_oil=cfg.filter_oil_keywords,
        ), "live"

    if name == "reddit":
        from oil_sentiment_pipeline.data_ingestion.reddit_collector import fetch_all_reddit

        return fetch_all_reddit(
            max_per_subreddit=cfg.max_per_source,
            allow_mock=cfg.allow_mock,
        ), "live"

    if name == "twitter":
        from oil_sentiment_pipeline.data_ingestion.twitter_scraper import fetch_all_tweets

        return fetch_all_tweets(
            max_per_query=cfg.max_per_source,
            allow_mock=cfg.allow_mock,
        ), "live"

    if name == "edgar":
        from oil_sentiment_pipeline.data_ingestion.edgar_parser import fetch_all_edgar

        start, end = cfg.edgar_date_range()
        records = fetch_all_edgar(
            use_full_text_search=True,
            use_company_filings=False,
            start_date=start,
            end_date=end,
            max_results=cfg.max_per_source,
            allow_mock=cfg.allow_mock,
        )
        return records, "live"

    raise ValueError(f"Source inconnue : {name}")


def collect(
    sources: Optional[List[str]] = None,
    config: Optional[IngestionConfig] = None,
    save_csv: bool = True,
    output_dir: Optional[str] = None,
) -> CollectReport:
    """
    Collecte, normalise, filtre par date et déduplique toutes les sources demandées.
    """
    cfg = config or IngestionConfig.from_env()
    if sources is None:
        sources = sorted(VALID_SOURCES)
    else:
        unknown = set(sources) - VALID_SOURCES
        if unknown:
            raise ValueError(f"Sources invalides : {unknown}")

    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    ts = timestamp_suffix()
    report = CollectReport()

    all_normalized: List[IngestRecord] = []

    for name in sources:
        stats = SourceStats(source=name)
        logger.info("=== Collecte %s ===", name.upper())

        try:
            raw, mode = _fetch_source(name, cfg)
            stats.fetched = len(raw)
            stats.mode = "mock" if raw and any("mock" in r.get("source", "") for r in raw[:3]) else mode

            normalized = normalize_batch(raw, min_text_length=cfg.min_text_length)
            normalized = filter_by_date_range(
                normalized, cfg.start_date, cfg.end_date
            )
            stats.kept = len(normalized)
            all_normalized.extend(normalized)

            if save_csv and normalized:
                path = os.path.join(out_dir, f"{name}_{ts}.csv")
                save_records_csv(normalized, path)
                report.output_files.append(path)

        except Exception as exc:
            stats.mode = "error"
            stats.error = str(exc)
            logger.error("Erreur collecte %s : %s", name, exc)

        report.by_source[name] = stats

    merged = deduplicate_records(all_normalized)
    report.records = merged

    if save_csv and merged:
        path = os.path.join(out_dir, f"all_sources_{ts}.csv")
        save_records_csv(merged, path)
        report.output_files.append(path)

    _log_summary(report, sources)
    return report


def collect_all(
    max_per_source: int = 30,
    save_csv: bool = True,
    output_dir: str = None,
    sources: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    allow_mock: bool = True,
    min_text_length: int = 20,
) -> List[Dict]:
    """
    Interface compatible avec le pipeline existant — retourne la liste de records.
    """
    cfg = IngestionConfig.from_env(
        max_per_source=max_per_source,
        min_text_length=min_text_length,
        allow_mock=allow_mock,
        start_date=start_date,
        end_date=end_date,
    )
    report = collect(
        sources=sources,
        config=cfg,
        save_csv=save_csv,
        output_dir=output_dir,
    )
    return list(report.records)


def _log_summary(report: CollectReport, sources: List[str]) -> None:
    logger.info("=== COLLECTE TERMINÉE : %d records (après dédup globale) ===", report.total)
    for name in sources:
        st = report.by_source.get(name)
        if not st:
            continue
        extra = f" | erreur: {st.error}" if st.error else ""
        logger.info(
            "  %-8s : %3d bruts → %3d conservés [%s]%s",
            name,
            st.fetched,
            st.kept,
            st.mode,
            extra,
        )
    if report.output_files:
        logger.info("Fichiers : %s", report.output_files[-1])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    report = collect(
        config=IngestionConfig.from_env(max_per_source=10),
        save_csv=True,
    )
    print(f"\nTOTAL : {report.total} records\n")
    for name, st in report.by_source.items():
        print(f"  {name:10s}  fetched={st.fetched:4d}  kept={st.kept:4d}  mode={st.mode}")
