"""Schéma des enregistrements bruts collectés."""

from typing import List, Optional, TypedDict

# Colonnes persistées en CSV (ordre fixe)
CSV_FIELDNAMES: List[str] = ["text", "date", "source", "url", "lang"]

VALID_SOURCES = frozenset({"news", "reddit", "twitter", "edgar"})


class IngestRecord(TypedDict, total=False):
    """Record normalisé produit par toutes les sources d'ingestion."""

    text: str
    date: str  # ISO-8601 UTC
    source: str
    url: str
    lang: str
