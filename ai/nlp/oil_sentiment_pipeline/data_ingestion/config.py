"""Configuration centralisée de l'ingestion (variables d'environnement + défauts)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

OIL_KEYWORDS: List[str] = [
    "oil",
    "crude",
    "petroleum",
    "brent",
    "wti",
    "opec",
    "barrel",
    "energy",
    "gasoline",
    "refinery",
    "drilling",
    "offshore",
    "pipeline",
    "natural gas",
    "fossil fuel",
    "lng",
    "shale",
]

REQUEST_TIMEOUT = 15
REQUEST_DELAY_SEC = 0.5

REDDIT_CLIENT_ID_ENV = "REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET_ENV = "REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT_ENV = "REDDIT_USER_AGENT"
TWITTER_BEARER_TOKEN_ENV = "TWITTER_BEARER_TOKEN"
SEC_USER_AGENT_ENV = "SEC_USER_AGENT"

DEFAULT_SEC_USER_AGENT = "OilSentimentPipeline contact@example.com"


def load_env() -> None:
    """Charge le fichier .env à la racine du projet si python-dotenv est disponible."""
    try:
        from dotenv import load_dotenv

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        load_dotenv(os.path.join(project_root, ".env"))
    except ImportError:
        pass


def _today_utc() -> datetime:
    return datetime.now(timezone.utc)


def default_edgar_window(days_back: int = 365) -> tuple[str, str]:
    end = _today_utc().date()
    start = end - timedelta(days=days_back)
    return start.isoformat(), end.isoformat()


@dataclass
class IngestionConfig:
    """Paramètres d'une session de collecte."""

    max_per_source: int = 30
    min_text_length: int = 20
    allow_mock: bool = True
    filter_oil_keywords: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    edgar_days_back: int = 365
    request_timeout: int = REQUEST_TIMEOUT

    reddit_subreddits: List[str] = field(
        default_factory=lambda: [
            "energy",
            "oilandgas",
            "investing",
            "stocks",
            "finance",
            "commodities",
            "CrudeOil",
        ]
    )
    reddit_search_queries: List[str] = field(
        default_factory=lambda: [
            "crude oil",
            "WTI oil price",
            "OPEC production",
            "oil market",
        ]
    )
    twitter_queries: List[str] = field(
        default_factory=lambda: [
            "crude oil lang:en -is:retweet",
            "WTI oil price lang:en -is:retweet",
            "OPEC production lang:en -is:retweet",
            "Brent crude lang:en -is:retweet",
            "#OilMarket lang:en -is:retweet",
            "#OOTT lang:en -is:retweet",
        ]
    )

    def edgar_date_range(self) -> tuple[str, str]:
        if self.start_date and self.end_date:
            return self.start_date, self.end_date
        return default_edgar_window(self.edgar_days_back)

    @classmethod
    def from_env(cls, **overrides) -> "IngestionConfig":
        load_env()
        cfg = cls()
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def sec_user_agent(self) -> str:
        load_env()
        return os.getenv(SEC_USER_AGENT_ENV, DEFAULT_SEC_USER_AGENT)
