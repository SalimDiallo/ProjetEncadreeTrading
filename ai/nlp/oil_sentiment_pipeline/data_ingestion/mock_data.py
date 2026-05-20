"""Données simulées avec dates récentes (évite les mocks figés en 2024)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List


def _offset_iso(days_ago: int, hours: int = 12) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def with_recent_dates(records: List[Dict], days_spread: int = 14) -> List[Dict]:
    """Réattribue des dates récentes aux enregistrements mock."""
    if not records:
        return []
    step = max(1, days_spread // max(len(records), 1))
    out = []
    for i, rec in enumerate(records):
        out.append({**rec, "date": _offset_iso(min(i * step, days_spread))})
    return out


MOCK_TWEETS_TEMPLATE = [
    "BREAKING: OPEC+ agrees to cut production by 1M bbl/day. #OilMarket #OOTT crude oil bullish",
    "WTI crude oil drops below $75 on weak US jobs data. Energy sector selloff.",
    "Brent crude up 2.5% as Red Sea shipping disruptions worsen. Supply risk premium returns #OOTT",
    "US crude inventories rise 4.2M barrels — bearish surprise for oil market",
    "Goldman Sachs raises Brent crude forecast to $95/barrel by Q3. Bullish on energy",
    "China oil demand recovery slower than expected in Q1. Downward pressure on crude.",
    "Iraq oil exports hit new high as OPEC compliance falters. WTI pressured lower.",
    "Energy stocks outperforming as oil prices stabilize near $80.",
    "US shale production growth slowing. Permian rig count down 5%. Bullish for crude long term.",
    "IEA monthly report: Global oil demand growth revised down for 2024. Bearish.",
]

MOCK_REDDIT_TEMPLATE = [
    "OPEC+ cuts production again — crude oil prices expected to rise above $90/barrel.",
    "WTI crude down 3% after unexpected build in US oil inventories. Bearish sentiment growing.",
    "Brent crude oil hits $85 as geopolitical tensions escalate. Energy stocks surging.",
    "Oil demand outlook cut by IEA. Transition to renewables faster than expected.",
    "Saudi Arabia confirms no increase in oil production. OPEC discipline holds.",
    "US shale oil production hits record high. Supply glut fears return.",
    "Energy sector outperforms as oil prices stabilize. XOM and CVX both up 2%.",
    "Crude oil inventory report shows surprise draw. Bullish signal for WTI futures.",
]

MOCK_EDGAR_TEMPLATE = [
    {
        "text": "[ITEM 1A] Crude oil prices remain volatile due to OPEC decisions and geopolitical risk. [ITEM 7] WTI averaged elevated levels in the recent quarter.",
        "source": "edgar_10k_ExxonMobil",
    },
    {
        "text": "[ITEM 7] Brent crude averaged strong levels; refining margins compressed on refined product oversupply.",
        "source": "edgar_10q_BP",
    },
    {
        "text": "8-K: Major oil producer reports quarterly results; realized crude price and production in line with guidance.",
        "source": "edgar_8k_ConocoPhillips",
    },
]


def build_mock_tweets(n: int = 20) -> List[Dict]:
    texts = MOCK_TWEETS_TEMPLATE[:n]
    records = [
        {"text": t, "source": "twitter_mock", "lang": "en"} for t in texts
    ]
    return with_recent_dates(records)


def build_mock_reddit(n: int = 10) -> List[Dict]:
    texts = MOCK_REDDIT_TEMPLATE[:n]
    records = [{"text": t, "source": "reddit_mock", "lang": "en"} for t in texts]
    return with_recent_dates(records)


def build_mock_edgar(n: int = 5) -> List[Dict]:
    records = MOCK_EDGAR_TEMPLATE[:n]
    return with_recent_dates(records)
