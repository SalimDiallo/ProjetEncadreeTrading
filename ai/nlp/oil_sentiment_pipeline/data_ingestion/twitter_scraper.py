"""
twitter_scraper.py
------------------
Collecte de tweets liés au pétrole.

Stratégie :
  1. Tentative via l'API Twitter v2 (bearer token en variable d'env TWITTER_BEARER_TOKEN).
  2. Si non disponible → mode mock avec tweets réalistes pré-générés.

Format de sortie standard :
    {"text": str, "date": str (ISO-8601), "source": str}
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from oil_sentiment_pipeline.data_ingestion.config import (
    REQUEST_TIMEOUT,
    TWITTER_BEARER_TOKEN_ENV,
    IngestionConfig,
    load_env,
)
from oil_sentiment_pipeline.data_ingestion.mock_data import build_mock_tweets
from oil_sentiment_pipeline.data_ingestion.utils import deduplicate_records, normalize_batch

logger = logging.getLogger(__name__)

TWITTER_API_V2_SEARCH = "https://api.twitter.com/2/tweets/search/recent"

# ---------------------------------------------------------------------------
# Mode Scraper (Nitter RSS) - Contournement API Twitter
# ---------------------------------------------------------------------------
import requests

def fetch_tweets_scraper(
    query: str,
    max_results: int = 50,
) -> List[Dict]:
    """
    Scrape StockTwits (Le 'Twitter de la finance') pour récupérer les vrais messages 
    en temps réel sur le pétrole brut. Aucune clé API requise, aucune limite bloquante.
    """
    results = []
    
    # On cible directement les tickers du pétrole (CL_F = Crude Oil)
    url = "https://api.stocktwits.com/api/2/streams/symbol/CL_F.json"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        messages = data.get("messages", [])
        
        for msg in messages:
            text = msg.get("body", "")
            created_at = msg.get("created_at", "")
            
            # Format attendu : "2024-03-15T08:00:00Z"
            results.append({
                "text": text[:2000],
                "date": created_at,
                "source": "stocktwits",
                "lang": "en",
            })
            if len(results) >= max_results:
                break
                
        logger.info("StockTwits API : %d messages financiers collectés pour le pétrole.", len(results))
            
    except Exception as exc:
        logger.error("Échec API StockTwits : %s", exc)

    return results


def fetch_all_twitter_api(
    max_per_query: int = 30,
    queries: Optional[List[str]] = None,
) -> List[Dict]:
    """Point d'entrée du scraper Twitter (remplace l'API officielle)."""
    if queries is None:
        queries = IngestionConfig.from_env().twitter_queries

    all_tweets = []
    for query in queries:
        tweets = fetch_tweets_scraper(query, max_results=max_per_query)
        all_tweets.extend(tweets)
        time.sleep(2)  # respecte le serveur

    if not all_tweets:
        raise ValueError("Le scraper Twitter n'a pu récupérer aucune donnée (Instances hors-ligne).")

    return all_tweets


def fetch_mock_tweets(n: int = 20) -> List[Dict]:
    """Retourne des tweets simulés avec dates récentes."""
    logger.warning("Mode MOCK Twitter activé — %d tweets simulés utilisés.", n)
    return build_mock_tweets(n)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def fetch_all_tweets(max_per_query: int = 30, allow_mock: bool = False) -> List[Dict]:
    """
    Collecte les tweets pétrole (Via StockTwits pour garantir 100% de vraie donnée).
    Le mock est désactivé volontairement.
    """
    all_tweets: List[Dict] = []

    try:
        all_tweets = fetch_all_twitter_api(max_per_query=max_per_query)

    except ValueError as exc:
        raise ValueError(f"Erreur StockTwits: {exc}. Pas de mock autorisé.")

    except Exception as exc:
        raise RuntimeError(f"Erreur inattendue StockTwits: {exc}. Pas de mock autorisé.")

    normalized = normalize_batch(all_tweets)
    deduplicated = deduplicate_records(normalized)
    logger.info("Total StockTwits (après normalisation/dédup) : %d", len(deduplicated))
    return [dict(r) for r in deduplicated]


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import csv, os
    from datetime import datetime, timezone
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    tweets = fetch_all_tweets(max_per_query=10)
    for t in tweets[:5]:
        print(f"[{t['date']}] [{t['source']}] {t['text'][:120]}")
    print(f"\nTotal : {len(tweets)} tweets collectés.")

    from oil_sentiment_pipeline.paths import RAW_DIR

    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = str(RAW_DIR / f"twitter_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "date", "source"])
        writer.writeheader()
        writer.writerows(tweets)
    print(f"Sauvegardé : {path}")
