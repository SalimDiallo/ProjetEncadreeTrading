"""
reddit_collector.py
-------------------
Collecte des posts et commentaires Reddit liés au pétrole via l'API PRAW.
Si les credentials PRAW ne sont pas configurés, un mode mock est activé
automatiquement pour permettre l'exécution du pipeline sans clé API.

Format de sortie standard :
    {"text": str, "date": str (ISO-8601), "source": str}
"""

import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

from oil_sentiment_pipeline.data_ingestion.config import (
    REDDIT_CLIENT_ID_ENV,
    REDDIT_CLIENT_SECRET_ENV,
    REDDIT_USER_AGENT_ENV,
    IngestionConfig,
    load_env,
)
from oil_sentiment_pipeline.data_ingestion.config import OIL_KEYWORDS
from oil_sentiment_pipeline.data_ingestion.mock_data import build_mock_reddit
from oil_sentiment_pipeline.data_ingestion.utils import (
    deduplicate_records,
    is_oil_related,
    normalize_batch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp_to_iso(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")




# ---------------------------------------------------------------------------
# Collecte via RSS (Scraping public sans API Key)
# ---------------------------------------------------------------------------
import requests
import feedparser
from bs4 import BeautifulSoup
import time
import urllib.parse

def _get_reddit_rss(url: str) -> List[Dict]:
    """Fonction utilitaire pour parser un flux RSS Reddit"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        return feed.entries
    except Exception as e:
        logger.error("Erreur requête RSS (%s): %s", url, e)
        return []

def fetch_subreddit_posts(
    subreddit_name: str,
    reddit_client=None, # Gardé pour la compatibilité
    limit: int = 50,
    filter_oil: bool = True,
) -> List[Dict]:
    """Collecte les posts 'hot' d'un subreddit via RSS public."""
    results = []
    url = f"https://www.reddit.com/r/{subreddit_name}/hot.rss?limit={limit}"
    entries = _get_reddit_rss(url)
    
    for entry in entries:
        title = getattr(entry, "title", "")
        summary_html = getattr(entry, "summary", "")
        selftext = BeautifulSoup(summary_html, "html.parser").get_text(separator=" ")
        full_text = f"{title}. {selftext}".strip()

        if filter_oil and not is_oil_related(full_text):
            continue

        # Extraction de la date (fallback sur now si absent)
        dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            from time import mktime
            dt = datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)
            dt_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        results.append({
            "text": full_text[:2000],
            "date": dt_str,
            "source": f"reddit_r/{subreddit_name}",
            "url": getattr(entry, "link", ""),
            "lang": "en",
        })

    logger.info("r/%s (RSS) : %d posts oil-related collectés.", subreddit_name, len(results))
    time.sleep(1) # Respect du rate-limit public
    return results


def fetch_reddit_search(
    query: str,
    reddit_client=None, # Gardé pour la compatibilité
    subreddit: str = "all",
    limit: int = 50,
    sort: str = "new",
) -> List[Dict]:
    """Recherche globale Reddit via RSS."""
    results = []
    q_encoded = urllib.parse.quote(query)
    
    if subreddit == "all":
        url = f"https://www.reddit.com/search.rss?q={q_encoded}&sort={sort}&limit={limit}"
    else:
        url = f"https://www.reddit.com/r/{subreddit}/search.rss?q={q_encoded}&restrict_sr=on&sort={sort}&limit={limit}"

    entries = _get_reddit_rss(url)
    
    for entry in entries:
        title = getattr(entry, "title", "")
        summary_html = getattr(entry, "summary", "")
        selftext = BeautifulSoup(summary_html, "html.parser").get_text(separator=" ")
        full_text = f"{title}. {selftext}".strip()

        dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            from time import mktime
            dt = datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)
            dt_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        results.append({
            "text": full_text[:2000],
            "date": dt_str,
            "source": f"reddit_search/{subreddit}",
            "url": getattr(entry, "link", ""),
            "lang": "en",
        })

    logger.info("Reddit search '%s' (RSS) : %d posts collectés.", query, len(results))
    time.sleep(1)
    return results


# ---------------------------------------------------------------------------
# Mode mock (sans credentials)
# ---------------------------------------------------------------------------

def fetch_mock_reddit(n: int = 10) -> List[Dict]:
    """Retourne des données Reddit simulées avec dates récentes."""
    logger.warning("Mode MOCK Reddit activé — données simulées utilisées.")
    return build_mock_reddit(n)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def fetch_all_reddit(
    max_per_subreddit: int = 30,
    search_queries: List[str] = None,
    allow_mock: bool = True,
    subreddits: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Collecte les données Reddit depuis toutes les sources configurées via RSS.
    Bascule automatiquement en mode mock en cas d'erreur.
    """
    cfg = IngestionConfig.from_env()
    if search_queries is None:
        search_queries = cfg.reddit_search_queries
    if subreddits is None:
        subreddits = cfg.reddit_subreddits

    all_posts: List[Dict] = []

    try:
        logger.info("Scraping Reddit (via RSS) démarré.")

        for sub in subreddits:
            posts = fetch_subreddit_posts(sub, limit=max_per_subreddit)
            all_posts.extend(posts)

        for query in search_queries:
            posts = fetch_reddit_search(query, limit=max_per_subreddit)
            all_posts.extend(posts)

    except Exception as exc:
        if allow_mock:
            logger.error("Erreur inattendue Scraper Reddit : %s — basculement mock.", exc)
            all_posts = fetch_mock_reddit(n=10)
        else:
            raise

    normalized = normalize_batch(all_posts)
    deduplicated = deduplicate_records(normalized)
    logger.info("Total posts Reddit (après normalisation/dédup) : %d", len(deduplicated))
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
    posts = fetch_all_reddit(max_per_subreddit=10)
    for p in posts[:5]:
        print(f"[{p['date']}] [{p['source']}] {p['text'][:120]}...")
    print(f"\nTotal : {len(posts)} posts Reddit collectés.")

    from oil_sentiment_pipeline.paths import RAW_DIR

    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = str(RAW_DIR / f"reddit_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "date", "source"])
        writer.writeheader()
        writer.writerows(posts)
    print(f"Sauvegardé : {path}")
