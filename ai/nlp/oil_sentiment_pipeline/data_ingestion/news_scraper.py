"""
news_scraper.py
---------------
Collecte des articles financiers depuis des flux RSS (Yahoo Finance, Reuters,
Bloomberg via des proxies publics) ainsi que depuis l'API Yahoo Finance.

Chaque article retourné respecte le format standard :
    {"text": str, "date": str (ISO-8601), "source": str}
"""

import logging
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Dict, Optional

from oil_sentiment_pipeline.data_ingestion.config import OIL_KEYWORDS, REQUEST_TIMEOUT
from oil_sentiment_pipeline.data_ingestion.utils import (
    deduplicate_records,
    is_oil_related,
    normalize_batch,
    now_iso,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flux RSS publics liés au pétrole / énergie
# ---------------------------------------------------------------------------
RSS_FEEDS: Dict[str, str] = {
    "yahoo_finance_oil": "https://finance.yahoo.com/rss/headline?s=CL=F",
    "yahoo_finance_energy": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=XOM,CVX,BP&region=US&lang=en-US",
    "reuters_energy": "https://feeds.reuters.com/reuters/businessNews",
    "investing_oil": "https://www.investing.com/rss/news_25.rss",
    "oilprice": "https://oilprice.com/rss/main",
    "eia_news": "https://www.eia.gov/rss/press_rss.xml",
}



def _parse_feed_date(entry) -> str:
    """Extrait la date d'une entrée feedparser et la normalise en ISO-8601."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        try:
            dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return now_iso()


def _extract_article_text(url: str, max_chars: int = 2000) -> Optional[str]:
    """
    Tente de scraper le corps d'un article via BeautifulSoup.
    Retourne None si inaccessible.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OilSentimentBot/1.0)"}
        from oil_sentiment_pipeline.data_ingestion.http_utils import get_with_retry

        resp = get_with_retry(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Supprime les balises non pertinentes
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        return text[:max_chars] if text else None
    except Exception as exc:
        logger.debug("Impossible de scraper %s : %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Collecteurs par source
# ---------------------------------------------------------------------------

def fetch_rss_articles(
    feed_name: str,
    feed_url: str,
    max_articles: int = 50,
    filter_oil: bool = True,
) -> List[Dict]:
    """Collecte les articles d'un flux RSS donné."""
    results = []
    try:
        logger.info("Lecture du flux RSS [%s] : %s", feed_name, feed_url)
        # On utilise requests avec un User-Agent pour éviter les blocages de format
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
        resp = requests.get(feed_url, headers=headers, timeout=10)
        feed = feedparser.parse(resp.content)
        
        if feed.bozo and feed.bozo_exception:
            logger.warning("Flux RSS malformé [%s] : %s", feed_name, feed.bozo_exception)

        for entry in feed.entries[:max_articles]:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            link = getattr(entry, "link", "") or ""
            full_text = f"{title}. {summary}".strip()

            if filter_oil and not is_oil_related(full_text):
                continue

            # Tentative d'enrichissement via scraping du corps de l'article
            body = _extract_article_text(link) if link else None
            if body:
                full_text = f"{title}. {body}"

            results.append({
                "text": full_text,
                "date": _parse_feed_date(entry),
                "source": feed_name,
                "url": link,
                "lang": "en",
            })

        logger.info("[%s] %d articles collectés.", feed_name, len(results))

    except Exception as exc:
        logger.error("Erreur lors de la lecture du flux [%s] : %s", feed_name, exc)

    return results


def fetch_yahoo_finance_news(ticker: str = "CL=F", max_articles: int = 30) -> List[Dict]:
    """
    Récupère les news Yahoo Finance pour un ticker donné via l'API yfinance.
    Fallback sur le flux RSS si yfinance n'est pas disponible.
    """
    results = []
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news or []
        logger.info("Yahoo Finance API : %d news pour %s", len(news), ticker)

        for item in news[:max_articles]:
            title = item.get("title", "") or ""
            link = item.get("link", "") or ""
            pub_date = item.get("providerPublishTime")

            if pub_date:
                dt = datetime.fromtimestamp(pub_date, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                date_str = now_iso()

            body = _extract_article_text(link) if link else None
            full_text = f"{title}. {body}" if body else title

            results.append({
                "text": full_text,
                "date": date_str,
                "source": "yahoo_finance_api",
                "url": link,
                "lang": "en",
            })

    except ImportError:
        logger.warning("yfinance non installé — fallback sur le flux RSS Yahoo Finance.")
        results = fetch_rss_articles(
            "yahoo_finance_oil",
            RSS_FEEDS["yahoo_finance_oil"],
            max_articles=max_articles,
        )
    except Exception as exc:
        logger.error("Erreur Yahoo Finance API : %s", exc)

    return results


def fetch_all_news(max_per_source: int = 30, filter_oil: bool = True) -> List[Dict]:
    """
    Point d'entrée principal : collecte toutes les sources RSS + Yahoo Finance API.
    Retourne une liste normalisée et dédupliquée.
    """
    all_articles: List[Dict] = []

    all_articles.extend(fetch_yahoo_finance_news(max_articles=max_per_source))

    for name, url in RSS_FEEDS.items():
        articles = fetch_rss_articles(
            name, url, max_articles=max_per_source, filter_oil=filter_oil
        )
        all_articles.extend(articles)

    normalized = normalize_batch(all_articles)
    deduplicated = deduplicate_records(normalized)
    logger.info("Total articles news (après normalisation/dédup) : %d", len(deduplicated))
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
    articles = fetch_all_news(max_per_source=50)
    for a in articles[:5]:
        print(f"[{a['date']}] [{a['source']}] {a['text'][:120]}...")
    print(f"\nTotal : {len(articles)} articles collectés.")

    # Sauvegarde CSV
    from oil_sentiment_pipeline.paths import RAW_DIR

    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = str(RAW_DIR / f"news_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "date", "source"])
        writer.writeheader()
        writer.writerows(articles)
    print(f"Sauvegardé : {path}")
