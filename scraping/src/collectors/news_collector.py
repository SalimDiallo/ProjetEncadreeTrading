import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from scraping.src.collectors.base import BaseCollector
from scraping.src import config

class OilNewsCollector(BaseCollector):
    """
    Collecteur standardisé pour le scraping des actualités pétrolières
    migré depuis ai/nlp/oil_sentiment_pipeline/data_ingestion/
    """
    def __init__(self, source_key: str = "oilprice"):
        self.source_key = source_key
        # Si NEWS_SOURCES n'est pas encore dans ton config.py, ajoute-le :
        # NEWS_SOURCES = {"oilprice": "https://oilprice.com/Energy/Oil-Prices"}
        self.url = getattr(config, "NEWS_SOURCES", {}).get(source_key, "https://oilprice.com/Energy/Oil-Prices")
        self.headers = getattr(config, "HEADERS", {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })

    def fetch(self) -> pd.DataFrame:
        """Exécute le scraping et retourne un DataFrame standardisé."""
        try:
            response = requests.get(self.url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Sauvegarde de la donnée brute (JSON/HTML) dans le dossier RAW
            self._archive_raw_data({"html": response.text})

            soup = BeautifulSoup(response.text, 'html.parser')
            news_data = []

            # Extraction spécifique pour OilPrice
            articles = soup.find_all('div', class_='categoryArticle')
            for article in articles:
                title_el = article.find('h2')
                desc_el = article.find('p')
                
                title = title_el.get_text(strip=True) if title_el else None
                summary = desc_el.get_text(strip=True) if desc_el else None
                
                if title:
                    news_data.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "title": title,
                        "content": summary,
                        "source": self.source_key
                    })

            return pd.DataFrame(news_data)

        except Exception as e:
            print(f" [!] Erreur lors du scraping de {self.source_key}: {e}")
            return pd.DataFrame()

    def _archive_raw_data(self, data: dict):
        """Archive le payload brut."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_{self.source_key}_{timestamp}.json"
        
        # Vérification si RAW_DATA_DIR existe dans config, sinon fallback
        raw_dir = getattr(config, "RAW_DATA_DIR", None)
        if raw_dir:
            file_path = raw_dir / filename
        else:
            from pathlib import Path
            file_path = Path(__file__).resolve().parent.parent / "data" / "raw" / filename
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
