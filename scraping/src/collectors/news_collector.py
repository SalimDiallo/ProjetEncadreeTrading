import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import json
import config
from collectors.base import BaseCollector

class OilNewsCollector(BaseCollector):
    """
    Scrapes headlines and summaries from oil-specific news sites. 
    """
    def __init__(self, source_key: str = "oilprice"):
        self.source_key = source_key
        self.url = config.NEWS_SOURCES.get(source_key)
        self.headers = config.HEADERS

    def fetch(self) -> pd.DataFrame:
        """Fetches news and returns a DataFrame with titles and dates."""
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Archive raw HTML for reproducibility 
            self._archive_raw_data({"html": response.text})

            soup = BeautifulSoup(response.text, 'html.parser')
            news_data = []

            # Specific logic for OilPrice.com (example)
            # Note: Selectors may need adjustment based on site changes
            articles = soup.find_all('div', class_='categoryArticle')
            
            for article in articles:
                title = article.find('h2').get_text(strip=True) if article.find('h2') else None
                summary = article.find('p').get_text(strip=True) if article.find('p') else None
                
                if title:
                    news_data.append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "title": title,
                        "content": summary,
                        "source": self.source_key
                    })

            return pd.DataFrame(news_data)

        except Exception as e:
            print(f" [News Module] Error scraping {self.source_key}: {e}")
            return pd.DataFrame()

    def _archive_raw_data(self, data: dict):
        """Saves the raw HTML response for auditing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_{self.source_key}_{timestamp}.json"
        file_path = config.RAW_DATA_DIR / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
