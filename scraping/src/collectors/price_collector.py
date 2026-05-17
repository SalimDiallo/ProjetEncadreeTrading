import requests
import json
from datetime import datetime
import pandas as pd
import config  # Importing your central config
from .base import BaseCollector

class EIAPriceCollector(BaseCollector):
    """
    Handles connection to EIA v2 for Petroleum prices.
    Ref: https://www.eia.gov/opendata/v2/petroleum/pri/spt/data/
    """
    def __init__(self, asset_key: str = "WTI"):
        self.api_key = config.EIA_API_KEY
        self.series_id = config.ASSETS.get(asset_key)
        self.base_url = config.EIA_BASE_URL
        self.asset_key = asset_key

    def fetch(self) -> pd.DataFrame:
        """Fetches and performs initial parsing of EIA data."""
        params = {
            "api_key": self.api_key,
            "frequency": config.DEFAULT_FREQUENCY,
            "data[0]": config.DEFAULT_DATA_COLUMN,
            "facets[series][]": [self.series_id],
            "sort[0][column]": "period",
            "sort[0][direction]": "desc"
        }


        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            raw_json= response.json()

            """"save to the raw subfolder for traceability"""
            self._archive_raw_data(raw_json)
            
            raw_data = raw_json.get('response', {}).get('data', [])
            if not raw_data:
                print(f"Warning: No data found for {self.series_id}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            # Map EIA columns to our internal standard
            # 'period' is the EIA date column, 'value' is the price
            df = df.rename(columns={'period': 'date', 'value': 'price'})
            
            # Return raw data for the Processor layer to handle
            return df[['date', 'price']]

        except requests.exceptions.RequestException as e:
            print(f"Network error fetching EIA data: {e}")
            return pd.DataFrame()

    def _archive_raw_data(self, data: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_{self.asset_key}_{timestamp}.json"
        file_path = config.RAW_DATA_DIR / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        
        print(f"Raw data archived at: {file_path}")
