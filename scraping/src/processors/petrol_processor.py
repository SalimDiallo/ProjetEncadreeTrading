import requests
import pandas as pd
from typing import Optional

class PetrolPriceCollector:
    """Fetches daily petrol (WTI/Brent) prices from EIA API v2."""
    
    BASE_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_daily_data(self, series_id: str = "RWTC") -> pd.DataFrame:
        """RWTC is WTI Spot Price. Returns a cleaned DataFrame."""
        params = {
            "api_key": self.api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": [series_id],
            "sort[0][column]": "period",
            "sort[0][direction]": "desc"
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status() # Basic error handling
        
        data = response.json()['response']['data']
        df = pd.DataFrame(data)
        
        # Immediate cleanup
        df['period'] = pd.to_datetime(df['period'])
        df = df.rename(columns={'period': 'date', 'value': 'price'})
        return df[['date', 'price']]
