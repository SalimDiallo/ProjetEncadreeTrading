import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- EIA API Settings ---
EIA_API_KEY = os.getenv("EIA_API_KEY")
EIA_BASE_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

# Petroleum Series IDs (EIA v2)
# RWTC: WTI Spot Price
# RBRTE: Brent Spot Price
ASSETS = {
    "WTI": "RWTC",
    "BRENT": "RBRTE"
}

# --- Default Parameters ---
DEFAULT_FREQUENCY = "daily"
DEFAULT_DATA_COLUMN = "value"

# --- Storage Settings ---
# Using Parquet for performance as per technical constraints 
OUTPUT_FORMAT = "parquet"
# --- News Settings ---
NEWS_SOURCES = {
    "oilprice": "https://oilprice.com/Energy/Oil-Prices",
    "reuters_energy": "https://www.reuters.com/business/energy/"
}

# User-Agent to avoid being blocked by basic scrapers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
