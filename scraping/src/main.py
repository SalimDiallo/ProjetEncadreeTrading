import pandas as pd
import config
from collectors.price_collector import EIAPriceCollector
from collectors.news_collector import OilNewsCollector

def run_data_pipeline():
    """
    Main orchestrator for the Data Module. 
    Handles both Price and News collection for the backtesting platform. 
    """
    print(" [Data Module] Starting Unified Pipeline...")

    # --- 1. Price Collection (WTI, Brent, etc.) ---
    # Iterating through assets defined in config makes it easy to scale.
    for asset_name in config.ASSETS.keys():
        print(f" [Price] Fetching {asset_name} data...")
        price_coll = EIAPriceCollector(asset_key=asset_name)
        df_prices = price_coll.fetch()
        
        if not df_prices.empty:
            output_path = config.PROCESSED_DATA_DIR / f"petrol_{asset_name.lower()}_daily.{config.OUTPUT_FORMAT}"
            df_prices.to_parquet(output_path, index=False)
            print(f" [Price] Successfully saved to: {output_path}")
            if df_prices.isnull().values.any():
                print(" [!] Warning: Processed data contains NaNs. Module ML might fail.")
        else:
            print(f" [Price] Warning: No data retrieved for {asset_name}")

    # --- 2. News Collection (OilPrice, Reuters, etc.) ---
    # Scrapes headlines to provide raw text for the Module NLP. 
    for source_name in config.NEWS_SOURCES.keys():
        print(f" [News] Scraping {source_name}...")
        news_coll = OilNewsCollector(source_key=source_name)
        df_news = news_coll.fetch()
        
        if not df_news.empty:
            output_path = config.PROCESSED_DATA_DIR / f"petrol_news_{source_name}.{config.OUTPUT_FORMAT}"
            df_news.to_parquet(output_path, index=False)
            print(f" [News] Successfully saved to: {output_path}")
        else:
            print(f" [News] Warning: No news headlines found for {source_name}")

    print(" [Data Module] Pipeline execution complete. Data is ready for ML/NLP modules. ")

if __name__ == "__main__":
    run_data_pipeline()
