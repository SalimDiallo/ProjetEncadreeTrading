from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Imports avec des chemins absolus depuis la racine du projet
from scraping.src.collectors.price_collector import EIAPriceCollector
from scraping.src.collectors.news_collector import OilNewsCollector
from scraping.src.database.db_client import DatabaseClient

default_args = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_disabled': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def job_extract_prices():
    """Extraction EIA -> BDD"""
    db = DatabaseClient()
    for asset in ["WTI", "BRENT"]:
        collector = EIAPriceCollector(asset_key=asset)
        df = collector.fetch()
        if not df.empty:
            db.save_prices(df, table_name=f"prices_{asset.lower()}")

def job_extract_news():
    """Scraping NLP -> BDD"""
    db = DatabaseClient()
    collector = OilNewsCollector(source_key="oilprice")
    df = collector.fetch()
    if not df.empty:
        db.save_news(df, table_name="oil_market_news")

# Initialisation du DAG (Déclenchement quotidien)
with DAG(
    'trading_daily_ingestion',
    default_args=default_args,
    description='Pipeline unifié ETL pour prix et news pétrolières',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    task_prices = PythonOperator(
        task_id='extract_oil_prices',
        python_callable=job_extract_prices,
    )

    task_news = PythonOperator(
        task_id='scrape_oil_news',
        python_callable=job_extract_news,
    )

    # Indépendance des tâches (peuvent tourner en parallèle)
    [task_prices, task_news]
