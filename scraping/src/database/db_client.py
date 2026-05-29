from sqlalchemy import create_engine
import os
import pandas as pd

class DatabaseClient:
    """Gère l'ingestion des données (prix et NLP) en Base de Données."""
    def __init__(self):
        # Par défaut, utilise une base SQLite locale si DATABASE_URL n'est pas définie
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///trading_platform.db")
        self.engine = create_engine(self.db_url)

    def save_prices(self, df: pd.DataFrame, table_name: str = "oil_prices"):
        if not df.empty:
            df.to_sql(table_name, con=self.engine, if_exists='append', index=False)
            print(f" [DB] {len(df)} lignes de PRIX insérées dans {table_name}.")

    def save_news(self, df: pd.DataFrame, table_name: str = "oil_news"):
        if not df.empty:
            df.to_sql(table_name, con=self.engine, if_exists='append', index=False)
            print(f" [DB] {len(df)} articles NEWS insérés dans {table_name}.")
