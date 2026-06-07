import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timezone

# Ensure project directories are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NLP_DIR = os.path.join(PROJECT_ROOT, "ai", "nlp")
SCRAPING_DIR = os.path.join(PROJECT_ROOT, "scraping", "src")

for path in [PROJECT_ROOT, NLP_DIR, SCRAPING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Force the database URL to point to a central database in the project root
DB_PATH = os.path.join(PROJECT_ROOT, "trading_platform.db")
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(NLP_DIR, ".env"))

from scraping.src.database.db_client import DatabaseClient
from scraping.src.collectors.price_collector import EIAPriceCollector
from scraping.src.collectors.historical_news_collector import HistoricalNewsCollector
from scraping.src import config

from oil_sentiment_pipeline.data_ingestion.utils import normalize_batch, deduplicate_records

def enrich_prices():
    """Récupère l'historique complet des prix EIA et les insère dans SQLite et Parquet."""
    print("\n" + "="*70)
    print(" [PRIX] ENRICHISSEMENT DES PRIX SPOT (WTI & BRENT) DEPUIS 2013")
    print("="*70)
    
    db = DatabaseClient()
    
    for asset in ["WTI", "BRENT"]:
        print(f"\n [Prix] Récupération de l'historique des prix pour {asset}...")
        collector = EIAPriceCollector(asset_key=asset)
        df = collector.fetch()
        
        if not df.empty:
            # S'assurer que les dates sont triées chronologiquement
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            # Filtrer à partir du 1er janvier 2013
            df = df[df["date"] >= pd.Timestamp("2013-01-01")].reset_index(drop=True)
            
            # 1. Sauvegarde en Base de Données SQLite
            table_name = f"prices_{asset.lower()}"
            # Nettoyer l'ancienne table si elle existe pour éviter les doublons au premier run
            try:
                with db.engine.connect() as conn:
                    # SQLAlchemy 2.0 transaction block execution
                    from sqlalchemy import text
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    print(f" [DB] Table {table_name} nettoyée.")
            except Exception as e:
                pass
                
            db.save_prices(df, table_name=table_name)
            
            # 2. Sauvegarde en Parquet pour le backtest/dashboard
            output_path = config.PROCESSED_DATA_DIR / f"petrol_{asset.lower()}_daily.{config.OUTPUT_FORMAT}"
            df.to_parquet(output_path, index=False)
            print(f" [Parquet] Sauvegardé dans {output_path} ({len(df)} lignes)")
        else:
            print(f" [!] Aucune donnée de prix récupérée pour {asset}")

def enrich_news(from_year: int, to_year: int, dry_run: bool = False, limit_snapshots: int = 15):
    """Récupère et enrichit la base avec les actualités historiques (2013-Présent)."""
    print("\n" + "="*70)
    print(f" [NEWS] ENRICHISSEMENT DES ACTUALITES PETROLE ({from_year}-{to_year})")
    print("="*70)
    
    collector = HistoricalNewsCollector()
    
    # 1. Collecter toutes les données historiques
    print(f" [Collecte] Lancement des collecteurs (Wayback snapshots limite: {limit_snapshots})...")
    # Surcharge temporaire des limites de snapshots dans le collecteur pour le run
    raw_records = collector.collect_all_historical(from_year=from_year, to_year=to_year)
    
    if not raw_records:
        print(" [!] Aucun record collecté. Fin du pipeline d'enrichissement.")
        return
        
    print(f"\n [Traitement] {len(raw_records)} articles bruts collectés.")
    
    # 2. Normalisation et Déduplication
    normalized = normalize_batch(raw_records, min_text_length=15)
    deduplicated = deduplicate_records(normalized)
    
    print(f" [Traitement] Après normalisation : {len(normalized)} articles.")
    print(f" [Traitement] Après déduplication globale : {len(deduplicated)} articles.")
    
    if dry_run:
        print("\n [Dry-Run] Mode Dry-Run actif. Aucune écriture en base.")
        print(" Échantillon de records :")
        df_sample = pd.DataFrame(deduplicated).head(5)
        print(df_sample)
        return
        
    # Convertir en DataFrame pour l'enregistrement
    df_news = pd.DataFrame(deduplicated)
    
    if df_news.empty:
        print(" [!] Aucun article après traitement.")
        return
        
    # Ajouter une colonne 'title' et 'content' pour compatibilité avec le client DB existant
    # Si 'text' contient le titre et la description séparés par un point, on les sépare
    titles = []
    contents = []
    for txt in df_news["text"]:
        parts = txt.split(".", 1)
        titles.append(parts[0].strip())
        contents.append(parts[1].strip() if len(parts) > 1 else txt)
        
    df_news["title"] = titles
    df_news["content"] = contents
    
    # Sélectionner les colonnes standard pour la base de données
    df_db = df_news[["date", "title", "content", "source"]]
    
    # 3. Enregistrement en Base de Données
    db = DatabaseClient()
    table_name = "oil_news"
    
    # Nettoyer l'ancienne table pour éviter les doublons lors de l'initialisation
    try:
        with db.engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.execute(text(f"DROP TABLE IF EXISTS oil_market_news"))
            print(f" [DB] Tables d'actualités nettoyées.")
    except Exception:
        pass

    # Sauvegarder dans oil_news (et dans oil_market_news pour l'Airflow DAG)
    db.save_news(df_db, table_name=table_name)
    db.save_news(df_db, table_name="oil_market_news")
    
    # 4. Enregistrement en Parquet subdivisé par source
    print("\n [Parquet] Exportation des actualités par source pour le Dashboard...")
    # Regrouper les sources majeures sous le nom attendu par le dashboard
    # Le dashboard charge 'petrol_news_oilprice.parquet'
    oilprice_data = df_db[df_db["source"].str.contains("oilprice", case=False)]
    if not oilprice_data.empty:
        op_path = config.PROCESSED_DATA_DIR / "petrol_news_oilprice.parquet"
        oilprice_data.to_parquet(op_path, index=False)
        print(f"  -> {len(oilprice_data)} articles de journaux exportés vers {op_path}")
        
    # Sauvegarder le dump global dans processed/
    global_path = config.PROCESSED_DATA_DIR / "petrol_news_all_sources.parquet"
    df_db.to_parquet(global_path, index=False)
    print(f"  -> Total {len(df_db)} articles exportés vers {global_path}")

    print("\n [SUCCESS] Enrichissement de la base de donnees termine avec succes !")
    print(f" Fichier SQLite créé : {DB_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Enrichir la base WTI avec prix et actualités 2013-Présent.")
    parser.add_argument("--prices-only", action="store_true", help="Enrichir uniquement les prix.")
    parser.add_argument("--news-only", action="store_true", help="Enrichir uniquement les actualités.")
    parser.add_argument("--dry-run", action="store_true", help="Exécuter sans écrire en base.")
    parser.add_argument("--from-year", type=int, default=2013, help="Année de départ pour la collecte.")
    parser.add_argument("--to-year", type=int, default=2026, help="Année de fin pour la collecte.")
    parser.add_argument("--limit-snapshots", type=int, default=15, help="Nombre de captures Wayback à échantillonner par source.")
    
    args = parser.parse_args()
    
    if args.prices_only:
        enrich_prices()
    elif args.news_only:
        enrich_news(args.from_year, args.to_year, args.dry_run, args.limit_snapshots)
    else:
        enrich_prices()
        enrich_news(args.from_year, args.to_year, args.dry_run, args.limit_snapshots)

if __name__ == "__main__":
    main()
