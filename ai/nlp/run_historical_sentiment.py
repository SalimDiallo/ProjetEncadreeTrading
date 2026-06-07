import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

# Add paths to sys.path
NLP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(NLP_DIR, "..", ".."))

for path in [NLP_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Setup central database path
DB_PATH = os.path.join(PROJECT_ROOT, "trading_platform.db")
db_url = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
engine = create_engine(db_url)

from oil_sentiment_pipeline.preprocessing.pipeline import run_preprocessing
from oil_sentiment_pipeline.modeling.pipeline import run_sentiment_analysis
from oil_sentiment_pipeline.aggregation.aggregator import aggregate_daily_sentiment
from oil_sentiment_pipeline.paths import SHARED_PROCESSED_DIR
from oil_sentiment_pipeline.settings import PipelineSettings

def run_historical_pipeline(model_type: str = "lexical"):
    """
    Charge les actualités brutes depuis SQLite, exécute le nettoyage (preprocessing),
    calcule le sentiment et exporte les résultats vers SQLite et Parquet.
    """
    print("\n" + "="*70)
    print(f" [NLP] PIPELINE DE SENTIMENT HISTORIQUE (MODELE: {model_type.upper()})")
    print("="*70)
    
    # 1. Chargement des actualités brutes depuis SQLite
    print(f" [DB] Lecture des actualites depuis {DB_PATH}...")
    try:
        df = pd.read_sql("SELECT date, title, content, source FROM oil_news", con=engine)
    except Exception as e:
        print(f" [!] Erreur lors de la lecture de la table oil_news : {e}")
        print("     Avez-vous bien execute la phase de collecte ?")
        print("     Commande : venv\\Scripts\\python scraping/src/enrich_db.py")
        sys.exit(1)
        
    if df.empty:
        print(" [!] La table oil_news est vide. Veuillez d'abord collecter des actualites.")
        sys.exit(1)
        
    print(f" [DB] {len(df)} actualites brutes chargees.")
    
    # Reconstruire les records textuels pour le preprocessing
    raw_records = []
    for _, row in df.iterrows():
        title = str(row["title"] or "").strip()
        content = str(row["content"] or "").strip()
        # Concaténer titre et contenu
        text_full = f"{title}. {content}" if title and content else (title or content)
        
        raw_records.append({
            "text": text_full,
            "date": str(row["date"]),
            "source": str(row["source"])
        })
        
    # 2. Nettoyage et Normalisation (Preprocessing)
    print("\n [Preprocessing] Nettoyage et normalisation des textes...")
    # min_tokens=3 pour filtrer les textes trop courts, min_oil_density=0.0 pour tout garder
    processed_records = run_preprocessing(
        records=raw_records,
        save_csv=False,
        min_tokens=3,
        min_oil_density=0.0
    )
    print(f" [Preprocessing] {len(processed_records)}/{len(raw_records)} articles conserves apres nettoyage.")
    
    if not processed_records:
        print(" [!] Aucun article restant apres preprocessing. Fin du pipeline.")
        return
        
    # 3. Calcul de Sentiment
    print(f"\n [Sentiment] Calcul des scores de sentiment (Modèle: {model_type})...")
    sentiment_records = run_sentiment_analysis(
        processed_records,
        model=model_type,
        text_field="text_clean", # Le preprocessing nettoie les textes
        save_results=False
    )
    
    # 4. Enregistrement des actualités individuelles scorées dans SQLite
    print("\n [DB] Enregistrement des actualites scorees dans la table 'oil_news_sentiment'...")
    df_sent = pd.DataFrame(sentiment_records)
    
    # Nettoyage des colonnes complexes (comme les listes de tokens qui ne passent pas en SQL)
    if "tokens" in df_sent.columns:
        df_sent = df_sent.drop(columns=["tokens"])
        
    try:
        # Remplacer la table existante si elle existe
        df_sent.to_sql("oil_news_sentiment", con=engine, if_exists="replace", index=False)
        print(f" [DB] {len(df_sent)} actualites individuelles avec score de sentiment sauvegardees.")
    except Exception as e:
        print(f" [!] Erreur lors de l'ecriture en base de donnees : {e}")
        
    # 5. Agrégation journalière et Export Parquet pour le Dashboard/ML
    print("\n [Aggregation] Compilation et aggregation journaliere du sentiment...")
    cfg = PipelineSettings.from_yaml()
    aggregated_data = aggregate_daily_sentiment(sentiment_records, cfg)
    
    print("\n [Export] Sauvegarde des parquets d'aggregation journaliere...")
    for asset, df_asset in aggregated_data.items():
        out_path = SHARED_PROCESSED_DIR / f"sentiment_{asset.lower()}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # S'assurer que le format de date est correct et trié
        df_asset["date"] = pd.to_datetime(df_asset["date"])
        df_asset = df_asset.sort_values("date").reset_index(drop=True)
        
        df_asset.to_parquet(out_path, index=False)
        print(f"  -> {asset} : {len(df_asset)} jours de sentiment exportes vers {out_path}")
        
    print("\n [SUCCESS] Pipeline de sentiment historique execute avec succes !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean et calcule le sentiment sur la base historique.")
    parser.add_argument(
        "--model",
        default="lexical",
        choices=["lexical", "finbert", "logistic_regression", "auto"],
        help="Modèle d'analyse de sentiment à utiliser (défaut: lexical)."
    )
    args = parser.parse_args()
    run_historical_pipeline(model_type=args.model)
