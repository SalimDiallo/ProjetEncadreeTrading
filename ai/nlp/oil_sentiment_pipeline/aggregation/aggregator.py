import pandas as pd
import logging

logger = logging.getLogger(__name__)

def aggregate_daily_sentiment(records: list[dict], settings) -> dict[str, pd.DataFrame]:
    """
    Groups article sentiments by day, applies configuration weights, 
    and splits them into WTI and Brent datasets.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return {}

    # Ensure datetime format and extract date
    df["date"] = pd.to_datetime(df["date"])
    df["date_only"] = df["date"].dt.date

    # Compute weights for each record
    source_weights = settings.source_weights

    def calculate_weight(row):
        w_conf = row.get("sentiment_confidence", 1.0) if settings.weight_by_confidence else 1.0
        w_dens = row.get("oil_density", 1.0) if settings.weight_by_oil_density else 1.0
        
        source = row.get("source", "default")
        w_src = source_weights.get(source, source_weights.get("default", 1.0)) if settings.weight_by_source else 1.0
        
        return float(w_conf * w_dens * w_src)

    df["weight"] = df.apply(calculate_weight, axis=1)
    df["weighted_score"] = df["sentiment_score"] * df["weight"]

    # Classify articles by asset keywords (fallback to both if general news)
    # Convert text to string first to handle any non-string values
    text_series = df["text"].astype(str)
    wti_mask = text_series.str.contains("wti|west texas|cl=f|crude|petroleum", case=False, na=False)
    brent_mask = text_series.str.contains("brent|london|bz=f", case=False, na=False)
    general_mask = ~(wti_mask | brent_mask)

    wti_df = df[wti_mask | general_mask]
    brent_df = df[brent_mask | general_mask]

    results = {}
    for asset, asset_df in [("wti", wti_df), ("brent", brent_df)]:
        if asset_df.empty:
            continue
        
        # Weighted mean score grouping by date
        # Group by 'date_only' and apply daily aggregation logic
        grouped = asset_df.groupby("date_only").apply(
            lambda x: pd.Series({
                "sentiment_score": (x["weighted_score"].sum() / x["weight"].sum()) if x["weight"].sum() > 0 else x["sentiment_score"].mean(),
                "n_articles": len(x)
            }),
            include_groups=False
        ).reset_index()

        grouped = grouped.rename(columns={"date_only": "date"})
        grouped["date"] = pd.to_datetime(grouped["date"])
        results[asset] = grouped

    return results
