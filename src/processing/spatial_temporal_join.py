"""Spatial-Temporal Alignment and Analytical Base Table (ABT) Construction.

Merges daily gridded thermal anomaly observations with gridded meteorological covariates
across India (2020-2025). Formulates the binary target variable (fire presence/absence)
and quantitative intensity targets (Fire Radiative Power).
"""

from pathlib import Path
from typing import Optional, Union

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.config import ANALYTICAL_BASE_DIR
from src.processing.spark_session import get_spark_session, is_spark_available
from src.utils.logger import get_logger

logger = get_logger("spatial_temporal_join")


def build_analytical_table_pandas(
    df_hotspots_gridded: pd.DataFrame,
    df_weather_gridded: pd.DataFrame,
) -> pd.DataFrame:
    """Join gridded weather covariates with active hotspot presence in Pandas.

    Performs a left join from weather grid-dates onto hotspot records, creating
    verified zero-fire background instances for machine learning modeling.

    Args:
        df_hotspots_gridded: Daily aggregated hotspot detections (grid_id, acq_date, ...).
        df_weather_gridded: Daily weather records per grid cell (grid_id, date, ...).

    Returns:
        Unified Analytical Base Table (ABT).
    """
    if df_weather_gridded.empty:
        logger.warning("Weather dataset is empty. Cannot construct Analytical Base Table.")
        return pd.DataFrame()

    weather = df_weather_gridded.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.date

    hotspots = df_hotspots_gridded.copy() if not df_hotspots_gridded.empty else pd.DataFrame()
    if not hotspots.empty:
        hotspots["acq_date"] = pd.to_datetime(hotspots["acq_date"]).dt.date

        # Left join weather (universe of space-time cells) with observed hotspots
        merged = pd.merge(
            weather,
            hotspots,
            left_on=["grid_id", "date"],
            right_on=["grid_id", "acq_date"],
            how="left",
            suffixes=("", "_hotspot"),
        )
    else:
        merged = weather.copy()
        merged["hotspot_count"] = 0
        merged["max_frp"] = 0.0
        merged["sum_frp"] = 0.0

    # Fill non-detection records with zeros (negative background class)
    merged["hotspot_count"] = merged["hotspot_count"].fillna(0).astype(int)
    merged["max_frp"] = merged["max_frp"].fillna(0.0).astype(float)
    merged["sum_frp"] = merged["sum_frp"].fillna(0.0).astype(float)
    merged["high_conf_count"] = (
        merged["high_conf_count"].fillna(0).astype(int)
        if "high_conf_count" in merged.columns
        else 0
    )

    # Binary classification target: 1 = Active hotspot observed, 0 = No hotspot observed
    merged["has_hotspot"] = (merged["hotspot_count"] > 0).astype(int)

    # Clean redundant join columns
    drop_cols = [c for c in ["acq_date", "grid_lat_hotspot", "grid_lon_hotspot"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    # Sort chronologically by space and time
    merged = merged.sort_values(by=["grid_id", "date"]).reset_index(drop=True)

    positive_events = merged["has_hotspot"].sum()
    total_records = len(merged)
    event_rate = (positive_events / max(total_records, 1)) * 100

    logger.info(
        f"Analytical Base Table constructed: {total_records:,} spatio-temporal rows. "
        f"Active fire events: {positive_events:,} ({event_rate:.2f}% prevalence)."
    )

    return merged


def build_analytical_table_pyspark(
    weather_parquet_path: Union[str, Path],
    hotspots_parquet_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
):
    """PySpark implementation for joining large-scale gridded weather and hotspots.

    Args:
        weather_parquet_path: Path to weather Parquet directory.
        hotspots_parquet_path: Path to hotspots Parquet directory.
        output_dir: Output storage directory.

    Returns:
        PySpark DataFrame representing the Analytical Base Table.
    """
    from pyspark.sql import functions as F

    spark = get_spark_session()
    w_path = str(Path(weather_parquet_path).resolve())
    h_path = str(Path(hotspots_parquet_path).resolve())

    logger.info(f"Loading weather Parquet from: {w_path}")
    df_weather = spark.read.parquet(w_path)

    logger.info(f"Loading hotspots Parquet from: {h_path}")
    df_hotspots = spark.read.parquet(h_path)

    # Perform distributed left join
    df_joined = df_weather.join(
        df_hotspots,
        (df_weather["grid_id"] == df_hotspots["grid_id"])
        & (df_weather["date"] == df_hotspots["acq_date"]),
        how="left",
    ).drop(df_hotspots["grid_id"]).drop(df_hotspots["acq_date"])

    # Impute missing values for background non-fire observations
    df_abt = (
        df_joined.withColumn("hotspot_count", F.coalesce(F.col("hotspot_count"), F.lit(0)))
        .withColumn("max_frp", F.coalesce(F.col("max_frp"), F.lit(0.0)))
        .withColumn("sum_frp", F.coalesce(F.col("sum_frp"), F.lit(0.0)))
        .withColumn("high_conf_count", F.coalesce(F.col("high_conf_count"), F.lit(0)))
        .withColumn(
            "has_hotspot",
            F.when(F.col("hotspot_count") > 0, 1).otherwise(0),
        )
    )

    out_path = Path(output_dir or ANALYTICAL_BASE_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_out = str(out_path / "abt.parquet")

    logger.info(f"Writing joined Analytical Base Table to: {parquet_out}")
    df_abt.write.mode("overwrite").parquet(parquet_out)

    return df_abt
