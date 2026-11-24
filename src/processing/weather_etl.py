"""Meteorological Data Processing and Spatiotemporal Alignment.

Standardizes daily weather variables from NASA POWER, verifies physical bounds,
handles missing-value imputation, and maps weather stations/points to the canonical
0.25° spatial grid.
"""

from pathlib import Path
from typing import Optional, Union

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.config import (
    METEOROLOGICAL_PARAMS,
    SPATIAL_GRID_DEGREE,
    WEATHER_GRIDDED_DIR,
)
from src.processing.hotspot_etl import compute_grid_cell
from src.processing.spark_session import get_spark_session, is_spark_available
from src.utils.logger import get_logger

logger = get_logger("weather_etl")

# Physical validity bounds for quality control
PHYSICAL_BOUNDS = {
    "T2M": (-20.0, 60.0),       # Degrees Celsius
    "T2M_MAX": (-20.0, 60.0),
    "T2M_MIN": (-25.0, 50.0),
    "RH2M": (0.0, 100.0),       # Percentage
    "PRECTOTCORR": (0.0, 800.0),# mm/day (maximum recorded daily rain in India)
    "WS10M": (0.0, 75.0),       # m/s (up to extreme cyclone strength)
    "PS": (50.0, 110.0),        # Surface pressure in kPa
}


def sanitize_meteorological_values(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce physical sanity bounds on weather variables, replacing anomalies with NaN.

    Args:
        df: Raw meteorological DataFrame.

    Returns:
        Sanitized DataFrame.
    """
    clean_df = df.copy()
    for col, (lower, upper) in PHYSICAL_BOUNDS.items():
        if col in clean_df.columns:
            # Replace out-of-bounds or fill values with NaN
            invalid_mask = (clean_df[col] < lower) | (clean_df[col] > upper)
            outlier_count = invalid_mask.sum()
            if outlier_count > 0:
                logger.debug(f"Variable '{col}': masked {outlier_count} values outside [{lower}, {upper}]")
                clean_df.loc[invalid_mask, col] = np.nan

    return clean_df


def clean_weather_pandas(
    df: pd.DataFrame,
    grid_degree: float = SPATIAL_GRID_DEGREE,
) -> pd.DataFrame:
    """Process, clean, and map meteorological DataFrame onto canonical spatial grid cells.

    Args:
        df: Ingested meteorological DataFrame.
        grid_degree: Spatial grid size in degrees.

    Returns:
        Cleaned, gridded daily weather DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Ensure date is standard date object
    df_clean["date"] = pd.to_datetime(df_clean["date"]).dt.date

    # Sanitize physical bounds
    df_clean = sanitize_meteorological_values(df_clean)

    # Assign canonical spatial grid cell
    grid_tuples = [
        compute_grid_cell(lat, lon, grid_degree=grid_degree)
        for lat, lon in zip(df_clean["latitude"], df_clean["longitude"])
    ]
    df_clean["grid_id"] = [t[0] for t in grid_tuples]
    df_clean["grid_lat"] = [t[1] for t in grid_tuples]
    df_clean["grid_lon"] = [t[2] for t in grid_tuples]

    # Impute small temporal gaps per grid cell via forward/backward fill
    numeric_weather_cols = [c for c in METEOROLOGICAL_PARAMS if c in df_clean.columns]
    df_clean = df_clean.sort_values(by=["grid_id", "date"]).reset_index(drop=True)
    df_clean[numeric_weather_cols] = (
        df_clean.groupby("grid_id")[numeric_weather_cols]
        .transform(lambda group: group.ffill().bfill())
    )

    logger.info(
        f"Processed gridded weather data: {len(df_clean):,} records across "
        f"{df_clean['grid_id'].nunique()} unique spatial cells."
    )
    return df_clean


def process_weather_pyspark(
    csv_or_parquet_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    grid_degree: float = SPATIAL_GRID_DEGREE,
):
    """PySpark ETL pipeline for scalable meteorological dataset gridding.

    Args:
        csv_or_parquet_path: Input data path.
        output_dir: Interim storage directory.
        grid_degree: Grid size in degrees.

    Returns:
        PySpark DataFrame of gridded daily weather.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType

    spark = get_spark_session()
    path_str = str(Path(csv_or_parquet_path).resolve())
    logger.info(f"Reading weather data into PySpark from: {path_str}")

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(path_str)

    # Map to canonical grid cell
    grid_lat_expr = (
        F.floor(F.col("latitude") / grid_degree) * grid_degree + (grid_degree / 2.0)
    ).cast(DoubleType())
    grid_lon_expr = (
        F.floor(F.col("longitude") / grid_degree) * grid_degree + (grid_degree / 2.0)
    ).cast(DoubleType())

    df_gridded = (
        df.withColumn("grid_lat", F.round(grid_lat_expr, 4))
        .withColumn("grid_lon", F.round(grid_lon_expr, 4))
        .withColumn(
            "grid_id",
            F.concat(
                F.lit("CELL_"),
                F.format_number(grid_lat_expr, 2),
                F.lit("_"),
                F.format_number(grid_lon_expr, 2),
            ),
        )
        .withColumn("date", F.to_date(F.col("date")))
    )

    out_path = Path(output_dir or WEATHER_GRIDDED_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_out = str(out_path / "weather_daily_gridded.parquet")

    logger.info(f"Writing daily gridded weather to: {parquet_out}")
    df_gridded.write.mode("overwrite").parquet(parquet_out)

    return df_gridded
