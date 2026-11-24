"""PySpark Distributed ETL for NASA FIRMS VIIRS Hotspot Observations.

Processes raw satellite active fire detections across India (2020-2025):
- Spatial bounding box filtering
- Discretization into standardized geospatial grid cells (0.25° x 0.25°)
- Daily spatiotemporal aggregation of fire presence, count, and Fire Radiative Power (FRP)
"""

import math
from pathlib import Path
from typing import Optional, Union

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.config import (
    GEO_BOUNDS_INDIA,
    HOTSPOTS_GRIDDED_DIR,
    SPATIAL_GRID_DEGREE,
    TEMPORAL_SCOPE,
)
from src.processing.spark_session import get_spark_session, is_spark_available
from src.utils.logger import get_logger

logger = get_logger("hotspot_etl")


def compute_grid_cell(
    lat: float,
    lon: float,
    grid_degree: float = SPATIAL_GRID_DEGREE,
) -> tuple:
    """Calculate the centroid coordinate and identifier for a spatial grid cell.

    Args:
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        grid_degree: Dimension of the grid cell in degrees (default 0.25°).

    Returns:
        Tuple of (grid_id: str, grid_lat: float, grid_lon: float).
    """
    grid_lat = math.floor(lat / grid_degree) * grid_degree + (grid_degree / 2.0)
    grid_lon = math.floor(lon / grid_degree) * grid_degree + (grid_degree / 2.0)
    grid_lat = round(float(grid_lat), 4)
    grid_lon = round(float(grid_lon), 4)
    grid_id = f"CELL_{grid_lat:.2f}_{grid_lon:.2f}"
    return grid_id, grid_lat, grid_lon


def process_hotspots_pyspark(
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    grid_degree: float = SPATIAL_GRID_DEGREE,
):
    """Distributed PySpark ETL pipeline for raw NASA FIRMS VIIRS data.

    Args:
        csv_path: Path to raw FIRMS CSV or folder of CSVs.
        output_dir: Directory to save interim gridded output.
        grid_degree: Spatial grid size in degrees.

    Returns:
        PySpark DataFrame of daily gridded hotspot aggregations.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType

    spark = get_spark_session()
    path_str = str(Path(csv_path).resolve())
    logger.info(f"Reading raw FIRMS data into PySpark from: {path_str}")

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(path_str)

    # Standardize column casing
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.strip().lower())

    # 1. Geographic filtering (India bounds)
    b = GEO_BOUNDS_INDIA
    df_filtered = df.filter(
        (F.col("latitude") >= b["lat_min"])
        & (F.col("latitude") <= b["lat_max"])
        & (F.col("longitude") >= b["lon_min"])
        & (F.col("longitude") <= b["lon_max"])
    )

    # 2. Temporal filtering (2020 - 2025)
    t = TEMPORAL_SCOPE
    df_filtered = df_filtered.withColumn("acq_date", F.to_date(F.col("acq_date"), "yyyy-MM-dd"))
    df_filtered = df_filtered.filter(
        (F.col("acq_date") >= F.lit(t["start_date"]))
        & (F.col("acq_date") <= F.lit(t["end_date"]))
    )

    # 3. Spatial Grid Binning
    # Center of grid cell: floor(coord / grid_degree) * grid_degree + (grid_degree / 2)
    grid_lat_expr = (
        F.floor(F.col("latitude") / grid_degree) * grid_degree + (grid_degree / 2.0)
    ).cast(DoubleType())
    grid_lon_expr = (
        F.floor(F.col("longitude") / grid_degree) * grid_degree + (grid_degree / 2.0)
    ).cast(DoubleType())

    df_gridded = df_filtered.withColumn("grid_lat", F.round(grid_lat_expr, 4)).withColumn(
        "grid_lon", F.round(grid_lon_expr, 4)
    )

    df_gridded = df_gridded.withColumn(
        "grid_id",
        F.concat(
            F.lit("CELL_"),
            F.format_number(F.col("grid_lat"), 2),
            F.lit("_"),
            F.format_number(F.col("grid_lon"), 2),
        ),
    )

    # Cast FRP and brightness temperatures
    df_gridded = (
        df_gridded.withColumn("frp", F.col("frp").cast(DoubleType()))
        .withColumn("bright_ti4", F.col("bright_ti4").cast(DoubleType()))
        .withColumn("confidence", F.lower(F.trim(F.col("confidence"))))
    )

    # 4. Spatiotemporal Aggregation (Daily per Grid Cell)
    agg_df = df_gridded.groupBy("grid_id", "grid_lat", "grid_lon", "acq_date").agg(
        F.count("*").alias("hotspot_count"),
        F.round(F.max("frp"), 2).alias("max_frp"),
        F.round(F.sum("frp"), 2).alias("sum_frp"),
        F.round(F.avg("frp"), 2).alias("mean_frp"),
        F.round(F.avg("bright_ti4"), 2).alias("mean_bright_ti4"),
        F.sum(F.when(F.col("confidence") == "high", 1).otherwise(0)).alias("high_conf_count"),
        F.sum(F.when(F.col("daynight") == "D", 1).otherwise(0)).alias("day_detections"),
        F.sum(F.when(F.col("daynight") == "N", 1).otherwise(0)).alias("night_detections"),
    )

    out_path = Path(output_dir or HOTSPOTS_GRIDDED_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_out = str(out_path / "hotspots_daily_gridded.parquet")

    logger.info(f"Writing daily gridded hotspots to: {parquet_out}")
    agg_df.write.mode("overwrite").parquet(parquet_out)

    return agg_df


def process_hotspots_pandas(
    df: pd.DataFrame,
    grid_degree: float = SPATIAL_GRID_DEGREE,
) -> pd.DataFrame:
    """Vectorized Pandas implementation of the spatial-temporal grid aggregation.

    Serves as an in-memory processor and unit-testing baseline for PySpark pipeline.

    Args:
        df: Cleaned FIRMS DataFrame.
        grid_degree: Grid dimension in degrees.

    Returns:
        Aggregated daily gridded DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()

    # Spatial binning
    grid_lat = np.floor(df_copy["latitude"] / grid_degree) * grid_degree + (grid_degree / 2.0)
    grid_lon = np.floor(df_copy["longitude"] / grid_degree) * grid_degree + (grid_degree / 2.0)

    df_copy["grid_lat"] = grid_lat.round(4)
    df_copy["grid_lon"] = grid_lon.round(4)
    df_copy["grid_id"] = (
        "CELL_"
        + df_copy["grid_lat"].apply(lambda x: f"{x:.2f}")
        + "_"
        + df_copy["grid_lon"].apply(lambda x: f"{x:.2f}")
    )
    df_copy["acq_date"] = pd.to_datetime(df_copy["acq_date"]).dt.date

    # Aggregations
    grouped = df_copy.groupby(["grid_id", "grid_lat", "grid_lon", "acq_date"])

    agg_df = grouped.agg(
        hotspot_count=("latitude", "count"),
        max_frp=("frp", "max"),
        sum_frp=("frp", "sum"),
        mean_frp=("frp", "mean"),
        mean_bright_ti4=("bright_ti4", "mean") if "bright_ti4" in df_copy.columns else ("frp", "count"),
        high_conf_count=("confidence", lambda s: (s == "high").sum()),
        day_detections=("daynight", lambda s: (s == "D").sum()) if "daynight" in df_copy.columns else ("frp", "count"),
        night_detections=("daynight", lambda s: (s == "N").sum()) if "daynight" in df_copy.columns else ("frp", "count"),
    ).reset_index()

    return agg_df
