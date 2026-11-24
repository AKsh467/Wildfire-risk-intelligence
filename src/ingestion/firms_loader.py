"""NASA FIRMS VIIRS S-NPP Active Fire Data Ingestion & Validator.

Loads, validates, and standardizes 375m active fire hotspot observation records
originating from the NASA FIRMS (Fire Information for Resource Management System) archive.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.config import FIRMS_RAW_DIR, GEO_BOUNDS_INDIA, TEMPORAL_SCOPE, VIIRS_SPEC
from src.utils.logger import get_logger

logger = get_logger("firms_loader")

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "confidence",
    "frp",
]


def validate_firms_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that incoming DataFrame contains the expected NASA FIRMS columns.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple of (is_valid: bool, missing_columns: list).
    """
    df_cols = [c.lower().strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df_cols]
    is_valid = len(missing) == 0
    return is_valid, missing


def clean_firms_hotspots(
    df: pd.DataFrame,
    filter_india: bool = True,
    filter_dates: bool = True,
) -> pd.DataFrame:
    """Standardize column names, types, and geographic/temporal bounds.

    Args:
        df: Raw FIRMS DataFrame.
        filter_india: Whether to clip records to the India bounding box.
        filter_dates: Whether to clip records to the 2020-2025 study period.

    Returns:
        Cleaned, type-casted DataFrame.
    """
    if df.empty:
        return df.copy()

    # Standardize column casing
    cleaned = df.copy()
    cleaned.columns = [c.lower().strip() for c in cleaned.columns]

    initial_count = len(cleaned)

    # Cast coordinates and FRP to numeric
    numeric_cols = ["latitude", "longitude", "frp", "bright_ti4", "bright_ti5", "scan", "track"]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Drop records with invalid coordinates
    cleaned = cleaned.dropna(subset=["latitude", "longitude"])

    # Standardize acq_date to datetime
    cleaned["acq_date"] = pd.to_datetime(cleaned["acq_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["acq_date"])

    # Standardize confidence (VIIRS uses 'l', 'n', 'h' or 'low', 'nominal', 'high')
    if "confidence" in cleaned.columns:
        cleaned["confidence"] = (
            cleaned["confidence"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"l": "low", "n": "nominal", "h": "high"})
        )

    # Filter by geographical bounding box (India)
    if filter_india:
        lat_mask = (cleaned["latitude"] >= GEO_BOUNDS_INDIA["lat_min"]) & (
            cleaned["latitude"] <= GEO_BOUNDS_INDIA["lat_max"]
        )
        lon_mask = (cleaned["longitude"] >= GEO_BOUNDS_INDIA["lon_min"]) & (
            cleaned["longitude"] <= GEO_BOUNDS_INDIA["lon_max"]
        )
        cleaned = cleaned[lat_mask & lon_mask]

    # Filter by temporal bounds (2020-2025)
    if filter_dates:
        start_dt = pd.to_datetime(TEMPORAL_SCOPE["start_date"])
        end_dt = pd.to_datetime(TEMPORAL_SCOPE["end_date"])
        date_mask = (cleaned["acq_date"] >= start_dt) & (cleaned["acq_date"] <= end_dt)
        cleaned = cleaned[date_mask]

    # Sort deterministically by date and time
    sort_cols = ["acq_date"]
    if "acq_time" in cleaned.columns:
        sort_cols.append("acq_time")
    cleaned = cleaned.sort_values(by=sort_cols).reset_index(drop=True)

    retained_count = len(cleaned)
    logger.info(
        f"Cleaned FIRMS hotspots: {retained_count:,} records retained out of {initial_count:,} "
        f"({(retained_count / max(initial_count, 1)) * 100:.1f}%)"
    )

    return cleaned


def load_firms_csv(
    file_path: Union[str, Path],
    filter_india: bool = True,
    filter_dates: bool = True,
) -> pd.DataFrame:
    """Load and validate a single NASA FIRMS VIIRS CSV file.

    Args:
        file_path: Path to the CSV file.
        filter_india: Whether to filter to India geographic bounds.
        filter_dates: Whether to filter to 2020-2025 temporal scope.

    Returns:
        Cleaned pandas DataFrame.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"FIRMS data file not found at: {path}")

    logger.info(f"Loading NASA FIRMS data from: {path.name}")
    raw_df = pd.read_csv(path, low_memory=False)

    is_valid, missing_cols = validate_firms_schema(raw_df)
    if not is_valid:
        raise ValueError(
            f"FIRMS CSV is missing required columns: {missing_cols}. "
            f"Found: {list(raw_df.columns)}"
        )

    return clean_firms_hotspots(raw_df, filter_india=filter_india, filter_dates=filter_dates)


def load_firms_directory(
    dir_path: Optional[Union[str, Path]] = None,
    pattern: str = "*.csv",
    filter_india: bool = True,
    filter_dates: bool = True,
) -> pd.DataFrame:
    """Load and concatenate all matching FIRMS CSV files from a directory.

    Args:
        dir_path: Target directory (defaults to data/raw/firms).
        pattern: Glob pattern to match CSV files.
        filter_india: Whether to filter to India bounds.
        filter_dates: Whether to filter to 2020-2025.

    Returns:
        Combined cleaned pandas DataFrame.
    """
    target_dir = Path(dir_path or FIRMS_RAW_DIR)
    csv_files = sorted(list(target_dir.glob(pattern)))

    if not csv_files:
        raise FileNotFoundError(
            f"No FIRMS CSV files matching '{pattern}' found in {target_dir.resolve()}. "
            "Please download the VIIRS S-NPP archive files and place them in data/raw/firms/."
        )

    logger.info(f"Found {len(csv_files)} FIRMS CSV files in {target_dir.name}")
    frames = []
    for f in csv_files:
        df_part = load_firms_csv(f, filter_india=filter_india, filter_dates=filter_dates)
        if not df_part.empty:
            frames.append(df_part)

    if not frames:
        logger.warning("All matching FIRMS CSV files were empty after filtering.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    logger.info(f"Successfully aggregated {len(combined):,} total unique hotspot records.")
    return combined


def summarize_hotspots(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate diagnostic summary metrics for academic evaluation.

    Args:
        df: Cleaned FIRMS DataFrame.

    Returns:
        Dictionary of summary metrics.
    """
    if df.empty:
        return {"total_records": 0, "status": "empty"}

    summary = {
        "total_records": int(len(df)),
        "date_range": {
            "start": str(df["acq_date"].min().date()),
            "end": str(df["acq_date"].max().date()),
            "total_days_observed": int(df["acq_date"].nunique()),
        },
        "spatial_extent": {
            "lat_min": float(df["latitude"].min()),
            "lat_max": float(df["latitude"].max()),
            "lon_min": float(df["longitude"].min()),
            "lon_max": float(df["longitude"].max()),
        },
        "confidence_distribution": df["confidence"].value_counts().to_dict()
        if "confidence" in df.columns
        else {},
        "frp_mw": {
            "mean": float(df["frp"].mean()) if "frp" in df.columns else None,
            "median": float(df["frp"].median()) if "frp" in df.columns else None,
            "max": float(df["frp"].max()) if "frp" in df.columns else None,
            "p95": float(df["frp"].quantile(0.95)) if "frp" in df.columns else None,
        },
    }
    return summary


if __name__ == "__main__":
    import json

    logger.info("Running FIRMS loader verification against data/raw/firms/...")
    try:
        hotspots_df = load_firms_directory()
        summary_info = summarize_hotspots(hotspots_df)
        print("\n--- NASA FIRMS VIIRS Hotspot Ingestion Summary ---")
        print(json.dumps(summary_info, indent=2))
    except FileNotFoundError as err:
        logger.info(f"Status: {err}")

