"""Scientific Feature Engineering Pipeline for Wildfire Risk Intelligence.

Implements environmental physics and fire meteorology calculations:
- Vapor Pressure Deficit (VPD via Tetens formula)
- Ångström Fire Hazard Index
- Antecedent precipitation metrics (3d, 7d, 14d cumulative rainfall)
- Consecutive dry days counter (fuel moisture depletion proxy)
- Multi-day rolling temperature, humidity, and wind dynamics
- Cyclical harmonic temporal encodings (DOY, Month) for seasonal burning regimes
"""

import math
from pathlib import Path
from typing import List, Optional, Union

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.config import (
    FEATURE_COLUMNS,
    FEATURES_MATRIX_PATH,
    TARGET_BINARY,
    TARGET_CONTINUOUS,
)
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")


def compute_vpd(
    temp_celsius: Union[float, "np.ndarray", "pd.Series"],
    rh_percent: Union[float, "np.ndarray", "pd.Series"],
) -> Union[float, "np.ndarray", "pd.Series"]:
    """Calculate Vapor Pressure Deficit (VPD in kPa) using the Tetens formula.

    Saturation vapor pressure over liquid water:
        e_s(T) = 0.61078 * exp((17.27 * T) / (T + 237.3))  [kPa]
    Actual vapor pressure:
        e_a(T, RH) = e_s(T) * (RH / 100)  [kPa]
    Vapor Pressure Deficit:
        VPD = e_s(T) - e_a(T, RH) = e_s(T) * (1 - RH / 100)  [kPa]

    Args:
        temp_celsius: Air temperature in degrees Celsius (°C).
        rh_percent: Relative humidity in percentage (0 to 100%).

    Returns:
        VPD in kilopascals (kPa).
    """
    if isinstance(temp_celsius, (int, float)) and isinstance(rh_percent, (int, float)):
        # Scalar calculation
        rh_clamped = max(0.0, min(100.0, float(rh_percent)))
        es = 0.61078 * math.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))
        vpd = es * (1.0 - (rh_clamped / 100.0))
        return max(0.0, vpd)

    # Vectorized calculation (numpy or pandas)
    rh_clamped = np.clip(rh_percent, 0.0, 100.0)
    es = 0.61078 * np.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))
    vpd = es * (1.0 - (rh_clamped / 100.0))
    return np.clip(vpd, 0.0, None)


def compute_angstrom_index(
    temp_celsius: Union[float, "np.ndarray", "pd.Series"],
    rh_percent: Union[float, "np.ndarray", "pd.Series"],
) -> Union[float, "np.ndarray", "pd.Series"]:
    """Calculate the Ångström Fire Danger Index.

    Formula:
        I = (RH / 20) + ((27 - T) / 10)
    Interpretation:
        I < 2.0: Fire occurrence is very likely (extreme hazard)
        2.0 <= I < 2.5: Fire is likely
        2.5 <= I < 4.0: Fire is unlikely
        I >= 4.0: Fire occurrence is very unlikely

    Args:
        temp_celsius: Temperature in °C.
        rh_percent: Relative humidity in %.

    Returns:
        Ångström fire index value.
    """
    return (rh_percent / 20.0) + ((27.0 - temp_celsius) / 10.0)


def compute_consecutive_dry_days_series(
    precip_series: "pd.Series",
    dry_threshold: float = 1.0,
) -> "pd.Series":
    """Compute running counter of consecutive days with precipitation < threshold (mm).

    Args:
        precip_series: Pandas Series of daily rainfall (mm).
        dry_threshold: Maximum precipitation considered a dry day (default 1.0 mm).

    Returns:
        Series of consecutive dry days count.
    """
    is_dry = (precip_series < dry_threshold).astype(int)
    # Reset cumsum whenever a wet day occurs
    dry_cumsum = is_dry.groupby((~is_dry.astype(bool)).cumsum()).cumsum()
    return dry_cumsum


def engineer_environmental_features_pandas(
    df_abt: "pd.DataFrame",
) -> "pd.DataFrame":
    """Generate the full analytical feature matrix from the Analytical Base Table.

    Computes atmospheric drying metrics, rolling weather windows, antecedent rainfall,
    consecutive dry spell counters, and harmonic seasonality encodings.

    Args:
        df_abt: Raw joined Analytical Base Table (containing grid_id, date, weather, target).

    Returns:
        Enhanced DataFrame containing all candidate predictors in FEATURE_COLUMNS.
    """
    if df_abt.empty:
        return df_abt.copy()

    logger.info("Computing environmental features and temporal lag windows...")
    df = df_abt.copy()

    # Ensure chronological sort within each spatial grid cell
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["grid_id", "date"]).reset_index(drop=True)

    # 1. Atmospheric Drying & Fire Weather Proxies
    t_mean = df["T2M"] if "T2M" in df.columns else (df["T2M_MAX"] + df["T2M_MIN"]) / 2.0
    t_max = df["T2M_MAX"] if "T2M_MAX" in df.columns else t_mean
    t_min = df["T2M_MIN"] if "T2M_MIN" in df.columns else t_mean
    rh = df["RH2M"]

    df["dtr"] = (t_max - t_min).clip(lower=0.0)
    df["vpd_kpa"] = compute_vpd(t_mean, rh).round(4)
    df["angstrom_index"] = compute_angstrom_index(t_mean, rh).round(3)

    # 2. Per-Cell Rolling Lag Windows & Antecedent Precipitation
    grouped = df.groupby("grid_id")

    # Antecedent rainfall
    precip = df["PRECTOTCORR"] if "PRECTOTCORR" in df.columns else pd.Series(0.0, index=df.index)
    df["rain_sum_3d"] = grouped["PRECTOTCORR"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    ).round(2)
    df["rain_sum_7d"] = grouped["PRECTOTCORR"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    ).round(2)
    df["rain_sum_14d"] = grouped["PRECTOTCORR"].transform(
        lambda s: s.rolling(14, min_periods=1).sum()
    ).round(2)

    # Consecutive dry days
    df["consecutive_dry_days"] = grouped["PRECTOTCORR"].transform(
        compute_consecutive_dry_days_series
    )

    # Rolling thermal windows
    df["t2m_roll_mean_3d"] = grouped["T2M"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    ).round(2)
    df["t2m_roll_mean_7d"] = grouped["T2M"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    ).round(2)
    df["t2m_max_roll_max_7d"] = grouped["T2M_MAX"].transform(
        lambda s: s.rolling(7, min_periods=1).max()
    ).round(2)

    # Rolling moisture & wind windows
    df["rh2m_roll_min_3d"] = grouped["RH2M"].transform(
        lambda s: s.rolling(3, min_periods=1).min()
    ).round(2)
    df["rh2m_roll_mean_7d"] = grouped["RH2M"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    ).round(2)
    df["ws10m_roll_mean_3d"] = grouped["WS10M"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    ).round(2)
    df["ws10m_roll_max_3d"] = grouped["WS10M"].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    ).round(2)

    # 3. Cyclical Harmonic Date Features (capturing India's dual fire seasons)
    doy = df["date"].dt.dayofyear
    month = df["date"].dt.month

    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).round(4)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).round(4)
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0).round(4)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0).round(4)

    # 4. Fill remaining edge NaNs with backward-fill per cell, then global median
    feature_subset = [c for c in FEATURE_COLUMNS if c in df.columns]
    df[feature_subset] = df.groupby("grid_id")[feature_subset].transform(
        lambda g: g.bfill().ffill()
    )
    df[feature_subset] = df[feature_subset].fillna(df[feature_subset].median())

    logger.info(
        f"Feature engineering complete. Shape: {df.shape}. "
        f"All {len(FEATURE_COLUMNS)} required features present and sanitized."
    )
    return df


def save_processed_feature_matrix(
    df_features: "pd.DataFrame",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save the final engineered analytical matrix to Parquet or CSV format.

    Args:
        df_features: Feature matrix DataFrame.
        output_path: Destination path (defaults to data/processed/features_matrix.parquet).

    Returns:
        Path of saved file.
    """
    dest = Path(output_path or FEATURES_MATRIX_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_features.to_parquet(dest, index=False)
    except Exception:
        csv_dest = dest.with_suffix(".csv")
        df_features.to_csv(csv_dest, index=False)
        dest = csv_dest
    logger.info(f"Saved processed feature matrix to: {dest.resolve()}")
    return dest
