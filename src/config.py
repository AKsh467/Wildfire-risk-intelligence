"""Central configuration module for Wildfire Risk Intelligence and Hotspot Analytics.

Defines project directory paths, study area boundaries (India), temporal bounds (2020-2025),
spatial binning parameters, and modeling constants.
"""

import os
from pathlib import Path

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FIRMS_RAW_DIR = RAW_DATA_DIR / "firms"
POWER_RAW_DIR = RAW_DATA_DIR / "power"
INTERIM_DATA_DIR = DATA_DIR / "interim"
HOTSPOTS_GRIDDED_DIR = INTERIM_DATA_DIR / "hotspots_gridded"
WEATHER_GRIDDED_DIR = INTERIM_DATA_DIR / "weather_gridded"
ANALYTICAL_BASE_DIR = INTERIM_DATA_DIR / "analytical_base_table"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# NASA FIRMS API Key (Free registration at https://firms.modaps.eosdis.nasa.gov/api/map_key/)
NASA_FIRMS_MAP_KEY = os.environ.get("NASA_FIRMS_MAP_KEY", "").strip()

# Geographic Scope: India Approximate Bounding Box
# Extended slightly to ensure full coverage of territorial borders and island groups
GEO_BOUNDS_INDIA = {
    "lat_min": 6.0,
    "lat_max": 38.0,
    "lon_min": 68.0,
    "lon_max": 98.0,
}

# Temporal Scope: 2020-2025
TEMPORAL_SCOPE = {
    "start_year": 2020,
    "end_year": 2025,
    "start_date": "2020-01-01",
    "end_date": "2025-12-31",
}

# Satellite Dataset Specifications: VIIRS S-NPP
VIIRS_SPEC = {
    "sensor": "VIIRS",
    "platform": "Suomi NPP",
    "spatial_resolution_m": 375,
    "i_band_channels": ["I4 (3.74 um)", "I5 (11.45 um)"],
    "columns_expected": [
        "latitude",
        "longitude",
        "bright_ti4",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence",
        "version",
        "bright_ti5",
        "frp",
        "daynight",
    ],
}

# Meteorological Variables: NASA POWER API
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_COMMUNITY = "AG"  # Agroclimatology community
METEOROLOGICAL_PARAMS = [
    "T2M",          # Temperature at 2 Meters (C)
    "T2M_MAX",      # Maximum Temperature at 2 Meters (C)
    "T2M_MIN",      # Minimum Temperature at 2 Meters (C)
    "RH2M",         # Relative Humidity at 2 Meters (%)
    "PRECTOTCORR",  # Corrected Total Precipitation (mm/day)
    "WS10M",        # Wind Speed at 10 Meters (m/s)
    "PS",           # Surface Pressure (kPa)
]

# Spatial & Temporal Aggregation Settings
SPATIAL_GRID_DEGREE = 0.25  # Grid bin size (~27.5 km at equator)
TEMPORAL_AGGREGATION = "D"  # Daily frequency

# Feature Engineering Definitions
FEATURE_COLUMNS = [
    # Static & Geospatial
    "grid_lat",
    "grid_lon",
    # Instantaneous Weather
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "WS10M",
    "PS",
    # Derived Atmospheric Drying & Drought Indices
    "dtr",                # Diurnal Temperature Range (T2M_MAX - T2M_MIN)
    "vpd_kpa",            # Vapor Pressure Deficit (Tetens equation)
    "angstrom_index",     # Fire danger index proxy (RH/20 + (27-T)/10)
    # Antecedent Rainfall & Dry Spells
    "rain_sum_3d",        # 3-day cumulative rainfall (mm)
    "rain_sum_7d",        # 7-day cumulative rainfall (mm)
    "rain_sum_14d",       # 14-day cumulative rainfall (mm)
    "consecutive_dry_days", # Consecutive days with rain < 1.0 mm
    # Multi-Day Rolling Environmental Lags
    "t2m_roll_mean_3d",
    "t2m_roll_mean_7d",
    "t2m_max_roll_max_7d",
    "rh2m_roll_min_3d",
    "rh2m_roll_mean_7d",
    "ws10m_roll_mean_3d",
    "ws10m_roll_max_3d",
    # Seasonality Harmonic Encoding
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]

TARGET_BINARY = "has_hotspot"
TARGET_CONTINUOUS = "max_frp"
FEATURES_MATRIX_PATH = PROCESSED_DATA_DIR / "features_matrix.parquet"

# Reproducibility Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
SPATIAL_CV_SPLITS = 5
