"""NASA POWER Climatology & Meteorology API Client.

Interfaces with the NASA Prediction Of Worldwide Energy Resources (POWER) API
to ingest daily surface meteorological covariates (temperature, humidity, precipitation,
wind speed, and surface pressure) with local filesystem caching, rate-limit courtesy delays,
and sentinel fill-value handling.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    import requests
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from src.config import (
    METEOROLOGICAL_PARAMS,
    NASA_POWER_BASE_URL,
    NASA_POWER_COMMUNITY,
    POWER_RAW_DIR,
    TEMPORAL_SCOPE,
)
from src.utils.logger import get_logger

logger = get_logger("power_api_client")

# NASA POWER uses -999.0 as the missing data sentinel
NASA_POWER_FILL_VALUE = -999.0


class NasaPowerClient:
    """Client for retrieving daily point meteorology from NASA POWER API."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout_seconds: int = 45,
        max_retries: int = 3,
        request_delay: float = 0.5,
    ):
        """Initialize client with cache directory and request settings."""
        self.cache_dir = Path(cache_dir or POWER_RAW_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.request_delay = request_delay

    def _get_cache_path(
        self,
        lat: float,
        lon: float,
        start_str: str,
        end_str: str,
    ) -> Path:
        """Generate a deterministic cache filename for a given location and date range."""
        clean_lat = f"{lat:.3f}".replace(".", "p").replace("-", "m")
        clean_lon = f"{lon:.3f}".replace(".", "p").replace("-", "m")
        filename = f"power_lat{clean_lat}_lon{clean_lon}_{start_str}_{end_str}.json"
        return self.cache_dir / filename

    def fetch_point_raw(
        self,
        lat: float,
        lon: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Fetch raw JSON data for a single geographic point from cache or API.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).
            start_date: Start date as 'YYYYMMDD' or 'YYYY-MM-DD'.
            end_date: End date as 'YYYYMMDD' or 'YYYY-MM-DD'.
            parameters: List of NASA POWER parameter codes.
            use_cache: Whether to read/write to local cache.

        Returns:
            Raw dictionary response.
        """
        # Format dates as YYYYMMDD
        s_date = (start_date or TEMPORAL_SCOPE["start_date"]).replace("-", "")
        e_date = (end_date or TEMPORAL_SCOPE["end_date"]).replace("-", "")
        params_list = parameters or METEOROLOGICAL_PARAMS
        params_str = ",".join(params_list)

        cache_path = self._get_cache_path(lat, lon, s_date, e_date)

        # 1. Check local cache
        if use_cache and cache_path.is_file():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"Loaded NASA POWER data from cache: {cache_path.name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path.name}: {e}. Re-fetching.")

        # 2. Build query parameters
        query_params = {
            "parameters": params_str,
            "community": NASA_POWER_COMMUNITY,
            "longitude": f"{lon:.4f}",
            "latitude": f"{lat:.4f}",
            "start": s_date,
            "end": e_date,
            "format": "JSON",
        }

        # 3. Execute HTTP request with retry logic
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Querying NASA POWER API for ({lat:.3f}, {lon:.3f}) [{s_date}-{e_date}] "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                response = requests.get(
                    NASA_POWER_BASE_URL,
                    params=query_params,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    # Write to cache
                    if use_cache:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                    time.sleep(self.request_delay)
                    return data
                elif response.status_code in [429, 500, 502, 503, 504]:
                    wait_time = 2**attempt
                    logger.warning(
                        f"NASA POWER API returned HTTP {response.status_code}. "
                        f"Backing off for {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise requests.HTTPError(
                        f"NASA POWER API request failed with status {response.status_code}: {response.text}"
                    )
            except Exception as e:
                last_exception = e
                time.sleep(2**attempt)

        raise RuntimeError(
            f"Failed to fetch NASA POWER data after {self.max_retries} attempts: {last_exception}"
        )

    def fetch_point_dataframe(
        self,
        lat: float,
        lon: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily meteorology for a point and parse into a clean pandas DataFrame.

        Args:
            lat: Latitude.
            lon: Longitude.
            start_date: Start date.
            end_date: End date.
            parameters: Meteorological parameters.
            use_cache: Whether to use local cache.

        Returns:
            DataFrame with columns: date, latitude, longitude, and meteorological variables.
        """
        raw_json = self.fetch_point_raw(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            use_cache=use_cache,
        )

        return self.parse_power_json(raw_json, lat=lat, lon=lon)

    @staticmethod
    def parse_power_json(
        data: Dict[str, Any],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> pd.DataFrame:
        """Parse raw NASA POWER JSON response into a tabular DataFrame.

        Replaces fill values (-999.0) with np.nan and casts dates.
        """
        try:
            param_dict = data["properties"]["parameter"]
        except KeyError:
            raise ValueError("Malformed NASA POWER JSON response: missing 'properties.parameter'.")

        # Collect dates from the first parameter series
        if not param_dict:
            return pd.DataFrame()

        first_key = next(iter(param_dict))
        date_keys = sorted(list(param_dict[first_key].keys()))

        records = []
        for d_str in date_keys:
            row: Dict[str, Any] = {"date": pd.to_datetime(d_str, format="%Y%m%d")}
            if lat is not None:
                row["latitude"] = lat
            if lon is not None:
                row["longitude"] = lon

            for p_name, p_series in param_dict.items():
                val = p_series.get(d_str, np.nan)
                # Replace NASA fill values (-999.0)
                if val == NASA_POWER_FILL_VALUE or val <= -900:
                    val = np.nan
                row[p_name] = val
            records.append(row)

        df = pd.DataFrame(records)
        return df

    def fetch_points_grid(
        self,
        coords: List[Tuple[float, float]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily meteorology across multiple coordinates and concatenate.

        Args:
            coords: List of (latitude, longitude) tuples.
            start_date: Start date.
            end_date: End date.
            use_cache: Whether to use cached files.

        Returns:
            Combined pandas DataFrame.
        """
        frames = []
        total = len(coords)
        for idx, (lat, lon) in enumerate(coords, 1):
            logger.info(f"Processing point {idx}/{total}: ({lat:.2f}, {lon:.2f})")
            df_point = self.fetch_point_dataframe(
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )
            frames.append(df_point)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    logger.info("Initializing NASA POWER API Client demo check...")
    client = NasaPowerClient()
    logger.info(f"Target cache directory: {client.cache_dir.resolve()}")
    logger.info("Ready for automated or on-demand meteorological data acquisition.")

