"""Unit tests for Phase 2: Ingestion Module (FIRMS & NASA POWER)."""

import json
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

if HAS_PANDAS:
    from src.ingestion.firms_loader import (
        clean_firms_hotspots,
        summarize_hotspots,
        validate_firms_schema,
    )
    from src.ingestion.power_api_client import NasaPowerClient


class TestIngestion(unittest.TestCase):
    """Test suite for NASA FIRMS parser and NASA POWER API client."""

    @unittest.skipUnless(HAS_PANDAS, "pandas/numpy required for schema validation test")
    def test_firms_schema_validation(self):
        """Test schema validation on valid and incomplete DataFrames."""
        sample_df = pd.DataFrame(
            {
                "latitude": [22.5, 30.1],
                "longitude": [78.2, 76.5],
                "acq_date": ["2023-03-15", "2023-04-01"],
                "acq_time": ["0730", "0815"],
                "confidence": ["nominal", "high"],
                "frp": [12.4, 35.8],
            }
        )
        is_valid, missing = validate_firms_schema(sample_df)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)

        # Incomplete schema test
        incomplete_df = sample_df.drop(columns=["frp", "confidence"])
        is_valid_inc, missing_inc = validate_firms_schema(incomplete_df)
        self.assertFalse(is_valid_inc)
        self.assertIn("frp", missing_inc)
        self.assertIn("confidence", missing_inc)

    @unittest.skipUnless(HAS_PANDAS, "pandas/numpy required for hotspot cleaning test")
    def test_clean_firms_hotspots(self):
        """Verify coordinate filtering, confidence normalization, and NaN removal."""
        raw_df = pd.DataFrame(
            {
                "latitude": [22.5, 30.1, 10.5, 55.0, np.nan],  # 55.0 is outside India
                "longitude": [78.2, 76.5, 77.2, 10.0, 78.0],   # 10.0 is outside India
                "bright_ti4": [330.5, 345.2, 312.0, 310.0, 300.0],
                "acq_date": ["2023-03-15", "2023-04-01", "2024-05-10", "2023-03-15", "2023-03-15"],
                "acq_time": ["0730", "0815", "1845", "0700", "0700"],
                "confidence": ["nominal", "high", "l", "nominal", "low"],
                "frp": [12.4, 35.8, 5.2, 8.0, 4.0],
            }
        )
        cleaned = clean_firms_hotspots(raw_df, filter_india=True, filter_dates=True)

        self.assertEqual(len(cleaned), 3)
        self.assertTrue(all(cleaned["latitude"] >= 6.0))
        self.assertTrue(all(cleaned["latitude"] <= 38.0))
        self.assertTrue(all(cleaned["longitude"] >= 68.0))
        self.assertTrue(all(cleaned["longitude"] <= 98.0))

    @unittest.skipUnless(HAS_PANDAS, "pandas/numpy required for NASA POWER parser test")
    def test_parse_power_json(self):
        """Verify NASA POWER JSON parsing and -999.0 fill-value conversion."""
        sample_power_response = {
            "properties": {
                "parameter": {
                    "T2M": {"20230101": 24.5, "20230102": -999.0},
                    "RH2M": {"20230101": 45.2, "20230102": 50.0},
                    "PRECTOTCORR": {"20230101": 0.0, "20230102": 2.5},
                }
            }
        }

        client = NasaPowerClient()
        df = client.parse_power_json(sample_power_response, lat=20.0, lon=78.0)

        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.loc[0, "T2M"], 24.5)
        self.assertTrue(np.isnan(df.loc[1, "T2M"]))
        self.assertAlmostEqual(df.loc[1, "RH2M"], 50.0)


if __name__ == "__main__":
    unittest.main()
