"""Unit tests for Phase 4: Feature Engineering Module."""

import math
import unittest

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.features.feature_engineering import compute_angstrom_index, compute_vpd

if HAS_PANDAS:
    from src.config import FEATURE_COLUMNS
    from src.features.feature_engineering import (
        compute_consecutive_dry_days_series,
        engineer_environmental_features_pandas,
    )


class TestFeatures(unittest.TestCase):
    """Test suite for environmental indices, lags, and feature transformations."""

    def test_compute_vpd_scalar(self):
        """Verify VPD Tetens calculation in pure Python."""
        # At 100% relative humidity, evaporative deficit must be 0
        vpd_saturated = compute_vpd(temp_celsius=30.0, rh_percent=100.0)
        self.assertAlmostEqual(vpd_saturated, 0.0, places=4)

        # At 30°C and 50% RH, e_s ~ 4.246 kPa, VPD ~ 2.123 kPa
        vpd_dry = compute_vpd(temp_celsius=30.0, rh_percent=50.0)
        self.assertGreater(vpd_dry, 2.0)
        self.assertLess(vpd_dry, 2.3)

    def test_compute_angstrom_index_scalar(self):
        """Verify Ångström fire hazard formula in pure Python."""
        # At T=27°C and RH=40%, index should equal exactly 2.0
        val = compute_angstrom_index(temp_celsius=27.0, rh_percent=40.0)
        self.assertAlmostEqual(val, 2.0, places=2)

    @unittest.skipUnless(HAS_PANDAS, "pandas required for consecutive dry days test")
    def test_compute_consecutive_dry_days(self):
        """Verify running counter of dry days."""
        precip = pd.Series([0.0, 0.0, 0.5, 12.0, 0.0, 0.0])
        dry_counts = compute_consecutive_dry_days_series(precip, dry_threshold=1.0)
        # Sequence: 1, 2, 3, 0 (rain!), 1, 2
        self.assertEqual(list(dry_counts), [1, 2, 3, 0, 1, 2])

    @unittest.skipUnless(HAS_PANDAS, "pandas required for full feature pipeline test")
    def test_engineer_environmental_features_pandas(self):
        """Verify all FEATURE_COLUMNS are generated and free of NaNs."""
        dates = pd.date_range("2023-03-01", periods=10, freq="D")
        abt_dummy = pd.DataFrame(
            {
                "grid_id": ["CELL_22.13_78.38"] * 10,
                "grid_lat": [22.125] * 10,
                "grid_lon": [78.375] * 10,
                "date": dates,
                "T2M": [30.0, 31.0, 32.5, 33.0, 34.0, 35.0, 34.5, 33.0, 32.0, 31.0],
                "T2M_MAX": [36.0, 37.0, 38.5, 39.0, 40.0, 41.0, 40.5, 39.0, 38.0, 37.0],
                "T2M_MIN": [24.0, 25.0, 25.5, 26.0, 27.0, 28.0, 27.5, 26.0, 25.0, 24.0],
                "RH2M": [40.0, 38.0, 35.0, 30.0, 25.0, 22.0, 28.0, 35.0, 40.0, 45.0],
                "PRECTOTCORR": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                "WS10M": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 4.0, 3.5, 3.0, 2.5],
                "PS": [98.5] * 10,
                "has_hotspot": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                "max_frp": [0.0, 0.0, 0.0, 15.2, 28.4, 42.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

        features_df = engineer_environmental_features_pandas(abt_dummy)

        # Verify all feature columns are present
        for col in FEATURE_COLUMNS:
            self.assertIn(col, features_df.columns)
            # Verify no nulls in engineered features
            self.assertEqual(features_df[col].isna().sum(), 0)

        # Verify cyclical bounds [-1, 1]
        self.assertTrue((features_df["doy_sin"] >= -1.0).all())
        self.assertTrue((features_df["doy_sin"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
