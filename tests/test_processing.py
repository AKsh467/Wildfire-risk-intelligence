"""Unit tests for Phase 3: Processing & Spatial-Temporal Join."""

import unittest

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.processing.hotspot_etl import compute_grid_cell

if HAS_PANDAS:
    from src.processing.hotspot_etl import process_hotspots_pandas
    from src.processing.weather_etl import clean_weather_pandas, sanitize_meteorological_values
    from src.processing.spatial_temporal_join import build_analytical_table_pandas


class TestProcessing(unittest.TestCase):
    """Test suite for ETL, gridding, and spatiotemporal table construction."""

    def test_compute_grid_cell(self):
        """Verify grid cell centroid and identifier calculation (pure Python)."""
        grid_id, g_lat, g_lon = compute_grid_cell(lat=22.10, lon=78.30, grid_degree=0.25)
        self.assertTrue(grid_id.startswith("CELL_"))
        self.assertAlmostEqual(g_lat, 22.125, places=2)
        self.assertAlmostEqual(g_lon, 78.375, places=2)

    @unittest.skipUnless(HAS_PANDAS, "pandas/numpy required for tabular processing test")
    def test_process_hotspots_pandas(self):
        """Verify daily gridded aggregation of hotspot observations."""
        df_firms = pd.DataFrame(
            {
                "latitude": [22.1, 22.15, 22.2],
                "longitude": [78.3, 78.32, 78.35],
                "acq_date": ["2023-03-15", "2023-03-15", "2023-03-15"],
                "frp": [10.5, 25.0, 15.2],
                "confidence": ["nominal", "high", "low"],
                "daynight": ["D", "D", "N"],
                "bright_ti4": [330.0, 340.0, 320.0],
            }
        )
        agg_df = process_hotspots_pandas(df_firms, grid_degree=0.25)

        self.assertEqual(len(agg_df), 1)  # All 3 fall in the same 0.25° cell
        self.assertEqual(agg_df.loc[0, "hotspot_count"], 3)
        self.assertAlmostEqual(agg_df.loc[0, "max_frp"], 25.0)
        self.assertEqual(agg_df.loc[0, "high_conf_count"], 1)

    @unittest.skipUnless(HAS_PANDAS, "pandas/numpy required for tabular processing test")
    def test_sanitize_meteorological_values(self):
        """Verify out-of-bounds weather values are masked."""
        df_weather = pd.DataFrame(
            {
                "T2M": [25.0, 150.0, -99.0],
                "RH2M": [50.0, -10.0, 120.0],
            }
        )
        clean_df = sanitize_meteorological_values(df_weather)
        self.assertFalse(np.isnan(clean_df.loc[0, "T2M"]))
        self.assertTrue(np.isnan(clean_df.loc[1, "T2M"]))
        self.assertTrue(np.isnan(clean_df.loc[2, "T2M"]))
        self.assertTrue(np.isnan(clean_df.loc[1, "RH2M"]))

    @unittest.skipUnless(HAS_PANDAS, "pandas/numpy required for tabular processing test")
    def test_build_analytical_table_pandas(self):
        """Verify left join produces 1 for fire days and 0 for non-fire background days."""
        df_weather = pd.DataFrame(
            {
                "grid_id": ["CELL_22.13_78.38", "CELL_22.13_78.38"],
                "grid_lat": [22.125, 22.125],
                "grid_lon": [78.375, 78.375],
                "date": ["2023-03-15", "2023-03-16"],
                "T2M": [32.5, 33.0],
                "RH2M": [35.0, 30.0],
                "PRECTOTCORR": [0.0, 0.0],
                "WS10M": [4.2, 5.0],
            }
        )

        df_hotspots = pd.DataFrame(
            {
                "grid_id": ["CELL_22.13_78.38"],
                "grid_lat": [22.125],
                "grid_lon": [78.375],
                "acq_date": ["2023-03-15"],
                "hotspot_count": [2],
                "max_frp": [18.5],
                "sum_frp": [30.0],
                "high_conf_count": [1],
            }
        )

        abt = build_analytical_table_pandas(df_hotspots, df_weather)
        self.assertEqual(len(abt), 2)

        row_fire = abt[abt["date"] == pd.to_datetime("2023-03-15").date()].iloc[0]
        self.assertEqual(row_fire["has_hotspot"], 1)
        self.assertEqual(row_fire["hotspot_count"], 2)

        row_nofire = abt[abt["date"] == pd.to_datetime("2023-03-16").date()].iloc[0]
        self.assertEqual(row_nofire["has_hotspot"], 0)


if __name__ == "__main__":
    unittest.main()
