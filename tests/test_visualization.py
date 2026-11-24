"""Unit tests for Phase 6: Visualization Modules."""

import unittest

try:
    import pandas as pd
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if HAS_PLOTLY:
    from src.visualization.geo_plots import plot_hotspots_mapbox
    from src.visualization.model_plots import plot_feature_importance_bar, plot_precision_recall_curve
    from src.visualization.temporal_plots import plot_monthly_seasonality


class TestVisualization(unittest.TestCase):
    """Test suite for interactive Plotly visualization generation."""

    @unittest.skipUnless(HAS_PLOTLY, "plotly required for mapbox visualization test")
    def test_plot_hotspots_mapbox_empty_and_valid(self):
        """Verify mapbox figure creation on empty and populated data."""
        empty_fig = plot_hotspots_mapbox(pd.DataFrame())
        self.assertIsInstance(empty_fig, go.Figure)

        df = pd.DataFrame(
            {
                "latitude": [22.5, 23.0],
                "longitude": [78.5, 79.0],
                "frp": [12.0, 35.0],
                "confidence": ["nominal", "high"],
                "acq_date": ["2023-03-15", "2023-03-16"],
            }
        )
        fig = plot_hotspots_mapbox(df)
        self.assertIsInstance(fig, go.Figure)

    @unittest.skipUnless(HAS_PLOTLY, "plotly required for seasonality test")
    def test_plot_monthly_seasonality(self):
        """Verify seasonality bar chart generation."""
        df = pd.DataFrame(
            {
                "acq_date": ["2023-03-15", "2023-04-10", "2023-10-25"],
            }
        )
        fig = plot_monthly_seasonality(df)
        self.assertIsInstance(fig, go.Figure)

    @unittest.skipUnless(HAS_PLOTLY, "plotly required for model plots test")
    def test_plot_model_diagnostics(self):
        """Verify PR curve and feature importance bar plots."""
        fig_pr = plot_precision_recall_curve(
            precisions=[1.0, 0.8, 0.6],
            recalls=[0.1, 0.5, 0.9],
            pr_auc=0.75,
        )
        self.assertIsInstance(fig_pr, go.Figure)

        df_imp = pd.DataFrame(
            {
                "feature": ["vpd_kpa", "T2M", "consecutive_dry_days"],
                "importance": [0.45, 0.30, 0.25],
            }
        )
        fig_imp = plot_feature_importance_bar(df_imp)
        self.assertIsInstance(fig_imp, go.Figure)


if __name__ == "__main__":
    unittest.main()
