"""Unit tests for Phase 5: Modeling, Spatial Cross-Validation & Diagnostics."""

import unittest

try:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.modeling.predict import assign_risk_tier

if HAS_SKLEARN:
    from src.config import FEATURE_COLUMNS
    from src.modeling.evaluate import compute_comprehensive_metrics, extract_feature_importances
    from src.modeling.predict import WildfireRiskPredictor
    from src.modeling.train import WildfireModelTrainer, create_spatial_blocks


class TestModeling(unittest.TestCase):
    """Test suite for spatial cross-validation, models, metrics, and inference."""

    def test_assign_risk_tier_pure_python(self):
        """Verify risk tier mapping boundaries in pure Python."""
        self.assertEqual(assign_risk_tier(0.05), "Low")
        self.assertEqual(assign_risk_tier(0.19), "Low")
        self.assertEqual(assign_risk_tier(0.20), "Moderate")
        self.assertEqual(assign_risk_tier(0.49), "Moderate")
        self.assertEqual(assign_risk_tier(0.50), "High")
        self.assertEqual(assign_risk_tier(0.74), "High")
        self.assertEqual(assign_risk_tier(0.75), "Extreme")
        self.assertEqual(assign_risk_tier(0.99), "Extreme")

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn required for spatial blocks test")
    def test_create_spatial_blocks(self):
        """Verify spatial partitioning across latitude and longitude bins."""
        df_coords = pd.DataFrame(
            {
                "grid_lat": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
                "grid_lon": [70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
            }
        )
        blocks = create_spatial_blocks(df_coords, n_splits=3)
        self.assertEqual(len(blocks), 6)
        self.assertTrue(set(blocks.unique()).issubset({0, 1, 2}))

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn required for evaluation metrics test")
    def test_compute_comprehensive_metrics(self):
        """Verify metrics computation on separable probabilities."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_prob = [0.05, 0.12, 0.20, 0.85, 0.90, 0.95]

        metrics = compute_comprehensive_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["f1_at_default"], 1.0)
        self.assertIn("confusion_matrix", metrics)

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn required for end-to-end training test")
    def test_model_training_and_prediction(self, tmp_path=None):
        """Verify model fitting, serialization, and predictor loading."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build miniature dataset
            np.random.seed(42)
            n_rows = 50
            dummy_data = {col: np.random.randn(n_rows) for col in FEATURE_COLUMNS}
            dummy_data["grid_lat"] = np.random.uniform(10.0, 30.0, n_rows)
            dummy_data["grid_lon"] = np.random.uniform(70.0, 90.0, n_rows)
            dummy_data["has_hotspot"] = np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])
            df_train = pd.DataFrame(dummy_data)

            # Train Logistic Regression
            trainer = WildfireModelTrainer(
                model_type="logistic_regression",
                models_dir=tmp_dir,
            )
            cv_results = trainer.train_spatial_cv(df_train, n_splits=3)
            self.assertEqual(len(cv_results), 3)

            # Fit final model and save
            trainer.fit_final(df_train)

            # Load into predictor
            predictor = WildfireRiskPredictor(
                model_type="logistic_regression",
                models_dir=tmp_dir,
            )
            df_pred = predictor.predict_risk(df_train)

            self.assertIn("risk_score", df_pred.columns)
            self.assertIn("risk_tier", df_pred.columns)
            self.assertTrue(df_pred["risk_score"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
