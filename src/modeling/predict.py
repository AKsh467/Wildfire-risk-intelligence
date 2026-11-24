"""Inference Pipeline for Relative Wildfire Risk Scoring.

Loads trained model artifacts, verifies input feature schemas, computes relative
wildfire risk probabilities, and maps predictions to academic operational risk tiers:
- Low Risk      (< 0.20)
- Moderate Risk (0.20 - 0.49)
- High Risk     (0.50 - 0.74)
- Extreme Risk  (>= 0.75)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import joblib
    import numpy as np
    import pandas as pd
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.features.feature_engineering import compute_angstrom_index, compute_vpd
from src.utils.logger import get_logger

logger = get_logger("model_predict")


def assign_risk_tier(probability: float) -> str:
    """Map continuous probability score to categorical operational risk tier.

    Args:
        probability: Value in [0.0, 1.0].

    Returns:
        One of 'Low', 'Moderate', 'High', 'Extreme'.
    """
    if probability < 0.20:
        return "Low"
    elif probability < 0.50:
        return "Moderate"
    elif probability < 0.75:
        return "High"
    else:
        return "Extreme"


class WildfireRiskPredictor:
    """Production inference engine for wildfire risk estimation."""

    def __init__(
        self,
        model_type: str = "random_forest",
        models_dir: Optional[Union[str, Path]] = None,
    ):
        """Load serialized model, scaler, and metadata."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn and joblib are required for prediction.")

        self.model_type = model_type.lower()
        self.models_dir = Path(models_dir or MODELS_DIR)

        self.model_path = self.models_dir / f"wildfire_{self.model_type}.joblib"
        self.scaler_path = self.models_dir / f"scaler_{self.model_type}.joblib"
        self.meta_path = self.models_dir / f"metadata_{self.model_type}.json"

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Please run model training first via src.modeling.train."
            )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.is_file() else None

        if self.meta_path.is_file():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata.get("feature_columns", FEATURE_COLUMNS)
        else:
            self.metadata = {}
            self.feature_columns = FEATURE_COLUMNS

        logger.info(f"Loaded '{self.model_type}' model for inference.")

    def predict_risk(self, df_features: "pd.DataFrame") -> "pd.DataFrame":
        """Compute relative wildfire risk scores and assign risk tiers.

        Args:
            df_features: DataFrame containing required feature columns.

        Returns:
            DataFrame with 'risk_score' and 'risk_tier' columns added.
        """
        missing_features = [col for col in self.feature_columns if col not in df_features.columns]
        if missing_features:
            raise ValueError(f"Input DataFrame is missing required features: {missing_features}")

        X = df_features[self.feature_columns]

        if self.scaler is not None:
            X_proc = self.scaler.transform(X)
            probs = self.model.predict_proba(X_proc)[:, 1]
        else:
            probs = self.model.predict_proba(X)[:, 1]

        result_df = df_features.copy()
        result_df["risk_score"] = probs.round(4)
        result_df["risk_tier"] = [assign_risk_tier(p) for p in probs]
        return result_df

    def predict_single_scenario(self, scenario_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict relative risk for an interactive 'what-if' weather scenario.

        Derives required environmental indices (VPD, Ångström index) if not explicitly provided.

        Args:
            scenario_dict: Dictionary containing meteorological readings.

        Returns:
            Dictionary with risk score, tier, and derived indices.
        """
        data = scenario_dict.copy()

        # Compute derived indices if missing
        t = data.get("T2M", 30.0)
        rh = data.get("RH2M", 40.0)
        if "vpd_kpa" not in data:
            data["vpd_kpa"] = float(compute_vpd(t, rh))
        if "angstrom_index" not in data:
            data["angstrom_index"] = float(compute_angstrom_index(t, rh))
        if "dtr" not in data:
            data["dtr"] = float(data.get("T2M_MAX", t + 5) - data.get("T2M_MIN", t - 5))

        # Fill default placeholders for any missing rolling lag windows
        for col in self.feature_columns:
            if col not in data:
                data[col] = 0.0

        df_single = pd.DataFrame([data])
        res_df = self.predict_risk(df_single)

        score = float(res_df.loc[0, "risk_score"])
        tier = str(res_df.loc[0, "risk_tier"])

        return {
            "risk_score": score,
            "risk_tier": tier,
            "vpd_kpa": data["vpd_kpa"],
            "angstrom_index": data["angstrom_index"],
        }
