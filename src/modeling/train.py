"""Machine Learning Training Pipeline with Spatial Block Cross-Validation.

Implements rigorous geospatial model training for relative wildfire risk estimation:
- Spatial Block Cross-Validation (prevents spatial autocorrelation data leakage)
- Class imbalance mitigation via balanced loss weights
- Comparative model training: Baseline Logistic Regression, Random Forest, HistGradientBoosting
- Model artifact serialization with metadata tracking
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    SPATIAL_CV_SPLITS,
    TARGET_BINARY,
)
from src.utils.logger import get_logger

logger = get_logger("model_train")


def create_spatial_blocks(
    df: "pd.DataFrame",
    lat_col: str = "grid_lat",
    lon_col: str = "grid_lon",
    n_splits: int = SPATIAL_CV_SPLITS,
) -> "pd.Series":
    """Partition geographic coordinates into spatial blocks for cross-validation.

    Prevents spatial autocorrelation data leakage (Tobler's First Law of Geography)
    where points from the same regional heatwave or fire cluster leak across folds.

    Args:
        df: Input DataFrame containing spatial coordinates.
        lat_col: Column name for latitude.
        lon_col: Column name for longitude.
        n_splits: Number of spatial blocks/folds.

    Returns:
        Series containing spatial block fold IDs (0 to n_splits - 1).
    """
    # 2D spatial binning over latitude and longitude grids
    n_lat_bins = int(np.ceil(np.sqrt(n_splits)))
    n_lon_bins = int(np.ceil(n_splits / n_lat_bins))

    lat_cuts = pd.qcut(df[lat_col], q=n_lat_bins, labels=False, duplicates="drop")
    lon_cuts = pd.qcut(df[lon_col], q=n_lon_bins, labels=False, duplicates="drop")

    block_id = (lat_cuts * n_lon_bins + lon_cuts) % n_splits
    return block_id.astype(int)


class WildfireModelTrainer:
    """Orchestrates model training, spatial validation, and artifact serialization."""

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = RANDOM_STATE,
        models_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize trainer with selected model family and output directory.

        Supported model_type values:
            - 'logistic_regression': Interpretable linear baseline with L2 penalty
            - 'random_forest': Non-linear ensemble of decision trees
            - 'gradient_boosting': Histogram-based gradient boosted decision trees
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for model training. Run: pip install scikit-learn"
            )

        self.model_type = model_type.lower()
        self.random_state = random_state
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Instantiate classifier with balanced class weighting to address rare events."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=self.random_state,
            )
        else:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                "Choose from: 'logistic_regression', 'random_forest', 'gradient_boosting'."
            )

    def train_spatial_cv(
        self,
        df: "pd.DataFrame",
        feature_cols: Optional[List[str]] = None,
        target_col: str = TARGET_BINARY,
        n_splits: int = SPATIAL_CV_SPLITS,
    ) -> List[Dict[str, Any]]:
        """Execute Spatial Block Cross-Validation across distinct geographic regions.

        Args:
            df: Feature DataFrame.
            feature_cols: Predictor column names.
            target_col: Binary target column.
            n_splits: Number of spatial folds.

        Returns:
            List of validation metrics per spatial fold.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        features = feature_cols or [c for c in FEATURE_COLUMNS if c in df.columns]
        spatial_blocks = create_spatial_blocks(df, n_splits=n_splits)

        fold_results = []
        logger.info(f"Starting {n_splits}-fold Spatial Block CV for '{self.model_type}'...")

        for fold in range(n_splits):
            train_idx = spatial_blocks != fold
            val_idx = spatial_blocks == fold

            X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, target_col]
            X_val, y_val = df.loc[val_idx, features], df.loc[val_idx, target_col]

            # Fit model
            if self.model_type == "logistic_regression":
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train)
                X_val_proc = scaler.transform(X_val)
                clf = self._initialize_model()
                clf.fit(X_train_proc, y_train)
                y_prob = clf.predict_proba(X_val_proc)[:, 1]
            else:
                clf = self._initialize_model()
                clf.fit(X_train, y_train)
                y_prob = clf.predict_proba(X_val)[:, 1]

            roc_auc = float(roc_auc_score(y_val, y_prob)) if len(np.unique(y_val)) > 1 else None
            pr_auc = float(average_precision_score(y_val, y_prob)) if len(np.unique(y_val)) > 1 else None

            fold_metric = {
                "fold": fold,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
            fold_results.append(fold_metric)
            logger.info(
                f"Fold {fold + 1}/{n_splits} - ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}"
            )

        return fold_results

    def fit_final(
        self,
        df: "pd.DataFrame",
        feature_cols: Optional[List[str]] = None,
        target_col: str = TARGET_BINARY,
    ):
        """Fit final model on full historical dataset and save serialized artifacts."""
        features = feature_cols or [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[features]
        y = df[target_col]

        logger.info(f"Fitting final '{self.model_type}' on {len(X):,} records...")

        if self.model_type == "logistic_regression":
            X_proc = self.scaler.fit_transform(X)
            self.model.fit(X_proc, y)
        else:
            self.model.fit(X, y)

        # Save artifacts
        model_filename = f"wildfire_{self.model_type}.joblib"
        scaler_filename = f"scaler_{self.model_type}.joblib"
        meta_filename = f"metadata_{self.model_type}.json"

        joblib.dump(self.model, self.models_dir / model_filename)
        if self.model_type == "logistic_regression":
            joblib.dump(self.scaler, self.models_dir / scaler_filename)

        metadata = {
            "model_type": self.model_type,
            "trained_at": datetime.utcnow().isoformat(),
            "n_samples": int(len(X)),
            "positive_prevalence": float(y.mean()),
            "feature_columns": features,
            "random_state": self.random_state,
        }
        with open(self.models_dir / meta_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model and metadata successfully saved to: {self.models_dir.resolve()}")
        return self.model
