"""Model Evaluation, Diagnostic Metrics & Feature Importance.

Tailored for severe class imbalance and spatial anomaly modeling:
- Precision-Recall Area Under the Curve (PR-AUC / Average Precision)
- ROC-AUC and Brier Score
- Optimal decision threshold discovery maximizing F1-score
- Permutation and Gini impurity feature importance rankings
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.utils.logger import get_logger

logger = get_logger("model_evaluate")


def find_optimal_threshold(
    y_true: "np.ndarray",
    y_prob: "np.ndarray",
) -> Tuple[float, float]:
    """Find classification probability threshold that maximizes F1-score.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted positive class probabilities.

    Returns:
        Tuple of (optimal_threshold: float, best_f1: float).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = np.argmax(f1_scores)
    # thresholds array has length len(precisions) - 1
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    return best_thresh, best_f1


def compute_comprehensive_metrics(
    y_true: Union[List, "np.ndarray", "pd.Series"],
    y_prob: Union[List, "np.ndarray", "pd.Series"],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate diagnostic evaluation metrics for imbalanced hotspot predictions.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold for discrete classification.

    Returns:
        Dictionary of performance metrics.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for evaluation metrics.")

    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    y_pred = (y_p >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else None
    pr_auc = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else None
    brier = float(brier_score_loss(y_t, y_p))

    opt_thresh, opt_f1 = find_optimal_threshold(y_t, y_p)

    cm = confusion_matrix(y_t, y_pred).tolist()

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "default_threshold": threshold,
        "precision_at_default": float(precision_score(y_t, y_pred, zero_division=0)),
        "recall_at_default": float(recall_score(y_t, y_pred, zero_division=0)),
        "f1_at_default": float(f1_score(y_t, y_pred, zero_division=0)),
        "optimal_threshold": opt_thresh,
        "max_f1_at_optimal": opt_f1,
        "confusion_matrix": cm,
    }
    return metrics


def extract_feature_importances(
    model: Any,
    feature_names: List[str],
    X_val: Optional["pd.DataFrame"] = None,
    y_val: Optional["pd.Series"] = None,
    n_repeats: int = 5,
) -> "pd.DataFrame":
    """Extract or calculate feature importance rankings.

    Uses MDI (Mean Decrease in Impurity) for tree models and Permutation Importance
    if validation data is provided.

    Args:
        model: Trained scikit-learn estimator.
        feature_names: Names of input features.
        X_val: Optional validation features for permutation importance.
        y_val: Optional validation targets.
        n_repeats: Number of permutations for permutation importance.

    Returns:
        Sorted DataFrame with columns: feature, importance, method.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for feature importance extraction.")

    # 1. Permutation Importance (most unbiased across different feature types)
    if X_val is not None and y_val is not None:
        logger.info(f"Computing permutation importance over {len(X_val)} validation rows...")
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=42,
            scoring="average_precision",
        )
        df_imp = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": result.importances_mean,
                "std": result.importances_std,
                "method": "Permutation Importance (PR-AUC)",
            }
        )
    # 2. Tree MDI Importance
    elif hasattr(model, "feature_importances_"):
        df_imp = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
                "method": "Mean Decrease in Impurity (Gini)",
            }
        )
    # 3. Linear Coefficients
    elif hasattr(model, "coef_"):
        df_imp = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.abs(model.coef_[0]),
                "method": "Absolute Logistic Regression Weight",
            }
        )
    else:
        df_imp = pd.DataFrame({"feature": feature_names, "importance": 0.0, "method": "None"})

    df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1
    return df_imp
