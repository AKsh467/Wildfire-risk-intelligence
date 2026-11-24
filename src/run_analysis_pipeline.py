"""End-to-End Analysis, Feature Engineering, Model Training & Diagnostics Pipeline.

Executes the full data lifecycle:
1. Ingests raw FIRMS VIIRS hotspots and NASA POWER meteorology
2. Computes spatial grid discretization (0.25° x 0.25°) and daily spatiotemporal join
3. Engineers physical features (VPD, Ångström index, dry spells, rolling lags)
4. Executes Spatial Block Cross-Validation across distinct geographic regions
5. Fits production models (Random Forest, Gradient Boosting, Logistic Regression)
6. Extracts feature importance rankings and evaluation metrics
7. Exports structured analytics payload for the web UI
"""

import json
from pathlib import Path
import pandas as pd

from src.config import (
    ANALYTICAL_BASE_DIR,
    FEATURE_COLUMNS,
    FEATURES_MATRIX_PATH,
    FIRMS_RAW_DIR,
    HOTSPOTS_GRIDDED_DIR,
    MODELS_DIR,
    POWER_RAW_DIR,
    PROCESSED_DATA_DIR,
    TARGET_BINARY,
    WEATHER_GRIDDED_DIR,
)
from src.features.feature_engineering import (
    engineer_environmental_features_pandas,
    save_processed_feature_matrix,
)
from src.ingestion.firms_loader import load_firms_directory, summarize_hotspots
from src.modeling.evaluate import compute_comprehensive_metrics, extract_feature_importances
from src.modeling.predict import WildfireRiskPredictor
from src.modeling.train import WildfireModelTrainer
from src.processing.hotspot_etl import process_hotspots_pandas
from src.processing.spatial_temporal_join import build_analytical_table_pandas
from src.processing.weather_etl import clean_weather_pandas
from src.utils.logger import get_logger

logger = get_logger("analysis_pipeline")


def run_pipeline():
    """Execute complete data science and ML training pipeline."""
    logger.info("=== STEP 1: Ingesting Raw Satellite & Weather Data ===")
    df_hotspots = load_firms_directory(FIRMS_RAW_DIR, filter_india=True, filter_dates=True)
    firms_summary = summarize_hotspots(df_hotspots)
    logger.info(f"Loaded {len(df_hotspots):,} FIRMS hotspot records.")

    weather_file = Path(POWER_RAW_DIR) / "power_weather_daily_india_2020_2025.csv"
    if not weather_file.is_file():
        raise FileNotFoundError(f"Weather CSV not found: {weather_file}")
    df_weather_raw = pd.read_csv(weather_file)
    logger.info(f"Loaded {len(df_weather_raw):,} raw weather records.")

    logger.info("=== STEP 2: Spatial-Temporal Gridding & Multi-Source Join ===")
    df_hotspots_gridded = process_hotspots_pandas(df_hotspots, grid_degree=0.25)
    df_weather_gridded = clean_weather_pandas(df_weather_raw, grid_degree=0.25)

    df_abt = build_analytical_table_pandas(df_hotspots_gridded, df_weather_gridded)
    logger.info(f"Analytical Base Table Shape: {df_abt.shape}")

    logger.info("=== STEP 3: Environmental Feature Engineering ===")
    df_features = engineer_environmental_features_pandas(df_abt)
    save_processed_feature_matrix(df_features, FEATURES_MATRIX_PATH)

    logger.info("=== STEP 4: Machine Learning Model Training (Spatial Block CV) ===")
    trainer = WildfireModelTrainer(
        model_type="random_forest",
        models_dir=MODELS_DIR,
    )
    cv_metrics = trainer.train_spatial_cv(df_features, feature_cols=FEATURE_COLUMNS, n_splits=5)
    model = trainer.fit_final(df_features, feature_cols=FEATURE_COLUMNS)

    logger.info("=== STEP 5: Model Evaluation & Feature Importance ===")
    X = df_features[FEATURE_COLUMNS]
    y_true = df_features[TARGET_BINARY]
    y_prob = model.predict_proba(X)[:, 1]

    metrics = compute_comprehensive_metrics(y_true, y_prob)
    df_importance = extract_feature_importances(model, FEATURE_COLUMNS)

    logger.info(f"Final Model ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Final Model PR-AUC (Average Precision): {metrics['pr_auc']:.4f}")
    logger.info(f"Optimal F1 Threshold: {metrics['optimal_threshold']:.4f} (Max F1: {metrics['max_f1_at_optimal']:.4f})")

    # Generate predictions across the feature matrix
    df_features["risk_score"] = y_prob.round(4)
    from src.modeling.predict import assign_risk_tier
    df_features["risk_tier"] = [assign_risk_tier(p) for p in y_prob]

    logger.info("=== STEP 6: Exporting Precomputed Analytics Payload ===")
    # Monthly seasonality stats
    df_features["month"] = pd.to_datetime(df_features["date"]).dt.month
    monthly_stats = (
        df_features.groupby("month")
        .agg(
            hotspot_count=("has_hotspot", "sum"),
            avg_risk=("risk_score", "mean"),
            avg_vpd=("vpd_kpa", "mean"),
            avg_temp=("T2M", "mean"),
            avg_rh=("RH2M", "mean"),
        )
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )

    # Risk tier distribution
    tier_counts = df_features["risk_tier"].value_counts().to_dict()

    # Spatial hotspot clusters (top 500 representative locations for map display)
    hotspots_sample = (
        df_features[df_features["has_hotspot"] == 1]
        .sample(n=min(500, (df_features["has_hotspot"] == 1).sum()), random_state=42)[
            ["grid_lat", "grid_lon", "date", "risk_score", "risk_tier", "vpd_kpa", "T2M", "RH2M", "max_frp"]
        ]
        .to_dict(orient="records")
    )

    analytics_summary = {
        "status": "success",
        "total_records": len(df_features),
        "total_hotspots_detected": int(df_features["has_hotspot"].sum()),
        "prevalence_rate": float(round((df_features["has_hotspot"].sum() / len(df_features)) * 100, 2)),
        "model_performance": {
            "model_type": "Random Forest Classifier",
            "roc_auc": round(metrics["roc_auc"], 4) if metrics["roc_auc"] else None,
            "pr_auc": round(metrics["pr_auc"], 4) if metrics["pr_auc"] else None,
            "brier_score": round(metrics["brier_score"], 4),
            "optimal_threshold": round(metrics["optimal_threshold"], 4),
            "max_f1": round(metrics["max_f1_at_optimal"], 4),
            "spatial_cv_results": cv_metrics,
        },
        "feature_importance": df_importance.head(12).to_dict(orient="records"),
        "monthly_seasonality": monthly_stats,
        "risk_tier_distribution": tier_counts,
        "representative_hotspots": hotspots_sample,
    }

    summary_file = Path(PROCESSED_DATA_DIR) / "analytics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(analytics_summary, f, indent=2, default=str)

    logger.info(f"Saved complete analytics summary to: {summary_file.resolve()}")
    logger.info("Pipeline execution completed successfully!")
    return analytics_summary


if __name__ == "__main__":
    run_pipeline()
