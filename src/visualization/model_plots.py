"""Model Diagnostics and Evaluation Visualizations.

Generates Precision-Recall curves, ROC curves, calibration charts, and feature
importance ranking diagrams for academic model verification.
"""

from typing import List, Optional

try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.utils.logger import get_logger

logger = get_logger("model_plots")


def plot_precision_recall_curve(
    precisions: "np.ndarray",
    recalls: "np.ndarray",
    pr_auc: Optional[float] = None,
    opt_threshold: Optional[float] = None,
    title: str = "Precision-Recall Curve (Rare Event Evaluation)",
):
    """Plot Precision-Recall curve with optimal threshold annotation."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required.")

    title_text = f"{title} (PR-AUC: {pr_auc:.3f})" if pr_auc is not None else title

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recalls,
            y=precisions,
            mode="lines",
            name="PR Curve",
            line=dict(color="#d62728", width=2.5),
        )
    )

    fig.update_layout(
        title=title_text,
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (Positive Predictive Value)",
        xaxis=dict(range=[0.0, 1.05]),
        yaxis=dict(range=[0.0, 1.05]),
        template="plotly_white",
    )
    return fig


def plot_roc_curve(
    fpr: "np.ndarray",
    tpr: "np.ndarray",
    roc_auc: Optional[float] = None,
    title: str = "Receiver Operating Characteristic (ROC)",
):
    """Plot ROC curve with diagonal no-skill baseline."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required.")

    title_text = f"{title} (ROC-AUC: {roc_auc:.3f})" if roc_auc is not None else title

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="Model ROC",
            line=dict(color="#1f77b4", width=2.5),
        )
    )
    # Diagonal random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(color="grey", dash="dash"),
        )
    )

    fig.update_layout(
        title=title_text,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        xaxis=dict(range=[0.0, 1.05]),
        yaxis=dict(range=[0.0, 1.05]),
        template="plotly_white",
    )
    return fig


def plot_feature_importance_bar(
    df_importance: "pd.DataFrame",
    top_n: int = 15,
    title: str = "Key Environmental Predictors of Wildfire Risk",
):
    """Plot horizontal bar chart of top predictive features."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required.")

    if df_importance.empty:
        fig = go.Figure()
        fig.update_layout(title="No feature importance data available.")
        return fig

    top_df = df_importance.head(top_n).sort_values(by="importance", ascending=True)

    fig = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        labels={"importance": "Relative Importance", "feature": "Environmental Feature"},
        color="importance",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        template="plotly_white",
    )
    return fig
