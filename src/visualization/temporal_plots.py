"""Temporal and Climatological Analysis Visualizations.

Decomposes seasonal fire cycles across India (pre-monsoon forest fires vs.
post-monsoon agricultural stubble burning), multi-year trends (2020-2025),
and meteorological driver correlations.
"""

from typing import Optional

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.utils.logger import get_logger

logger = get_logger("temporal_plots")

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


def plot_monthly_seasonality(
    df: "pd.DataFrame",
    date_col: str = "acq_date",
    title: str = "Annual Hotspot Seasonality across India (Dual-Peak Regime)",
):
    """Plot monthly thermal anomaly distributions illustrating India's seasonal peaks.

    Highlights:
        - March-May: Pre-monsoon dry season forest fires
        - June-September: Monsoon rain suppression
        - October-November: Post-monsoon agricultural burning
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for temporal plotting.")

    if df.empty or date_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available for seasonality plot.")
        return fig

    df_local = df.copy()
    df_local["month"] = pd.to_datetime(df_local[date_col]).dt.month

    monthly = df_local.groupby("month").size().reset_index(name="detection_count")
    # Reindex across all 12 calendar months
    all_months = pd.DataFrame({"month": list(range(1, 13))})
    monthly = pd.merge(all_months, monthly, on="month", how="left").fillna(0)
    monthly["month_name"] = [MONTH_NAMES[m - 1] for m in monthly["month"]]

    # Annotate seasonal regimes
    regimes = []
    for m in monthly["month"]:
        if m in [3, 4, 5]:
            regimes.append("Pre-monsoon Forest Fires")
        elif m in [10, 11]:
            regimes.append("Post-monsoon Residue Burning")
        elif m in [7, 8]:
            regimes.append("Monsoon Suppression")
        else:
            regimes.append("Intermediate / Baseline")
    monthly["regime"] = regimes

    color_discrete_map = {
        "Pre-monsoon Forest Fires": "#d62728",
        "Post-monsoon Residue Burning": "#ff7f0e",
        "Monsoon Suppression": "#1f77b4",
        "Intermediate / Baseline": "#7f7f7f",
    }

    fig = px.bar(
        monthly,
        x="month_name",
        y="detection_count",
        color="regime",
        color_discrete_map=color_discrete_map,
        labels={"month_name": "Month", "detection_count": "Thermal Anomaly Observations", "regime": "Seasonal Regime"},
        title=title,
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Observed Thermal Detections",
        legend_title="Fire Regime",
        template="plotly_white",
    )
    return fig


def plot_multi_year_comparison(
    df: "pd.DataFrame",
    date_col: str = "acq_date",
    title: str = "Multi-Year Hotspot Trends (India: 2020–2025)",
):
    """Plot annual detection totals across the 2020-2025 study period."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required.")

    if df.empty or date_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No multi-year data available.")
        return fig

    df_local = df.copy()
    df_local["year"] = pd.to_datetime(df_local[date_col]).dt.year

    annual = df_local.groupby("year").size().reset_index(name="total_detections")

    fig = px.line(
        annual,
        x="year",
        y="total_detections",
        markers=True,
        title=title,
        labels={"year": "Year", "total_detections": "Annual Hotspot Observations"},
    )
    fig.update_traces(line_color="#d62728", line_width=3, marker_size=8)
    fig.update_layout(template="plotly_white")
    return fig


def plot_vpd_vs_hotspots(
    df: "pd.DataFrame",
    vpd_col: str = "vpd_kpa",
    count_col: str = "hotspot_count",
    title: str = "Atmospheric Drying (VPD) vs. Fire Anomaly Density",
):
    """Plot relationship between Vapor Pressure Deficit and fire frequency."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required.")

    if df.empty or vpd_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="VPD feature not available in dataset.")
        return fig

    fig = px.scatter(
        df,
        x=vpd_col,
        y=count_col if count_col in df.columns else "frp",
        opacity=0.6,
        trendline="ols",
        title=title,
        labels={vpd_col: "Vapor Pressure Deficit (kPa)", count_col: "Hotspot Count / FRP"},
    )
    fig.update_layout(template="plotly_white")
    return fig
