"""Interactive Geospatial Plotly Visualizations.

Generates interactive geospatial maps for satellite thermal anomaly observations
and gridded relative wildfire risk predictions across the Indian subcontinent.
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

logger = get_logger("geo_plots")

# Center coordinate for India mapbox view
INDIA_CENTER = {"lat": 22.0, "lon": 79.5}
DEFAULT_ZOOM = 4.0


def _create_scatter_map(df: "pd.DataFrame", **kwargs):
    """Universal scatter map builder compatible with Plotly 5.x and 6.x."""
    if hasattr(px, "scatter_map"):
        # Plotly 6.0+
        if "mapbox_style" in kwargs:
            kwargs["map_style"] = kwargs.pop("mapbox_style")
        return px.scatter_map(df, **kwargs)
    elif hasattr(px, "scatter_mapbox"):
        # Plotly 5.x
        return px.scatter_mapbox(df, **kwargs)
    else:
        for k in ["zoom", "center", "mapbox_style", "map_style"]:
            kwargs.pop(k, None)
        return px.scatter_geo(df, **kwargs)


def plot_hotspots_mapbox(
    df: "pd.DataFrame",
    color_col: str = "frp",
    max_points: int = 5000,
    title: str = "NASA FIRMS VIIRS Active Fire Hotspot Observations (India)",
):
    """Generate an interactive Plotly scatter mapbox of satellite thermal anomalies.

    Args:
        df: DataFrame containing latitude, longitude, acq_date, frp, confidence.
        color_col: Column to use for marker color scale (default: 'frp').
        max_points: Maximum points to render to preserve smooth browser interaction.
        title: Plot title.

    Returns:
        plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for geospatial rendering. Run: pip install plotly")

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No hotspot data available for selected filters.")
        return fig

    # Subsample if dataset is exceptionally large for smooth interactive browser rendering
    render_df = df.copy()
    if len(render_df) > max_points:
        logger.info(f"Subsampling {len(render_df):,} points to {max_points:,} for map responsiveness.")
        render_df = render_df.sample(n=max_points, random_state=42)

    hover_cols = [c for c in ["acq_date", "acq_time", "confidence", "daynight", "bright_ti4"] if c in render_df.columns]

    fig = _create_scatter_map(
        render_df,
        lat="latitude",
        lon="longitude",
        color=color_col if color_col in render_df.columns else None,
        color_continuous_scale="YlOrRd",
        size="frp" if "frp" in render_df.columns else None,
        size_max=12,
        hover_data=hover_cols,
        zoom=DEFAULT_ZOOM,
        center=INDIA_CENTER,
        mapbox_style="carto-positron",
        title=title,
    )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="FRP (MW)"),
    )
    return fig


def plot_risk_grid_mapbox(
    df_risk: "pd.DataFrame",
    lat_col: str = "grid_lat",
    lon_col: str = "grid_lon",
    risk_col: str = "risk_score",
    tier_col: str = "risk_tier",
    title: str = "Relative Wildfire Risk Projection Grid (India)",
):
    """Generate a gridded relative risk map color-coded by operational risk tiers.

    Args:
        df_risk: DataFrame with grid coordinates, risk_score, and risk_tier.
        lat_col: Latitude column.
        lon_col: Longitude column.
        risk_col: Continuous risk score column ([0, 1]).
        tier_col: Risk tier column (Low, Moderate, High, Extreme).
        title: Plot title.

    Returns:
        plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for geospatial rendering.")

    if df_risk.empty:
        fig = go.Figure()
        fig.update_layout(title="No risk projection data available.")
        return fig

    tier_color_map = {
        "Low": "#2ca02c",       # Green
        "Moderate": "#ff7f0e",  # Orange
        "High": "#d62728",      # Red
        "Extreme": "#7b001a",   # Dark Maroon/Crimson
    }

    fig = _create_scatter_map(
        df_risk,
        lat=lat_col,
        lon=lon_col,
        color=tier_col if tier_col in df_risk.columns else risk_col,
        color_discrete_map=tier_color_map if tier_col in df_risk.columns else None,
        color_continuous_scale="YlOrRd" if tier_col not in df_risk.columns else None,
        hover_data=[c for c in [risk_col, "T2M", "RH2M", "vpd_kpa"] if c in df_risk.columns],
        zoom=DEFAULT_ZOOM,
        center=INDIA_CENTER,
        mapbox_style="carto-positron",
        title=title,
    )

    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig
