"""Folium layer utilities for forecasted temperatures & risk.

Adds two layers:
- Future predicted temperature/risk (gradient colors)
- Future hotspots (High Risk only)

We keep point-based rendering using sampled points to avoid heavy tile generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd


def _risk_color(score_0_100: float) -> str:
    """Simple gradient mapping for risk score."""
    s = float(score_0_100)
    # Interpolate between blue->cyan->yellow->orange->red
    # Breakpoints: 0, 30, 55, 70, 85, 100
    if s <= 30:
        return "#2E86C1"  # blue
    if s <= 55:
        return "#00CED1"  # cyan
    if s <= 70:
        return "#F1C40F"  # yellow
    if s <= 85:
        return "#E67E22"  # orange
    return "#E74C3C"  # red


def add_forecast_layers(
    m: folium.Map,
    df_with_predictions: pd.DataFrame,
    *,
    future_hotspot_only: bool = False,
    risk_col: str = "risk_score",
    temp_col: str = "LST_Celsius_pred",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> None:
    """Add forecast layer with point markers."""

    if df_with_predictions is None or df_with_predictions.empty:
        return

    if {risk_col, lon_col, lat_col}.difference(df_with_predictions.columns):
        return

    marker_df = df_with_predictions

    layer_name = "Future Temperatures & Heat Risk"
    if future_hotspot_only:
        layer_name = "Future Heat Hotspots (High Risk)"
        marker_df = marker_df[marker_df[risk_col] >= 71]

    grp = folium.FeatureGroup(name=layer_name)

    for _, row in marker_df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        if pd.isna(lat) or pd.isna(lon):
            continue
        score = float(row.get(risk_col, 0.0))
        color = _risk_color(score)
        temp = row.get(temp_col, None)
        tooltip = f"Predicted Temp: {temp:.1f}°C" if temp is not None and not pd.isna(temp) else ""
        tooltip += f"<br>Risk Score: {score:.0f}/100"

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            weight=2,
            fill=True,
            fill_opacity=0.75,
            tooltip=tooltip,
        ).add_to(grp)

    grp.add_to(m)

