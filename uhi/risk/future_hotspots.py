"""Future hotspot detection.

A 'future hotspot' is defined as a point predicted to have High risk.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def detect_future_hotspots(df_pred: pd.DataFrame, *, risk_col: str = "risk_score") -> pd.DataFrame:
    """Return subset of points classified as High Risk (>= 71)."""
    if df_pred is None or df_pred.empty:
        return df_pred
    if risk_col not in df_pred.columns:
        return pd.DataFrame()
    return df_pred[df_pred[risk_col] >= 71].copy()

