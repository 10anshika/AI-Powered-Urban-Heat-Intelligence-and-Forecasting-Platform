"""Heat Risk scoring utilities.

Converts predicted temperature into a 0-100 heat risk score and categorizes it.

Risk bands (inclusive/exclusive):
- Low Risk: 0-30
- Medium Risk: 31-70
- High Risk: 71-100
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeatRiskBand:
    low: int = 0
    medium: int = 31
    high: int = 71


def temperature_to_risk_score(
    temp_c: float | np.ndarray,
    *,
    min_temp_c: float,
    max_temp_c: float,
    clip: bool = True,
) -> float | np.ndarray:
    """Normalize temperature to 0-100.

    If max_temp_c == min_temp_c, returns 0 for all inputs to avoid division by zero.
    """
    if max_temp_c == min_temp_c:
        return np.zeros_like(temp_c, dtype=float) if isinstance(temp_c, np.ndarray) else 0.0

    score = (np.asarray(temp_c, dtype=float) - min_temp_c) / (max_temp_c - min_temp_c)
    score = score * 100.0
    if clip:
        score = np.clip(score, 0.0, 100.0)

    if isinstance(temp_c, np.ndarray):
        return score
    return float(score)


def categorize_risk(score_0_100: float | np.ndarray, *, band: HeatRiskBand = HeatRiskBand()):
    """Categorize risk score into Low/Medium/High."""
    arr = np.asarray(score_0_100, dtype=float)

    labels = np.empty(arr.shape, dtype=object)
    labels[(arr >= band.low) & (arr <= 30)] = "Low Risk"
    labels[(arr >= band.medium) & (arr <= 70)] = "Medium Risk"
    labels[(arr >= band.high) & (arr <= 100)] = "High Risk"

    # Handle NaNs or out-of-band values
    labels[np.isnan(arr)] = "Unknown"

    if isinstance(score_0_100, np.ndarray):
        return labels
    return str(labels.item())

