"""XGBoost-based heat (LST) forecasting.

This model is trained on the existing sampled point dataset across years.
Each record contains static urban/environmental features (NDVI, NDBI, land cover,
terrain features, etc.) plus a time feature (Year) added from the analysis year.

Outputs:
- predicted future LST (°C)
- risk score & categorical heat risk
- evaluation metrics for the forecasting model

Note: We preserve existing UHI analysis behavior in the main pipeline; forecasting is
an additive module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required for forecasting features") from exc


@dataclass(frozen=True)
class ForecastConfig:
    random_state: int = 42
    test_size: float = 0.2

    # Basic model hyperparameters (kept modest for runtime stability)
    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0


def _add_time_feature(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = int(year)
    return out


def _prepare_supervised_data(
    df_all_years: pd.DataFrame,
    *,
    target_col: str = "LST_Celsius",
    time_col: str = "Year",
    non_feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    non_feature_cols = list(non_feature_cols or [])
    non_feature_cols = [c for c in non_feature_cols if c in df_all_years.columns]

    df = df_all_years.copy()

    # Ensure time feature exists
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}'. Forecasting requires Year.")

    # One-hot encode land cover
    feature_df = df.drop(columns=[c for c in non_feature_cols if c in df.columns])
    if "Land_Cover_Type" in feature_df.columns:
        feature_df = pd.get_dummies(feature_df, columns=["Land_Cover_Type"], prefix="LC", dummy_na=False)

    if target_col not in feature_df.columns:
        raise ValueError(f"Missing target column '{target_col}'.")

    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    # Basic cleanup
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if X.empty:
        raise ValueError("Forecasting dataset became empty after preprocessing.")

    return X, y


def train_xgboost_forecaster(
    df_all_years: pd.DataFrame,
    *,
    cfg: ForecastConfig = ForecastConfig(),
    non_feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[object, Dict[str, float], pd.DataFrame]:
    """Train and evaluate XGBoost on combined multi-year samples."""

    X, y = _prepare_supervised_data(
        df_all_years,
        target_col="LST_Celsius",
        time_col="Year",
        non_feature_cols=non_feature_cols or ["LST_Cluster", "LST_Zone", "Gi_ZScore", "Gi_PValue", "Hot_Spot_Class"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    model = XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return model, metrics, X


def predict_future(
    model: object,
    df_future: pd.DataFrame,
    *,
    non_feature_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Predict future LST for future timestamps using same feature preprocessing."""

    # The forecasting preprocessing includes one-hot encoding and drops target.
    non_feature_cols = non_feature_cols or [
        "LST_Celsius",
        "LST_Cluster",
        "LST_Zone",
        "Gi_ZScore",
        "Gi_PValue",
        "Hot_Spot_Class",
    ]

    df = df_future.copy()
    # Drop any known target if present
    if "LST_Celsius" in df.columns:
        df = df.drop(columns=["LST_Celsius"])

    # Encode land cover if present
    if "Land_Cover_Type" in df.columns:
        df = pd.get_dummies(df, columns=["Land_Cover_Type"], prefix="LC", dummy_na=False)

    # Drop non-feature columns (present ones)
    for c in non_feature_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Align columns to model
    # XGBoost models have feature names stored in booster
    booster = model.get_booster()
    feature_names = booster.feature_names
    if feature_names is None:
        # Fallback: just predict with current columns
        return model.predict(df)

    # Add missing columns with zeros
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return model.predict(df)


def build_future_dataframe(
    df_current_features: pd.DataFrame,
    *,
    future_years: Sequence[int],
) -> pd.DataFrame:
    """Create a future dataframe by reusing static features from current samples.

    Assumption: Environmental/urban features are treated as approximately constant
    across near-term horizons for this initial implementation.
    """
    frames = []
    base = df_current_features.copy()
    base = base.drop(columns=[c for c in ["LST_Celsius"] if c in base.columns])

    for y in future_years:
        tmp = _add_time_feature(base, int(y))
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_forecast_outputs(
    outputs_dir: str,
    *,
    metrics: Dict[str, float],
    metrics_filename: str = "forecast_model_evaluation.json",
) -> str:
    os.makedirs(outputs_dir, exist_ok=True)
    path = os.path.join(outputs_dir, metrics_filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path

