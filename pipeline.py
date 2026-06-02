import logging
import os
from typing import Dict, List, Optional, Tuple

import ee
import numpy as np
import pandas as pd

from config import AnalysisConfig
from utils import ensure_dir, setup_logging, get_full_year_date_range, get_city_roi_bounds, generate_summary_report
import gee as gee_mod
from sampling import sample_to_dataframe
from eda import create_eda_plots
from clustering import choose_optimal_clusters, kmeans_and_label, plot_uhi_distribution
from spatial import run_hot_spot_analysis, plot_hot_spots
from map import create_map
from ml import prepare_features, train_random_forest, evaluate_and_plot, shap_analysis

# Forecasting & climate intelligence (additive)
from uhi.forecasting.xgboost_forecast import (
    ForecastConfig,
    build_future_dataframe,
    predict_future,
    train_xgboost_forecaster,
)
from uhi.risk.scoring import HeatRiskBand, categorize_risk, temperature_to_risk_score
from uhi.risk.future_hotspots import detect_future_hotspots
from uhi.visualization.forecast_map_layer import add_forecast_layers
from uhi.dashboard.forecast_dashboard import ForecastDashboardStats, render_dashboard_html




def _compute_current_and_future_climate_intelligence(
    *,
    df_all: pd.DataFrame,
    outputs_dir: str,
    random_state: int,
    future_years: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Train XGBoost forecaster and compute risk/hotspots for future years.

    Returns:
      - df_all: original combined dataframe (unchanged)
      - df_future_pred: dataframe with predictions + risk for all future points
      - forecast_metrics: dict with RMSE/MAE/R2
    """
    if df_all is None or df_all.empty:
        return df_all, pd.DataFrame(), {}

    # Train on combined samples
    forecast_cfg = ForecastConfig(random_state=random_state)
    # Provide non-feature cols as per the forecasting module defaults + what we know exists.
    model, metrics, _X = train_xgboost_forecaster(
        df_all,
        cfg=forecast_cfg,
        non_feature_cols=[
            "LST_Celsius",
            "LST_Cluster",
            "LST_Zone",
            "Gi_ZScore",
            "Gi_PValue",
            "Hot_Spot_Class",
            # latitude/longitude are present but are also geospatial; keep them as features.
        ],
    )

    # Build a future dataframe by reusing static features.
    # Note: this assumes landscape features remain relatively stable for the forecasting horizon.
    # We add time feature (Year) inside build_future_dataframe.
    # Use df_all as the base features; it already includes many static columns.
    df_future_features = build_future_dataframe(df_all, future_years=future_years)

    preds = predict_future(
        model,
        df_future_features,
        non_feature_cols=[
            "LST_Celsius",
            "LST_Cluster",
            "LST_Zone",
            "Gi_ZScore",
            "Gi_PValue",
            "Hot_Spot_Class",
        ],
    )

    df_future_pred = df_future_features.copy()
    df_future_pred["LST_Celsius_pred"] = preds

    # Risk scaling: use training combined data range for stability.
    min_temp = float(df_all["LST_Celsius"].min())
    max_temp = float(df_all["LST_Celsius"].max())

    df_future_pred["risk_score"] = temperature_to_risk_score(
        df_future_pred["LST_Celsius_pred"].values,
        min_temp_c=min_temp,
        max_temp_c=max_temp,
        clip=True,
    )

    # Categorize into required labels
    df_future_pred["risk_band"] = categorize_risk(
        df_future_pred["risk_score"].values,
        band=HeatRiskBand(),
    )

    return df_all, df_future_pred, metrics


def run_analysis(cfg: AnalysisConfig) -> None:

    setup_logging(cfg.log_level, cfg.outputs_dir)
    ensure_dir(cfg.outputs_dir)

    if not gee_mod.initialize_earth_engine():
        return

    roi = None
    if cfg.roi_bounds and len(cfg.roi_bounds) == 4:
        logging.info("Using user-provided ROI bounds: %s", cfg.roi_bounds)
        roi = ee.Geometry.Rectangle(cfg.roi_bounds)
    elif cfg.city_name:
        logging.info("Attempting to find ROI bounds for city: '%s'", cfg.city_name)
        bounds = get_city_roi_bounds(cfg.city_name)
        if bounds:
            cfg.roi_bounds = bounds
            roi = ee.Geometry.Rectangle(bounds)
            logging.info("Found bounds for %s: %s", cfg.city_name, bounds)

    if roi is None:
        logging.error("Could not determine Region of Interest. Provide --roi-bounds or a valid --city.")
        return

    all_year_frames: List[pd.DataFrame] = []
    yearly_reports = {}

    # FIX: The main loop now iterates through years, not seasons.
    for year in cfg.analysis_years:
        period_id = str(year)
        date_range = get_full_year_date_range(year)
        if not date_range:
            continue
        start_date, end_date = date_range

        logging.info("\n" + "#"*80)
        logging.info("PROCESSING PERIOD: Year %s (%s to %s)", period_id, start_date, end_date)
        logging.info("#"*80)

        # --- Data Fetching & Cleaning ---
        lst_img = (
            gee_mod.get_modis_lst(roi, start_date, end_date)
            if cfg.primary_lst_source.upper() == "MODIS"
            else gee_mod.get_landsat_lst(roi, start_date, end_date)
        )
        lst_band = "LST_Celsius" if cfg.primary_lst_source.upper() == "MODIS" else "LST_Landsat_Celsius"
        
        ndvi_img, ndbi_img = gee_mod.get_landsat_indices(roi, start_date, end_date)
        lc_img = gee_mod.get_worldcover_landcover(roi, cfg.worldcover_year)
        topo_img = gee_mod.get_topographic_features(roi)

        if any(x is None for x in [lst_img, ndvi_img, ndbi_img, lc_img, topo_img]):
            logging.warning("Skipping year %s due to missing GEE data.", period_id)
            continue

        combined = lst_img.addBands([ndvi_img, ndbi_img, lc_img, topo_img])
        df = sample_to_dataframe(combined, roi, cfg.num_pixels_for_ml, cfg.common_resolution_m, cfg.random_state)
        if df is None or df.empty:
            logging.warning("Empty sample for %s; skipping year.", period_id)
            continue

        if lst_band in df.columns and lst_band != "LST_Celsius":
            df.rename(columns={lst_band: "LST_Celsius"}, inplace=True)
        df.dropna(inplace=True)
        if "Map" in df.columns:
            lc_map = {10:"Trees", 20:"Shrubland", 30:"Grassland", 40:"Cropland", 50:"Built-up", 60:"Bare_Vegetation", 80:"Water", 90:"Wetland", 95:"Mangroves"}
            df["Land_Cover_Type"] = df["Map"].map(lc_map)
            df.drop(columns=["Map"], inplace=True)
        
        # --- Perform Full Analysis for this Year ---
        logging.info("--- Starting Analysis for Year %s ---", period_id)
        
        # 1. Clustering
        k = choose_optimal_clusters(df, cfg.uhi_cluster_range, cfg.random_state) if cfg.uhi_clusters == "auto" else int(cfg.uhi_clusters)
        df, ordered_zones, zone_colors, _ = kmeans_and_label(df, k, cfg.random_state)
        plot_uhi_distribution(df, ordered_zones, zone_colors, os.path.join(cfg.outputs_dir, f"{period_id}_uhi_zones.png"))

        # 2. UHI Intensity
        urban_mean = df[df["Land_Cover_Type"] == "Built-up"]["LST_Celsius"].mean()
        rural_types = ["Trees", "Shrubland", "Grassland", "Cropland", "Water"]
        rural_mean = df[df["Land_Cover_Type"].isin(rural_types)]["LST_Celsius"].mean()
        uhi_intensity = urban_mean - rural_mean if pd.notna(urban_mean) and pd.notna(rural_mean) else None
        if uhi_intensity: logging.info("[%s] Annual UHI Intensity: %.2f°C", period_id, uhi_intensity)

        # 3. Hot Spot Analysis
        df = run_hot_spot_analysis(df)
        plot_hot_spots(df, os.path.join(cfg.outputs_dir, f"{period_id}_hot_spots.png"))

        yearly_reports[period_id] = {
            "uhi_intensity": uhi_intensity,
            "hot_spots": (df["Hot_Spot_Class"] == "Hot Spot").sum(),
            "cold_spots": (df["Hot_Spot_Class"] == "Cold Spot").sum(),
        }
        all_year_frames.append(df)

    if not all_year_frames:
        logging.error("No data successfully processed for any year. Exiting.")
        return

    df_all = pd.concat(all_year_frames, ignore_index=True)
    logging.info("\nCombined DataFrame for all years has shape: %s", df_all.shape)

    # --- Post-Loop Analysis on COMBINED data ---
    create_eda_plots(df_all, cfg.outputs_dir)

    # ---------------- Forecasting & Climate Intelligence (additive) ----------------
    # If user provided at least 2 analysis years, forecasting gets trained on the time slices.
    # We forecast for the next year beyond max(analysis_years) by default.
    try:
        analysis_years_sorted = sorted([int(y) for y in cfg.analysis_years])
        if analysis_years_sorted:
            future_years = [analysis_years_sorted[-1] + 1]
        else:
            future_years = []

        df_all, df_future_pred, forecast_metrics = _compute_current_and_future_climate_intelligence(
            df_all=df_all,
            outputs_dir=cfg.outputs_dir,
            random_state=cfg.random_state,
            future_years=future_years,
        )

        if not df_future_pred.empty:
            # Derive current hotspots from observed hot/cold classes (already computed per-year)
            # Future hotspots are computed from predicted risk.
            df_future_pred["future_hotspot"] = df_future_pred["risk_score"] >= 71
            df_future_hotspots = detect_future_hotspots(df_future_pred, risk_col="risk_score")

            current_avg_temp = float(df_all["LST_Celsius"].mean())
            predicted_avg_temp = float(df_future_pred["LST_Celsius_pred"].mean())

            current_hotspot_count = int((df_all.get("Hot_Spot_Class") == "Hot Spot").sum()) if "Hot_Spot_Class" in df_all.columns else 0
            future_hotspot_count = int((df_future_pred["future_hotspot"] == True).sum())

            # Highest risk zone: use risk_band frequency ordering.
            if "risk_band" in df_future_pred.columns and (df_future_pred["risk_band"] == "High Risk").any():
                highest_risk_zone = "High Risk"
            else:
                highest_risk_zone = str(df_future_pred["risk_band"].value_counts().idxmax()) if "risk_band" in df_future_pred.columns and not df_future_pred["risk_band"].empty else "Unknown"

            dashboard_stats = ForecastDashboardStats(
                current_avg_temp_c=current_avg_temp,
                predicted_avg_temp_c=predicted_avg_temp,
                current_hotspot_count=current_hotspot_count,
                future_hotspot_count=future_hotspot_count,
                highest_risk_zone=highest_risk_zone,
            )

            # Save forecast evaluation metrics
            if forecast_metrics:
                # Save as json
                import json

                forecast_metrics_path = os.path.join(cfg.outputs_dir, "forecast_model_evaluation.json")
                with open(forecast_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(forecast_metrics, f, indent=2)

            # Patch forecast dashboard overlay + layers into the map after create_map call.
            # We inject these after the base map is created below.
        else:
            df_future_hotspots = pd.DataFrame()
            dashboard_stats = None
            df_future_pred = pd.DataFrame()
    except Exception as exc:
        logging.warning("Forecasting/Climate intelligence step skipped due to error: %s", exc)
        df_future_pred = pd.DataFrame()
        df_future_hotspots = pd.DataFrame()
        dashboard_stats = None



    logging.info("Fetching GEE layers for overall map...")
    primary_df = all_year_frames[0]
    _, p_ord_zones, p_zone_colors, _ = kmeans_and_label(primary_df.copy(), 3, cfg.random_state)
    lst_map = (gee_mod.get_modis_lst(roi, cfg.start_date, cfg.end_date) if cfg.primary_lst_source.upper() == "MODIS" else gee_mod.get_landsat_lst(roi, cfg.start_date, cfg.end_date))
    ndvi_map, ndbi_map = gee_mod.get_landsat_indices(roi, cfg.start_date, cfg.end_date)
    lc_map = gee_mod.get_worldcover_landcover(roi, cfg.worldcover_year)
    topo_map = gee_mod.get_topographic_features(roi)
    map_path = create_map(
        roi,
        lst_map,
        ndvi_map,
        ndbi_map,
        lc_map,
        topo_map,
        primary_df,
        p_ord_zones,
        p_zone_colors,
        cfg.outputs_dir,
        cfg.start_date,
        cfg.end_date,
        cfg.city_name,
    )

    # --- Add forecast visualization layers and dashboard overlay ---
    # We re-open/augment the saved HTML by injecting additional folium layers.
    # For simplicity and robustness, we generate a second map HTML that includes forecast layers.
    # This preserves original map functionality.
    try:
        if not df_future_pred.empty:
            # Reload base map as a new folium object by regenerating from existing inputs.
            # (We avoid parsing HTML; instead we add layers on top of an identical base map.)
            import folium
            from folium import plugins

            try:
                center = roi.centroid().coordinates().getInfo()[::-1]
            except Exception:
                center = [0.0, 0.0]

            m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
            folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
            folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite').add_to(m)

            # Add the same base EE layers using create_map internals would be ideal,
            # but to keep scope contained we only add forecast layers + dashboard.
            # Users still have full UHI layers in the original map HTML at map_path.

            add_forecast_layers(m, df_future_pred, future_hotspot_only=False)
            add_forecast_layers(m, df_future_pred, future_hotspot_only=True)

            if dashboard_stats is not None:
                m.get_root().html.add_child(folium.Element(render_dashboard_html(dashboard_stats)))

            plugins.Fullscreen().add_to(m)
            plugins.MiniMap(toggle_display=True).add_to(m)
            folium.LayerControl().add_to(m)

            forecast_map_path = os.path.join(
                cfg.outputs_dir,
                f"{cfg.city_name.replace(' ', '_').lower()}_forecast_interactive_map.html",
            )
            m.save(forecast_map_path)
            logging.info("Saved forecast interactive map: %s", forecast_map_path)
    except Exception as exc:
        logging.warning("Forecast map layer injection skipped: %s", exc)


    metrics, model = None, None
    try:
        X, y = prepare_features(df_all)
        default_rf = {"n_estimators": cfg.rf_n_estimators, "max_depth": cfg.rf_max_depth, "min_samples_split": cfg.rf_min_samples_split, "min_samples_leaf": cfg.rf_min_samples_leaf}
        model = train_random_forest(X, y, cfg.random_state, cfg.perform_hyperparameter_tuning, cfg.tuning_iterations, default_rf)
        metrics = evaluate_and_plot(model, X, y, cfg.outputs_dir, cfg.random_state)
        logging.info("Model metrics on combined data: %s", metrics)
        shap_analysis(model, X, cfg.outputs_dir, cfg.random_state)
    except Exception as exc:
        logging.error("ML pipeline failed: %s", exc, exc_info=True)

    generate_summary_report(cfg, df_all, metrics, model, yearly_reports)

    logging.info("Analysis complete. Outputs in '%s'", cfg.outputs_dir)
