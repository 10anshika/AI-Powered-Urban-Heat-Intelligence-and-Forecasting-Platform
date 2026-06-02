# TODO - AI-Powered Urban Heat Intelligence and Forecasting Platform

## Plan (approved)
1. Repo understanding & requirements mapping
   - Inspect existing pipeline, ML, and mapping outputs.
   - Identify where to plug in: forecasting model, heat risk scoring, future hotspot detection, and dashboard cards.

2. Code refactor into clean modules (keep existing functionality)
   - Introduce new package structure under `uhi/` (forecasting, risk, visualization, dashboard, reporting).
   - Keep old modules working by wrapping or reusing existing functions.

3. Heat Risk Forecasting (XGBoost)
   - Add XGBoost regressor training to predict future LST.
   - Use existing features + add time features (Year as numeric) for supervised learning.
   - Support predicting for future timestamps.

4. Heat Risk Score (0-100)
   - Convert predicted temperature to a normalized 0-100 risk score.
   - Categorize: Low (0-30), Medium (31-70), High (71-100).

5. Future Hotspot Detection
   - Define “future hotspots” as points with High risk in predicted future timestamps.
   - Display separately from current hotspots.

6. Interactive Forecast Map layer
   - Extend existing folium map to add a new layer for predicted future temperatures/risk.
   - Use color gradients for risk levels.

7. Climate Intelligence Dashboard
   - Add dashboard cards (in generated HTML map) with:
     - Current Average Temperature
     - Predicted Average Temperature
     - Number of Heat Hotspots (current)
     - Number of Future Hotspots
     - Highest Risk Zone

8. Model Evaluation
   - Compute RMSE, MAE, R² for forecasting model.
   - Save evaluation plot and metrics to outputs.

9. Resume-oriented output
   - Update README to reflect the expanded capabilities.
   - Add documentation and comments.

10. Rename project
   - Update `AnalysisConfig.project_name` and README header/title.

## Progress Tracking
- [x] Step 1: Repo understanding
- [x] Step 2: Refactor skeleton modules

- [x] Step 3: Add forecasting XGBoost

- [x] Step 4: Add risk score + categorization
- [x] Step 5: Add future hotspots
- [x] Step 6: Add forecast map layer
- [x] Step 7: Add dashboard cards
- [x] Step 8: Add evaluation metrics
- [x] Step 9: README + docs
- [x] Step 10: Project rename



