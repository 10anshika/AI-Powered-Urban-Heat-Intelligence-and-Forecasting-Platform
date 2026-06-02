"""HTML dashboard snippet generation for the forecast map.

The project uses a Folium map which can be extended with custom HTML.
We inject a small dashboard overlay with:
- current avg temperature
- predicted avg temperature
- # current hotspots
- # future hotspots
- highest risk zone
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ForecastDashboardStats:
    current_avg_temp_c: float
    predicted_avg_temp_c: float
    current_hotspot_count: int
    future_hotspot_count: int
    highest_risk_zone: str


def render_dashboard_html(stats: ForecastDashboardStats) -> str:
    # Keep styles simple/robust (works inside folium).
    return f"""
    <div id="forecast-dashboard" style="position: fixed; top: 10px; right: 10px; z-index: 9999; width: 320px;">
      <div style="background: rgba(255,255,255,0.95); border: 1px solid #ddd; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); padding: 12px;">
        <div style="font-family: Arial, sans-serif;">
          <div style="font-weight: 700; font-size: 16px; margin-bottom: 10px;">Climate Intelligence Dashboard</div>
          <div style="display: grid; grid-template-columns: 1fr; gap: 8px;">
            <div><b>Current Avg Temp</b>: {stats.current_avg_temp_c:.2f}°C</div>
            <div><b>Predicted Avg Temp</b>: {stats.predicted_avg_temp_c:.2f}°C</div>
            <div><b># Heat Hotspots</b>: {stats.current_hotspot_count}</div>
            <div><b># Future Hotspots</b>: {stats.future_hotspot_count}</div>
            <div><b>Highest Risk Zone</b>: {stats.highest_risk_zone}</div>
          </div>
          <div style="margin-top: 10px; font-size: 12px; color: #666;">
            Forecast uses XGBoost with existing geospatial features + Year.
          </div>
        </div>
      </div>
    </div>
    """

