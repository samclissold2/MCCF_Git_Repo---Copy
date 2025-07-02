# Alternative Python Mapping Approaches

This document analyses Folium performance bottlenecks and proposes several alternative stacks that can reproduce existing map features while improving performance.

## Folium Performance Issues

* **Large DOM size** – each CircleMarker or polygon becomes an individual SVG/HTML element. Thousands of elements inflate the page and slow down the browser.
* **Data embedded in HTML** – Folium inlines feature data in the output, leading to multi‑MB files.
* **Synchronous rendering** – all features load at once which causes a slow first paint and sluggish pop‑ups.
* **Limited client‑side interactivity** – Folium relies on Python to generate JavaScript, so dynamic behaviours (e.g. lazy loading) are harder to implement.

## Candidate Stacks

The table below compares four options that can match the required visuals: variable-radius markers, clustering, heat maps, rich tooltips/pop-ups, layer toggles, and custom HTML legends.

| Stack | Pros | Cons | Offline Support |
|------|------|------|---------------|
| **pydeck / kepler.gl** | • WebGL rendering handles thousands of features smoothly.<br>• Easy heat-map, scatter, and path layers.<br>• `to_html()` creates self-contained files or hosted apps.<br>• Supports custom legends and tooltips via HTML templates.| • Marker clustering not built in (needs grid aggregation).<br>• Requires Mapbox token for basemaps (or custom tiles). | Self-contained HTML possible; Mapbox tiles cached or replaced with open tiles for offline. |
| **ipyleaflet** | • Maintains Leaflet plugin ecosystem.<br>• Data streamed via Jupyter widgets reduces initial payload.<br>• Supports clustering and heatmaps.| • Primarily Jupyter based; harder to export static HTML.<br>• Slightly heavier dependency chain.| Offline tiles required; Jupyter needed for full interactivity. |
| **Bokeh + Datashader** | • Designed for large datasets; renders only visible pixels.<br>• Good for heat maps and point aggregation.<br>• Generates standalone HTML.| • Less native support for Leaflet-style controls.<br>• Requires extra code for legends and layer toggles.| Fully offline capable using local tiles or none. |
| **Plotly + Dash + Mapbox** | • Rich interactive components and Dash app framework.<br>• Built-in clustering and heatmap features.<br>• Simple to add password protection in Flask/Dash.| • WebGL layers require Mapbox token.<br>• Larger bundle size than pydeck.| Can export static HTML or run a small web server; offline mode needs local tiles. |

## Recommended Approach

`pydeck` with deck.gl offers the best balance of performance and portability. WebGL keeps interaction smooth even with many features, and `to_html()` outputs a single file for offline use. Optional Flask hosting can be added without affecting existing Folium scripts. The example below also shows how to visualise solar irradiance points and wind power density as heat maps alongside projects and substations.

### Code Sketch

```python
import os
import json
import pandas as pd
import pydeck as pdk
from flask import Flask, render_template_string, request, abort
from MCCF.PDP8 import map_utils as utils

def _hex_to_rgb(hex_color: str) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]

# ---- Data loading via existing utilities ----
power_lines = utils.get_power_lines()
substations = utils.read_substation_data()
projects, name_col = utils.read_and_clean_power_data()
solar_df = utils.read_solar_irradiance_points()
wind_heat = utils.create_wind_power_density_layer()

# ---- Layer definitions ----
projects["radius"] = (projects["mw"].abs() ** 0.5) * 1500
project_layer = pdk.Layer(
    "ScatterplotLayer",
    data=projects,
    get_position="[lon, lat]",
    get_radius="radius",
    get_fill_color="[255,0,0]",
    pickable=True,
)
sub_layer = pdk.Layer(
    "ScatterplotLayer",
    data=substations,
    get_position="[longitude, latitude]",
    get_radius=1000,
    get_fill_color="[0,0,255]",
    pickable=True,
)
solar_layer = pdk.Layer(
    "HeatmapLayer",
    data=solar_df,
    get_position="[lon, lat]",
    get_weight="irradiance",
    radiusPixels=40,
)
wind_df = pd.DataFrame(wind_heat.data, columns=["lat", "lon", "value"])
wind_layer = pdk.Layer(
    "HeatmapLayer",
    data=wind_df,
    get_position="[lon, lat]",
    get_weight="value",
    radiusPixels=40,
)
line_layer = pdk.Layer(
    "GeoJsonLayer",
    data=json.loads(power_lines.to_json()),
    get_line_color=[0, 0, 0],
    get_line_width=2,
)

deck = pdk.Deck(
    layers=[line_layer, sub_layer, project_layer, solar_layer, wind_layer],
    initial_view_state=pdk.ViewState(
        latitude=projects.lat.mean(),
        longitude=projects.lon.mean(),
        zoom=6,
    ),
    map_style="mapbox://styles/mapbox/light-v10",
    tooltip={"text": "{name}"},
)

offline_html = deck.to_html("offline_map.html", open_browser=False)

def create_app(token: str) -> Flask:
    with open("offline_map.html", "r", encoding="utf-8") as f:
        html_body = f.read()

    app = Flask(__name__)

    @app.route("/")
    def index():
        if token and request.args.get("token") != token:
            abort(401)
        return render_template_string(html_body)

    return app

if __name__ == "__main__":
    app_token = os.environ.get("MAP_APP_TOKEN", "")
    app = create_app(app_token)
    app.run(host="0.0.0.0", port=8050)
```

### Running

1. Install dependencies:
   ```bash
   pip install pydeck flask geopandas
   ```
2. Generate the offline HTML:
   ```bash
   python path/to/this_script.py
   # produces offline_map.html (self-contained)
   ```
3. To host online with token protection:
   ```bash
   export MAP_APP_TOKEN=secret123
   python path/to/this_script.py
   # Access http://localhost:8050/?token=secret123
   ```

### Additional Notes
* Set `MAPBOX_API_KEY` in your environment for custom Mapbox styles. Use open tiles or cached tiles for full offline mode.
* Keep tokens and credentials outside version control by using environment variables or a `.env` file loaded at runtime.
