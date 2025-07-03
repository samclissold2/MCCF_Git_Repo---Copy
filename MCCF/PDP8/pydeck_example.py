"""Example pydeck map generation using MCCF's processed data."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import pydeck as pdk
from flask import Flask, render_template_string, request, abort

from PDP8.map_utils import get_power_lines, read_substation_data, read_and_clean_power_data, read_solar_irradiance_points, create_wind_power_density_layer

# NEW ───────────────────────────────────────────────────────────────────────────
# Mapbox key (same env-var Folium expects when using Mapbox tiles)
MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN", "")
if not MAPBOX_TOKEN:
    print(
        "[pydeck]  ❗  No Mapbox token found – "
        "set MAPBOX_API_KEY or MAPBOX_TOKEN for basemap tiles"
    )

pdk.settings.mapbox_api_key = MAPBOX_TOKEN

# Give every layer a stable id so JS controls can find it quickly
LAYER_IDS = {
    "PROJECTS": "power-projects",
    "SUBS": "substations",
    "LINES": "transmission-lines",
    "SOLAR": "solar-irradiance",
    "WIND": "wind-power-density",
}

# Colours used in Folium / create_map.py (trimmed to the ones we plot here)
TECH_COLOURS = {
    "Solar": "#FF0000",
    "Hydro": "#003366",
    "Onshore": "#87CEEB",
    "LNG-Fired Gas": "#808080",
    "Domestic Gas-Fired": "#333333",
    "Pumped-Storage": "#4682B4",
    "Nuclear": "#800080",
    "Biomass": "#228B22",
    "Waste-To-Energy": "#8B6F22",
    "Flexible": "#1A1A1A",
}

# Voltage‐category colours (hex codes, **not** names!)
VOLTAGE_COLOURS = {
    "500kV": "#FF0000",   # red
    "220kV": "#FFA500",   # orange
    "115kV": "#800080",   # purple
    "110kV": "#0000FF",   # blue
    "50kV":  "#008000",   # green
    "33kV":  "#A52A2A",   # brown
    "25kV":  "#FFC0CB",   # pink
    "22kV":  "#808080",   # gray
    "<22kV": "#000000",   # black
    "Unknown": "#000000",
}
# ───────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> list[int]:
    """Convert "#RRGGBB" to ``[R, G, B]`` for pydeck."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


def build_deck(force_recompute: bool = False) -> pdk.Deck:
    """Create a pydeck ``Deck`` from datasets used in ``create_map``."""
    # Load the same datasets that Folium maps use
    power_lines = get_power_lines()
    substations = read_substation_data(force_recompute=force_recompute)
    projects, name_col = read_and_clean_power_data(force_recompute=force_recompute)
    solar_df = read_solar_irradiance_points(force_recompute=force_recompute)
    wind_heat = create_wind_power_density_layer(force_recompute=force_recompute)

    # --- Prepare project markers ---
    projects["name"] = projects[name_col]
    projects["radius"] = (projects["mw"].abs() ** 0.5) * 1500
    projects["colour_rgb"] = projects["tech"].map(
        lambda t: _hex_to_rgb(TECH_COLOURS.get(t, "#888888"))
    )
    projects["tooltip"] = projects.apply(
        lambda r: f"{r['name']} ({r['tech']}) — {r['mw']:.0f} MW", axis=1
    )

    project_layer = pdk.Layer(
        "ScatterplotLayer",
        id=LAYER_IDS["PROJECTS"],
        data=projects,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="colour_rgb",
        pickable=True,
        auto_highlight=True,
    )

    # --- Solar irradiance heat map ---
    solar_layer = None
    if solar_df is not None and not solar_df.empty:
        solar_layer = pdk.Layer(
            "HeatmapLayer",
            id=LAYER_IDS["SOLAR"],
            data=solar_df,
            get_position="[lon, lat]",
            get_weight="irradiance",
            radiusPixels=40,
            pickable=False,
        )

    # --- Prepare substation markers ---
    substations["radius"] = 1000
    substations["colour_rgb"] = [[0, 0, 255]] * len(substations)
    substations["tooltip"] = substations.apply(
        lambda r: f"Substation {r.get('max_voltage', '-')} V", axis=1
    )

    sub_layer = pdk.Layer(
        "ScatterplotLayer",
        id=LAYER_IDS["SUBS"],
        data=substations,
        get_position="[longitude, latitude]",
        get_radius="radius",
        get_fill_color="colour_rgb",
        pickable=True,
        auto_highlight=True,
    )

    # --- Wind power density heat map ---
    wind_layer = None
    if wind_heat is not None:
        wind_df = pd.DataFrame(wind_heat.data, columns=["lat", "lon", "value"])
        wind_layer = pdk.Layer(
            "HeatmapLayer",
            id=LAYER_IDS["WIND"],
            data=wind_df,
            get_position="[lon, lat]",
            get_weight="value",
            radiusPixels=40,
            pickable=False,
        )

    # --- Transmission-line layer --------------------------------------------
    # Categorise numeric voltages → label, then map to colour
    def voltage_category(val: float) -> str:
        try:
            v = float(val)
            if v >= 500_000:
                return "500kV"
            elif v >= 220_000:
                return "220kV"
            elif v >= 115_000:
                return "115kV"
            elif v >= 110_000:
                return "110kV"
            elif v >= 50_000:
                return "50kV"
            elif v >= 33_000:
                return "33kV"
            elif v >= 25_000:
                return "25kV"
            elif v >= 22_000:
                return "22kV"
            else:
                return "<22kV"
        except Exception:
            return "Unknown"

    if "max_voltage" in power_lines.columns:
        power_lines["voltage_cat"] = power_lines["max_voltage"].apply(voltage_category)
    else:
        power_lines["voltage_cat"] = "Unknown"

    power_lines["tooltip"] = power_lines["voltage_cat"] + " line"
    power_lines["colour_rgb"] = power_lines["voltage_cat"].map(
        lambda cat: _hex_to_rgb(VOLTAGE_COLOURS.get(cat, "#000000"))
    )

    line_layer = pdk.Layer(
        "GeoJsonLayer",
        id=LAYER_IDS["LINES"],
        data=json.loads(power_lines.to_json()),
        get_line_color="properties.colour_rgb",
        get_line_width=2,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=projects.lat.mean(),
        longitude=projects.lon.mean(),
        zoom=6,
    )

    layers = [line_layer, sub_layer, project_layer]
    # if solar_layer is not None:
    #     layers.append(solar_layer)
    # if wind_layer is not None:
    #     layers.append(wind_layer)

    tooltip = {
        "html": "{tooltip}",
        "style": {
            "backgroundColor": "rgba(0,0,0,0.8)",
            "color": "white",
            "fontSize": "11px",
        },
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="mapbox",
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip=tooltip,
    )
    return deck


def _legend_html() -> str:
    """Return a collapsible legend based on Folium's colour dictionaries."""
    rows = "\n".join(
        f'<div><span style="display:inline-block;width:12px;height:12px;background:{c};border-radius:50%"></span> {t}</div>'
        for t, c in TECH_COLOURS.items()
    )
    return f"""
    <!-- Legend -->
    <div id="legend" style="
         position:fixed;bottom:30px;left:30px;z-index:9999;
         background:#fff;border:2px solid grey;border-radius:6px;
         font-size:12px;padding:8px;max-width:180px">
       <div onclick="(function(el){{{{el.classList.toggle('open');}}}})(this)"
            style="cursor:pointer;font-weight:bold;margin-bottom:4px">
         Technology Types <span style="float:right">▼</span>
       </div>
       <div id="legend-body">{rows}</div>
    </div>
    <script>
      /* collapse logic */
      (function(){{{{
        const body = document.getElementById('legend-body');
        body.parentElement.addEventListener('click', () => {{{{
          body.style.display = body.style.display === 'none' ? 'block' : 'none';
        }}}});
      }}}})();
    </script>
    """
    
def _layer_control_html() -> str:
    """Simple check-box UI for toggling layer visibility."""
    checks = "\n".join(
        f'<label><input type="checkbox" id="ck-{lid}" checked '
        f'onchange="toggleLayer(\'{lid}\',this.checked)"> {title}</label><br>'
        for lid, title in [
            (LAYER_IDS["PROJECTS"], "Power Projects"),
            (LAYER_IDS["SUBS"], "Substations"),
            (LAYER_IDS["LINES"], "Transmission Lines"),
            (LAYER_IDS["SOLAR"], "Solar Irradiance"),
            (LAYER_IDS["WIND"], "Wind Power Density"),
        ]
    )
    return f"""
    <!-- Layer control -->
    <div id="layers" style="
         position:fixed;top:10px;right:10px;z-index:9999;
         background:#fff;border:2px solid grey;border-radius:6px;
         font-size:12px;padding:8px">
       <b>Layers</b><br>{checks}</div>
    <script>
      function toggleLayer(id, visible){{
          const deck = window.deckgl || window.deck || window._deckgl;
          if(!deck) return;
          const layers = deck.props.layers.map(l=>
              l.id===id ? l.clone({{visible:visible}}) : l);
          deck.setProps({{layers}});
      }}
    </script>
    """
# ───────────────────────────────────────────────────────────────────────────────

def create_offline_html(deck: pdk.Deck, output_html: str, open_browser: bool = True) -> str:
    """Write deck to HTML, inject legend + layer control, return the path."""
    deck.to_html(output_html, open_browser=False)
    # Inject custom HTML right before </body>
    with open(output_html, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", _legend_html() + _layer_control_html() + "</body>")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    if open_browser:
        import webbrowser, pathlib
        webbrowser.open(pathlib.Path(output_html).resolve().as_uri())

    return output_html


def create_flask_app(html_path: str, token: str) -> Flask:
    with open(html_path, "r", encoding="utf-8") as f:
        html_body = f.read()

    app = Flask(__name__)

    @app.route("/")
    def index():
        if token and request.args.get("token") != token:
            abort(401)
        return render_template_string(html_body)

    return app


if __name__ == "__main__":
    deck = build_deck()
    html_file = create_offline_html(deck, "offline_map.html")
    token = os.environ.get("MAP_APP_TOKEN", "")
    app = create_flask_app(html_file, token)
    app.run(host="0.0.0.0", port=8050)
