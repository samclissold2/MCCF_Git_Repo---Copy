"""Example pydeck map generation using MCCF's processed data."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import pydeck as pdk
from flask import Flask, render_template_string, request, abort

from PDP8.map_utils import get_power_lines, read_substation_data, read_and_clean_power_data, read_solar_irradiance_points, create_wind_power_density_layer


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

    # ---- Prepare project markers ----
    tech_colors = {
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
    projects["radius"] = (projects["mw"].abs() ** 0.5) * 1500
    projects["color"] = projects["tech"].map(lambda t: _hex_to_rgb(tech_colors.get(t, "#888888")))

    project_layer = pdk.Layer(
        "ScatterplotLayer",
        data=projects,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        name="Power Projects",
    )

    #---- Solar irradiance heat map ----
    solar_layer = None
    if solar_df is not None and not solar_df.empty:
        solar_layer = pdk.Layer(
            "HeatmapLayer",
            data=solar_df,
            get_position="[lon, lat]",
            get_weight="irradiance",
            radiusPixels=40,
            name="Solar Irradiance",
        )

    # ---- Prepare substation markers ----
    substations["radius"] = 1000
    substations["color"] = [[0, 0, 255]] * len(substations)

    sub_layer = pdk.Layer(
        "ScatterplotLayer",
        data=substations,
        get_position="[longitude, latitude]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        name="Substations",
    )

    # ---- Wind power density heat map ----
    wind_layer = None
    if wind_heat is not None:
        wind_df = pd.DataFrame(wind_heat.data, columns=["lat", "lon", "value"])
        wind_layer = pdk.Layer(
            "HeatmapLayer",
            data=wind_df,
            get_position="[lon, lat]",
            get_weight="value",
            radiusPixels=40,
            name="Wind Power Density",
        )

    # ---- Transmission line layer ----
    line_layer = pdk.Layer(
        "GeoJsonLayer",
        data=json.loads(power_lines.to_json()),
        get_line_color=[0, 0, 0],
        get_line_width=2,
        pickable=True,
        name="Transmission Lines",
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

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={"text": "{name}"},
    )
    return deck


def create_offline_html(deck: pdk.Deck, output_html: str) -> str:
    """Write deck to HTML and return the path."""
    deck.to_html(output_html, open_browser=True)
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
