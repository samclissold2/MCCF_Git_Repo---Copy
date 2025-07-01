import webbrowser
from pathlib import Path

import pandas as pd
import pydeck as pdk

from . import kepler_data_utils as kutils
from .config import KEPLER_COMPREHENSIVE_MAP


_voltage_colors = {
    "500kV": [255, 0, 0],
    "220kV": [255, 165, 0],
    "115kV": [160, 32, 240],
    "110kV": [0, 0, 255],
    "50kV": [0, 128, 0],
    "33kV": [139, 69, 19],
    "25kV": [255, 192, 203],
    "22kV": [128, 128, 128],
    "<22kV": [0, 0, 0],
    "Unknown": [0, 0, 0],
}

_tech_colors = {
    "Solar": [255, 0, 0],
    "Hydro": [0, 51, 102],
    "Onshore": [135, 206, 235],
    "Wind": [135, 206, 235],
    "LNG-Fired Gas": [211, 211, 211],
    "Domestic Gas-Fired": [51, 51, 51],
    "Pumped-Storage": [70, 130, 180],
    "Nuclear": [128, 0, 128],
    "Biomass": [34, 139, 34],
    "Waste-To-Energy": [139, 111, 34],
    "Flexible": [0, 0, 0],
    "LNG Terminal": [211, 211, 211],
    "Coal": [0, 0, 0],
    "Coal Terminal": [139, 69, 19],
}


def _make_path_layer(df: pd.DataFrame) -> pdk.Layer:
    df = df.copy()
    df["color"] = df["voltage"].map(_voltage_colors).fillna([0, 0, 0])
    return pdk.Layer(
        "PathLayer",
        df,
        get_path="path",
        get_color="color",
        width_scale=5,
        width_min_pixels=2,
        pickable=True,
    )

def create_comprehensive_deck(force_recompute: bool = False) -> pdk.Deck:
    """Build a deck.gl map approximating ``create_comprehensive_map``."""
    data = kutils.load_comprehensive_data(force_recompute)

    layers = []

    # Transmission lines
    if not data["power_lines"].empty:
        layers.append(_make_path_layer(data["power_lines"]))

    # Substations
    if not data["substations"].empty:
        sdf = data["substations"].copy()
        sdf["color"] = sdf["voltage_cat"].map(_voltage_colors).fillna([0, 0, 0])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                sdf,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius=3000,
                pickable=True,
            )
        )

    # GEM assets
    if not data["gem_assets"].empty:
        gdf = data["gem_assets"].copy()
        gdf["color"] = gdf["type"].map(_tech_colors).fillna([100, 100, 100])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                gdf,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius=4000,
                pickable=True,
            )
        )

    # Planned power projects
    if not data["projects"].empty:
        pdf = data["projects"].copy()
        pdf["color"] = pdf["tech"].map(_tech_colors).fillna([120, 120, 120])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                pdf,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=4000,
                pickable=True,
            )
        )

    # Planned substations
    if not data["planned_substations"].empty:
        tdf = data["planned_substations"].copy()
        tdf["color"] = tdf["voltage"].apply(lambda x: _voltage_colors.get(str(x), [0, 0, 0]))
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                tdf,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=3000,
                pickable=True,
            )
        )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=16.0, longitude=106.0, zoom=5),
        map_style="mapbox://styles/mapbox/light-v9",
    )
    return deck


def save_comprehensive_deck(deck: pdk.Deck, output_file: Path | None = None) -> None:
    if output_file is None:
        output_file = KEPLER_COMPREHENSIVE_MAP
    html = deck.to_html(as_string=True)
    Path(output_file).write_text(html, encoding="utf-8")
    webbrowser.open(f"file://{Path(output_file).resolve()}")
