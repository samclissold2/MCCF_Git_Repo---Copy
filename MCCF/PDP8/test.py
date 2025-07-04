import pandas as pd
import folium
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, MeasureControl, HeatMap
from pathlib import Path
import webbrowser
import re
import numpy as np
import geopandas as gpd
import logging

from config import (
    INFRASTRUCTURE_DATA,
    PDP8_PROJECT_DATA,
    VNM_GPKG,
    DATA_DIR,
    RESULTS_DIR,
    NEW_TRANSMISSION_DATA
)
from .map_utils import (
    read_new_transmission_data,   # step a)
    split_circuit_km,             # only needed if you want to redo or test it
    annotate_planned_lines,        # step b)
    read_substation_data,
    read_planned_substation_data,   # NEW
    get_power_lines,
    get_voltage_color,              # already in map_utils.py
    load_from_cache,
    save_to_cache,
)

# ------------------------------------------------------------------
# STEP 1 – read the PDP-8 planned-line points (cached for speed)
planned_df = read_new_transmission_data()

# If you want to double-check what split_circuit_km() produced:

# ------------------------------------------------------------------
# OPTIONAL – rerun split_circuit_km() yourself (not normally needed)
planned_df[['circuits', 'route_km', 'circuit_km']] = (planned_df['Number of circuits × kilometres'].apply(split_circuit_km))

# ------------------------------------------------------------------
# STEP 2 – enrich the points with proximity information
planned_df = planned_df.rename(columns={"Longitude": "lon", "Latitude": "lat"})

# -----------------------------------------------------------------------------  
# Voltage helper copied from map_utils.py
# -----------------------------------------------------------------------------
def voltage_category(val):
    """Return a categorical kV label for a numerical/max-voltage value."""
    try:
        # strip anything non-numeric just in case (‘220kV’ → ‘220’)
        if isinstance(val, str):
            val = re.sub(r"[^\d.]", "", val)
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
    except (ValueError, TypeError):
        return "Unknown"

# -----------------------------------------------------------------------------  
# STEP 2 – load supporting datasets ONCE
# -----------------------------------------------------------------------------
substations = read_substation_data()
lines_raw   = get_power_lines()
planned_subs = read_planned_substation_data()        # NEW

# ---- apply voltage_category to all dataframes --------------------------------
lines_raw["voltage_cat"]      = lines_raw["max_voltage"].apply(voltage_category)
substations["voltage_cat"]    = substations["max_voltage"].apply(voltage_category)
planned_subs["voltage_cat"]   = planned_subs["voltage"].apply(voltage_category)

# Only one call to annotate_planned_lines(), using the pre-loaded datasets
planned_gdf = annotate_planned_lines(
    planned_df,
    subs=substations,
    lines=lines_raw,
    substation_buffer=1_000,   # 1 km
    line_buffer=250            # 250 m
)

# -----------------------------------------------------------------------------  
# attach nearest-existing-line voltage to every planned point
# -----------------------------------------------------------------------------
if "nearest_line_kV" not in planned_gdf.columns:
    # make sure we have a numeric voltage column to join on
    lines_raw["numeric_voltage"] = lines_raw["max_voltage"].apply(
        lambda v: np.nan if pd.isna(v) else float(v)
    )

    # Ensure both layers are in the SAME projected CRS (EPSG:3857)
    lines_3857 = lines_raw.to_crs("EPSG:3857")

    nearest = gpd.sjoin_nearest(
        planned_gdf,                                # left – already EPSG:3857
        lines_3857[["numeric_voltage", "geometry"]], # right
        how="left",
        distance_col="line_dist_m"
    )

    planned_gdf["nearest_line_kV"] = nearest["numeric_voltage"]

planned_gdf["voltage_cat"] = planned_gdf["nearest_line_kV"].apply(voltage_category)

# convenience boolean: likely connectable?
planned_gdf["connectable"] = (
    planned_gdf["near_existing_sub"] | planned_gdf["touches_existing_line"]
)

# -----------------------------------------------------------------------------  
# 2. Create a Folium map
# -----------------------------------------------------------------------------
centre = [planned_gdf.lat.mean(), planned_gdf.lon.mean()]        # Vietnam
m = folium.Map(location=centre, zoom_start=6,
               tiles="CartoDB Positron", attr="© OpenStreetMap")

# ---------- layer: existing transmission lines ------------------------------
line_layer = folium.FeatureGroup(name="Existing Lines", show=False)
for _, row in lines_raw.iterrows():
    if row.geometry.is_empty:
        continue
    colour = get_voltage_color(row.max_voltage)
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda _, col=colour: dict(color=col,
                                                  weight=2, opacity=0.8),
        tooltip=folium.Tooltip(f"{row.max_voltage if not pd.isna(row.max_voltage) else 'Unknown'} kV line", sticky=False)  # NEW
    ).add_to(line_layer)
line_layer.add_to(m)

# ---------- layer: existing substations --------------------------------------
sub_layer = folium.FeatureGroup(name="Substations")
for _, row in substations.iterrows():
    colour = get_voltage_color(row.max_voltage)
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=3, color=colour, fill=True, fill_opacity=.9
    ).add_child(folium.Tooltip(
        f"Substation {row.substation_type} ({row.max_voltage} kV)",
        sticky=True
    )).add_to(sub_layer)
sub_layer.add_to(m)

# ---------- layer: planned end-points ----------------------------------------
plan_layer = folium.FeatureGroup(name="Planned Line Points", show=True)
for _, row in planned_gdf.iterrows():
    col = "green" if row.connectable else "red"
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=4, color=col, fill=True, fill_opacity=.9
    ).add_child(folium.Tooltip(
        f"{row.get('project','Planned point')}"
        f"<br>Nearest substation: {row.nearest_substation_dist_m:,.0f} m"
        f"<br>Touches line: {row.touches_existing_line}",
        sticky=True
    )).add_to(plan_layer)
plan_layer.add_to(m)

# ---------- layer: planned substations --------------------------------------
planned_sub_layer = folium.FeatureGroup(name="Planned Substations", show=True)
for _, row in planned_subs.iterrows():
    colour = get_voltage_color(row.voltage)
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=3, color=colour, fill=True, fill_opacity=.9
    ).add_child(folium.Tooltip(
        f"Planned substation ({row.voltage})",
        sticky=True
    )).add_to(planned_sub_layer)
planned_sub_layer.add_to(m)

# ---------- legend & controls ------------------------------------------------
legend_html = """
<div style='position: fixed; bottom:30px; left:30px; z-index:9999;
            background:white; padding:10px; border:2px solid grey;
            border-radius:6px; font-size:13px'>
<b>Legend</b><br>
<span style='color:green'>&#9679;</span> Planned point – likely connectable<br>
<span style='color:red'  >&#9679;</span> Planned point – isolated<br>
Line / substation colours scale with voltage (blue → red)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)

# -----------------------------------------------------------------------------  
# 3. Save & open
# -----------------------------------------------------------------------------
out = Path("planned_connections_map.html").resolve()
m.save(out)
logging.info(f"Map saved to {out}")
import webbrowser; webbrowser.open(out.as_uri())

def get_power_lines(force_recompute: bool = False):
    """
    Reads (and now caches) the power-line layer from the GPKG file.
    """
    if not force_recompute:
        cached = load_from_cache("get_power_lines")
        if cached is not None:
            return cached

    if not VNM_GPKG.exists():
        logging.error(f"Could not find {VNM_GPKG}")
        raise FileNotFoundError(f"Could not find {VNM_GPKG}")

    try:
        logging.info(f"Reading powerline data from {VNM_GPKG}")
        gdf = gpd.read_file(VNM_GPKG, layer="power_line")
        logging.info(f"Read {len(gdf)} powerline records")
        save_to_cache("get_power_lines", gdf)
        return gdf
    except Exception as e:
        logging.error(f"Error reading powerline data: {str(e)}", exc_info=True)
        return gpd.GeoDataFrame()
