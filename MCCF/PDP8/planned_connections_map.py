import pandas as pd
import folium
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, MeasureControl, HeatMap
from pathlib import Path
import webbrowser
import re
import numpy as np
import geopandas as gpd
import logging
from shapely.geometry import LineString
import time
import os, importlib, sys

from config import (
    INFRASTRUCTURE_DATA,
    PDP8_PROJECT_DATA,
    VNM_GPKG,
    DATA_DIR,
    RESULTS_DIR,
    NEW_TRANSMISSION_DATA,
)

# ------------------------------------------------------------------
# Robust import of map_utils (same pattern as create_map.py)
# ------------------------------------------------------------------
try:
    from . import map_utils as utils  # package-relative import (preferred)
except ImportError:  # stand-alone execution fallback
    _current_dir = os.path.dirname(os.path.abspath(__file__))        # …/MCCF/PDP8
    _project_root = os.path.dirname(os.path.dirname(_current_dir))   # repo root
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    try:
        utils = importlib.import_module("MCCF.PDP8.map_utils")  # absolute import
    except ModuleNotFoundError:
        utils = importlib.import_module("map_utils")            # same folder

# Pull the individual helpers used below directly into namespace
read_new_transmission_data   = utils.read_new_transmission_data
split_circuit_km             = utils.split_circuit_km
annotate_planned_lines       = utils.annotate_planned_lines
read_substation_data         = utils.read_substation_data
read_planned_substation_data = utils.read_planned_substation_data
get_power_lines              = utils.get_power_lines
get_voltage_color            = utils.get_voltage_color
load_from_cache              = utils.load_from_cache
save_to_cache                = utils.save_to_cache
cache_polylines              = utils.cache_polylines
population_data = utils.read_vnm_pd_2020_1km()


def features_to_gdf(features):
    """Convert cache_polylines() output to a GeoDataFrame."""
    records = []
    for f in features:
        # 1. Build a Shapely LineString  (swap lat/lon → lon,lat for Shapely)
        geom = LineString([(lon, lat) for lat, lon in f["geometry"]["coordinates"]])

        # 2. Voltage label that cache_polylines already stored (e.g. '220kV')
        vlabel = f["properties"]["voltage"]

        # 3. Derive a numeric max-voltage value (needed by existing utils)
        m = re.search(r"\d+", vlabel)
        max_voltage = float(m.group()) * 1_000 if m else None   # 220 → 220 000

        records.append(
            {
                "voltage_cat": vlabel,      # keep the categorical label
                "max_voltage": max_voltage, # numeric for colouring, etc.
                "cluster": f["properties"]["cluster"],
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")
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
        # Allow inputs like '220kV' or numeric kV values
        if isinstance(val, str):
            val = re.sub(r"[^\d.]", "", val)  # keep digits & decimal point
        v = float(val)

        # If the number looks like kV (eg 220) rather than volts (220000),
        # scale it up by 1 000 so that threshold checks work uniformly.
        if v < 1_000:  # assume value is in kV
            v *= 1_000

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
substations = substations[substations['substation_type'].notna() & (substations['substation_type'].astype(str).str.strip() != '')]
features    = cache_polylines(get_power_lines(), cache_file='powerline_polylines.geojson',
                              eps=0.0025, min_samples=3, force_recompute=False)
lines_raw   = features_to_gdf(features)       # <— GeoDataFrame again
planned_subs = read_planned_substation_data()

# ---- apply voltage_category to all dataframes --------------------------------

# Ensure voltage_cat present for safety
if "voltage_cat" not in lines_raw.columns:
    lines_raw["voltage_cat"] = lines_raw["max_voltage"].apply(voltage_category)
if "voltage_cat" not in substations.columns and "max_voltage" in substations.columns:
    substations["voltage_cat"] = substations["max_voltage"].apply(voltage_category)

if "voltage_cat" not in planned_subs.columns and "voltage" in planned_subs.columns:
    planned_subs["voltage_cat"] = planned_subs["voltage"].apply(voltage_category)

# ------------------------------------------------------------------
# Logging configuration (console only)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ------------------------------------------------------------------
# CACHING annotate_planned_lines() output
# ------------------------------------------------------------------
start_anno = time.time()
cached_ann = load_from_cache("annotate_planned_lines_result")

# If cache exists but is missing expected columns, we force a recompute
def _needs_recompute(df):
    req_cols = {"nearest_substation_dist_m", "near_existing_sub", "touches_existing_line"}
    return df is None or not req_cols.issubset(df.columns)

if not _needs_recompute(cached_ann):
    logging.info("Loaded annotated planned-line dataframe from cache")
    planned_gdf = cached_ann
else:
    logging.info("Running annotate_planned_lines … (cache miss / schema update)")
    planned_gdf = annotate_planned_lines(
        planned_df,
        subs=substations,
        lines=lines_raw,
        substation_buffer=1_000,   # 1 km
        line_buffer=250            # 250 m
    )
    save_to_cache("annotate_planned_lines_result", planned_gdf)
    logging.info("Saved annotate_planned_lines result to cache")
logging.info(f"annotate_planned_lines overall time: {time.time()-start_anno:.2f}s")

# -----------------------------------------------------------------------------
# Add proximity to *planned* substations
# -----------------------------------------------------------------------------

planned_subs_gdf = gpd.GeoDataFrame(
    planned_subs,
    geometry=gpd.points_from_xy(planned_subs["lon"], planned_subs["lat"]),
    crs="EPSG:4326",
).to_crs("EPSG:3857")

if "near_planned_sub" not in planned_gdf.columns:
    t0 = time.time()
    nearest_planned = gpd.sjoin_nearest(
        planned_gdf,
        planned_subs_gdf[["geometry"]],
        how="left",
        distance_col="planned_sub_dist_m",
    )

    # collapse duplicates – keep first match per planned point
    planned_dist = (
        nearest_planned.groupby(level=0)["planned_sub_dist_m"].first()
    )

    planned_gdf["planned_sub_dist_m"] = planned_dist
    planned_gdf["near_planned_sub"] = planned_gdf["planned_sub_dist_m"] <= 1_000  # 1 km
    logging.info(
        "Computed distance to planned substations in %.2fs (<=1 km: %d)" % (
            time.time()-t0,
            planned_gdf["near_planned_sub"].sum(),
        )
    )

# -----------------------------------------------------------------------------
# Determine overall connection type
# -----------------------------------------------------------------------------

def classify(row):
    if row.get("near_existing_sub", False):
        return "existing_sub"
    if row.get("touches_existing_line", False):
        return "existing_line"
    if row.get("near_planned_sub", False):
        return "planned_sub"
    return "isolated"

planned_gdf["connection_type"] = planned_gdf.apply(classify, axis=1)

# ensure voltage_cat present for planned_gdf (derived from nearest_line_kV if missing)

if "voltage_cat" not in planned_gdf.columns:
    planned_gdf["voltage_cat"] = planned_gdf["kV"].apply(voltage_category)

# -----------------------------------------------------------------------------  
# 2. Create a Folium map
# -----------------------------------------------------------------------------
centre = [planned_gdf.lat.mean(), planned_gdf.lon.mean()]        # Vietnam
m = folium.Map(location=centre, zoom_start=6,
               tiles="CartoDB Positron", attr="© OpenStreetMap")

# Unified voltage → colour map (same as comprehensive_map)
voltage_colors = {
    "500kV": "red",
    "220kV": "orange",
    "115kV": "purple",
    "110kV": "blue",
    "50kV": "green",
    "33kV": "brown",
    "25kV": "pink",
    "22kV": "gray",
    "<22kV": "black",
    "Unknown": "black",
}

# ---------- layer: existing transmission lines ------------------------------
line_layer = folium.FeatureGroup(name="Existing Lines", show=False)
for _, row in lines_raw.iterrows():
    if row.geometry.is_empty:
        continue
    colour = voltage_colors.get(row.get("voltage_cat", "Unknown"), "black")
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda _, col=colour: dict(color=col,
                                                  weight=2, opacity=0.8),
        tooltip=folium.Tooltip(f"{row['voltage_cat']} line", sticky=False)
    ).add_to(line_layer)
line_layer.add_to(m)

# ---------- layer: existing substations --------------------------------------
sub_layer = folium.FeatureGroup(name="Substations")
for _, row in substations.iterrows():
    colour = voltage_colors.get(row.get("voltage_cat", "Unknown"), "black")
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=4, color=colour, fill=True, fill_opacity=.9
    ).add_child(folium.Tooltip(
        f"Substation {row.substation_type} ({row['voltage_cat']})",
        sticky=True
    )).add_to(sub_layer)
sub_layer.add_to(m)

# ---------- layer: planned end-points ----------------------------------------
plan_layer = folium.FeatureGroup(name="Planned Line Points", show=True)
for _, row in planned_gdf.iterrows():
    colour = voltage_colors.get(row.get("voltage_cat", "Unknown"), "black")
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=5, color=colour, fill=True, fill_opacity=.9
    ).add_child(folium.Tooltip(
        f"{row.get('project','Planned point')}"
        f"<br>Voltage: {row.get('voltage_cat', 'Unknown')}"
        f"<br>Connection type: {row.connection_type}"
        f"<br>Existing sub distance: {row.nearest_substation_dist_m:,.0f} m"
        f"<br>Planned sub distance: {row.planned_sub_dist_m:,.0f} m"
        f"<br>Touches line: {row.touches_existing_line}",
        sticky=True
    )).add_to(plan_layer)
plan_layer.add_to(m)

# ---------- layer: planned substations --------------------------------------
planned_sub_layer = folium.FeatureGroup(name="Planned Substations", show=True)
for _, row in planned_subs.iterrows():
    colour = voltage_colors.get(row.get("voltage_cat", "Unknown"), "black")
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=4, color=colour, fill=True, fill_opacity=.9
    ).add_child(folium.Tooltip(
        f"Planned substation ({row.get('voltage_cat', 'Unknown')})",
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

