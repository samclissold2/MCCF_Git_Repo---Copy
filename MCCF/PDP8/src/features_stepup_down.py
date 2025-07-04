import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point
from typing import Optional

from MCCF.PDP8.map_utils import load_from_cache, save_to_cache


def _detect_capacity_col(df: gpd.GeoDataFrame) -> str:
    for col in ["mw", "capacity", "capacity_mw", "capacity_mw_", "expected capacity mw", "Capacity (MW)"]:
        if col in df.columns:
            return col
    raise KeyError("No capacity column found in generation dataframe")


def _detect_tech_col(df: gpd.GeoDataFrame) -> Optional[str]:
    for col in ["tech", "type", "fuel", "technology"]:
        if col in df.columns:
            return col
    return None


THERMAL_PATTERN = r"coal|gas|oil|diesel|thermal"


def build_stepup_down_features(
    subs_df: gpd.GeoDataFrame,
    gen_df: gpd.GeoDataFrame,
    pop_raster_path: Path,
    hv_lines: Optional[gpd.GeoDataFrame] = None,
    buffers: Optional[dict[str, int]] = None,
) -> gpd.GeoDataFrame:
    """Compute step-up/step-down feature set for substations."""

    cached = load_from_cache("stepup_down_features")
    if cached is not None:
        return cached

    buffers = buffers or {"sum5": 5000, "sum10": 10000}

    # Work in metric projection
    subs_proj = subs_df.to_crs("EPSG:3857").copy()
    gens_proj = gen_df.to_crs("EPSG:3857").copy()

    cap_col = _detect_capacity_col(gens_proj)
    tech_col = _detect_tech_col(gens_proj)

    # Nearest generation
    nearest = gpd.sjoin_nearest(
        subs_proj, gens_proj[[cap_col, tech_col] if tech_col else [cap_col, "geometry"]],
        how="left", distance_col="dist_m",
    )
    subs_proj["dist_to_nearest_gen"] = nearest["dist_m"] / 1000.0
    subs_proj["nearest_gen_capacity"] = nearest[cap_col]
    if tech_col:
        subs_proj["nearest_gen_thermal"] = nearest[tech_col].str.contains(
            THERMAL_PATTERN, case=False, na=False
        ).astype(int)
    else:
        subs_proj["nearest_gen_thermal"] = 0

    # Sum generation within buffers
    buf5 = subs_proj.geometry.buffer(buffers["sum5"])
    buf10 = subs_proj.geometry.buffer(buffers["sum10"])

    join5 = gpd.sjoin(gens_proj[[cap_col, "geometry"]], gpd.GeoDataFrame(geometry=buf5, index=subs_proj.index), predicate="intersects")
    join10 = gpd.sjoin(gens_proj[[cap_col, tech_col] if tech_col else [cap_col, "geometry"]], gpd.GeoDataFrame(geometry=buf10, index=subs_proj.index), predicate="intersects")

    sum5 = join5.groupby("index_right")[cap_col].sum()
    sum10 = join10.groupby("index_right")[cap_col].sum()

    subs_proj["sum_gen_5km"] = subs_proj.index.map(sum5).fillna(0)
    subs_proj["sum_gen_10km"] = subs_proj.index.map(sum10).fillna(0)

    if tech_col:
        thermal_sum = join10[join10[tech_col].str.contains(THERMAL_PATTERN, case=False, na=False)]
        thermal_cap = thermal_sum.groupby("index_right")[cap_col].sum()
        subs_proj["thermal_frac_10km"] = (
            subs_proj.index.map(thermal_cap).fillna(0) / subs_proj["sum_gen_10km"].replace(0, np.nan)
        ).fillna(0)
    else:
        subs_proj["thermal_frac_10km"] = 0.0

    # Population density
    with rasterio.open(pop_raster_path) as src:
        sub_coords = [Point(xy) for xy in zip(subs_df.geometry.x, subs_df.geometry.y)]
        vals = list(src.sample([(p.x, p.y) for p in gpd.GeoSeries(sub_coords, crs=subs_df.crs).to_crs(src.crs).geometry]))
        subs_proj["pop_density"] = [v[0] if v is not None else np.nan for v in vals]

    # HV corridor
    if hv_lines is not None and not hv_lines.empty:
        lines_proj = hv_lines.to_crs("EPSG:3857")
        if "max_voltage" in lines_proj.columns:
            lines_proj = lines_proj[lines_proj["max_voltage"] >= 400_000]
        subs_proj["is_on_ehv_corridor"] = subs_proj.geometry.intersects(lines_proj.unary_union)
    else:
        subs_proj["is_on_ehv_corridor"] = False

    if "MVA_220kV" in subs_proj.columns:
        mva = pd.to_numeric(subs_proj["MVA_220kV"], errors="coerce")
        subs_proj["MVA_norm"] = (mva - mva.mean()) / mva.std(ddof=0)
    else:
        subs_proj["MVA_norm"] = np.nan

    result = subs_proj.to_crs(subs_df.crs)
    save_to_cache("stepup_down_features", result)
    return result
