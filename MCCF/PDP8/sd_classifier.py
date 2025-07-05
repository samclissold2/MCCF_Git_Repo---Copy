# ─── sd_classifier.py ────────────────────────────────────────────────────────
import geopandas as gpd, numpy as np, pandas as pd, logging
from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path
from shapely.ops import nearest_points
from scipy.spatial import cKDTree

from map_utils import (
    read_wind_power_data, read_solar_power_data, read_oil_gas_plant_data,
    read_coal_plant_data, read_hydropower_data, read_vnm_pd_2020_1km,
    get_power_lines, read_substation_data, save_to_cache, load_from_cache,
    DATA_DIR, RESULTS_DIR
)

RESULTS_DIR.mkdir(exist_ok=True)

def _assemble_generation_sites() -> gpd.GeoDataFrame:
    dfs = [
        read_wind_power_data(), read_solar_power_data(),
        read_oil_gas_plant_data(), read_coal_plant_data(),
        read_hydropower_data()
    ]
    gdf = pd.concat(dfs, ignore_index=True, sort=False)
    gdf = gdf.dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        gdf,
        geometry=gpd.points_from_xy(gdf["longitude"], gdf["latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    # boolean flag for baseload vs variable
    gdf["is_variable"] = gdf["type"].str.contains("Wind|Solar", case=False, na=False)
    return gdf

def build_sd_features(subs: gpd.GeoDataFrame, force_recompute=False) -> gpd.GeoDataFrame:
    """
    Returns subs with engineered columns used for SD classification.
    Caches by 'sd_features'.
    """
    cached = load_from_cache("sd_features")
    if cached is not None and not force_recompute:
        return cached

    subs = subs.to_crs("EPSG:3857").copy()
    gens = _assemble_generation_sites()
    pop  = read_vnm_pd_2020_1km()                          # DataFrame lat/lon/density
    pop_gdf = gpd.GeoDataFrame(
        pop,
        geometry=gpd.points_from_xy(pop["longitude"], pop["latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    # ---- nearest-generation features ---------------------------------------
    gen_sindex = gens.sindex
    dist_list, cap_list, var_list = [], [], []
    for geom in subs.geometry:
        # index of closest generation point
        idx = list(gen_sindex.nearest(geom.bounds, 1))[0]
        dist  = geom.distance(gens.geometry.iloc[idx]) / 1_000  # km
        dist_list.append(dist)
        cap_list.append(gens.Capacity_MW.iloc[idx])
        var_list.append(int(gens.is_variable.iloc[idx]))

    subs["dist_gen_km"]      = dist_list
    subs["nearest_gen_mw"]   = cap_list
    subs["nearest_gen_var"]  = var_list

    # ---- sum of generation within 5 km radius ------------------------------
    buffer_5 = gpd.GeoSeries(subs.geometry).buffer(5_000)
    tree     = cKDTree(np.column_stack([gens.geometry.x, gens.geometry.y]))
    sums5, varfrac = [], []
    for geom in subs.geometry:
        idxs = list(tree.query_ball_point([geom.x, geom.y], r=5_000))
        if idxs:
            sel = gens.iloc[idxs]
            sums5.append(sel.Capacity_MW.sum())
            varfrac.append(sel.is_variable.mean())
        else:
            sums5.append(0)
            varfrac.append(0)
    subs["sum_gen_5km"]       = sums5
    subs["var_frac_5km"]      = varfrac

    # ---- population density at substation pixel ---------------------------
    pop_sindex = pop_gdf.sindex
    pop_vals   = []
    for geom in subs.geometry:
        idxs = list(pop_sindex.nearest(geom.bounds, 1))
        pop_vals.append(pop_gdf.population_density.iloc[idxs[0]])
    subs["pop_density"] = pop_vals

    # ---- on 500-kV corridor flag ------------------------------------------
    lines = get_power_lines()
    ehv   = lines[lines.max_voltage >= 400_000].to_crs("EPSG:3857").buffer(100)
    union = ehv.unary_union
    subs["on_500kV_corridor"] = subs.geometry.intersects(union).astype(int)

    # ---- normalised MVA (if present) --------------------------------------
    if "MVA_220kV" in subs.columns:
        subs["MVA_norm"] = (subs.MVA_220kV - subs.MVA_220kV.mean()) / subs.MVA_220kV.std(ddof=0)
    else:
        subs["MVA_norm"] = 0

    save_to_cache("sd_features", subs)
    return subs

def classify_step(sd_features: gpd.GeoDataFrame, *, retrain=False) -> gpd.GeoDataFrame:
    """
    Adds ‘sd_class’ and ‘sd_conf’ columns to sd_features and returns GeoDataFrame.
    A tiny GBDT is re-trained only if no model exists or retrain==True.
    """
    model_path = RESULTS_DIR / "sd_gbm.joblib"
    feature_cols = [
        "dist_gen_km", "nearest_gen_mw", "nearest_gen_var",
        "sum_gen_5km", "var_frac_5km", "pop_density",
        "on_500kV_corridor", "MVA_norm"
    ]

    try:
        from joblib import load, dump
        if not model_path.exists() or retrain:
            raise FileNotFoundError
        clf = load(model_path)
    except FileNotFoundError:
        # ---- minimal hand-label seed --------------------------------------
        seed = sd_features.sample(40, random_state=1).copy()
        # Manually tag a few for first train → user can refine later
        seed["Label"] = np.where(seed.dist_gen_km < 3, "Step-Up",
                           np.where(seed.pop_density > 400, "Step-Down",
                                    "Interconnection"))
        X = seed[feature_cols]
        y = seed.Label
        clf = GradientBoostingClassifier().fit(X, y)
        from joblib import dump
        dump(clf, model_path)

    prob  = clf.predict_proba(sd_features[feature_cols])
    sd_features["sd_class"] = clf.classes_[prob.argmax(axis=1)]
    sd_features["sd_conf"]  = prob.max(axis=1)

    return sd_features
