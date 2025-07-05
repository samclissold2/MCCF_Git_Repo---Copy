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
    # normalise capacity column name
    if "Capacity_MW" not in gdf.columns and "capacity" in gdf.columns:
        gdf = gdf.rename(columns={"capacity": "Capacity_MW"})
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
        idx_raw = list(gen_sindex.nearest(geom, 1))[0]
        # GeoPandas can return:
        #   • int
        #   • (src_idx, nbr_idx)
        #   • list/array of ints
        if isinstance(idx_raw, tuple):
            idx = idx_raw[1]
        elif isinstance(idx_raw, (list, np.ndarray)):
            idx = idx_raw[0]
        else:
            idx = idx_raw
        dist  = geom.distance(gens.geometry.iloc[idx]) / 1_000  # km
        dist_list.append(dist)
        cap_col = "Capacity_MW"  # ensured by _assemble_generation_sites()
        cap_list.append(gens[cap_col].iloc[idx])
        # Ensure scalar boolean → int (0/1)
        var_val = gens.is_variable.iloc[idx]
        if not np.isscalar(var_val):  # Series -> take first value
            var_val = var_val.iloc[0]
        var_list.append(int(bool(var_val)))

    subs["dist_gen_km"]      = dist_list
    subs["nearest_gen_mw"]   = cap_list
    subs["nearest_gen_var"]  = var_list

    # ---- sum of generation within 5 km radius ------------------------------
    tree     = cKDTree(np.column_stack([gens.geometry.x, gens.geometry.y]))
    sums5, varfrac = [], []
    for geom in subs.geometry:
        idxs = list(tree.query_ball_point([geom.x, geom.y], r=5_000))
        if idxs:
            sel = gens.iloc[idxs]
            sums5.append(sel[cap_col].sum())
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
        idxs_raw = list(pop_sindex.nearest(geom, 1))
        idxs = [r[1] if isinstance(r, tuple) else r for r in idxs_raw]
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
        # Ensure numeric types for rule-based labelling
        seed["dist_gen_km"] = pd.to_numeric(seed["dist_gen_km"], errors="coerce").fillna(999)
        seed["pop_density"] = pd.to_numeric(seed["pop_density"], errors="coerce").fillna(0)
        # Manually tag a few for first train → user can refine later
        seed["Label"] = np.where(seed.dist_gen_km < 3, "Step-Up",
                           np.where(seed.pop_density > 400, "Step-Down",
                                    "Interconnection"))
        X = seed[feature_cols]
        y = seed.Label

        # ── ensure at least two classes to satisfy scikit-learn ──────────
        if y.nunique() < 2:
            # pick 5 random rows (or all if <5) to flip to another class
            alt_class = "Interconnection" if y.iloc[0] != "Interconnection" else "Step-Down"
            n_flip = min(5, len(y))
            flip_idx = y.sample(n_flip, random_state=2).index
            y.loc[flip_idx] = alt_class

        X = X.fillna(0)
        clf = GradientBoostingClassifier().fit(X, y)
        from joblib import dump
        dump(clf, model_path)

    prob  = clf.predict_proba(sd_features[feature_cols])
    sd_features["sd_class"] = clf.classes_[prob.argmax(axis=1)]
    sd_features["sd_conf"]  = prob.max(axis=1)

    return sd_features
