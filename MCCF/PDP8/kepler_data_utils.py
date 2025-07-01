import pandas as pd
from . import map_utils as utils


def load_comprehensive_data(force_recompute: bool = False) -> dict:
    """Load and transform data required for the deck.gl comprehensive map."""
    data = {}

    # Transmission lines
    gdf = utils.get_power_lines()

    def voltage_category(val: object) -> str:
        try:
            v = float(val)
            if v >= 500000:
                return "500kV"
            elif v >= 220000:
                return "220kV"
            elif v >= 115000:
                return "115kV"
            elif v >= 110000:
                return "110kV"
            elif v >= 50000:
                return "50kV"
            elif v >= 33000:
                return "33kV"
            elif v >= 25000:
                return "25kV"
            elif v >= 22000:
                return "22kV"
            else:
                return "<22kV"
        except Exception:
            return "Unknown"

    if "max_voltage" in gdf.columns:
        gdf["voltage_cat"] = gdf["max_voltage"].apply(voltage_category)
    else:
        gdf["voltage_cat"] = "Unknown"

    features = utils.cache_polylines(
        gdf,
        cache_file="powerline_polylines.geojson",
        eps=0.0025,
        min_samples=3,
        force_recompute=force_recompute,
    )

    # Convert features to DataFrame for pydeck PathLayer
    line_records = [
        {
            "path": [(lat, lon) for lat, lon in f["geometry"]["coordinates"].copy()],
            "voltage": f["properties"]["voltage"],
        }
        for f in features
    ]
    data["power_lines"] = pd.DataFrame(line_records)

    # Substations
    sdf = utils.read_substation_data(force_recompute=force_recompute)
    data["substations"] = sdf

    # GEM existing assets
    gem_frames = [
        utils.read_coal_plant_data(force_recompute),
        utils.read_coal_terminal_data(force_recompute),
        utils.read_wind_power_data(force_recompute),
        utils.read_oil_gas_plant_data(force_recompute),
        utils.read_lng_terminal_data(force_recompute),
        utils.read_hydropower_data(force_recompute),
        utils.read_solar_power_data(force_recompute),
    ]
    gem_df = pd.concat([df for df in gem_frames if not df.empty], ignore_index=True)
    type_mapping = {
        "Coal Plant": "Coal",
        "Wind Power": "Wind",
        "Oil/Gas Plant - fossil gas: natural gas": "Domestic Gas-Fired",
        "Oil/Gas Plant - fossil gas: LNG": "LNG-Fired Gas",
        "Oil/Gas Plant - fossil gas: natural gas, fossil liquids: fuel oil": "Domestic Gas-fired/Fuel Oil",
        "Hydropower Plant": "Hydro",
        "Solar Farm": "Solar",
    }
    gem_df["type"] = gem_df["type"].map(type_mapping).fillna(gem_df["type"])
    data["gem_assets"] = gem_df

    # PDP8 power projects
    power_df, name_col = utils.read_and_clean_power_data(force_recompute)
    power_df["project_name"] = power_df[name_col]
    data["projects"] = power_df

    # Planned substations
    transformer_df = utils.read_planned_substation_data()
    data["planned_substations"] = transformer_df

    return data
