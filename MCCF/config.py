from pathlib import Path

# Base directory (where this config file is located)
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Input files
INFRASTRUCTURE_DATA = DATA_DIR / "infrastructure_data.xlsx"
PDP8_PROJECT_DATA = DATA_DIR / "PDP8 project data (2).xlsx"
PDP8_POWER_LINES = DATA_DIR / "PDP8 power lines and grid.xlsx"
VNM_GPKG = DATA_DIR / "VNM.gpkg"

# Solar irradiance data (PVOUT.asc)
SOLAR_PVOUT_ASC = DATA_DIR / "solar" / "yearly geotiff" / "Vietnam_GISdata_LTAy_YearlyMonthlyTotals_GlobalSolarAtlas-v2_AAIGRID" / "PVOUT.asc"

# Wind data (GeoJSON)
WIND_GEOJSON = DATA_DIR / "wind" / "vietnam.geojson"

# Output files
PROJECTS_MAP = RESULTS_DIR / "vn_projects_with_layers.html"
SUBSTATION_MAP = RESULTS_DIR / "vn_substation_map.html"
TRANSMISSION_MAP = RESULTS_DIR / "vn_transmission_map.html"
INTEGRATED_MAP = RESULTS_DIR / "vn_integrated_map.html"
WIND_MAP = RESULTS_DIR / "vn_wind_power_density_map.html"
NEW_TRANSFORMER_MAP = RESULTS_DIR / "vn_new_transformers_map.html"
GEM_MAP = RESULTS_DIR / "vn_gem_assets_map.html"
TZ_SOLAR_MAP = RESULTS_DIR / "vn_tz_solar_map.html"
EXISTING_GENERATOR_MAP = RESULTS_DIR / "vn_existing_generators_map.html"

# Kepler output files
KEPLER_POWER_MAP = RESULTS_DIR / "kepler_power_map.html"
KEPLER_SUBSTATION_MAP = RESULTS_DIR / "kepler_substation_map.html"
KEPLER_TRANSMISSION_MAP = RESULTS_DIR / "kepler_transmission_map.html"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True) 