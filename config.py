from pathlib import Path

# Base directory (where this config file is located)
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "MCCF" / "PDP8" / "data"
RESULTS_DIR = BASE_DIR / "MCCF" / "PDP8" / "results"

# Input files
INFRASTRUCTURE_DATA = DATA_DIR / "infrastructure_data.xlsx"
PDP8_PROJECT_DATA = DATA_DIR / "PDP8 project data (2).xlsx"
OPENINFRA_EXISTING_GENERATOR_DATA = DATA_DIR / "openinfra_existing_generator_data.xlsx"
PDP8_POWER_LINES = DATA_DIR / "PDP8 power lines and grid.xlsx"
VNM_GPKG = DATA_DIR / "VNM.gpkg"

# Solar irradiance data (PVOUT.asc)
SOLAR_PVOUT_ASC = DATA_DIR / "solar" / "yearly geotiff" / "Vietnam_GISdata_LTAy_YearlyMonthlyTotals_GlobalSolarAtlas-v2_AAIGRID" / "PVOUT.asc"

# Wind data (GeoJSON)
WIND_GEOJSON = DATA_DIR / "wind" / "vietnam.geojson"

# Output files
PROJECTS_MAP = RESULTS_DIR / "PDP8_power_projects_map.html"
SUBSTATION_MAP = RESULTS_DIR / "vn_substation_map.html"
TRANSMISSION_MAP = RESULTS_DIR / "vn_transmission_map.html"
INTEGRATED_MAP = RESULTS_DIR / "vn_integrated_map.html"
WIND_MAP = RESULTS_DIR / "vn_wind_power_density_map.html"
NEW_TRANSFORMER_MAP = RESULTS_DIR / "vn_new_transformers_map.html"
GEM_MAP = RESULTS_DIR / "vn_gem_assets_map.html"
OPENINFRA_EXISTING_GENERATOR_MAP = RESULTS_DIR / "vn_openinfra_existing_generators_map.html"
NEW_TRANSMISSION_DATA = DATA_DIR / "PDP8_new_transmission_lines.xlsx"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True) 