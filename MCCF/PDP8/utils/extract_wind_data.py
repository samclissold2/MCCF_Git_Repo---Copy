import geopandas as gpd
import pandas as pd
from pathlib import Path
from config import WIND_GEOJSON, DATA_DIR
import numpy as np

# Output directory for extracted data
extracted_data_dir = DATA_DIR / "extracted_data"
extracted_data_dir.mkdir(exist_ok=True)
output_csv = extracted_data_dir / "wind_power_density_points.csv"

# Read the wind GeoJSON file
if not WIND_GEOJSON.exists():
    print(f"Wind GeoJSON file not found: {WIND_GEOJSON}")
    exit(1)

gdf = gpd.read_file(WIND_GEOJSON)
gdf = gdf.explode(index_parts=False)

# Vectorized extraction of coordinates
coords_series = gdf.geometry.apply(lambda geom: np.array(geom.exterior.coords))
lengths = coords_series.apply(len)
all_coords = np.vstack(coords_series.to_list())
wind_vals = np.repeat(gdf.get('wind_power_density', None).values, lengths)

wind_df = pd.DataFrame({
    'lat': all_coords[:, 1],
    'lon': all_coords[:, 0],
    'wind_power_density': wind_vals
})
print(wind_df.head())
print(f"Extracted {len(wind_df)} wind power density points.")

# Save for later use
wind_df.to_csv(output_csv, index=False)
print(f"Saved extracted wind power density data to {output_csv}") 