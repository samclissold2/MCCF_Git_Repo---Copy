import geopandas as gpd
import pandas as pd
from pathlib import Path
from config import WIND_GEOJSON, DATA_DIR

# Output directory for extracted data
extracted_data_dir = DATA_DIR / "extracted_data"
extracted_data_dir.mkdir(exist_ok=True)
output_csv = extracted_data_dir / "wind_power_density_points.csv"

# Read the wind GeoJSON file
if not WIND_GEOJSON.exists():
    print(f"Wind GeoJSON file not found: {WIND_GEOJSON}")
    exit(1)

gdf = gpd.read_file(WIND_GEOJSON)

# Extract coordinates and wind power density values
records = []
for _, row in gdf.iterrows():
    # Assume wind power density is in the properties as 'wind_power_density'
    wind_val = row.get('wind_power_density', None)
    geom = row.geometry
    if geom.type == 'MultiPolygon':
        for polygon in geom.geoms:
            for x, y in polygon.exterior.coords:
                records.append({'lat': y, 'lon': x, 'wind_power_density': wind_val})
    elif geom.type == 'Polygon':
        for x, y in geom.exterior.coords:
            records.append({'lat': y, 'lon': x, 'wind_power_density': wind_val})
    # Add more geometry types if needed

# Create DataFrame
wind_df = pd.DataFrame(records)
print(wind_df.head())
print(f"Extracted {len(wind_df)} wind power density points.")

# Save for later use
wind_df.to_csv(output_csv, index=False)
print(f"Saved extracted wind power density data to {output_csv}") 