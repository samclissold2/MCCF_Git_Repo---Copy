import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from config import SOLAR_PVOUT_ASC, DATA_DIR

# Read the PVOUT.asc file (ASCII grid)
asc_path = SOLAR_PVOUT_ASC
if not asc_path.exists():
    print(f"PVOUT.asc file not found: {asc_path}")
    exit(1)

print(f"Using file: {asc_path}")

# Output directory for extracted data
extracted_data_dir = DATA_DIR / "extracted_data"
extracted_data_dir.mkdir(exist_ok=True)
output_csv = extracted_data_dir / "solar_irradiance_points.csv"

# Read with rasterio
with rasterio.open(asc_path) as src:
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Width, Height:", src.width, src.height)
    data = src.read(1)
    print("Min/Max:", data.min(), data.max())

    # Quick plot
    plt.imshow(data, cmap='viridis')
    plt.title("Solar Irradiance (PVOUT)")
    plt.colorbar(label="Irradiance (kWh/m²/year)")
    plt.show()

    # Get transform (affine) for pixel <-> geo conversion
    transform = src.transform
    rows, cols = np.where(data != src.nodata)
    lons, lats = rasterio.transform.xy(transform, rows, cols, offset='center')
    irradiance = data[rows, cols]
    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'irradiance': irradiance
    })

print(df.head())
print(f"Extracted {len(df)} valid grid points.")

# Save for later use if desired
df.to_csv(output_csv, index=False)
print(f"Saved extracted solar irradiance data to {output_csv}")