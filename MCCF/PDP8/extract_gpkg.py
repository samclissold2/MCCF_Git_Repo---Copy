import geopandas as gpd
import os
from pathlib import Path
import pandas as pd
from shapely.geometry import Point, LineString, MultiPolygon
from config import (
    DATA_DIR,
    INFRASTRUCTURE_DATA,
    VNM_GPKG
)

def list_available_layers(gpkg_path):
    """
    List all available layers in the GeoPackage file
    """
    try:
        # Get all layers in the GeoPackage
        layers = gpd.read_file(gpkg_path, driver="GPKG").layers
        print("\nAvailable layers in the GeoPackage:")
        for layer in layers:
            # Get basic info about each layer
            try:
                gdf = gpd.read_file(gpkg_path, layer=layer)
                print(f"- {layer}:")
                print(f"  Features: {len(gdf)}")
                print(f"  Geometry type: {gdf.geometry.type[0]}")
                print(f"  Columns: {gdf.columns.tolist()}")
                print()
            except Exception as e:
                print(f"- {layer}: Error reading layer - {str(e)}")
        return layers
    except Exception as e:
        print(f"Error listing layers: {str(e)}")
        return []

def extract_coordinates(gdf):
    """
    Extract coordinates from GeoDataFrame while preserving original data.
    For polygons/lines, reproject to a projected CRS before calculating centroid.
    """
    df = gdf.copy()
    if isinstance(df.geometry.iloc[0], Point):
        df['longitude'] = df.geometry.x
        df['latitude'] = df.geometry.y
    elif isinstance(df.geometry.iloc[0], LineString) or isinstance(df.geometry.iloc[0], MultiPolygon):
        # Reproject to UTM 48N (EPSG:32648) for Vietnam
        gdf_proj = df.to_crs(epsg=32648)
        centroids = gdf_proj.geometry.centroid
        # Convert centroids back to WGS84
        centroids_wgs = centroids.to_crs(epsg=4326)
        df['longitude'] = centroids_wgs.x
        df['latitude'] = centroids_wgs.y
    else:
        # Fallback: try centroid with warning
        df['longitude'] = df.geometry.centroid.x
        df['latitude'] = df.geometry.centroid.y
    df = df.drop(columns=['geometry'])
    return df

def extract_gpkg_data():
    # Path to the GeoPackage file
    if not VNM_GPKG.exists():
        raise FileNotFoundError(f"Could not find {VNM_GPKG}")
    
    # Dictionary to store all extracted data
    infrastructure_data = {}
    all_data = []
    
    try:
        # List of layers to extract with their correct names
        power_layers = [
            'power_generator_point',
            'power_generator_polygon',
            'power_plant',
            'power_line',
            'power_substation_point',
            'power_substation_polygon',
            'power_transformer',
            'power_switch',
            'power_compensator',
            'power_tower'
        ]
        
        oil_gas_layers = [
            'petroleum_well',
            'petroleum_site',
            'pipeline',
            'pipeline_feature'
        ]
        
        # Extract power infrastructure
        print("\nExtracting Power Infrastructure...")
        for layer in power_layers:
            try:
                gdf = gpd.read_file(VNM_GPKG, layer=layer)
                infrastructure_data[layer] = gdf
                
                # Convert to DataFrame with coordinates
                df = extract_coordinates(gdf)
                all_data.append(df)
                
                print(f"\n{layer.replace('_', ' ').title()} Information:")
                print(f"Number of features: {len(gdf)}")
                print(f"Columns: {gdf.columns.tolist()}")
                print(f"Geometry type: {gdf.geometry.type[0]}")
            except Exception as e:
                print(f"Could not read layer {layer}: {str(e)}")
        
        # Extract oil and gas infrastructure
        print("\nExtracting Oil & Gas Infrastructure...")
        for layer in oil_gas_layers:
            try:
                gdf = gpd.read_file(VNM_GPKG, layer=layer)
                infrastructure_data[layer] = gdf
                
                # Convert to DataFrame with coordinates
                df = extract_coordinates(gdf)
                all_data.append(df)
                
                print(f"\n{layer.replace('_', ' ').title()} Information:")
                print(f"Number of features: {len(gdf)}")
                print(f"Columns: {gdf.columns.tolist()}")
                print(f"Geometry type: {gdf.geometry.type[0]}")
            except Exception as e:
                print(f"Could not read layer {layer}: {str(e)}")
        
        # Combine all data into a single DataFrame
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            # Save to Excel for use with map.py
            combined_df.to_excel(INFRASTRUCTURE_DATA, index=False)
            print(f"\nSaved combined data to {INFRASTRUCTURE_DATA}")
            return combined_df
        else:
            print("No data was successfully extracted")
            return None
        
    except Exception as e:
        print(f"Error reading the GeoPackage file: {str(e)}")
        return None

def save_extracted_data(data_dict, output_dir=None):
    """
    Save the extracted data to separate GeoPackage files
    """
    if output_dir is None:
        output_dir = DATA_DIR / "extracted_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each layer to a separate file
    for layer_name, gdf in data_dict.items():
        output_path = os.path.join(output_dir, f"{layer_name}.gpkg")
        gdf.to_file(output_path, driver="GPKG")
        print(f"Saved {layer_name} to {output_path}")

def inventory_gpkg(gpkg_path):
    """
    Print a detailed inventory of all layers, columns, types, and sample values in the GeoPackage.
    """
    import geopandas as gpd
    import pandas as pd
    import fiona
    # List all layers
    layers = fiona.listlayers(gpkg_path)
    print(f"\nGeoPackage: {gpkg_path}")
    print(f"Layers found: {layers}")
    for layer in layers:
        print(f"\n--- Layer: {layer} ---")
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer)
            print(f"Number of features: {len(gdf)}")
            print(f"Geometry type: {gdf.geometry.geom_type.unique()}")
            print("Columns and types:")
            print(gdf.dtypes)
            print("\nSample data:")
            print(gdf.head(5))
            # For each object/text column, print unique values (if not too many)
            for col in gdf.select_dtypes(include=['object']).columns:
                unique_vals = gdf[col].unique()
                if len(unique_vals) <= 20:
                    print(f"Unique values in '{col}': {unique_vals}")
                else:
                    print(f"Column '{col}' has {len(unique_vals)} unique values (showing first 20): {unique_vals[:20]}")
        except Exception as e:
            print(f"Error reading layer {layer}: {e}")

if __name__ == "__main__":
    # Extract the data
    data = extract_gpkg_data()
    
    # Check if data was successfully extracted
    if isinstance(data, pd.DataFrame) and not data.empty:
        print(f"\nSuccessfully extracted {len(data)} infrastructure features")
    else:
        print("\nNo data was extracted")
    # To run a full inventory, uncomment the following line:
    inventory_gpkg(VNM_GPKG) 