import pandas as pd
import folium
import branca.colormap as cm
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, MeasureControl, HeatMap
from pathlib import Path
import webbrowser
import os
import requests
import json
import re
import numpy as np
import argparse
import geopandas as gpd
import logging
import time
import pickle
from datetime import datetime
from config import (
    INFRASTRUCTURE_DATA,
    PDP8_PROJECT_DATA,
    VNM_GPKG,
    DATA_DIR,
    RESULTS_DIR,
    NEW_TRANSMISSION_DATA
)
import rasterio

#test for push

# Create cache directory
CACHE_DIR = RESULTS_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(function_name):
    """Get the cache file path for a given function."""
    return CACHE_DIR / f"{function_name}_cache.pkl"

def load_from_cache(function_name):
    """Load data from cache if it exists and is newer than source files."""
    cache_path = get_cache_path(function_name)
    
    if not cache_path.exists():
        return None
    
    try:
        # Load cached data
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Check if cache is still valid (you could add timestamp checking here)
        logging.info(f"Loaded {function_name} data from cache")
        return cached_data
    except Exception as e:
        logging.warning(f"Failed to load cache for {function_name}: {e}")
        return None

def save_to_cache(function_name, data):
    """Save data to cache."""
    cache_path = get_cache_path(function_name)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved {function_name} data to cache")
    except Exception as e:
        logging.warning(f"Failed to save cache for {function_name}: {e}")

def clear_cache(function_name=None):
    """Clear cache for a specific function or all caches."""
    if function_name:
        cache_path = get_cache_path(function_name)
        if cache_path.exists():
            cache_path.unlink()
            logging.info(f"Cleared cache for {function_name}")
    else:
        for cache_file in CACHE_DIR.glob("*_cache.pkl"):
            cache_file.unlink()
        logging.info("Cleared all caches")

# Configure logging
def setup_logging():
    """Configure logging with console output only"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler only
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    logging.info("Logging initialized. Console output only.")

# Initialize logging when module is imported
setup_logging()

def split_circuit_km(value):
    """Parse '#circuits×km' strings."""
    if value is None or str(value).strip() == "":
        return pd.Series([None, None, None])

    m = re.match(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)", str(value).strip())
    if not m:
        return pd.Series([None, None, None])

    circuits = int(m.group(1))
    km = float(m.group(2))
    return pd.Series([circuits, km, circuits * km])

def read_new_transmission_data(force_recompute=False):

    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_new_transmission_data')
        if cached_data is not None:
            return cached_data

    start_time = time.time()
    logging.info("Starting NEW PDP8 transmission data reading")

    if not NEW_TRANSMISSION_DATA.exists():
        logging.error(f"Could not find {NEW_TRANSMISSION_DATA}")
        raise FileNotFoundError(f"Could not find {NEW_TRANSMISSION_DATA}")

    try:
        logging.info(f"Reading transmission data from {NEW_TRANSMISSION_DATA}")
        sheet_names = [
            ("500kV North", "North"),
            ("220kV North", "North"),
            ("500kV Central", "Central"),
            ("220kV Central", "Central"),
            ("500kV South", "South"),
            ("220kV South", "South"),
        ]


        dfs = []
        for sheet, region in sheet_names:
            logging.info(f"Reading sheet: {sheet}")
            df_sheet = pd.read_excel(
                NEW_TRANSMISSION_DATA,
                sheet_name=sheet,
            )

            df_sheet.columns = df_sheet.columns.str.strip()

            if "Number of circuits × kilometres" in df_sheet.columns:
                df_sheet[["circuits", "route_km", "circuit_km"]] = (
                    df_sheet["Number of circuits × kilometres"].apply(split_circuit_km)
                )

            lat_col = next((c for c in df_sheet.columns if "lat" in c.lower()), None)
            lon_col = next((c for c in df_sheet.columns if "lon" in c.lower()), None)
            if lat_col and lon_col:
                df_sheet = df_sheet.rename(columns={lat_col: "lat", lon_col: "lon"})
                df_sheet["lat"] = pd.to_numeric(df_sheet["lat"], errors="coerce")
                df_sheet["lon"] = pd.to_numeric(df_sheet["lon"], errors="coerce")

            df_sheet["region"] = region
            df_sheet["sheet_source"] = sheet
            dfs.append(df_sheet)
        
        pdp_planned_transmission_lines = pd.concat(dfs, ignore_index=True)
        pdp_planned_transmission_lines = pdp_planned_transmission_lines.dropna(subset=["lat", "lon"])

        logging.info(
            f"Initial data shape: {pdp_planned_transmission_lines.shape}"
        )


        processing_time = time.time() - start_time
        logging.info(
            f"NEW PDP8 transmission data reading completed in {processing_time:.2f} seconds"
        )
        save_to_cache(
            "read_new_transmission_data",
            pdp_planned_transmission_lines,
        )

        logging.info(
            f"Final transmission records: {len(pdp_planned_transmission_lines)}"
        )

        pdp_planned_transmission_lines = pdp_planned_transmission_lines.drop_duplicates()

        return pdp_planned_transmission_lines

    except Exception as e:
        logging.error(f"Error reading PDP8 transmission data: {str(e)}", exc_info=True)
        raise

def annotate_planned_lines(planned_df,
                           subs=None,
                           lines=None,
                           substation_buffer: int = 1000,
                           line_buffer: int = 250):
    """Attach nearest substations and existing-line proximity flags.

    If *subs* or *lines* are supplied, those pre-loaded datasets are used;
    otherwise they are loaded internally (cached versions if available)."""
    if planned_df.empty:
        return gpd.GeoDataFrame(planned_df)

    # ---- use supplied datasets, or fall back to internal loaders -----------
    if subs is None:
        subs = read_substation_data()
    if lines is None:
        lines = get_power_lines()

    planned_gdf = gpd.GeoDataFrame(
        planned_df,
        geometry=gpd.points_from_xy(planned_df["lon"], planned_df["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    subs_gdf = gpd.GeoDataFrame(
        subs,
        geometry=gpd.points_from_xy(subs["longitude"], subs["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    lines_gdf = lines.to_crs("EPSG:3857")

    nearest = gpd.sjoin_nearest(
        planned_gdf, subs_gdf, how="left", distance_col="dist_m"
    )
    planned_gdf["nearest_substation_dist_m"] = nearest["dist_m"]
    planned_gdf["near_existing_sub"] = (
        planned_gdf["nearest_substation_dist_m"] <= substation_buffer
    )

    line_buffered = lines_gdf.buffer(line_buffer)
    planned_gdf["touches_existing_line"] = planned_gdf.geometry.intersects(
        line_buffered.unary_union
    )
    return planned_gdf

# GEM Data Processing Functions - Updated for cleaned data
def read_coal_plant_data(force_recompute=False):
    """
    Reads and processes the cleaned Global Coal Plant Tracker data.
    Returns a DataFrame with cleaned coal plant data.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_coal_plant_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting coal plant data processing")
    
    coal_file = DATA_DIR / "GEM" / "Global Coal Plant Tracker July 2024.xlsx"
    if not coal_file.exists():
        logging.error(f"Coal plant data file not found: {coal_file}")
        return pd.DataFrame()
    
    try:
        # Read the cleaned file
        logging.info("Reading coal plant Excel file")
        df = pd.read_excel(coal_file)
        logging.info(f"Initial data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Create standardized DataFrame
        logging.info("Standardizing DataFrame format")
        result_df = pd.DataFrame({
            'name': df['Plant name'],
            'capacity': pd.to_numeric(df['Capacity (MW)'], errors='coerce'),
            'latitude': pd.to_numeric(df['Latitude'], errors='coerce'),
            'longitude': pd.to_numeric(df['Longitude'], errors='coerce'),
            'status': df['Status'] if 'Status' in df.columns else 'Unknown'
        })
        
        # Clean up
        initial_len = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude'])
        result_df['type'] = 'Coal Plant'
        
        dropped_count = initial_len - len(result_df)
        logging.info(f"Dropped {dropped_count} rows with missing coordinates")
        logging.info(f"Final coal plant records: {len(result_df)}")
        
        processing_time = time.time() - start_time
        logging.info(f"Coal plant data processing completed in {processing_time:.2f} seconds")
        
        # Save to cache
        save_to_cache('read_coal_plant_data', result_df)
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error reading coal plant data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def read_coal_terminal_data(force_recompute=False):
    """
    Reads and processes the cleaned Global Coal Terminals Tracker data.
    Returns a DataFrame with cleaned coal terminal data.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_coal_terminal_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting coal terminal data processing")
    
    terminal_file = DATA_DIR / "GEM" / "Global-Coal-Terminals-Tracker-December-2024.xlsx"
    if not terminal_file.exists():
        logging.error(f"Coal terminal data file not found: {terminal_file}")
        return pd.DataFrame()
    
    try:
        # Read the cleaned file
        logging.info("Reading coal terminal Excel file")
        df = pd.read_excel(terminal_file)
        logging.info(f"Initial data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Create standardized DataFrame
        logging.info("Standardizing DataFrame format")
        result_df = pd.DataFrame({
            'name': df['Coal Terminal Name'],
            'capacity': pd.to_numeric(df['Capacity (Mt)'], errors='coerce'),
            'latitude': pd.to_numeric(df['Latitude'], errors='coerce'),
            'longitude': pd.to_numeric(df['Longitude'], errors='coerce'),
            'status': df['Status'] if 'Status' in df.columns else 'Unknown'
        })
        
        # Clean up
        initial_len = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude'])
        result_df['type'] = 'Coal Terminal'
        
        dropped_count = initial_len - len(result_df)
        logging.info(f"Dropped {dropped_count} rows with missing coordinates")
        logging.info(f"Final coal terminal records: {len(result_df)}")
        
        processing_time = time.time() - start_time
        logging.info(f"Coal terminal data processing completed in {processing_time:.2f} seconds")
        
        # Save to cache
        save_to_cache('read_coal_terminal_data', result_df)
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error reading coal terminal data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def read_wind_power_data(force_recompute=False):
    """
    Reads and processes the cleaned Global Wind Power Tracker data.
    Returns a DataFrame with cleaned wind power data.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_wind_power_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting wind power data processing")
    
    wind_file = DATA_DIR / "GEM" / "Global-Wind-Power-Tracker-February-2025.xlsx"
    if not wind_file.exists():
        logging.error(f"Wind power data file not found: {wind_file}")
        return pd.DataFrame()
    
    try:
        # Read the cleaned file
        logging.info("Reading wind power Excel file")
        df = pd.read_excel(wind_file)
        logging.info(f"Initial data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Filter for Vietnam if Country/Area column exists
        initial_len = len(df)
        if 'Country/Area' in df.columns:
            df = df[df['Country/Area'].str.lower().str.contains('vietnam', na=False)]
            logging.info(f"Filtered {initial_len - len(df)} non-Vietnam records")
        
        # Create standardized DataFrame
        logging.info("Standardizing DataFrame format")
        result_df = pd.DataFrame({
            'name': df['Project Name'] if 'Project Name' in df.columns else df.get('Name', 'Unknown'),
            'capacity': pd.to_numeric(df['Capacity (MW)'], errors='coerce') if 'Capacity (MW)' in df.columns else pd.NA,
            'latitude': pd.to_numeric(df['Latitude'], errors='coerce'),
            'longitude': pd.to_numeric(df['Longitude'], errors='coerce'),
            'status': df['Status'] if 'Status' in df.columns else 'Unknown'
        })
        
        # Clean up
        initial_len = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude'])
        result_df['type'] = 'Wind Power'
        
        dropped_count = initial_len - len(result_df)
        logging.info(f"Dropped {dropped_count} rows with missing coordinates")
        logging.info(f"Final wind power records: {len(result_df)}")
        
        processing_time = time.time() - start_time
        logging.info(f"Wind power data processing completed in {processing_time:.2f} seconds")
        
        # Save to cache
        save_to_cache('read_wind_power_data', result_df)
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error reading wind power data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def read_oil_gas_plant_data(force_recompute=False):
    """
    Reads and processes the cleaned Global Oil and Gas Plant Tracker data.
    Returns a DataFrame with cleaned oil and gas plant data.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_oil_gas_plant_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting oil and gas plant data processing")
    
    plant_file = DATA_DIR / "GEM" / "Global-Oil-and-Gas-Plant-Tracker-GOGPT-January-2025.xlsx"
    if not plant_file.exists():
        logging.error(f"Oil and gas plant data file not found: {plant_file}")
        return pd.DataFrame()
    
    try:
        # Read the cleaned file
        logging.info("Reading oil and gas plant Excel file")
        df = pd.read_excel(plant_file)
        logging.info(f"Initial data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Filter for Vietnam if Country/Area column exists
        initial_len = len(df)
        if 'Country/Area' in df.columns:
            df = df[df['Country/Area'].str.lower().str.contains('vietnam', na=False)]
            logging.info(f"Filtered {initial_len - len(df)} non-Vietnam records")
        
        # Create standardized DataFrame
        logging.info("Standardizing DataFrame format")
        result_df = pd.DataFrame({
            'name': df['Plant name'] if 'Plant name' in df.columns else df.get('Name', 'Unknown'),
            'capacity': pd.to_numeric(df['Capacity (MW)'], errors='coerce') if 'Capacity (MW)' in df.columns else pd.NA,
            'latitude': pd.to_numeric(df['Latitude'], errors='coerce'),
            'longitude': pd.to_numeric(df['Longitude'], errors='coerce'),
            'status': df['Status'] if 'Status' in df.columns else 'Unknown',
            'fuel': df['Fuel'] if 'Fuel' in df.columns else 'Unknown'
        })
        
        # Clean up
        initial_len = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude'])
        result_df['type'] = 'Oil/Gas Plant - ' + result_df['fuel'].astype(str)
        
        dropped_count = initial_len - len(result_df)
        logging.info(f"Dropped {dropped_count} rows with missing coordinates")
        logging.info(f"Final oil and gas plant records: {len(result_df)}")
        
        processing_time = time.time() - start_time
        logging.info(f"Oil and gas plant data processing completed in {processing_time:.2f} seconds")
        
        # Save to cache
        save_to_cache('read_oil_gas_plant_data', result_df)
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error reading oil and gas plant data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def read_lng_terminal_data(force_recompute=False):
    """
    Reads and processes the LNG Terminals data from DBF file.
    Returns a DataFrame with cleaned LNG terminal data.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_lng_terminal_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting LNG terminal data processing")
    
    lng_file = DATA_DIR / "GEM" / "GEM-GGIT-LNG-Terminals-2024-09.dbf"
    if not lng_file.exists():
        logging.error(f"LNG terminal data file not found: {lng_file}")
        return pd.DataFrame()
    
    try:
        # Read DBF file using geopandas
        logging.info("Reading LNG terminal DBF file")
        gdf = gpd.read_file(lng_file)
        logging.info(f"Initial data shape: {gdf.shape}, Columns: {list(gdf.columns)}")
        
        # First try to filter by country name
        logging.info("Attempting country-based filtering")
        vietnam_mask = pd.Series(False, index=gdf.index)
        country_cols = [col for col in gdf.columns if 'country' in col.lower()]
        logging.info(f"Found {len(country_cols)} potential country columns: {country_cols}")
        
        if country_cols:
            for col in country_cols:
                vietnam_variations = ['vietnam', 'viet nam', 'vietn nam', 'vn']
                for variation in vietnam_variations:
                    vietnam_mask |= gdf[col].astype(str).str.lower().str.contains(variation, na=False)
        
        initial_len = len(gdf)
        if vietnam_mask.any():
            gdf = gdf[vietnam_mask]
            logging.info(f"Found {len(gdf)} LNG terminal records for Vietnam using country filter")
        else:
            logging.info("No Vietnam records found using country filter, applying coordinate filter")
            # Filter by Vietnam's geographic boundaries
            vietnam_bounds = {
                'min_lat': 8.0,   # Southern tip
                'max_lat': 24.0,  # Northern border
                'min_lon': 102.0, # Western border  
                'max_lon': 110.0  # Eastern border including offshore areas
            }
            
            # Convert CRS to WGS84 if not already
            if gdf.crs and gdf.crs != "EPSG:4326":
                logging.info(f"Converting CRS from {gdf.crs} to EPSG:4326")
                gdf = gdf.to_crs("EPSG:4326")
            
            # Extract coordinates from geometry if it exists
            if hasattr(gdf, 'geometry') and gdf.geometry is not None and not gdf.geometry.isna().all():
                logging.info("Using geometry bounds for filtering")
                bounds = gdf.bounds
                
                # Filter geometries that intersect with Vietnam's bounding box
                vietnam_mask = (
                    (bounds['minx'] <= vietnam_bounds['max_lon']) &
                    (bounds['maxx'] >= vietnam_bounds['min_lon']) &
                    (bounds['miny'] <= vietnam_bounds['max_lat']) &
                    (bounds['maxy'] >= vietnam_bounds['min_lat'])
                )
                
                gdf = gdf[vietnam_mask]
                logging.info(f"Found {len(gdf)} LNG terminal records within Vietnam's coordinate boundaries")
            else:
                logging.info("No valid geometry found, attempting to use lat/lon columns")
                # Try using lat/lon columns if no geometry
                lat_col = next((col for col in gdf.columns if 'lat' in col.lower()), None)
                lon_col = next((col for col in gdf.columns if 'lon' in col.lower()), None)
                
                if lat_col and lon_col:
                    logging.info(f"Using columns: {lat_col} and {lon_col} for coordinates")
                    gdf['temp_lat'] = pd.to_numeric(gdf[lat_col], errors='coerce')
                    gdf['temp_lon'] = pd.to_numeric(gdf[lon_col], errors='coerce')
                    
                    vietnam_mask = (
                        (gdf['temp_lat'] >= vietnam_bounds['min_lat']) & 
                        (gdf['temp_lat'] <= vietnam_bounds['max_lat']) &
                        (gdf['temp_lon'] >= vietnam_bounds['min_lon']) & 
                        (gdf['temp_lon'] <= vietnam_bounds['max_lon'])
                    )
                    
                    gdf = gdf[vietnam_mask]
                    logging.info(f"Found {len(gdf)} LNG terminal records within Vietnam's coordinate boundaries using lat/lon columns")
        
        if gdf.empty:
            logging.warning("No LNG terminal records found for Vietnam after filtering")
            return pd.DataFrame()
        
        # Extract coordinates
        logging.info("Extracting final coordinates")
        if hasattr(gdf, 'geometry') and gdf.geometry is not None and not gdf.geometry.isna().all():
            logging.info("Using geometry for final coordinates")
            gdf['latitude'] = gdf.geometry.y
            gdf['longitude'] = gdf.geometry.x
        else:
            logging.info("Using lat/lon columns for final coordinates")
            lat_col = next((col for col in gdf.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in gdf.columns if 'lon' in col.lower()), None)
            if lat_col and lon_col:
                gdf['latitude'] = pd.to_numeric(gdf[lat_col], errors='coerce')
                gdf['longitude'] = pd.to_numeric(gdf[lon_col], errors='coerce')
        
        # Double-check that coordinates are within Vietnam bounds
        initial_len = len(gdf)
        vietnam_coord_mask = (
            (gdf['latitude'] >= 8.0) & (gdf['latitude'] <= 24.0) &
            (gdf['longitude'] >= 102.0) & (gdf['longitude'] <= 110.0)
        )
        gdf = gdf[vietnam_coord_mask]
        logging.info(f"Filtered {initial_len - len(gdf)} records outside Vietnam's bounds")
        
        # Find relevant columns
        logging.info("Creating final standardized DataFrame")
        name_col = next((col for col in gdf.columns if 'name' in col.lower()), None)
        capacity_col = next((col for col in gdf.columns if 'capacity' in col.lower()), None)
        status_col = next((col for col in gdf.columns if 'status' in col.lower()), None)
        
        result_df = pd.DataFrame({
            'name': gdf[name_col] if name_col else 'LNG Terminal',
            'capacity': pd.to_numeric(gdf[capacity_col], errors='coerce') if capacity_col else pd.NA,
            'latitude': gdf['latitude'],
            'longitude': gdf['longitude'],
            'status': gdf[status_col] if status_col else 'Unknown'
        })
        
        initial_len = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude'])
        result_df['type'] = 'LNG Terminal'
        
        dropped_count = initial_len - len(result_df)
        logging.info(f"Dropped {dropped_count} rows with missing coordinates")
        logging.info(f"Final LNG terminal records: {len(result_df)}")
        
        processing_time = time.time() - start_time
        logging.info(f"LNG terminal data processing completed in {processing_time:.2f} seconds")
        return result_df
        
    except Exception as e:
        logging.error(f"Error reading LNG terminal data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def read_hydropower_data(force_recompute=False):
    """
    Reads and processes the Global Hydropower Tracker data.
    Returns a DataFrame with cleaned hydropower plant data.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_hydropower_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting hydropower data processing")
    
    hydro_file = DATA_DIR / "GEM" / "Global-Hydropower-Tracker-April-2025.xlsx"
    if not hydro_file.exists():
        logging.error(f"Hydropower data file not found: {hydro_file}")
        return pd.DataFrame()
    
    try:
        # Read the file
        logging.info("Reading hydropower Excel file")
        df = pd.read_excel(hydro_file)
        logging.info(f"Initial data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Filter for Vietnam if Country/Area column exists
        initial_len = len(df)
        if 'Country/Area 1' in df.columns:
            df = df[df['Country/Area 1'].str.lower().str.contains('vietnam', na=False)]
            logging.info(f"Filtered {initial_len - len(df)} non-Vietnam records")
        
        # Create standardized DataFrame
        logging.info("Standardizing DataFrame format")
        result_df = pd.DataFrame({
            'name': df['Project Name'] if 'Project Name' in df.columns else 'Hydropower Plant',
            'capacity': pd.to_numeric(df['Capacity (MW)'], errors='coerce') if 'Capacity (MW)' in df.columns else pd.NA,
            'latitude': pd.to_numeric(df['Latitude'], errors='coerce'),
            'longitude': pd.to_numeric(df['Longitude'], errors='coerce'),
            'status': df['Status'] if 'Status' in df.columns else 'Unknown'
        })
        
        # Clean up
        initial_len = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude'])
        result_df['type'] = 'Hydropower Plant'
        
        dropped_count = initial_len - len(result_df)
        logging.info(f"Dropped {dropped_count} rows with missing coordinates")
        logging.info(f"Final hydropower plant records: {len(result_df)}")
        
        processing_time = time.time() - start_time
        logging.info(f"Hydropower data processing completed in {processing_time:.2f} seconds")
        
        # Save to cache
        save_to_cache('read_hydropower_data', result_df)
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error reading hydropower data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def read_solar_power_data(force_recompute=False):
    """
    Reads and processes the Global Solar Power Tracker data.
    Returns a DataFrame with cleaned solar power plant data.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_solar_power_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting solar power data processing")
    
    solar_file = DATA_DIR / "GEM" / "Global-Solar-Power-Tracker-February-2025.xlsx"
    if not solar_file.exists():
        logging.error(f"Solar power data file not found: {solar_file}")
        return pd.DataFrame()
    
    try:
        # Read the file
        logging.info("Reading solar power Excel file")
        df = pd.read_excel(solar_file)
        logging.info(f"Initial data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # The file already appears to contain Vietnam data based on our test
        # Create standardized DataFrame
        logging.info("Standardizing DataFrame format")
        result_df = pd.DataFrame({
            'name': df['Project Name'] if 'Project Name' in df.columns else 'Solar Farm',
            'capacity': pd.to_numeric(df['Capacity (MW)'], errors='coerce') if 'Capacity (MW)' in df.columns else pd.NA,
            'latitude': pd.to_numeric(df['Latitude'], errors='coerce'),
            'longitude': pd.to_numeric(df['Longitude'], errors='coerce'),
            'status': df['Status'] if 'Status' in df.columns else 'Unknown'
        })
        
        # Clean up
        initial_len = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude'])
        result_df['type'] = 'Solar Farm'
        
        dropped_count = initial_len - len(result_df)
        logging.info(f"Dropped {dropped_count} rows with missing coordinates")
        logging.info(f"Final solar farm records: {len(result_df)}")
        
        processing_time = time.time() - start_time
        logging.info(f"Solar power data processing completed in {processing_time:.2f} seconds")
        
        # Save to cache
        save_to_cache('read_solar_power_data', result_df)
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error reading solar power data: {str(e)}", exc_info=True)
        return pd.DataFrame()

def create_collapsible_legend(position='left', title='Legend', content='', width=220):
    """
    Creates a collapsible legend div with specified position and content.
    
    Args:
        position: 'left' or 'right' initial position
        title: Legend title
        content: HTML content for the legend
        width: Width of the legend box in pixels
    """
    # Generate a unique ID for this legend
    legend_id = f'legend-{hash(title) % 10000}'
    
    return f'''
    <div id="{legend_id}" style="
        position: fixed; 
        bottom: 50px; 
        {position}: 50px; 
        width: {width}px; 
        z-index:9999; 
        background: white; 
        border:2px solid grey; 
        border-radius:6px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        font-size:12px; 
        padding: 10px;">
        <div onclick="toggleLegend('{legend_id}')" style="cursor:pointer;font-weight:bold;user-select:none;margin-bottom:5px;padding:5px;background:#f8f8f8;border-radius:4px;">
            {title} <span id="{legend_id}-arrow" style="float:right;">▶</span>
        </div>
        <div id="{legend_id}-content" style="display:none; margin-top:8px;">
            {content}
        </div>
    </div>
    '''

def add_legend_control_script():
    """
    Adds the JavaScript for controlling legend visibility only.
    """
    return '''
    <script>
    function toggleLegend(legendId) {
        const content = document.getElementById(legendId + '-content');
        const arrow = document.getElementById(legendId + '-arrow');
        
        if (content && arrow) {
            if (content.style.display === 'none' || content.style.display === '') {
                content.style.display = 'block';
                arrow.innerHTML = '▼';
            } else {
                content.style.display = 'none';
                arrow.innerHTML = '▶';
            }
        }
    }
    </script>
    '''

def read_infrastructure_data(force_recompute=False):
    """
    Reads the infrastructure_data.xlsx file and returns a DataFrame.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_infrastructure_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting infrastructure data reading")
    
    if not INFRASTRUCTURE_DATA.exists():
        logging.error(f"Could not find {INFRASTRUCTURE_DATA}")
        raise FileNotFoundError(f"Could not find {INFRASTRUCTURE_DATA}")
    
    try:
        logging.info(f"Reading infrastructure data from {INFRASTRUCTURE_DATA}")
        df = pd.read_excel(INFRASTRUCTURE_DATA)
        logging.info(f"Read {len(df)} rows of infrastructure data")
        
        processing_time = time.time() - start_time
        logging.info(f"Infrastructure data reading completed in {processing_time:.2f} seconds")
        
        # Save to cache
        save_to_cache('read_infrastructure_data', df)
        
        return df
    except Exception as e:
        logging.error(f"Error reading infrastructure data: {str(e)}", exc_info=True)
        raise

def assign_period(phase):
    """
    Maps operational phase values to standardized period ranges.
    
    Args:
        phase: The operational phase value from the data
        
    Returns:
        str: A standardized period range (e.g., '2025-2030', '2031-2035') or None if not mappable
    """
    if pd.isna(phase):
        return None
        
    phase = str(phase).lower().strip()
    
    # Map common phase patterns to periods
    if any(x in phase for x in ['2025', '2026', '2027', '2028', '2029', '2030']):
        return '2025-2030'
    elif any(x in phase for x in ['2031', '2032', '2033', '2034', '2035']):
        return '2031-2035'
    elif any(x in phase for x in ['2036', '2037', '2038', '2039', '2040']):
        return '2036-2040'
    elif any(x in phase for x in ['2041', '2042', '2043', '2044', '2045']):
        return '2041-2045'
    elif any(x in phase for x in ['2046', '2047', '2048', '2049', '2050']):
        return '2046-2050'
    
    # If no match found
    return None

def read_and_clean_power_data(force_recompute=False):
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_and_clean_power_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting power data reading and cleaning")
    
    sheet_names = [
        "solar",
        "onshore",
        "LNG-fired gas",
        'cogeneration',
        'domestic gas-fired',
        'hydro',
        'pumped-storage',
        'nuclear',
        'biomass',
        'waste-to-energy',
        'flexible'
    ]
    
    if not PDP8_PROJECT_DATA.exists():
        logging.error(f"Could not find {PDP8_PROJECT_DATA}")
        raise FileNotFoundError(f"Could not find {PDP8_PROJECT_DATA}")
    
    try:
        logging.info(f"Reading {len(sheet_names)} sheets from {PDP8_PROJECT_DATA}")
        df_dict = pd.read_excel(PDP8_PROJECT_DATA, sheet_name=sheet_names, engine="openpyxl")
        
        lat_col  = "longitude"
        lon_col  = "latitude"
        mw_col   = "expected capacity mw"
        name_col = "project"
        all_dfs = []
        
        for tech, df in df_dict.items():
            logging.info(f"Processing {tech} sheet with {len(df)} rows")
            
            # Clean column names
            df.columns = (
                df.columns.str.strip().str.lower()
                  .str.replace(r"[^\w\s]", "", regex=True)
                  .str.replace(r"\s+", " ", regex=True)
            )
            
            # Filter valid projects
            initial_len = len(df)
            df = df[df[name_col].astype(str).str.strip().astype(bool)]
            logging.info(f"Filtered {initial_len - len(df)} rows with empty project names")
            
            df["tech"] = tech.title()
            
            # Process phases
            phases = df["operational phase"].dropna().unique()
            logging.info(f"Found {len(phases)} unique operational phases")
            
            for phase in phases:
                col_name = f"phase_{str(phase).replace('-', '_')}"
                df[col_name] = df["operational phase"] == phase
            
            df["operational phase original"] = df["operational phase"]
            
            # Process periods
            df["period"] = df["operational phase"].apply(assign_period)
            initial_period_len = len(df)
            df = df.dropna(subset=["period"])
            logging.info(f"Filtered {initial_period_len - len(df)} rows with invalid periods")
            
            all_dfs.append(df)
            logging.info(f"Completed processing {tech} sheet")
        
        logging.info("Combining all processed sheets")
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Process coordinates and capacity
        logging.info("Processing coordinates and capacity")
        initial_len = len(df)
        df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
        df["mw"]  = (
            df[mw_col]
              .astype(str)
              .str.replace(",", ".")
              .pipe(pd.to_numeric, errors="coerce")
        )
        
        df = df.dropna(subset=["lat", "lon", "mw"])
        dropped_count = initial_len - len(df)
        logging.info(f"Dropped {dropped_count} rows with invalid coordinates or capacity")
        
        processing_time = time.time() - start_time
        logging.info(f"Power data processing completed in {processing_time:.2f} seconds")
        logging.info(f"Final dataset has {len(df)} rows")
        
        result = (df, name_col)
        
        # Save to cache
        save_to_cache('read_and_clean_power_data', result)
        
        return result
    except Exception as e:
        logging.error(f"Error processing power data: {str(e)}", exc_info=True)
        raise

def read_solar_irradiance_points(force_recompute: bool = False):
    """
    Reads (and now caches) extracted solar-irradiance points.
    Returns a DataFrame with columns  lat / lon / irradiance
    """
    if not force_recompute:
        cached = load_from_cache("solar_irradiance_points")
        if cached is not None:
            return cached

    start   = time.time()
    csvpath = DATA_DIR / "extracted_data" / "solar_irradiance_points.csv"
    if not csvpath.exists():
        logging.error(f"Solar irradiance CSV not found: {csvpath}")
        return None

    try:
        df = pd.read_csv(csvpath)
        logging.info(f"Read {len(df):,} solar-irradiance points  "
                     f"in {time.time()-start:,.1f}s")
        save_to_cache("solar_irradiance_points", df)        # ← NEW
        return df
    except Exception as e:
        logging.error("Error reading solar irradiance data", exc_info=True)
        return None

def create_folium_map(df):
    """Base map creation with collapsible legends."""
    start_time = time.time()
    logging.info("Starting base map creation")
    
    try:
        # Handle different column naming conventions
        if 'lat' in df.columns and 'lon' in df.columns:
            center_lat, center_lon = df.lat.mean(), df.lon.mean()
        elif 'latitude' in df.columns and 'longitude' in df.columns:
            center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
        else:
            # Default to center of Vietnam if no coordinate columns found
            center_lat, center_lon = 16.0, 106.0
            logging.warning("No coordinate columns found, using default center of Vietnam")
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles="CartoDB Positron",
            attr="© OpenStreetMap, © CartoDB",
        )
        
        logging.info("Adding map plugins")
        for Plugin in (Fullscreen, MeasureControl):  # Removed MiniMap
            Plugin().add_to(m)
        
        logging.info("Adding legend control script")
        m.get_root().html.add_child(folium.Element(add_legend_control_script()))
        
        processing_time = time.time() - start_time
        logging.info(f"Base map creation completed in {processing_time:.2f} seconds")
        return m
    except Exception as e:
        logging.error(f"Error creating base map: {str(e)}", exc_info=True)
        return None

def save_and_open_map(m, output_file=None):
    start_time = time.time()
    logging.info("Starting map saving process")
    
    if output_file is None:
        output_file = PROJECTS_MAP
    
    try:
        logging.info(f"Saving map to {output_file}")
        m.save(output_file)
        
        file_path = os.path.abspath(output_file)
        logging.info(f"Opening map in browser: {file_path}")
        webbrowser.open(f"file://{file_path}")
        
        processing_time = time.time() - start_time
        logging.info(f"Map saving and opening completed in {processing_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error saving/opening map: {str(e)}", exc_info=True)
        raise

def read_transmission_data(force_recompute=False):
    """
    Reads infrastructure_data.xlsx and returns a DataFrame of transmission-line
    records.  Keeps longitude / latitude columns and *max_voltage* so that the
    whole codebase uses the same voltage label.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_transmission_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting transmission data reading")
    
    if not INFRASTRUCTURE_DATA.exists():
        logging.error(f"Could not find {INFRASTRUCTURE_DATA}")
        raise FileNotFoundError(f"Could not find {INFRASTRUCTURE_DATA}")
    
    try:
        logging.info(f"Reading transmission data from {INFRASTRUCTURE_DATA}")
        df = pd.read_excel(INFRASTRUCTURE_DATA)
        logging.info(f"Initial data shape: {df.shape}")
        
        # Filter for rows where 'location' is not blank
        initial_len = len(df)
        df = df[df['location'].astype(str).str.strip().astype(bool)]
        logging.info(f"Filtered {initial_len - len(df)} rows with blank location")
        
        # Only keep rows where location is 'overhead' or 'underground'
        initial_len = len(df)
        df = df[df['location'].str.lower().isin(['overhead', 'underground'])]
        logging.info(f"Filtered {initial_len - len(df)} rows with invalid location type")
        
        # Only keep relevant columns – include any voltage-like column
        keep_cols = ['location', 'longitude', 'latitude',
                     'max_voltage',           # preferred name
                     'voltage', 'voltage_kv', 'kv', 'volt_kv']
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]
        logging.info(f"Kept columns: {keep_cols}")

        # ── STANDARDISE THE VOLTAGE LABEL  ─────────────────────────────────────
        # rename the first voltage-type column we find to *max_voltage*
        if 'max_voltage' not in df.columns:
            vcol = next((c for c in df.columns if 'volt' in c.lower()), None)
            if vcol:
                df = df.rename(columns={vcol: 'max_voltage'})
            else:
                df['max_voltage'] = pd.NA        # keep column present
        # ───────────────────────────────────────────────────────────────────────
        
        processing_time = time.time() - start_time
        logging.info(f"Transmission data reading completed in {processing_time:.2f} seconds")
        logging.info(f"Final transmission records: {len(df)}")
        
        # Save to cache
        save_to_cache('read_transmission_data', df)
        
        return df
        
    except Exception as e:
        logging.error(f"Error reading transmission data: {str(e)}", exc_info=True)
        raise

def get_source_color(source):
    """Get a consistent color for a given source type."""
    color_map = {
        'solar': '#FFA500',
        'hydro': '#003366',
        'onshore': '#87CEEB',
        'lng': '#D3D3D3',
        'gas': '#333333',
        'pumped': '#4682B4',
        'nuclear': '#800080',
        'biomass': '#228B22',
        'waste': '#8B6F22',
        'flexible': '#000000'
    }
    
    source_lower = source.lower()
    for key, color in color_map.items():
        if key in source_lower:
            return color
    return '#999999'  # Default color for unknown sources

def get_voltage_color(voltage):
    """Get a consistent color for a given voltage level."""
    if pd.isna(voltage) or voltage == 'unknown':
        return '#999999'  # Gray for unknown voltage
    
    try:
        voltage = float(voltage)
        if voltage >= 500:
            return '#FF0000'  # Red for highest voltage
        elif voltage >= 220:
            return '#FFA500'  # Orange
        elif voltage >= 110:
            return '#FFD700'  # Gold
        elif voltage >= 66:
            return '#32CD32'  # Lime green
        else:
            return '#4169E1'  # Royal blue for lower voltages
    except (ValueError, TypeError):
        return '#999999'  # Gray for unparseable voltage values

def read_powerline_data(force_recompute=False):
    """
    Reads and processes powerline data, returning clustered transmission lines.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_powerline_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting powerline data reading and processing")
    
    try:
        gdf = get_power_lines()
        
        if gdf.empty:
            logging.warning("No powerline data found")
            return []
        
        # Categorize by max_voltage for lines
        def voltage_category(val):
            try:
                v = float(val)
                if v >= 500000:
                    return '500kV'
                elif v >= 220000:
                    return '220kV'
                elif v >= 115000:
                    return '115kV'
                elif v >= 110000:
                    return '110kV'
                elif v >= 50000:
                    return '50kV'
                elif v >= 33000:
                    return '33kV'
                elif v >= 25000:
                    return '25kV'
                elif v >= 22000:
                    return '22kV'
                else:
                    return '<22kV'
            except:
                return 'Unknown'

        if 'max_voltage' in gdf.columns:
            gdf['voltage_cat'] = gdf['max_voltage'].apply(voltage_category)
        else:
            gdf['voltage_cat'] = 'Unknown'
        
        # Get clustered polylines
        features = cache_polylines(gdf, cache_file='powerline_polylines.geojson', eps=0.0025, min_samples=3, force_recompute=force_recompute)
        
        processing_time = time.time() - start_time
        logging.info(f"Powerline data processing completed in {processing_time:.2f} seconds")
        logging.info(f"Generated {len(features)} clustered transmission lines")
        
        # Save to cache
        save_to_cache('read_powerline_data', features)
        
        return features
        
    except Exception as e:
        logging.error(f"Error reading powerline data: {str(e)}", exc_info=True)
        return []

def get_power_lines(force_recompute: bool = False):
    """
    Reads (and now caches) the power-line layer from the GPKG file.
    """
    if not force_recompute:
        cached = load_from_cache("get_power_lines")
        if cached is not None:
            return cached

    if not VNM_GPKG.exists():
        logging.error(f"Could not find {VNM_GPKG}")
        raise FileNotFoundError(f"Could not find {VNM_GPKG}")

    try:
        logging.info(f"Reading powerline data from {VNM_GPKG}")
        gdf = gpd.read_file(VNM_GPKG, layer="power_line")
        logging.info(f"Read {len(gdf)} powerline records")
        save_to_cache("get_power_lines", gdf)
        return gdf
    except Exception as e:
        logging.error(f"Error reading powerline data: {str(e)}", exc_info=True)
        return gpd.GeoDataFrame()

def get_power_towers():
    """
    Reads the power tower data from the GPKG file and returns a GeoDataFrame.
    """
    if not VNM_GPKG.exists():
        logging.error(f"Could not find {VNM_GPKG}")
        raise FileNotFoundError(f"Could not find {VNM_GPKG}")
    
    try:
        logging.info(f"Reading power tower data from {VNM_GPKG}")
        gdf = gpd.read_file(VNM_GPKG, layer='power_tower')
        logging.info(f"Read {len(gdf)} power tower records")
        return gdf
    except Exception as e:
        logging.error(f"Error reading power tower data: {str(e)}", exc_info=True)
        return gpd.GeoDataFrame()

def cache_polylines(gdf, cache_file='powerline_polylines.geojson', eps=0.0025, min_samples=3, force_recompute=False):
    """
    Cluster power lines by voltage category, then order with greedy path within each voltage group.
    Returns a list of polylines (each is a list of (lat, lon) tuples).
    """
    cache_path = RESULTS_DIR / cache_file
    
    if cache_path.exists() and not force_recompute:
        logging.info(f"Loading polylines from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            geojson = json.load(f)
        return geojson['features']
    
    logging.info("Computing DBSCAN clusters and greedy paths for each voltage category...")
    features = []
    
    # Import DBSCAN here to avoid import errors if sklearn not available
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        logging.warning("scikit-learn not available, cannot perform clustering")
        return features
    
    # Group power lines by voltage category

    def voltage_category(val):
        try:
            v = float(val)
            if v >= 500000:
                return '500kV'
            elif v >= 220000:
                return '220kV'
            elif v >= 115000:
                return '115kV'
            elif v >= 110000:
                return '110kV'
            elif v >= 50000:
                return '50kV'
            elif v >= 33000:
                return '33kV'
            elif v >= 25000:
                return '25kV'
            elif v >= 22000:
                return '22kV'
            else:
                return '<22kV'
        except:
            return 'Unknown'
    if 'max_voltage' in gdf.columns:
        gdf['voltage_cat'] = gdf['max_voltage'].apply(voltage_category)
    else:
        gdf['voltage_cat'] = 'Unknown'

    voltage_groups = gdf.groupby('voltage_cat')
    
    for voltage_cat, group in voltage_groups:
        if len(group) < min_samples:
            continue
            
        coords_arrays = group.geometry.apply(
            lambda geom: np.vstack([
                np.array(line.coords) for line in (
                    geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
                )
            ])
        ).to_list()
        if not coords_arrays:
            continue
        coords = np.vstack(coords_arrays)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Process each cluster within this voltage category
        for cluster_id in set(db.labels_):
            if cluster_id == -1:
                continue  # noise
                
            cluster_points = coords[db.labels_ == cluster_id]
            # Greedy path ordering
            path = [cluster_points[0]]
            used = set([0])
            
            for _ in range(1, len(cluster_points)):
                last = path[-1]
                dists = np.linalg.norm(cluster_points - last, axis=1)
                dists[list(used)] = np.inf
                next_idx = np.argmin(dists)
                path.append(cluster_points[next_idx])
                used.add(next_idx)
                
            # Save as GeoJSON LineString with voltage category
            line_coords = [[float(y), float(x)] for x, y in path]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line_coords},
                "properties": {
                    "cluster": int(cluster_id),
                    "voltage": voltage_cat
                }
            })
    
    geojson = {"type": "FeatureCollection", "features": features}
    with open(cache_path, 'w') as f:
        json.dump(geojson, f)
    logging.info(f"Saved polylines to {cache_path}")
    return features

def read_substation_data(force_recompute=False):
    """
    Reads infrastructure_data.xlsx and returns a DataFrame with rows where 'substation_type' is not blank.
    Only keeps longitude and latitude columns and relevant info.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_substation_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting substation data reading")
    
    if not INFRASTRUCTURE_DATA.exists():
        logging.error(f"Could not find {INFRASTRUCTURE_DATA}")
        raise FileNotFoundError(f"Could not find {INFRASTRUCTURE_DATA}")
    
    try:
        logging.info(f"Reading substation data from {INFRASTRUCTURE_DATA}")
        df = pd.read_excel(INFRASTRUCTURE_DATA)
        logging.info(f"Initial data shape: {df.shape}")
        
        # Filter for rows where 'substation_type' is not blank
        initial_len = len(df)
        df = df[df['substation_type'].astype(str).str.strip().astype(bool)]
        logging.info(f"Filtered {initial_len - len(df)} rows with blank substation type")

        # Only keep relevant columns
        keep_cols = ['substation_type', 'max_voltage', 'longitude', 'latitude']
        keep_cols = [col for col in keep_cols if col in df.columns]
        df = df[keep_cols]
        logging.info(f"Kept columns: {keep_cols}")
        
        processing_time = time.time() - start_time
        logging.info(f"Substation data reading completed in {processing_time:.2f} seconds")
        logging.info(f"Final substation records: {len(df)}")
        
        # Save to cache
        save_to_cache('read_substation_data', df)
        
        return df
        
    except Exception as e:
        logging.error(f"Error reading substation data: {str(e)}", exc_info=True)
        raise

def read_tif_data(tif_path, sample_rate=0.1):
    """
    Reads a TIF file and returns data suitable for mapping.
    Returns a list of [lat, lon, value] for non-null pixels.
    
    Args:
        tif_path: Path to the TIF file
        sample_rate: Fraction of points to sample (0.1 = 10% of points) for performance
    """
    try:
        with rasterio.open(tif_path) as src:
            # Read the data
            data = src.read(1)  # Read first band
            
            # Get the transform to convert pixel coordinates to geographic coordinates
            transform = src.transform
            
            # Get valid data (non-null values)
            valid_mask = ~np.isnan(data) & (data != src.nodata) if src.nodata is not None else ~np.isnan(data)
            valid_indices = np.where(valid_mask)
            
            # Convert pixel coordinates to geographic coordinates
            rows, cols = valid_indices
            lons, lats = rasterio.transform.xy(transform, rows, cols)
            
            # Get the corresponding values
            values = data[valid_mask]
            
            # Sample points for better performance
            total_points = len(values)
            sample_size = int(total_points * sample_rate)
            
            if sample_size < total_points:
                # Randomly sample points
                indices = np.random.choice(total_points, sample_size, replace=False)
                lats = [lats[i] for i in indices]
                lons = [lons[i] for i in indices]
                values = values[indices]
            
            # Create list of [lat, lon, value]
            points = []
            for lat, lon, value in zip(lats, lons, values):
                if not np.isnan(value) and value > 0:  # Filter out invalid values
                    points.append([lat, lon, float(value)])
            
            print(f"Loaded {len(points)} valid data points from {tif_path} (sampled from {total_points} total points)")
            return points
            
    except Exception as e:
        print(f"Error reading TIF file {tif_path}: {e}")
        return []

def create_wind_power_density_layer(force_recompute: bool = False):
    """
    Creates (and now caches) a Folium HeatMap layer for wind-power-density.
    """
    if not force_recompute:
        cached = load_from_cache("wind_heatmap_layer")
        if cached is not None:
            return cached

    tif = DATA_DIR / "wind" / "VNM_power-density_100m.tif"
    if not tif.exists():
        logging.error(f"TIF not found: {tif}")
        return None

    pts = read_tif_data(tif, sample_rate=0.05)          # 5 % sampling
    if not pts:
        return None

    heat = HeatMap(
        pts, name="Wind Power Density (100 m)", show=True,
        min_opacity=.2, max_opacity=.9, radius=20, blur=25,
        gradient={0.0: 'blue', 0.2: 'cyan', 0.4: 'green',
                  0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
    )

    save_to_cache("wind_heatmap_layer", heat)           # ← NEW
    return heat

def read_planned_substation_data(force_recompute=False):
    """
    Reads the PDP8_new_transformer.xlsx file and extracts transformer data from all worksheets.
    Returns a DataFrame with transformer information including voltage categorization.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_planned_substation_data')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting planned substation data reading")
    
    transformer_file = DATA_DIR / "PDP8_new_transformer.xlsx"
    
    if not transformer_file.exists():
        logging.error(f"Planned substation file not found: {transformer_file}")
        return pd.DataFrame()
    
    try:
        # Read all sheets from the Excel file
        logging.info(f"Reading substation data from {transformer_file}")
        excel_file = pd.ExcelFile(transformer_file)
        logging.info(f"Found {len(excel_file.sheet_names)} worksheets: {excel_file.sheet_names}")
        
        all_dfs = []
        
        for sheet_name in excel_file.sheet_names:
            sheet_start_time = time.time()
            logging.info(f"Processing sheet: {sheet_name}")
            
            # Try to detect voltage from sheet name
            sheet_name_lower = sheet_name.lower()
            sheet_voltage = None
            
            # Map sheet names to voltage categories
            voltage_patterns = {
                '500kv': '500kV',
                '220kv': '220kV',
                '200kv': '220kV',  # Map 200kV to 220kV category
                '115kv': '115kV',
                '110kv': '110kV',
                '50kv': '50kV',
                '33kv': '33kV',
                '25kv': '25kV',
                '22kv': '22kV'
            }
            
            # Check for voltage in sheet name
            for pattern, voltage in voltage_patterns.items():
                if pattern in sheet_name_lower or pattern.replace('kv', '_kv') in sheet_name_lower:
                    sheet_voltage = voltage
                    logging.info(f"Detected voltage {voltage} from sheet name")
                    break
            
            # Read the sheet
            df = pd.read_excel(transformer_file, sheet_name=sheet_name)
            logging.info(f"Initial data shape for {sheet_name}: {df.shape}")
            
            # Clean column names
            df.columns = (
                df.columns.str.strip().str.lower()
                  .str.replace(r"[^\w\s]", "", regex=True)
                  .str.replace(r"\s+", "_", regex=True)
            )
            
            # Look for key columns
            lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
            name_col = next((col for col in df.columns if any(x in col.lower() for x in ['name', 'transformer', 'station'])), None)
            voltage_col = next((col for col in df.columns if any(x in col.lower() for x in ['voltage', 'kv'])), None)
            
            # Only process if we have location data
            if lat_col and lon_col:
                logging.info(f"Found location columns: {lat_col}, {lon_col}")
                
                # Create standardized DataFrame
                result_df = pd.DataFrame({
                    'name': df[name_col] if name_col else f"Transformer_{sheet_name}",
                    'lat': pd.to_numeric(df[lat_col], errors='coerce'),
                    'lon': pd.to_numeric(df[lon_col], errors='coerce'),
                    'voltage': df[voltage_col] if voltage_col else 'Unknown'
                })
                
                # Apply sheet voltage if needed
                if sheet_voltage and (voltage_col is None or result_df['voltage'].isna().all() or (result_df['voltage'] == 'Unknown').all()):
                    result_df['voltage'] = sheet_voltage
                    logging.info(f"Applied sheet voltage {sheet_voltage} to all transformers")
                
                # Filter valid coordinates
                initial_len = len(result_df)
                result_df = result_df.dropna(subset=['lat', 'lon'])
                dropped_count = initial_len - len(result_df)
                logging.info(f"Dropped {dropped_count} rows with missing coordinates")
                
                if not result_df.empty:
                    result_df['sheet_source'] = sheet_name
                    all_dfs.append(result_df)
                    logging.info(f"Added {len(result_df)} valid transformer records from {sheet_name}")
                else:
                    logging.warning(f"No valid coordinates found in {sheet_name}")
            else:
                logging.warning(f"No location columns found in {sheet_name}")
        
            sheet_processing_time = time.time() - sheet_start_time
            logging.info(f"Sheet {sheet_name} processing completed in {sheet_processing_time:.2f} seconds")
        
        # Combine all processed dataframes
        if all_dfs:
            logging.info("Combining all processed sheets")
            combined_df = pd.concat(all_dfs, ignore_index=True)
            logging.info(f"Final combined shape: {combined_df.shape}")
            
            processing_time = time.time() - start_time
            logging.info(f"Transformer data processing completed in {processing_time:.2f} seconds")
            
            # Save to cache
            save_to_cache('read_planned_substation_data', combined_df)
            
            return combined_df
        else:
            logging.warning("No valid transformer data found in any worksheet")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error reading transformer data: {str(e)}", exc_info=True)
        return pd.DataFrame()
