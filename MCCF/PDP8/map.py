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
    PDP8_POWER_LINES,
    VNM_GPKG,
    PROJECTS_MAP,
    SUBSTATION_MAP,
    TRANSMISSION_MAP,
    INTEGRATED_MAP,
    WIND_MAP,
    NEW_TRANSFORMER_MAP,
    DATA_DIR,
    RESULTS_DIR,
    GEM_MAP,
    OPENINFRA_EXISTING_GENERATOR_MAP,
    OPENINFRA_EXISTING_GENERATOR_DATA
)

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
            'name': df['Terminal Name'],
            'capacity': pd.to_numeric(df['Capacity (Mtpa)'], errors='coerce'),
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

def create_gem_map(force_recompute=False):
    """
    Creates a map combining all GEM data sources with status-based bucketing.
    For oil/gas plants, also buckets by fuel type.
    """
    start_time = time.time()
    logging.info("Starting GEM map creation")
    
    # Read all datasets
    logging.info("Reading GEM datasets...")
    datasets = [
        read_coal_plant_data(force_recompute=force_recompute),
        read_coal_terminal_data(force_recompute=force_recompute),
        read_wind_power_data(force_recompute=force_recompute),
        read_oil_gas_plant_data(force_recompute=force_recompute),
        read_lng_terminal_data(force_recompute=force_recompute),
        read_hydropower_data(force_recompute=force_recompute),
        read_solar_power_data(force_recompute=force_recompute)
    ]
    
    # Combine all non-empty datasets
    valid_dfs = [df for df in datasets if not df.empty]
    if not valid_dfs:
        logging.error("No valid GEM data found")
        return None
    
    logging.info(f"Found {len(valid_dfs)} non-empty GEM datasets")
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    logging.info(f"Combined dataset size: {len(combined_df)} rows")
    
    # Create base map
    logging.info("Creating base map")
    m = folium.Map(
        location=[16.0, 106.0],  # Center of Vietnam
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    
    # Add plugins
    logging.info("Adding map plugins")
    for Plugin in (Fullscreen, MiniMap, MeasureControl):
        Plugin().add_to(m)
    
    # Process feature groups and markers
    logging.info("Processing feature groups and markers")
    feature_groups = {}
    marker_count = 0
    
    for asset_type in combined_df['type'].unique():
        type_df = combined_df[combined_df['type'] == asset_type]
        logging.info(f"Processing {len(type_df)} assets of type: {asset_type}")
        
        # Add markers for each asset
        for _, row in type_df.iterrows():
            marker_count += 1
            if marker_count % 100 == 0:
                logging.info(f"Added {marker_count} markers to map")
            
            # Determine the appropriate feature group
            if row['type'].startswith('Oil/Gas Plant'):
                group_name = f"{row['type']} ({row['fuel']}) - {row['status']}"
            else:
                group_name = f"{row['type']} - {row['status']}"
            
            # Get feature group
            fg = feature_groups.get(group_name)
            if not fg:
                continue
            
            # Get base color for asset type
            base_color = type_colors.get(row['type'].split(' - ')[0], '#888888')
            # Get status color
            status_color = status_colors.get(row['status'], '#000000')
            
            # Scale marker size based on capacity if available
            radius = 8
            if 'capacity' in row and pd.notnull(row['capacity']):
                radius = max(6, min(20, (float(row['capacity']) ** 0.5) * 0.5))
            
            # Create marker with status-colored border and type-colored fill
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=status_color,  # Border color based on status
                weight=2,
                opacity=0.8,
                fill=True,
                fill_color=base_color,  # Fill color based on type
                fill_opacity=0.6,
                tooltip=f"{row['name']} ({row['type']})<br>Status: {row['status']}<br>Capacity: {row.get('capacity', 'N/A')} MW"
            ).add_to(fg)
    
    logging.info(f"Total markers added: {marker_count}")
    
    # Add legends
    logging.info("Adding legends and controls")
    m.get_root().html.add_child(folium.Element(add_legend_control_script()))
    
    processing_time = time.time() - start_time
    logging.info(f"GEM map creation completed in {processing_time:.2f} seconds")
    return m

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

def read_solar_irradiance_points():
    """
    Reads the extracted solar irradiance CSV and returns a DataFrame.
    """
    start_time = time.time()
    logging.info("Starting solar irradiance data reading")
    
    csv_path = DATA_DIR / "extracted_data" / "solar_irradiance_points.csv"
    if not csv_path.exists():
        logging.error(f"Solar irradiance CSV not found: {csv_path}")
        return None
    
    try:
        logging.info(f"Reading solar irradiance data from {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info(f"Read {len(df)} solar irradiance points")
        
        processing_time = time.time() - start_time
        logging.info(f"Solar irradiance data reading completed in {processing_time:.2f} seconds")
        return df
    except Exception as e:
        logging.error(f"Error reading solar irradiance data: {str(e)}", exc_info=True)
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

def get_power_lines():
    """
    Reads the powerline data from the GPKG file and returns a GeoDataFrame.
    """
    if not VNM_GPKG.exists():
        logging.error(f"Could not find {VNM_GPKG}")
        raise FileNotFoundError(f"Could not find {VNM_GPKG}")
    
    try:
        logging.info(f"Reading powerline data from {VNM_GPKG}")
        gdf = gpd.read_file(VNM_GPKG, layer='power_line')
        logging.info(f"Read {len(gdf)} powerline records")
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
    voltage_groups = gdf.groupby('voltage_cat')
    
    for voltage_cat, group in voltage_groups:
        if len(group) < min_samples:
            continue
            
        # Get coordinates of power line endpoints
        coords = []
        for geom in group.geometry:
            if geom.geom_type == 'LineString':
                coords.extend(geom.coords)  # coords are already (x,y) tuples
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords.extend(line.coords)  # coords are already (x,y) tuples
        
        if not coords:
            continue
            
        coords = np.array(coords)
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

def create_integrated_map(force_recompute=False):
    """
    Creates a map that overlays power projects, substations, and power lines.
    """
    # Create base map
    m = folium.Map(
        location=[21.0, 105.8],  # Center of Vietnam
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    
    # Add power lines
    gdf = get_power_lines()
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
    features = cache_polylines(gdf, cache_file='powerline_polylines.geojson', eps=0.0025, min_samples=3, force_recompute=force_recompute)
    voltage_colors = {
        '500kV': 'red',
        '220kV': 'orange',
        '115kV': 'purple',
        '110kV': 'blue',
        '50kV': 'green',
        '33kV': 'brown',
        '25kV': 'pink',
        '22kV': 'gray',
        '<22kV': 'black',
        'Unknown': 'black',
    }
    
    # Add power lines as a toggleable FeatureGroup
    transmission_fg = folium.FeatureGroup(name="Transmission Lines", show=True).add_to(m)
    for feat in features:
        coords = feat['geometry']['coordinates']
        voltage = feat['properties']['voltage']
        color = voltage_colors.get(voltage, 'black')
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in coords],
            color=color,
            weight=4,
            opacity=0.7,
            tooltip=f"{voltage} Line (Cluster {feat['properties']['cluster']})"
        ).add_to(transmission_fg)
    
    # Add substations
    sdf = read_substation_data()
    sdf = sdf[sdf['substation_type'].notna() & (sdf['substation_type'].astype(str).str.strip() != '')]
    # 3. Bucket substations by max_voltage and assign colors
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
    if 'max_voltage' in sdf.columns:
        sdf['voltage_cat'] = sdf['max_voltage'].apply(voltage_category)
    else:
        sdf['voltage_cat'] = 'Unknown'
    voltage_colors = {
        '500kV': 'red',
        '220kV': 'orange',
        '115kV': 'purple',
        '110kV': 'blue',
        '50kV': 'green',
        '33kV': 'brown',
        '25kV': 'pink',
        '22kV': 'gray',
        '<22kV': 'black',
        'Unknown': 'black',
    }
    if not sdf.empty:
        sub_fg = folium.FeatureGroup(name="Substations", show=False).add_to(m)
        for _, row in sdf.iterrows():
            color = voltage_colors.get(row['voltage_cat'], 'black')
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f'<div style="font-size:10px; color:{color}; font-weight:bold;">×</div>'),
                tooltip=f"Substation ({row['voltage_cat']})"
            ).add_to(sub_fg)
    # # Add solar irradiance heatmap to integrated map, toggled off by default
    # solar_df = read_solar_irradiance_points()
    # if solar_df is not None and not solar_df.empty:
    #     heat_data = [
    #         [row['lat'], row['lon'], row['irradiance']] for _, row in solar_df.iterrows()
    #     ]
    #     heatmap_layer = HeatMap(
    #         heat_data,
    #         name="Solar Irradiance Heatmap",
    #         min_opacity=0.005,
    #         max_opacity=0.015,
    #         radius=8,
    #         blur=12,
    #         gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'},
    #         show=False
    #     )
    #     heatmap_fg = folium.FeatureGroup(name="Solar Irradiance Heatmap", show=False)
    #     heatmap_fg.add_child(heatmap_layer)
    #     m.add_child(heatmap_fg)
    
    # Add power projects
    df, name_col = read_and_clean_power_data()
    tech_colors = {
        'Solar': '#FF0000',            # red
        'Hydro': '#003366',            # dark blue
        'Onshore': '#87CEEB',          # light blue
        'LNG-Fired Gas': '#D3D3D3',    # light grey
        'Domestic Gas-Fired': '#333333', # dark grey
        'Pumped-Storage': '#4682B4',   # medium blue
        'Nuclear': '#800080',          # purple
        'Biomass': '#228B22',          # green
        'Waste-To-Energy': '#8B6F22',  # dirty green/brown
        'Flexible': '#000000',         # black
    }
    
    for tech in df.tech.unique():
        for period in df[df.tech == tech]["period"].unique():
            name = f"{tech} {period}"
            show = (tech == 'Solar' and period in ['2025-2030', '2031-2035'])
            fg = folium.FeatureGroup(name=name, show=show).add_to(m)
            tech_df = df[(df.tech == tech) & (df.period == period)]
            color = tech_colors.get(tech, '#888888')
            for _, row in tech_df.iterrows():
                folium.CircleMarker(
                    location=[row.lat, row.lon],
                    radius=max(4, (row.mw ** 0.5) * 0.5),
                    color='#222',  # faint outline
                    opacity=0.3,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1,
                    tooltip=f"{row[name_col]} ({row.tech}, {row['period']}) — {row.mw:.0f} MW"
                ).add_to(fg)
    
    # Add planned transformers
    transformer_df = read_planned_substation_data()
    if not transformer_df.empty:
        # Group transformers by voltage category - use same categories as existing substations
        def voltage_category(val):
            try:
                v = float(str(val).replace('kV', '').replace('KV', ''))
                if v >= 500:
                    return '500kV'
                elif v >= 220:
                    return '220kV'
                elif v >= 115:
                    return '115kV'
                elif v >= 110:
                    return '110kV'
                elif v >= 50:
                    return '50kV'
                elif v >= 33:
                    return '33kV'
                elif v >= 25:
                    return '25kV'
                elif v >= 22:
                    return '22kV'   
                else:
                    return '<22kV'
            except:
                return 'Unknown'

        transformer_df.rename(columns={'voltage': 'voltage_cat'}, inplace=True)
        transformer_df['voltage_cat'] = transformer_df['voltage_cat'].apply(voltage_category)
        
        # Create feature group for planned transformers
        transformer_fg = folium.FeatureGroup(name="Planned Transformers", show=False).add_to(m)
        
        for _, row in transformer_df.iterrows():
            # Extract numeric voltage value for color coding directly from voltage_cat
            try:
                numeric_voltage = float(str(row['voltage_cat']).replace('kV', '').replace('KV', ''))
            except:
                numeric_voltage = 'unknown'
            
            # Use the same color coding system as substation map
            color = get_voltage_color(numeric_voltage)
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.DivIcon(html=f'<div style="font-size:10px; color:{color}; font-weight:bold;">×</div>'),
                tooltip=f"Planned Transformer ({row['voltage_cat']}) - {row['name']} - {row['sheet_source']}"
            ).add_to(transformer_fg)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legends using the standardized collapsible legend system
    powerline_legend_content = '''
        <div><span style="display:inline-block; width:24px; height:4px; background:red; margin-right:4px;"></span> 500kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:orange; margin-right:4px;"></span> 220kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:purple; margin-right:4px;"></span> 115kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:blue; margin-right:4px;"></span> 110kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:green; margin-right:4px;"></span> 50kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:brown; margin-right:4px;"></span> 33kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:pink; margin-right:4px;"></span> 25kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:gray; margin-right:4px;"></span> 22kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:black; margin-right:4px;"></span> <22kV or Unknown Lines</div>
    '''
    
    substation_legend_content = '''
        <div><span style="display:inline-block; width:16px; height:16px; background:red; border-radius:50%; margin-right:4px;"></span> 500kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:orange; border-radius:50%; margin-right:4px;"></span> 220kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:purple; border-radius:50%; margin-right:4px;"></span> 115kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:blue; border-radius:50%; margin-right:4px;"></span> 110kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:green; border-radius:50%; margin-right:4px;"></span> 50kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:brown; border-radius:50%; margin-right:4px;"></span> 33kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:pink; border-radius:50%; margin-right:4px;"></span> 25kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:gray; border-radius:50%; margin-right:4px;"></span> 22kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:black; border-radius:50%; margin-right:4px;"></span> <22kV or Unknown</div>
    '''
    
    project_legend_content = '''
        <div><span style="background:#FF0000; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Solar</div>
        <div><span style="background:#003366; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Hydro</div>
        <div><span style="background:#87CEEB; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Onshore</div>
        <div><span style="background:#D3D3D3; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> LNG-Fired Gas</div>
        <div><span style="background:#333333; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Domestic Gas-Fired</div>
        <div><span style="background:#4682B4; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Pumped-Storage</div>
        <div><span style="background:#800080; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Nuclear</div>
        <div><span style="background:#228B22; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Biomass</div>
        <div><span style="background:#8B6F22; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Waste-To-Energy</div>
        <div><span style="background:#000000; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Flexible</div>
    '''
    
    # Create custom legends stacked vertically on the left side with good spacing
    powerline_legend_id = 'legend-powerline'
    substation_legend_id = 'legend-substation'
    project_legend_id = 'legend-project'
    
    powerline_legend = f'''
    <div id="{powerline_legend_id}" style="
        position: fixed; 
        bottom: 50px; 
        left: 50px; 
        width: 200px; 
        z-index:9999; 
        background: white; 
        border:2px solid grey; 
        border-radius:6px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        font-size:12px; 
        padding: 10px;">
        <div onclick="toggleLegend('{powerline_legend_id}')" style="cursor:pointer;font-weight:bold;user-select:none;margin-bottom:5px;padding:5px;background:#f8f8f8;border-radius:4px;">
            Transmission Line Legend <span id="{powerline_legend_id}-arrow" style="float:right;">▶</span>
        </div>
        <div id="{powerline_legend_id}-content" style="display:none; margin-top:8px;">
            {powerline_legend_content}
        </div>
    </div>
    '''
    
    substation_legend = f'''
    <div id="{substation_legend_id}" style="
        position: fixed; 
        bottom: 300px; 
        left: 50px; 
        width: 200px; 
        z-index:9999; 
        background: white; 
        border:2px solid grey; 
        border-radius:6px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        font-size:12px; 
        padding: 10px;">
        <div onclick="toggleLegend('{substation_legend_id}')" style="cursor:pointer;font-weight:bold;user-select:none;margin-bottom:5px;padding:5px;background:#f8f8f8;border-radius:4px;">
            Substation Voltage Legend <span id="{substation_legend_id}-arrow" style="float:right;">▶</span>
        </div>
        <div id="{substation_legend_id}-content" style="display:none; margin-top:8px;">
            {substation_legend_content}
        </div>
    </div>
    '''
    
    project_legend = f'''
    <div id="{project_legend_id}" style="
        position: fixed; 
        bottom: 650px; 
        left: 50px; 
        width: 200px; 
        z-index:9999; 
        background: white; 
        border:2px solid grey; 
        border-radius:6px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        font-size:12px; 
        padding: 10px;">
        <div onclick="toggleLegend('{project_legend_id}')" style="cursor:pointer;font-weight:bold;user-select:none;margin-bottom:5px;padding:5px;background:#f8f8f8;border-radius:4px;">
            Power Project Legend <span id="{project_legend_id}-arrow" style="float:right;">▶</span>
        </div>
        <div id="{project_legend_id}-content" style="display:none; margin-top:8px;">
            {project_legend_content}
        </div>
    </div>
    '''
    
    # Add the legend control script and legends to map
    m.get_root().html.add_child(folium.Element(add_legend_control_script()))
    m.get_root().html.add_child(folium.Element(powerline_legend))
    m.get_root().html.add_child(folium.Element(substation_legend))
    m.get_root().html.add_child(folium.Element(project_legend))
    
    return m

def read_openinfra_existing_generators(force_recompute=False):
    """
    Reads the dedicated existing generators data file.
    Returns a DataFrame with the following columns:
    - source: The source/type of the generator
    - latitude: Latitude coordinate
    - longitude: Longitude coordinate
    - output_mw: Power output in megawatts
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('read_openinfra_existing_generators')
        if cached_data is not None:
            return cached_data
    
    start_time = time.time()
    logging.info("Starting OpenInfra existing generators data reading")
    
    if not OPENINFRA_EXISTING_GENERATOR_DATA.exists():
        logging.error(f"Could not find {OPENINFRA_EXISTING_GENERATOR_DATA}")
        raise FileNotFoundError(f"Could not find {OPENINFRA_EXISTING_GENERATOR_DATA}")
    
    try:
        logging.info(f"Reading generator data from {OPENINFRA_EXISTING_GENERATOR_DATA}")
        df = pd.read_excel(OPENINFRA_EXISTING_GENERATOR_DATA)
        logging.info(f"Initial data shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Map actual columns to expected columns
        column_mapping = {
            'source': 'source',
            'latitude': 'latitude', 
            'longitude': 'longitude',
            'output': 'output_mw'  # Map 'output' to 'output_mw'
        }
        
        # Check for required columns (using actual column names)
        actual_required_cols = ['source', 'latitude', 'longitude', 'output']
        missing_cols = [col for col in actual_required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns in {OPENINFRA_EXISTING_GENERATOR_DATA}: {missing_cols}")
        
        # Keep only required columns and rename
        df = df[actual_required_cols].copy()
        df = df.rename(columns=column_mapping)
        logging.info(f"Kept and renamed {len(actual_required_cols)} required columns")
        
        processing_time = time.time() - start_time
        logging.info(f"OpenInfra existing generators data reading completed in {processing_time:.2f} seconds")
        logging.info(f"Final generator records: {len(df)}")
        
        # Save to cache
        save_to_cache('read_openinfra_existing_generators', df)
        
        return df
        
    except Exception as e:
        logging.error(f"Error reading OpenInfra existing generators data: {str(e)}", exc_info=True)
        raise

def create_openinfra_existing_generator_map(force_recompute=False):
    """Map for OpenInfra existing generators with collapsible legends."""
    df = read_openinfra_existing_generators(force_recompute=force_recompute)
    
    if df.empty:
        print("No existing generator data found")
        return None
    
    m = create_folium_map(df)
    
    # Create feature groups for each source type
    source_groups = {}
    for source in df['source'].unique():
        source_groups[source] = folium.FeatureGroup(name=source, show=True).add_to(m)
    
    # Add markers
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color='#222',
            weight=1,
            opacity=0.8,
            fill=True,
            fill_color=get_source_color(row['source']),
            fill_opacity=0.6,
            tooltip=f"{row['source']}<br>Output: {row['output_mw']} MW"
        ).add_to(source_groups[row['source']])
    
    # Add layer control
    folium.LayerControl(
        collapsed=True,
        position='topright',
        autoZIndex=True
    ).add_to(m)
    
    # Create legends
    source_legend_content = ''.join(
        f'<div><span style="background:{get_source_color(source)}; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> {source}</div>'
        for source in sorted(df['source'].unique())
    )
    
    layer_control_content = '''
        <div style="margin-bottom:8px;"><strong>Layer Controls</strong></div>
        <div style="font-size:10px; color:#666;">
            Toggle layers to show/hide different generator types
        </div>
    '''
    
    # Add legends
    left_legend = create_collapsible_legend(
        position='left',
        title='Generator Types',
        content=source_legend_content,
        width=250
    )
    
    right_legend = create_collapsible_legend(
        position='right',
        title='Layer Controls',
        content=layer_control_content,
        width=250
    )
    
    m.get_root().html.add_child(folium.Element(left_legend))
    m.get_root().html.add_child(folium.Element(right_legend))
    
    return m

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

def create_wind_power_density_layer():
    """
    Creates a wind power density layer from the TIF file.
    Returns a HeatMap layer for folium.
    """
    tif_path = DATA_DIR / "wind" / "VNM_power-density_100m.tif"
    
    if not tif_path.exists():
        print(f"Wind power density TIF file not found: {tif_path}")
        return None
    
    # Read the TIF data with sampling for better performance
    points = read_tif_data(tif_path, sample_rate=0.05)  # Use 5% of points for better performance
    
    if not points:
        print("No valid data points found in TIF file")
        return None
    
    # Normalize values for better visualization
    values = [point[2] for point in points]
    min_val, max_val = min(values), max(values)
    
    print(f"Wind power density range: {min_val:.2f} - {max_val:.2f} W/m²")
    
    # Create heatmap layer with optimized parameters
    heatmap_layer = HeatMap(
        points,
        name="Wind Power Density (100m)",
        min_opacity=0.2,
        max_opacity=0.9,
        radius=20,
        blur=25,
        gradient={
            0.0: 'blue',
            0.2: 'cyan', 
            0.4: 'green',
            0.6: 'yellow',
            0.8: 'orange',
            1.0: 'red'
        },
        show=True  # Start with the layer ON for the dedicated wind map
    )
    
    return heatmap_layer

def create_wind_power_density_map():
    """
    Creates a dedicated map for wind power density visualization.
    """
    # Create base map centered on Vietnam
    m = folium.Map(
        location=[21.0, 105.8],  # Center of Vietnam
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    
    # Add plugins
    for Plugin in (Fullscreen, MiniMap, MeasureControl):
        Plugin().add_to(m)
    
    # Create wind power density layer
    wind_layer = create_wind_power_density_layer()
    if wind_layer is not None:
        # Add the wind layer directly to the map (not as a FeatureGroup)
        m.add_child(wind_layer)
        
        # Create wind power density legend content
        wind_legend_content = '''
            <div><span style="display:inline-block; width:24px; height:6px; background:blue; margin-right:6px;"></span> Low (0-200 W/m²)</div>
            <div><span style="display:inline-block; width:24px; height:6px; background:cyan; margin-right:6px;"></span> Low-Medium (200-400 W/m²)</div>
            <div><span style="display:inline-block; width:24px; height:6px; background:green; margin-right:6px;"></span> Medium (400-600 W/m²)</div>
            <div><span style="display:inline-block; width:24px; height:6px; background:yellow; margin-right:6px;"></span> Medium-High (600-800 W/m²)</div>
            <div><span style="display:inline-block; width:24px; height:6px; background:orange; margin-right:6px;"></span> High (800-1000 W/m²)</div>
            <div><span style="display:inline-block; width:24px; height:6px; background:red; margin-right:6px;"></span> Very High (>1000 W/m²)</div>
            <div style="font-size:10px; color:#666; margin-top:6px; border-top:1px solid #ccc; padding-top:6px;">Wind Power Density at 100m height</div>
        '''
        
        # Add legend
        left_legend = create_collapsible_legend(
            position='left',
            title='Wind Power Density Legend',
            content=wind_legend_content,
            width=250
        )
        
        # Add the legend and control script
        m.get_root().html.add_child(folium.Element(add_legend_control_script()))
        m.get_root().html.add_child(folium.Element(left_legend))
    else:
        print("Warning: Could not load wind power density data")
    
    # Add layer control
    folium.LayerControl(
        collapsed=True,
        position='topright',
        autoZIndex=True
    ).add_to(m)
    
    return m

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


def asset_analysis(force_recompute=False):
    """
    Analyzes potential overlaps between GEM data and existing infrastructure data.
    Creates both an Excel report and a map showing overlapping assets.
    Groups comparisons by power technology type to identify likely duplicates.
    """
    import time
    start_time = time.time()
    
    print("\n=== Starting Asset Analysis ===")
    
    # Read GEM data
    print("\nReading GEM datasets...")
    gem_datasets = [
        read_coal_plant_data(force_recompute=force_recompute),
        read_coal_terminal_data(force_recompute=force_recompute),
        read_wind_power_data(force_recompute=force_recompute),
        read_oil_gas_plant_data(force_recompute=force_recompute),
        read_lng_terminal_data(force_recompute=force_recompute),
        read_hydropower_data(force_recompute=force_recompute),
        read_solar_power_data(force_recompute=force_recompute)
    ]
    
    print("\nProcessing GEM datasets...")
    # Combine all GEM datasets with proper handling of empty/NA columns
    gem_dfs = [df for df in gem_datasets if not df.empty]
    if not gem_dfs:
        print("No valid GEM data found")
        return None
    
    print(f"Found {len(gem_dfs)} non-empty GEM datasets")
    
    # Ensure all dataframes have the same columns before concatenation
    required_columns = ['name', 'type', 'latitude', 'longitude']
    print("\nStandardizing GEM dataset columns...")
    for i, df in enumerate(gem_dfs, 1):
        print(f"Processing dataset {i}/{len(gem_dfs)}...")
        # Add missing required columns with NA values
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
        # Drop any columns that aren't in the required set
        df = df[required_columns]
        # Ensure consistent dtypes
        df['name'] = df['name'].astype(str)
        df['type'] = df['type'].astype(str)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    print("\nConcatenating GEM datasets...")
    # Now concatenate the cleaned dataframes
    gem_df = pd.concat(gem_dfs, ignore_index=True)
    
    # Drop any rows with NA in critical columns
    initial_rows = len(gem_df)
    gem_df = gem_df.dropna(subset=['latitude', 'longitude'])
    print(f"Removed {initial_rows - len(gem_df)} rows with missing coordinates")
    print(f"Final GEM dataset size: {len(gem_df)} rows")
    
    print("\nReading infrastructure data...")
    # Read existing infrastructure data
    infra_df = read_openinfra_existing_generators(force_recompute=force_recompute)
    print(f"Infrastructure dataset size: {len(infra_df)} rows")
    
    # Define technology type mapping between GEM and OpenInfra datasets
    technology_mapping = {
        'Coal Plant': ['coal', 'thermal'],
        'Coal Terminal': ['coal terminal', 'port'],
        'Wind Power': ['wind', 'wind power'],
        'Oil/Gas Plant - Oil': ['oil', 'diesel', 'fuel oil'],
        'Oil/Gas Plant - Gas': ['gas', 'natural gas', 'lng'],
        'LNG Terminal': ['lng terminal', 'gas terminal'],
        'Hydropower Plant': ['hydro', 'hydropower', 'hydroelectric'],
        'Solar Farm': ['solar', 'pv', 'photovoltaic']
    }
    
    # Function to standardize technology types
    def get_standard_type(tech_str):
        tech_str = str(tech_str).lower()
        for standard_type, variations in technology_mapping.items():
            if any(var in tech_str for var in variations):
                return standard_type
        return 'Other'
    
    # Add standardized technology type columns
    gem_df['standard_type'] = gem_df['type'].apply(get_standard_type)
    infra_df['standard_type'] = infra_df['source'].apply(get_standard_type)
    
    # Function to calculate distance between two points in km
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    # Find overlapping assets (within 2km of each other)
    DISTANCE_THRESHOLD = 2.0  # kilometers
    overlaps = []
    
    print("\nSearching for overlapping assets by technology type...")
    
    # Get unique technology types
    tech_types = set(gem_df['standard_type'].unique()) & set(infra_df['standard_type'].unique())
    tech_types = [t for t in tech_types if t != 'Other']  # Exclude 'Other' category
    
    for tech_type in tech_types:
        print(f"\nAnalyzing {tech_type}...")
        
        # Filter datasets by technology type
        gem_tech = gem_df[gem_df['standard_type'] == tech_type]
        infra_tech = infra_df[infra_df['standard_type'] == tech_type]
        
        total_comparisons = len(gem_tech) * len(infra_tech)
        print(f"Comparing {len(gem_tech)} GEM assets with {len(infra_tech)} infrastructure assets")
        
        comparison_count = 0
        last_progress = time.time()
        
        for idx1, gem_row in gem_tech.iterrows():
            for idx2, infra_row in infra_tech.iterrows():
                comparison_count += 1
                
                # Show progress every 5% or if 5 seconds have passed
                if comparison_count % max(1, total_comparisons // 20) == 0 or time.time() - last_progress >= 5:
                    progress = (comparison_count / total_comparisons) * 100
                    elapsed = time.time() - start_time
                    print(f"Progress: {progress:.1f}% ({comparison_count:,}/{total_comparisons:,} comparisons) - Elapsed time: {elapsed:.1f}s")
                    last_progress = time.time()
                
                distance = haversine_distance(
                    gem_row['latitude'], gem_row['longitude'],
                    infra_row['latitude'], infra_row['longitude']
                )
                
                if distance <= DISTANCE_THRESHOLD:
                    overlaps.append({
                        'technology_type': tech_type,
                        'gem_name': gem_row['name'],
                        'gem_type': gem_row['type'],
                        'gem_lat': gem_row['latitude'],
                        'gem_lon': gem_row['longitude'],
                        'infra_source': infra_row['source'],
                        'infra_capacity': infra_row['output_mw'],
                        'infra_lat': infra_row['latitude'],
                        'infra_lon': infra_row['longitude'],
                        'distance_km': distance
                    })
    
    print(f"\nCompleted overlap search in {time.time() - start_time:.1f} seconds")
    
    if not overlaps:
        print("No overlapping assets found")
        return None
    
    print("\nCreating output files...")
    # Create DataFrame of overlaps
    overlaps_df = pd.DataFrame(overlaps)
    
    # Sort by technology type and distance
    overlaps_df = overlaps_df.sort_values(['technology_type', 'distance_km'])
    
    # Save to Excel
    output_file = RESULTS_DIR / 'asset_overlaps.xlsx'
    overlaps_df.to_excel(output_file, index=False)
    print(f"Saved overlap analysis to {output_file}")
    
    print("\nCreating visualization map...")
    # Create map
    m = folium.Map(
        location=[gem_df['latitude'].mean(), gem_df['longitude'].mean()],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    
    # Add plugins
    for Plugin in (Fullscreen, MiniMap, MeasureControl):
        Plugin().add_to(m)
    
    # Create feature groups for different technology types
    tech_groups = {}
    for tech_type in tech_types:
        tech_groups[tech_type] = {
            'gem': folium.FeatureGroup(name=f"{tech_type} - GEM", show=True),
            'infra': folium.FeatureGroup(name=f"{tech_type} - Infrastructure", show=True),
            'lines': folium.FeatureGroup(name=f"{tech_type} - Connections", show=True)
        }
    
    print("Adding markers to map...")
    # Plot overlapping assets
    for i, overlap in enumerate(overlaps, 1):
        if i % 100 == 0:  # Progress for large numbers of overlaps
            print(f"Added {i}/{len(overlaps)} markers to map")
        
        tech_type = overlap['technology_type']
        groups = tech_groups[tech_type]
        
        # Plot GEM asset
        folium.CircleMarker(
            location=[overlap['gem_lat'], overlap['gem_lon']],
            radius=8,
            color='red',
            fill=True,
            popup=f"GEM: {overlap['gem_name']} ({overlap['gem_type']})",
            tooltip=f"GEM Asset: {overlap['gem_name']}"
        ).add_to(groups['gem'])
        
        # Plot infrastructure asset
        folium.CircleMarker(
            location=[overlap['infra_lat'], overlap['infra_lon']],
            radius=8,
            color='blue',
            fill=True,
            popup=f"Infrastructure: {overlap['infra_source']} ({overlap['infra_capacity']} MW)",
            tooltip=f"Infrastructure: {overlap['infra_source']}"
        ).add_to(groups['infra'])
        
        # Draw line between overlapping assets
        folium.PolyLine(
            locations=[
                [overlap['gem_lat'], overlap['gem_lon']],
                [overlap['infra_lat'], overlap['infra_lon']]
            ],
            color='yellow',
            weight=2,
            opacity=0.8,
            popup=f"{tech_type}: Distance = {overlap['distance_km']:.2f} km"
        ).add_to(groups['lines'])
    
    # Add all feature groups to map
    for groups in tech_groups.values():
        for fg in groups.values():
            fg.add_to(m)
    
    # Add layer control
    folium.LayerControl(
        collapsed=True,
        position='topright',
        autoZIndex=True
    ).add_to(m)
    
    # Create legends
    overlap_legend_content = '''
        <div><span style="background:red; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> GEM Asset</div>
        <div><span style="background:blue; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Infrastructure Asset</div>
        <div><span style="background:yellow; width:12px; height:4px; display:inline-block;"></span> Overlap Connection</div>
        <div style="font-size:10px; color:#666; margin-top:6px;">
            Shows assets within {DISTANCE_THRESHOLD}km of each other
        </div>
        <div style="font-size:10px; color:#666; margin-top:6px;">
            Grouped by technology type
        </div>
    '''
    
    layer_control_content = '''
        <div style="margin-bottom:8px;"><strong>Layer Controls</strong></div>
        <div style="font-size:10px; color:#666;">
            Toggle layers to show/hide different asset types and their connections
        </div>
    '''
    
    # Add legends
    left_legend = create_collapsible_legend(
        position='left',
        title='Asset Overlap Legend',
        content=overlap_legend_content,
        width=250
    )
    
    right_legend = create_collapsible_legend(
        position='right',
        title='Layer Controls',
        content=layer_control_content,
        width=250
    )
    
    m.get_root().html.add_child(folium.Element(add_legend_control_script()))
    m.get_root().html.add_child(folium.Element(left_legend))
    m.get_root().html.add_child(folium.Element(right_legend))
    
    # Save map
    output_map = RESULTS_DIR / 'overlap_map.html'
    m.save(output_map)
    print(f"Saved overlap map to {output_map}")
    
    # Print summary statistics
    print("\nOverlap Analysis Summary:")
    print("\nBy Technology Type:")
    tech_summary = overlaps_df.groupby('technology_type').agg({
        'distance_km': ['count', 'mean', 'min', 'max']
    }).round(2)
    print(tech_summary)
    
    print("\nOverall Statistics:")
    print(f"Total overlapping pairs found: {len(overlaps)}")
    print(f"Average distance between overlapping assets: {overlaps_df['distance_km'].mean():.2f} km")
    print(f"Minimum distance: {overlaps_df['distance_km'].min():.2f} km")
    print(f"Maximum distance: {overlaps_df['distance_km'].max():.2f} km")
    
    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time:.1f} seconds")
    
    return m

def main():
    """
    Main function to create maps based on command line arguments.
    """
    if __name__ == '__main__':
        start_time = time.time()
        logging.info("Starting map generation process")
        
        parser = argparse.ArgumentParser(description="Vietnam Power Maps")
        parser.add_argument('--map', choices=[
            'power', 'substation', 'transmission', 'integrated', 'openinfra_existing_generator', 
            'wind', 'new_transformer', 'gem', 'overlap', 'powerline', 'all'
        ], default='power',
            help="Type of map to generate")
        parser.add_argument('--force-recompute', action='store_true',
            help="Force recomputation of data (ignore cache)")
        parser.add_argument('--clear-cache', action='store_true',
            help="Clear all cached data")
        
        args = parser.parse_args()
        logging.info(f"Generating map type: {args.map}")
        
                # Clear cache if requested
        if args.clear_cache:
            clear_cache()
            logging.info("Cache cleared")
        try:

            if args.map in ['integrated', 'all']:
                logging.info("Generating integrated map")
                m = create_integrated_map(force_recompute=args.force_recompute)
                if m:
                    save_and_open_map(m, INTEGRATED_MAP)

            if args.map in ['wind', 'all']:
                logging.info("Generating wind power density map")
                m = create_wind_power_density_map()
                if m:
                    save_and_open_map(m, WIND_MAP)

            if args.map in ['gem', 'all']:
                logging.info("Generating GEM map")
                m = create_gem_map(force_recompute=args.force_recompute)
                if m:
                    save_and_open_map(m, GEM_MAP)

            if args.map in ['overlap', 'all']:
                logging.info("Generating asset overlap map")
                m = asset_analysis(force_recompute=args.force_recompute)
                if m:
                    save_and_open_map(m, RESULTS_DIR / 'overlap_map.html')
            
            total_time = time.time() - start_time
            logging.info(f"Map generation completed in {total_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error creating map: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()