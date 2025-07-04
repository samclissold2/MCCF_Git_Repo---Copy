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
    PROJECTS_MAP,
    INTEGRATED_MAP,
    WIND_MAP,
    RESULTS_DIR,
    GEM_MAP,
)

# Attempt relative import first (works when executed with "python -m MCCF.PDP8.create_map")
# but gracefully fall back to absolute import paths when the file is run as a
# standalone script (e.g. "python MCCF/PDP8/create_map.py").
try:
    from . import map_utils as utils  # Package-relative import
except ImportError:  # pragma: no cover – stand-alone execution fallback
    import importlib
    import sys

    # Ensure the project root (two levels up) is on sys.path so that the
    # absolute package import below succeeds.
    _current_dir = os.path.dirname(os.path.abspath(__file__))        # …/MCCF/PDP8
    _project_root = os.path.dirname(os.path.dirname(_current_dir))   # …/ (repo root)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    try:
        # Try absolute import via package hierarchy
        utils = importlib.import_module("MCCF.PDP8.map_utils")
    except ModuleNotFoundError:
        # Last-ditch: import the module by filename (same directory)
        utils = importlib.import_module("map_utils")

def create_gem_map(force_recompute=False):
    """
    Creates a map combining all GEM data sources with status-based bucketing.
    For oil/gas plants, also buckets by fuel type.
    """
    start_time = time.time()
    logging.info("Starting GEM map creation")
    
    # Define color scheme for technology types only
    tech_colors = {
        'Solar': '#FF0000',            # red
        'Hydro': '#003366',            # dark blue
        'LNG-Fired Gas': '#808080',    # middle grey
        'Domestic Gas-Fired': '#333333', # dark grey
        'Coal': '#000000',       # black
        'Coal Terminal': '#8B4513',    # sienna
        'Wind': '#87CEEB',       # light blue (same as Onshore)
        'LNG Terminal': '#D3D3D3',     # light grey (same as LNG-Fired Gas)
    }
    
    # Read all datasets
    logging.info("Reading GEM datasets...")
    datasets = [
        utils.read_coal_plant_data(force_recompute=force_recompute),
        utils.read_coal_terminal_data(force_recompute=force_recompute),
        utils.read_wind_power_data(force_recompute=force_recompute),
        utils.read_oil_gas_plant_data(force_recompute=force_recompute),
        utils.read_lng_terminal_data(force_recompute=force_recompute),
        utils.read_hydropower_data(force_recompute=force_recompute),
        utils.read_solar_power_data(force_recompute=force_recompute)
    ]
    
    # Combine all non-empty datasets
    valid_dfs = [df for df in datasets if not df.empty]
    if not valid_dfs:
        logging.error("No valid GEM data found")
        return None
    
    logging.info(f"Found {len(valid_dfs)} non-empty GEM datasets")
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    type_mapping = {
        'Coal Plant': 'Coal',
        'Wind Power': 'Wind',
        'Oil/Gas Plant - fossil gas: natural gas': 'Domestic Gas-Fired',
        'Oil/Gas Plant - fossil gas: LNG': 'LNG-Fired Gas',
        'Oil/Gas Plant - fossil gas: natural gas, fossil liquids: fuel oil': 'Natural Gas/Fuel Oil',
        'LNG Terminal': 'LNG Terminal',
        'Hydropower Plant': 'Hydro',
        'Solar Farm': 'Solar'
    }
    
    combined_df['type'] = combined_df['type'].map(type_mapping).fillna(combined_df['type'])
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
    
    # Process markers by technology type only
    logging.info("Processing markers by technology type")
    combined_df["base_type"] = combined_df["type"].str.split(" - ").str[0]
    combined_df["color"] = combined_df["base_type"].map(tech_colors).fillna("#888888")
    combined_df["radius"] = np.maximum(4, np.sqrt(combined_df["capacity"].fillna(0).astype(float)) * 0.5)

    feature_groups = {}
    marker_count = 0

    for asset_type, type_df in combined_df.groupby('type'):
        logging.info(f"Processing {len(type_df)} assets of type: {asset_type}")
        
        # Create feature group for this technology type if it doesn't exist
        if asset_type not in feature_groups:
            feature_groups[asset_type] = folium.FeatureGroup(name=asset_type, show=True).add_to(m)
        
        fg = feature_groups[asset_type]
        
        # Add markers for each asset of this type
        for row in type_df.itertuples(index=False):
            marker_count += 1
            if marker_count % 100 == 0:
                logging.info(f"Added {marker_count} markers to map")
            
            # Get base color for asset type - map GEM types to integrated map colors
            
            # Create marker with technology color only, status shown in tooltip
            folium.CircleMarker(
                    location=[row.latitude, row.longitude],
                    radius=row.radius,
                    color='#222',  # No border
                    opacity=0.3,
                    fill=True,
                    fill_color=row.color,  # Fill color based on technology type only
                    fill_opacity=0.7,
                    weight=1,
                    tooltip=f"{row.name} ({row.type})<br>Status: {row.status}<br>Capacity: {row.capacity if pd.notna(row.capacity) else 'N/A'} MW"
                ).add_to(fg)
    
    logging.info(f"Total markers added: {marker_count}")
    
    # Add layer control
    folium.LayerControl(
        collapsed=True,
        position='topright',
        autoZIndex=True
    ).add_to(m)
    
    # Create legend content for technology types only
    tech_legend_content = ''.join(
        f'<div><span style="background:{color}; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></span> {tech}</div>'
        for tech, color in tech_colors.items()
    )
    
    # Create technology legend
    tech_legend = utils.create_collapsible_legend(
        position='left',
        title='Technology Types',
        content=tech_legend_content,
        width=250
    )
    
    # Add legend and control script
    m.get_root().html.add_child(folium.Element(utils.add_legend_control_script()))
    m.get_root().html.add_child(folium.Element(tech_legend))
    
    processing_time = time.time() - start_time
    logging.info(f"GEM map creation completed in {processing_time:.2f} seconds")
    return m

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
    wind_layer = utils.create_wind_power_density_layer(force_recompute=False)
    if wind_layer is not None:
        # For the dedicated wind map, we want the layer to be visible by default
        wind_layer.show = True
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
        left_legend = utils.create_collapsible_legend(
            position='left',
            title='Wind Power Density Legend',
            content=wind_legend_content,
            width=250
        )
        
        # Add the legend and control script
        m.get_root().html.add_child(folium.Element(utils.add_legend_control_script()))
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

def create_comprehensive_map(force_recompute: bool = False):
    """
    Everything from *integrated* map (but no solar-irradiance or wind-heatmap)
    plus all GEM technology layers, with the LayerControl split into
        • Existing Assets   • Planned Assets
    """

    m = folium.Map(
        location=[16.0, 106.0],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )

    # ---------- EXISTING  -------------------------------------------------
    existing_group = "Existing Assets"

    # Transmission lines
    # Add power lines
    gdf = utils.get_power_lines()
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
    features = utils.cache_polylines(gdf, cache_file='powerline_polylines.geojson', eps=0.0025, min_samples=3, force_recompute=force_recompute)
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
            tooltip=f"{voltage} Operating Transmission Line"
        ).add_to(transmission_fg)
    
    # Add substations
    sdf = utils.read_substation_data()

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
                location=[row.latitude, row.longitude],
                icon=folium.DivIcon(html=f'<div style="font-size:10px; color:{color}; font-weight:bold;">×</div>'),
                tooltip=f"Operating Transformer - ({row['voltage_cat']})"
            ).add_to(sub_fg)


    # GEM datasets (all are existing)
    gem_df = pd.concat(
        [
            utils.read_coal_plant_data(force_recompute),
            utils.read_coal_terminal_data(force_recompute),
            utils.read_wind_power_data(force_recompute),
            utils.read_oil_gas_plant_data(force_recompute),
            utils.read_lng_terminal_data(force_recompute),
            utils.read_hydropower_data(force_recompute),
            utils.read_solar_power_data(force_recompute),
        ],
        ignore_index=True,
    )
    
    type_mapping = {
        'Coal Plant': 'Coal',
        'Wind Power': 'Wind',
        'Oil/Gas Plant - fossil gas: natural gas': "Domestic Gas-Fired",
        'Oil/Gas Plant - fossil gas: LNG': "LNG-Fired Gas",
        'Oil/Gas Plant - fossil gas: natural gas, fossil liquids: fuel oil': 'Domestic Gas-fired/Fuel Oil',
        'Hydropower Plant': 'Hydro',
        'Solar Farm': 'Solar'
    }

    gem_df['type'] = gem_df['type'].map(type_mapping).fillna(gem_df['type'])

    tech_colour = {
        "Solar": "#FF0000",
        "Hydro": "#003366",
        "Wind": "#87CEEB",
        "Coal": "#000000",
        "Domestic Gas-Fired": "#333333",
        "LNG-Fired Gas": "#808080",
        "LNG Terminal": "#D3D3D3",
        "Coal Terminal": "#8B4513",
    }
    # Scale GEM marker sizes by capacity (larger capacity → larger radius)
    min_r, max_r = 4, 12  # radius bounds

    for tech in gem_df["type"].unique():
        fg = folium.FeatureGroup(name=tech, show=False).add_to(m)
        colour = tech_colour.get(tech, '#888888')
        tech_df = gem_df[gem_df.type == tech]
        if tech_df.empty:
            continue
        max_cap = tech_df.capacity.max(skipna=True)
        for _, row in tech_df.iterrows():
            cap = row.capacity if pd.notna(row.capacity) else 0
            rad = min_r if max_cap == 0 else min_r + (cap / max_cap) * (max_r - min_r)
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=rad,
                color='#222',  # faint outline
                # The 'opacity' parameter controls the transparency of the marker's border.
                opacity=0.3,
                fill=True,
                fill_color=colour,
                # The 'fill_opacity' parameter controls the transparency of the marker's fill color.
                fill_opacity=0.4,
                tooltip=f"{row.iloc[0]} - {row.capacity}MW - {row.status}"
            ).add_to(fg)

    # ---------- PLANNED  --------------------------------------------------
    planned_group = "Planned Assets"

    # PDP-8 power projects
    pwr_df, name_col = utils.read_and_clean_power_data(force_recompute)
    p_cols = {
        "Solar": "#FF0000",
        "Hydro": "#003366",
        "Onshore": "#87CEEB",
        "LNG-Fired Gas": "#808080",
        "Domestic Gas-Fired": "#333333",
    }
    for tech in pwr_df.tech.unique():
        for period in pwr_df[pwr_df.tech == tech]["period"].unique():
            fg = folium.FeatureGroup(
                name=f"{tech} {period}",
                group=planned_group,
                show=False,
            ).add_to(m)
            colour = p_cols.get(tech, "#888888")
            # Pre-compute max MW for scaling
            period_df = pwr_df[(pwr_df.tech == tech) & (pwr_df.period == period)]
            if period_df.empty:
                continue
            max_mw = period_df.mw.max(skipna=True)
            for _, r in period_df.iterrows():
                mw = r.mw if pd.notna(r.mw) else 0
                rad = min_r if max_mw == 0 else min_r + (mw / max_mw) * (max_r - min_r)
                folium.CircleMarker(
                    [r.lat, r.lon],
                    radius=rad,
                    color="#222",
                    opacity=0.3,
                    fill=True,
                    fill_color=colour,
                    fill_opacity=1.0,
                    tooltip=f"{r[name_col]} - Expected Completion in ({period}) — {r.mw:.0f}MW",
                ).add_to(fg)

    # Planned substations
    # Add planned transformers
    transformer_df = utils.read_planned_substation_data()
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
        transformer_fg = folium.FeatureGroup(name="Planned Substations", show=False).add_to(m)
        
        for _, row in transformer_df.iterrows():
            # Extract numeric voltage value for color coding directly from voltage_cat
            try:
                numeric_voltage = float(str(row['voltage_cat']).replace('kV', '').replace('KV', ''))
            except:
                numeric_voltage = 'unknown'
            
            # Use the same color coding system as substation map
            color = utils.get_voltage_color(numeric_voltage)
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=folium.DivIcon(html=f'<div style="font-size:10px; color:{color}; font-weight:bold;">×</div>'),
                tooltip=f"Planned Transformer ({row['voltage_cat']}) - {row['name']}"
            ).add_to(transformer_fg)
    
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
    
    existing_legend_content = ''.join(
        f'<div><span style="background: {colour}; opacity: 0.3; width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 4px;"></span> {tech}</div>'
        for tech, colour in tech_colour.items()
    )

    # Create custom legends stacked vertically on the left side with good spacing
    powerline_legend_id = 'legend-powerline'
    substation_legend_id = 'legend-substation'
    project_legend_id = 'legend-project'
    existing_legend_id = 'legend-existing'

    powerline_legend = f'''
    <div id="{powerline_legend_id}" style="
        position: fixed; 
        bottom: 20px; 
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
        bottom: 20px; 
        left: 250px; 
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
        bottom: 20px; 
        left: 450px; 
        width: 200px; 
        z-index:9999; 
        background: white; 
        border:2px solid grey; 
        border-radius:6px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        font-size:12px; 
        padding: 10px;">
        <div onclick="toggleLegend('{project_legend_id}')" style="cursor:pointer;font-weight:bold;user-select:none;margin-bottom:5px;padding:5px;background:#f8f8f8;border-radius:4px;">
            PDP8 Planned Projects Legend <span id="{project_legend_id}-arrow" style="float:right;">▶</span>
        </div>
        <div id="{project_legend_id}-content" style="display:none; margin-top:8px;">
            {project_legend_content}
        </div>
    </div>
    '''
    

    existing_legend = f"""
    <div id="{existing_legend_id}" style="
        position: fixed;
        bottom: 20px;
        left: 650px;
        width: 200px;
        z-index: 9999;
        background: white;
        border: 2px solid grey;
        border-radius: 6px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        font-size: 12px;
        padding: 10px;
    ">
        <div onclick="toggleLegend('{existing_legend_id}')" style="
            cursor: pointer;
            font-weight: bold;
            user-select: none;
            margin-bottom: 5px;
            padding: 5px;
            background: #f8f8f8;
            border-radius: 4px;
        ">
            Existing Assets Legend <span id="{existing_legend_id}-arrow" style="float: right;">▶</span>
        </div>
        <div id="{existing_legend_id}-content" style="display: none; margin-top: 8px;">
            {existing_legend_content}
        </div>
    </div>
    """


    # Add the legend control script and legends to map
    m.get_root().html.add_child(folium.Element(utils.add_legend_control_script()))
    m.get_root().html.add_child(folium.Element(powerline_legend))
    m.get_root().html.add_child(folium.Element(substation_legend))
    m.get_root().html.add_child(folium.Element(project_legend))
    m.get_root().html.add_child(folium.Element(existing_legend))
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    for P in (Fullscreen, MeasureControl):
        P().add_to(m)

    return m

def create_new_transmission_data_map():
    """
    Create a Folium map visualizing new transmission data from read_new_transmission_data.
    """
    try:
        # Get the transmission data
        pdp8_transmission_data = utils.read_new_transmission_data(force_recompute=True)
        if pdp8_transmission_data is None or pdp8_transmission_data.empty:
            print("No transmission data available.")
            return None

        # Create a base map centered on Vietnam
        m = folium.Map(
            location=[16.0, 106.0],  # Center of Vietnam
            zoom_start=6,
            tiles="CartoDB Positron"
        )

        # Voltage color mapping (add more as needed)
        voltage_colors = {
            500: 'red',
            220: 'orange',

        }

        # Add a FeatureGroup for new transmission points
        fg = folium.FeatureGroup(name="New Transmission Points", show=True).add_to(m)

        # Plot each transmission point as a custom DivIcon marker
        for _, row in pdp8_transmission_data.iterrows():
            kv = int(row['kV'])
            color = voltage_colors.get(kv, 'black')
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:14px; color:{color}; font-weight:bold;">×</div>'
                ),
                tooltip=f"{row['Name']}<br>{row['kV']} kV"
            ).add_to(fg)

        # Add a legend for voltage levels
        legend_html = '''
        <div id="legend-transmission" style="
            position: fixed; 
            bottom: 20px; 
            left: 50px; 
            width: 200px; 
            z-index:9999; 
            background: white; 
            border:2px solid grey; 
            border-radius:6px; 
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
            font-size:12px; 
            padding: 10px;">
            <div style="font-weight:bold; margin-bottom:5px;">Transmission Voltage Legend</div>
            <div><span style="background:red; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></span> 500kV</div>
            <div><span style="background:orange; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></span> 220kV</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl(collapsed=True, position="topright").add_to(m)
        return m

    except Exception as e:
        print(f"An error occurred during the map creation: {e}")
        return None

def create_population_density_map(force_recompute: bool = False):
    """Create a simple Folium heat-map of Vietnam 2020 population density.

    The raster is sampled via utils.read_vnm_pd_2020_1km() (which delegates to
    a cached read_tif_data helper).  The function returns *None* if the data
    cannot be loaded so that the caller can gracefully skip map creation.
    """

    # Fetch (and possibly sample) the population-density points
    try:
        # 2 % sampling gives ~20 000 pts for the 1 km grid → good performance
        df = utils.read_vnm_pd_2020_1km(sample_rate=0.02)  # type: ignore[arg-type]
    except Exception as e:
        logging.error("Failed to read population-density raster", exc_info=True)
        return None

    if df is None or df.empty:
        logging.error("Population-density DataFrame is empty – skipping map")
        return None

    # Base map centred on Vietnam
    m = folium.Map(
        location=[16.0, 106.0],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )

    # ── Increase visual contrast ────────────────────────────────────────
    # Cap extreme high values (99th percentile) to avoid skew, then
    # normalise and apply a square-root stretch so that mid-range densities
    # are easier to discern on the colour scale.
    dens = df["population_density"].astype(float)
    cap  = dens.quantile(0.99)           # ignore extreme outliers
    norm = np.clip(dens / cap, 0, 1) ** 0.5   # √-stretch for contrast

    heat_data = np.column_stack((
        df["latitude"].values,
        df["longitude"].values,
        norm.values,
    )).tolist()

    HeatMap(
        heat_data,
        name="Population Density (2020, 1 km)",
        min_opacity=0.2,
        max_opacity=0.9,
        radius=9,
        blur=10,
        gradient={
            0.0: "#0000ff",   # blue
            0.15: "#00ffff", # cyan
            0.3: "#00ff00",  # lime
            0.45: "#ffff00", # yellow
            0.6: "#ff9900",  # orange
            0.75: "#ff0000", # red
            0.9: "#7f0000",  # dark-red
        },
        show=True,
    ).add_to(m)

    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    for Plugin in (Fullscreen, MiniMap, MeasureControl):
        Plugin().add_to(m)

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
            'integrated',
            'comprehensive',      # ← NEW
            'wind',
            'gem',
            'all',
            'new_transmission',    # ← ADD THIS LINE
            'population_density'    # ← ADD THIS LINE
        ], default='comprehensive',
            help="Type of map to generate")
        parser.add_argument('--force-recompute', action='store_true',
            help="Force recomputation of data (ignore cache)")
        parser.add_argument('--clear-cache', action='store_true',
            help="Clear all cached data")


        args = parser.parse_args()
        logging.info(f"Generating map type: {args.map}")
        
        # Clear cache if requested
        if args.clear_cache:
            utils.clear_cache()
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

            if args.map in ['comprehensive', 'all']:
                logging.info("Generating comprehensive map")
                m = create_comprehensive_map(force_recompute=args.force_recompute)
                if m:
                    save_and_open_map(m, RESULTS_DIR / 'vn_comprehensive_map.html')

            if args.map == 'new_transmission':  # ← ADD THIS BLOCK
                logging.info("Generating new transmission map")
                m = create_new_transmission_data_map()
                if m:
                    save_and_open_map(m, RESULTS_DIR / 'vn_new_transmission_map.html')

            if args.map == 'population_density':  # ← ADD THIS BLOCK
                logging.info("Generating population density map")
                m = create_population_density_map(force_recompute=args.force_recompute)
                if m:
                    save_and_open_map(m, RESULTS_DIR / 'vn_population_density_map.html')

            total_time = time.time() - start_time
            logging.info(f"Map generation completed in {total_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error creating map: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
