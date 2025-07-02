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
            tooltip=f"{voltage} Line (Cluster {feat['properties']['cluster']})"
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
                tooltip=f"Substation ({row['voltage_cat']})"
            ).add_to(sub_fg)

    # # Add solar irradiance heatmap to integrated map, toggled off by default
    # solar_df = utils.read_solar_irradiance_points(force_recompute=force_recompute)
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
    df, name_col = utils.read_and_clean_power_data()
    tech_colors = {
        'Solar': '#FF0000',            # red
        'Hydro': '#003366',            # dark blue
        'Onshore': '#87CEEB',          # light blue
        'LNG-Fired Gas': '##808080',    # light grey
        'Domestic Gas-Fired': '#333333', # dark grey
        'Pumped-Storage': '#4682B4',   # medium blue
        'Nuclear': '#800080',          # purple
        'Biomass': '#228B22',          # green
        'Waste-To-Energy': '#8B6F22',  # dirty green/brown
        'Flexible': '#1A1A1A',         # very dark grey
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
                tooltip=f"Planned Transformer ({row['voltage_cat']}) - {row['name']} - {row['sheet_source']}"
            ).add_to(transformer_fg)
    
    # Add layer control
    folium.LayerControl(collapsed=True,
        position='topright',
        autoZIndex=True).add_to(m)
    
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
    m.get_root().html.add_child(folium.Element(utils.add_legend_control_script()))
    m.get_root().html.add_child(folium.Element(powerline_legend))
    m.get_root().html.add_child(folium.Element(substation_legend))
    m.get_root().html.add_child(folium.Element(project_legend))
    
    # # Add wind power density heatmap to integrated map with custom toggle
    # # Note: HeatMap plugin is not compatible with FeatureGroups, so we add it directly
    # # and provide a custom toggle button to preserve LayerControl functionality
    # wind_layer = utils.create_wind_power_density_layer(force_recompute=force_recompute)
    # if wind_layer is not None:
    #     # Add wind layer directly to map (not as FeatureGroup to preserve LayerControl)
    #     # Make it subtle so it doesn't interfere with other layers
    #     wind_layer.min_opacity = 0.1  # Very low opacity
    #     wind_layer.max_opacity = 0.3  # Low max opacity
    #     wind_layer.radius = 15        # Smaller radius
    #     wind_layer.blur = 20          # More blur for subtlety
    #     m.add_child(wind_layer)
        
    #     # Add custom wind toggle button
    #     wind_toggle_html = '''
    #     <div id="wind-toggle" style="
    #         position: fixed; 
    #         top: 120px; 
    #         right: 50px; 
    #         z-index: 1000; 
    #         background: white; 
    #         border: 2px solid grey; 
    #         border-radius: 6px; 
    #         padding: 8px; 
    #         font-size: 12px; 
    #         box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
    #         <div style="font-weight: bold; margin-bottom: 5px;">Wind Power Density</div>
    #         <button onclick="toggleWindLayer()" id="wind-toggle-btn" style="
    #             background: #4CAF50; 
    #             color: white; 
    #             border: none; 
    #             padding: 5px 10px; 
    #             border-radius: 3px; 
    #             cursor: pointer; 
    #             font-size: 11px;">
    #             Hide Layer
    #         </button>
    #     </div>
    #     <script>
    #     function toggleWindLayer() {
    #         const windLayer = document.querySelector('.folium-heatmap-layer');
    #         const btn = document.getElementById('wind-toggle-btn');
    #         if (windLayer) {
    #             if (windLayer.style.display === 'none') {
    #                 windLayer.style.display = 'block';
    #                 btn.textContent = 'Hide Layer';
    #                 btn.style.background = '#4CAF50';
    #             } else {
    #                 windLayer.style.display = 'none';
    #                 btn.textContent = 'Show Layer';
    #                 btn.style.background = '#f44336';
    #             }
    #         }
    #     }
    #     </script>
    #     '''
    #     m.get_root().html.add_child(folium.Element(wind_toggle_html))
    
    return m

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
            tooltip=f"{voltage} Line (Cluster {feat['properties']['cluster']})"
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
                tooltip=f"Substation ({row['voltage_cat']})"
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
    gem_max_cap = gem_df["capacity"].max(skipna=True)
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
                opacity=0.3,
                fill=True,
                fill_color=colour,
                fill_opacity=0.4,
                tooltip=f"{row.name} - {row.capacity} - {row.status}"
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
                    tooltip=f"{r[name_col]} ({period}) — {r.mw:.0f} MW",
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
                tooltip=f"Planned Transformer ({row['voltage_cat']}) - {row['name']} - {row['sheet_source']}"
            ).add_to(transformer_fg)
    
    # Add layer control
    folium.LayerControl(collapsed=True,
        position='topright',
        autoZIndex=True).add_to(m)
    
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
    m.get_root().html.add_child(folium.Element(utils.add_legend_control_script()))
    m.get_root().html.add_child(folium.Element(powerline_legend))
    m.get_root().html.add_child(folium.Element(substation_legend))
    m.get_root().html.add_child(folium.Element(project_legend))
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    for P in (Fullscreen, MeasureControl):
        P().add_to(m)

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
            'all'
        ], default='integrated',
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

            total_time = time.time() - start_time
            logging.info(f"Map generation completed in {total_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error creating map: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
