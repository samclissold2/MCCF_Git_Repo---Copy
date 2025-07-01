import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, MeasureControl, HeatMap
import webbrowser
import logging
import time
import argparse
import os
from config import (PROJECTS_MAP, SUBSTATION_MAP, TRANSMISSION_MAP, INTEGRATED_MAP, WIND_MAP, GEM_MAP, RESULTS_DIR)
from map_utils import (clear_cache, load_from_cache, save_to_cache, read_coal_plant_data, read_coal_terminal_data, read_wind_power_data, read_oil_gas_plant_data, read_lng_terminal_data, read_hydropower_data, read_solar_power_data, read_infrastructure_data, read_and_clean_power_data, read_solar_irradiance_points, read_transmission_data, read_powerline_data, get_power_lines, get_power_towers, cache_polylines, read_substation_data, get_source_color, get_voltage_color, read_tif_data, read_planned_substation_data)

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
    feature_groups = {}
    marker_count = 0

    for asset_type in combined_df['type'].unique():
        type_df = combined_df[combined_df['type'] == asset_type]
        logging.info(f"Processing {len(type_df)} assets of type: {asset_type}")
        
        # Create feature group for this technology type if it doesn't exist
        if asset_type not in feature_groups:
            feature_groups[asset_type] = folium.FeatureGroup(name=asset_type, show=True).add_to(m)
        
        fg = feature_groups[asset_type]
        
        # Add markers for each asset of this type
        for _, row in type_df.iterrows():
            marker_count += 1
            if marker_count % 100 == 0:
                logging.info(f"Added {marker_count} markers to map")
            
            # Get base color for asset type - map GEM types to integrated map colors
            base_type = row['type'].split(' - ')[0] if ' - ' in row['type'] else row['type']
            base_color = tech_colors.get(base_type, '#888888')  # Default gray for unknown types
            
            # Create marker with technology color only, status shown in tooltip
            folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=max(4, (float(row['capacity']) ** 0.5) * 0.5),
                    color='#222',  # No border
                    opacity=0.3,
                    fill=True,
                    fill_color=base_color,  # Fill color based on technology type only
                    fill_opacity=0.7,
                    weight=1,
                    tooltip=f"{row['name']} ({row['type']})<br>Status: {row['status']}<br>Capacity: {row.get('capacity', 'N/A')} MW"
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
    tech_legend = create_collapsible_legend(
        position='left',
        title='Technology Types',
        content=tech_legend_content,
        width=250
    )
    
    # Add legend and control script
    m.get_root().html.add_child(folium.Element(add_legend_control_script()))
    m.get_root().html.add_child(folium.Element(tech_legend))
    
    processing_time = time.time() - start_time
    logging.info(f"GEM map creation completed in {processing_time:.2f} seconds")
    return m

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
    # solar_df = read_solar_irradiance_points(force_recompute=force_recompute)
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
        transformer_fg = folium.FeatureGroup(name="Planned Substations", show=False).add_to(m)
        
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
    
    wind_legend_content = '''
        <div><span style="display:inline-block; width:24px; height:6px; background:blue; margin-right:6px;"></span> Low (0-200 W/m²)</div>
        <div><span style="display:inline-block; width:24px; height:6px; background:cyan; margin-right:6px;"></span> Low-Medium (200-400 W/m²)</div>
        <div><span style="display:inline-block; width:24px; height:6px; background:green; margin-right:6px;"></span> Medium (400-600 W/m²)</div>
        <div><span style="display:inline-block; width:24px; height:6px; background:yellow; margin-right:6px;"></span> Medium-High (600-800 W/m²)</div>
        <div><span style="display:inline-block; width:24px; height:6px; background:orange; margin-right:6px;"></span> High (800-1000 W/m²)</div>
        <div><span style="display:inline-block; width:24px; height:6px; background:red; margin-right:6px;"></span> Very High (>1000 W/m²)</div>
        <div style="font-size:10px; color:#666; margin-top:6px; border-top:1px solid #ccc; padding-top:6px;">Wind Power Density at 100m height</div>
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
    
    # # Add wind power density heatmap to integrated map with custom toggle
    # # Note: HeatMap plugin is not compatible with FeatureGroups, so we add it directly
    # # and provide a custom toggle button to preserve LayerControl functionality
    # wind_layer = create_wind_power_density_layer(force_recompute=force_recompute)
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

def create_wind_power_density_layer(force_recompute=False):
    """
    Creates a wind power density layer from the TIF file.
    Returns a HeatMap layer for folium.
    """
    # Check cache first
    if not force_recompute:
        cached_data = load_from_cache('create_wind_power_density_layer')
        if cached_data is not None:
            return cached_data
    
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
        show=True  # Always visible when added directly to map
    )
    
    # Save to cache
    save_to_cache('create_wind_power_density_layer', heatmap_layer)
    
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
    wind_layer = create_wind_power_density_layer(force_recompute=False)
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
    start_time = time.time()
    logging.info("Starting map generation process")
    
    parser = argparse.ArgumentParser(description="Vietnam Power Maps")
    parser.add_argument('--map', choices=[
         'integrated', 
        'wind',  'gem', 'overlap',  'all'
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
