import geopandas as gpd
from config import VNM_GPKG

def read_gpkg_layer(layer_name):
    """Read a specific layer from the GeoPackage as a GeoDataFrame."""
    return gpd.read_file(VNM_GPKG, layer=layer_name)

def get_substations():
    """Return the substations (point) layer as a GeoDataFrame."""
    return read_gpkg_layer('power_substation_point')

def get_power_lines():
    """Return the power line layer as a GeoDataFrame."""
    return read_gpkg_layer('power_line')

def get_power_towers():
    """Return the power tower layer as a GeoDataFrame."""
    return read_gpkg_layer('power_tower')