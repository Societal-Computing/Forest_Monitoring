

import pandas as pd
import json
import rasterio
import os
import requests
import ast
from shapely.geometry import shape, Polygon, MultiPolygon, mapping, MultiLineString, MultiPoint, LineString, Point
from shapely.ops import transform
import pyproj
import numpy as np
from scipy import stats
from shapely import wkt
import geopandas as gpd
from bs4 import BeautifulSoup
from io import BytesIO
import time
from random import uniform
import zipfile


##################################################
### Plant_Planet_Meta_Data_preprocessing.ipynb ###
##################################################

# Removing sites without geometries (e.g. )
def remove_not_geom(site):
    try:
        site_data = ast.literal_eval(site)
        if isinstance(site_data, list) and all("geometry" in item for item in site_data):
            return site
        else:
            return None
    except:
        return None


# defining a function to count the total number of polygons as each row of our data has project information which in other cases are multi-polygons
def count_all_polygons(all_geom_data):
    try:
        all_geom_data = ast.literal_eval(all_geom_data)
        count_polygons = 0
        polygon_areas = []
        polygon_geometries = []

        project = pyproj.Transformer.from_crs(
            pyproj.CRS('EPSG:4326'),
            pyproj.CRS('EPSG:6933'), # This to project the data to a plane for easy and accurate area calculation,later we will change back the crs to 4326 before final saving 
            always_xy=True).transform

        for geom_data in all_geom_data:
            if geom_data is None or "geometry" not in geom_data:
                continue
            geometry = shape(geom_data["geometry"])
            if isinstance(geometry, Polygon):
                count_polygons += 1
                projected_polygon = transform(project, geometry)
                # calculating polygon area in metres square
                polygon_areas.append(projected_polygon.area)
                polygon_geometries.append(geometry)
            elif isinstance(geometry, MultiPolygon):
                count_polygons += len(geometry.geoms)
                for polygon in geometry.geoms:
                    projected_polygon = transform(project, polygon)
                    polygon_areas.append(projected_polygon.area)
                    polygon_geometries.append(polygon)

        return count_polygons, polygon_areas, polygon_geometries
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


def split_multipolygon(geometry_list):
    result = []
    for geometry in geometry_list:
        if isinstance(geometry, MultiPolygon):
            result.extend(list(geometry))
        elif isinstance(geometry, Polygon):
            result.append(geometry)
    return result


# converting 3D polygons to 2D
def convert_3d_to_2d(polygon):
    if polygon.has_z:
        
        new_coords = [(x, y) for x, y, z in polygon.exterior.coords]
        return Polygon(new_coords)
    return polygon



#######################################
### Tree_Nation-meta_data_pre.ipynb ###
#######################################

def swap_lat_lon(coords):
    """Swap the positions of latitude and longitude in a list of coordinates."""
    return [(lon, lat) for lat, lon in coords]

def convert_to_list(data):
    if data is None:
        return None
    if isinstance(data, list):
        return swap_lat_lon(data)
    if isinstance(data, str):
        try:
            if data.endswith(","):
                data = data[:-1]
            coords = ast.literal_eval(data)
            return swap_lat_lon(coords)
        except (SyntaxError, ValueError):
            print(f"Failed to convert string to list: {data}")
            return None
    return None

#################################################
### open_forest_projests_Data_filtering.ipynb ###
#################################################

def remove_trailing_zeros(s):
  
    if s.startswith('list(c(') and s.endswith(')'):
       
        s = s[7:-1]
  
    list_of_strings = s.split(',')
   
    list_of_strings = [s.strip().lstrip('c(').rstrip(')') for s in list_of_strings]
    
    list_of_floats = list(map(float, list_of_strings))
   
    while list_of_floats and list_of_floats[-1] == 0.0:
        list_of_floats.pop()
   
    return ', '.join(map(str, list_of_floats))


####################################
### extracting_verra_sites.ipynb ###
####################################


def kmz_to_kml(content):
    try:
        with zipfile.ZipFile(BytesIO(content)) as kmz:
            with kmz.open(next(f for f in kmz.namelist() if f.endswith('.kml'))) as kml_file:
                return kml_file.read().decode('utf-8')
    except zipfile.BadZipFile:
        return content.decode('utf-8')

# Function to fetch and parse KML file
def fetch_kml(uri):
    response = requests.get(uri)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to download KML file from {uri}")
        return None

# Function to parse KML and convert to geometries
def parse_kml(content):
    soup = BeautifulSoup(content, 'xml')
    geometries = []

    # Find all MultiGeometry elements, which can host multiple Polygons
    for multi_geom in soup.find_all('MultiGeometry'):
        polygons = []
        linestrings = []
        points = []
        for polygon in multi_geom.find_all('Polygon'):
            if polygon.find('coordinates').string == None:
                polygons.append(Polygon())
                continue
            coords = polygon.find('coordinates').string.strip().split()
            c_points = [tuple(map(float, c.split(','))) for c in coords]
            polygons.append(Polygon(c_points))
        for linestring in multi_geom.find_all('LineString'):
            if linestring.find('coordinates').string == None:
                linestrings.append(LineString())
                continue
            coords = linestring.find('coordinates').string.strip().split()
            c_points = [tuple(map(float, c.split(','))) for c in coords]
            if len(c_points) != 1:
                linestrings.append(LineString(c_points))
            else:
                geometries.append(Point(c_points))
        for point in multi_geom.find_all('Point'):
            if point.find('coordinates').string == None:
                points.append(Point())
                continue
            coords = point.find('coordinates').string.strip().split()
            c_points = [tuple(map(float, c.split(','))) for c in coords]
            points.append(Point(c_points))
        if polygons:
            geometries.append(MultiPolygon(polygons))
        if linestrings:
            geometries.append(MultiLineString(linestrings))
        if points:
            geometries.append(MultiPoint(points))

    # Also check for geometries that are not part of MultiGeometry
    # Check for Polygons
    for polygon in soup.find_all('Polygon'):
        if polygon.parent.name != 'MultiGeometry':
            if polygon.find('coordinates').string == None:
                geometries.append(Polygon())
                continue
            coords = polygon.find('coordinates').string.strip().split()
            c_points = [tuple(map(float, c.split(','))) for c in coords]
            geometries.append(Polygon(c_points))

    # Check for LineStrings
    for linestring in soup.find_all('LineString'):
        if linestring.parent.name != 'MultiGeometry':
            if linestring.find('coordinates').string == None:
                geometries.append(LineString())
                continue
            coords = linestring.find('coordinates').string.strip().split()
            c_points = [tuple(map(float, c.split(','))) for c in coords]
            if len(c_points) != 1:
                geometries.append(LineString(c_points))
            else:
                geometries.append(Point(c_points))

    # Check for Points
    for point in soup.find_all('Point'):
        if point.parent.name != 'MultiGeometry':
            if point.find('coordinates').string == None:
                geometries.append(Point())
                continue
            coords = point.find('coordinates').string.strip().split()
            c_points = [tuple(map(float, c.split(','))) for c in coords]
            geometries.append(Point(c_points))

    return geometries

# Main processing
def process_kml_uris(kml_uris):
    all_geometries = []
    for uri in kml_uris:
        uri_content = fetch_kml(uri)
        if uri_content:
            kml_content = kmz_to_kml(uri_content)
            geometries = parse_kml(kml_content)
            all_geometries.extend(geometries)
    return all_geometries

####################################
### weather_variables_extraction.ipynb ###
####################################
# Extracting Year of interest from our polygon "Planting date feature"
def extract_year(date_str):
    if pd.isna(date_str):
        return np.nan
    try:
      
        date_parsed = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(date_parsed):
            return date_parsed.year
    except ValueError:
        pass  
    
   
    if isinstance(date_str, str) and date_str.isdigit() and len(date_str) == 4:
        return int(date_str)
    
    return np.nan
##################################################
### biomass_data_extraction.ipynb ###
# extracting raster values without transforming CRS inside
def extract_raster_values(raster_path, transformed_centroids):
    with rasterio.open(raster_path) as src:
        values = []
        for point in transformed_centroids.geometry:
            row, col = src.index(point.x, point.y)
            if (0 <= row < src.height) and (0 <= col < src.width):
                value = src.read(1)[row, col]
                values.append(value)
            else:
                values.append(np.nan)
        return values

#  processing GeoDataFrame in chunks with CRS transformation before extraction
def process_in_chunks(gdf, chunk_size, raster_crs, raster_dir, period):
    results_df = pd.DataFrame()
    
    for start in range(0, len(gdf), chunk_size):
        end = start + chunk_size
        chunk = gdf.iloc[start:end]
        
        if chunk.crs.is_geographic:
            chunk = chunk.to_crs("EPSG:3395")
        chunk['centroid'] = chunk.geometry.centroid
        
        # Transform centroids to match the raster CRS
        transformed_centroids = chunk['centroid'].to_crs(raster_crs)
        
        biomass_values = [0] * len(chunk)
        
        for raster_file in os.listdir(raster_dir):
            if raster_file.endswith(".tif"):
                raster_path = os.path.join(raster_dir, raster_file)
                values = extract_raster_values(raster_path, transformed_centroids)
                biomass_values = [x + y if not np.isnan(y) else x for x, y in zip(biomass_values, values)]
        
        chunk[f'Biomass_change_{period}'] = biomass_values
        chunk.drop(columns=['centroid'], inplace=True)
        
        results_df = pd.concat([results_df, chunk])
    
    return results_df.to_crs(gdf.crs)
