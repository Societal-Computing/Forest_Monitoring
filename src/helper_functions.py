

import pandas as pd
import json
import rasterio
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import ast
from shapely.geometry import shape,Polygon, MultiPolygon, GeometryCollection, mapping, MultiLineString, MultiPoint, LineString, Point
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
import ee
import geemap
import os
import gc


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

    # Checking for Points
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
### gee_columns_generation.ipynb ###
####################################

# Calculate land cover class shares
def calculate_area(feature, classCodes, maskedLandCover):
    total_area = ee.Number(0)
    
    def calculate_class_area(class_code, total):
        class_area = maskedLandCover.eq(ee.Number(class_code)) \
            .multiply(ee.Image.pixelArea()) \
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=30,
                maxPixels=1e9
            ).get('remapped')

        area_value = ee.Number(class_area).divide(1e6)  # Converting the area to km^2
        return ee.Number(total).add(area_value)

    total_area = ee.List(classCodes).iterate(calculate_class_area, total_area)
    return feature.set('cover_area_2020', total_area)

# Calculate built area shares
def calculate_built_area(feature, builtImage):
    built_area = builtImage.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=feature.geometry(),
        scale=10,
        maxPixels=1e9
    ).get('built_characteristics')

    built_area = ee.Number(built_area).max(0)
    return feature.set('built_area_2018', built_area)

#  Calculating road length for each feature (polygon)
def calculate_road_length(polygon, roads):
    # Finding all roads that intersect the polygon
    intersecting_roads = roads.filterBounds(polygon.geometry())

    #  clipping road geometry and calculating its length within the polygon
    def clip_and_calculate_length(road):
        road_geom = road.geometry()
        polygon_geom = polygon.geometry()
        clipped = road_geom.intersection(polygon_geom, ee.ErrorMargin(1))
        return ee.Feature(clipped).set('length', clipped.length())

    # Mapping the clipping and length calculation over the intersecting roads
    clipped_roads = intersecting_roads.map(clip_and_calculate_length)

    # Summing the total road length in the polygon
    road_length_sum = clipped_roads.reduceColumns(
        reducer=ee.Reducer.sum(),
        selectors=['length']
    ).get('sum')

    # Counting the number of intersecting roads
    intersecting_roads_count = intersecting_roads.size()

    # ting total road length (in km) and road count as properties on the polygon
    return polygon.set({
        'total_road_length_km': ee.Number(road_length_sum).divide(1000),
        'intersecting_roads_count': intersecting_roads_count
    })

def calculate_forest_loss(feature, gfc2017):
    # Selecting the forest loss band and calculate loss area
    loss_image = gfc2017.select(['loss'])
    loss_area_image = loss_image.multiply(ee.Image.pixelArea())
    loss_year = gfc2017.select(['lossyear'])

    # Calculating forest loss area by year within the feature geometry
    loss_by_year = loss_area_image.addBands(loss_year).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1),
        geometry=feature.geometry(),
        scale=30,
        maxPixels=1e9
    )
 
    return feature.set(loss_by_year)

#  Calculating the mean elevation and slope
def calculate_elevation_and_slope(feature, elevation, slope):
    elevation_mean = elevation.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=feature.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('elevation')

    slope_mean = slope.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=feature.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('slope')

    return feature.set({
        'mean_elevation': elevation_mean,
        'mean_slope': slope_mean
    })



# Defining the function to extract Polygon and MultiPolygon geometries from a GeometryCollection
def extract_polygons(geometry):
    if isinstance(geometry, GeometryCollection):
        polygons = [geom for geom in geometry.geoms if isinstance(geom, (Polygon, MultiPolygon))]
        if polygons:
            return polygons[0]  
    return geometry


def mask_s2clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

# Define the process_month function
def process_month(month, chunks, S2):
    monthly_results = []

    for chunk_index, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_index + 1}/{len(chunks)} for month {month}...")

        # Converting the current GeoDataFrame chunk to GeoJSON format
        gdf_json_chunk = chunk.__geo_interface__

        try:
            # Converting the GeoJSON chunk to an Earth Engine FeatureCollection
            fc_chunk = geemap.geojson_to_ee(gdf_json_chunk)
        except Exception as e:
            print(f"Error converting chunk {chunk_index + 1} to Earth Engine FeatureCollection: {e}")
            continue

        # Filtering Sentinel-2 images by the specified month and calculate NDVI
        monthly_s2 = (S2
                      .filter(ee.Filter.calendarRange(month, month, 'month'))
                      .map(mask_s2clouds)
                      .map(lambda image: image.addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI'))))

        # Reducing the collection to a single NDVI image for the month
        monthly_ndvi = monthly_s2.select('NDVI').mean().rename('NDVI')

        retry_count = 0
        max_retries = 5
        backoff_time = 2

        while retry_count < max_retries:
            try:
                # Calculating the mean NDVI for each feature (polygon) in the chunk
                mean_ndvi = monthly_ndvi.reduceRegions(
                    collection=fc_chunk,
                    reducer=ee.Reducer.mean(),
                    scale=30
                )

                # Converting the results to a DataFrame
                if mean_ndvi:
                    temp_chunk_df = pd.DataFrame([feature['properties'] for feature in mean_ndvi.getInfo()['features']])
                else:
                    temp_chunk_df = pd.DataFrame()

                temp_chunk_df['month'] = month
                monthly_results.append(temp_chunk_df)
                break
            except Exception as e:
                print(f"Error processing chunk {chunk_index + 1}: {e}")
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff

    # Concatenating all chunk results for the month
    return pd.concat(monthly_results, ignore_index=True)

# Shadow Index
def calculate_shadow_index(image):
    shadow_index = image.expression(
        '(10000 - blue) * (10000 - green) * (10000 - red)', {
            'blue': image.select('B2'),
            'green': image.select('B3'),
            'red': image.select('B4')
        }).pow(1 / 3).divide(10000).rename('shadow_index')
    return image.addBands(shadow_index)
# SAVI
def calculate_savi(image):
    savi = image.expression(
        '(NIR - RED) / (NIR + RED + L) * (1 + L)', {
            'NIR': image.select('B8'),  
            'RED': image.select('B4'),  
            'L': 0.5  
        }).rename('savi_index')
    return image.addBands(savi)


# def get_shadow_index_for_month(feature, S2):

#     feature = ee.Feature(feature)
    
#     target_year = ee.Number(feature.get('planting_date_reported'))
#     month = ee.Number(feature.get('month'))
    
#     # Filtering S2 ImageCollection based on feature properties
#     monthly_s2 = (
#         S2.filter(ee.Filter.calendarRange(target_year, target_year, 'year'))
#           .filter(ee.Filter.calendarRange(month, month, 'month'))
#           .filterBounds(feature.geometry())
#           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
#           .map(mask_s2clouds)
#           .map(calculate_shadow_index)
#     )
       
#     # Using ee.Algorithms.If to handle empty ImageCollection
#     def compute_shadow_index():
#         # Calculating the median shadow index
#         monthly_si = monthly_s2.select('shadow_index').median().rename('shadow_index')
        
#         # Reducing the shadow index over the feature's geometry
#         shadow_index_value = monthly_si.reduceRegion(
#             reducer=ee.Reducer.median(),
#             geometry=feature.geometry(),
#             scale=10,
#             maxPixels=1e13
#         ).get('shadow_index')
        
#         return feature.set({'shadow_index': shadow_index_value})
    
#     # Returning feature with shadow index or null
#     return ee.Algorithms.If(
#         monthly_s2.size().eq(0),
#         feature.set({'shadow_index': None}),
#         compute_shadow_index()
#     )
def get_savi_for_month(feature, S2):
    feature = ee.Feature(feature)
    
    target_year = ee.Number(feature.get('planting_date_reported'))
    month = ee.Number(feature.get('month'))
    
    # Filtering S2 ImageCollection based on feature properties
    monthly_s2 = (
        S2.filter(ee.Filter.calendarRange(target_year, target_year, 'year'))
          .filter(ee.Filter.calendarRange(month, month, 'month'))
          .filterBounds(feature.geometry())
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .map(mask_s2clouds)
          .map(calculate_savi)  
    )
       
   
    def compute_savi():
        
        monthly_savi = monthly_s2.select('savi_index').median().rename('savi_index')
       
        savi_value = monthly_savi.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=feature.geometry(),
            scale=10,
            maxPixels=1e13
        ).get('savi_index')
        
        return feature.set({'savi_index': savi_value})
    
  
    return ee.Algorithms.If(
        monthly_s2.size().eq(0),
        feature.set({'savi_index': None}),
        compute_savi()
    )

# NDVI
def get_ndvi_for_month(feature, S2):

    feature = ee.Feature(feature)
    
    target_year = ee.Number(feature.get('planting_date_reported'))
    month = ee.Number(feature.get('month'))
    
    # Filtering S2 ImageCollection based on feature properties
    monthly_s2 = (
        S2.filter(ee.Filter.calendarRange(target_year, target_year, 'year'))
          .filter(ee.Filter.calendarRange(month, month, 'month'))
          .filterBounds(feature.geometry())
          .map(mask_s2clouds)
          .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('ndvi'))
    )
       
    # Using ee.Algorithms.If to handle empty ImageCollection
    def compute_ndvi():
        # Calculating mean ndvi
        monthly_si = monthly_s2.select('ndvi').mean().rename('ndvi')
        
        # Reducing the NDVI over the feature's geometry
        ndvi_value = monthly_si.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=10,
            maxPixels=1e13
        ).get('ndvi')
        
        return feature.set({'ndvi': ndvi_value})
    
    # Returning feature with NDVI or null
    return ee.Algorithms.If(
        monthly_s2.size().eq(0),
        feature.set({'ndvi': None}),
        compute_ndvi()
    )
# NDRE
def get_ndre_for_month(feature, S2):
    feature = ee.Feature(feature)
    
    target_year = ee.Number(feature.get('planting_date_reported'))
    month = ee.Number(feature.get('month'))
    
   
    monthly_s2 = (
        S2.filter(ee.Filter.calendarRange(target_year, target_year, 'year'))
          .filter(ee.Filter.calendarRange(month, month, 'month'))
          .filterBounds(feature.geometry())
          .map(mask_s2clouds)
          .map(lambda img: img.normalizedDifference(['B8', 'B5']).rename('ndre'))  # NDRE calculation
    )
       
  
    def compute_ndre():
       
        monthly_ndre = monthly_s2.select('ndre').mean().rename('ndre')
       
        ndre_value = monthly_ndre.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=10,
            maxPixels=1e13
        ).get('ndre')
        
        return feature.set({'ndre': ndre_value})
   
    return ee.Algorithms.If(
        monthly_s2.size().eq(0),
        feature.set({'ndre': None}),
        compute_ndre()
    )


##########################################
### weather_variables_extraction.ipynb ###
##########################################
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


#####################################
### biomass_data_extraction.ipynb ###
#####################################


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
#####################################
### nested_polygons_filtering.ipynb ###
#####################################
def clean_data(value):
    if isinstance(value, dict):
        return str(value)
    if isinstance(value, list):
        return str(value)
    return value if not pd.isna(value) else None
############
### Weathe_data_extraction.ipynb ###


def process_climate_variable(gdf, tif_folder, output_folder, combined_output_path, variable_name):
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif")]
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No .tif files found in the directory for {variable_name}.")

    tif_files_by_year = {}
    for tif_file in tif_files:
        year_month = tif_file.split("_")[-1].split(".")[0]  
        year, month = int(year_month.split("-")[0]), int(year_month.split("-")[1])
        if year not in tif_files_by_year:
            tif_files_by_year[year] = {}
        tif_files_by_year[year][month] = tif_file

    chunk_size = 200  

    for i in range(0, len(gdf), chunk_size):
        gdf_chunk = gdf.iloc[i:i + chunk_size].copy()
        variable_by_years_after_planting = {}

        for idx, polygon in gdf_chunk.iterrows():
            planting_year = polygon['planting_year']
            if pd.isna(planting_year):
                continue

            variable_by_years_after_planting[idx] = {
                'planting_year': {'sum': 0, 'count': 0},
                'year_1': {'sum': 0, 'count': 0},
                'year_2': {'sum': 0, 'count': 0},
                'year_5': {'sum': 0, 'count': 0}
            }

            centroid = polygon['geometry'].centroid
            centroid_point = [(centroid.x, centroid.y)]

            for year_offset in [0, 1, 2, 5]:
                current_year = planting_year + year_offset
                if current_year in tif_files_by_year:
                    for month in range(1, 13):
                        if month in tif_files_by_year[current_year]:
                            tif_file = tif_files_by_year[current_year][month]
                            tif_path = os.path.join(tif_folder, tif_file)

                            try:
                                with rasterio.open(tif_path) as src:
                                    for val in src.sample(centroid_point):
                                        valid_data = val[0]
                                        if not np.isnan(valid_data):
                                            key = f'year_{year_offset}' if year_offset > 0 else 'planting_year'
                                            variable_by_years_after_planting[idx][key]['sum'] += valid_data
                                            variable_by_years_after_planting[idx][key]['count'] += 1
                            except Exception as e:
                                print(f"Error processing {tif_file}: {e}")

        for idx, data in variable_by_years_after_planting.items():
            for key in data:
                sum_value = data[key]['sum']
                count_value = data[key]['count']
                avg_value = sum_value / count_value if count_value > 0 else np.nan
                gdf_chunk.at[idx, f"avg_{variable_name}_{key}"] = avg_value

        output_geojson_path = os.path.join(output_folder, f"{variable_name}_chunk_{i}.geojson")
        gdf_chunk.to_file(output_geojson_path, driver="GeoJSON")

        del gdf_chunk, variable_by_years_after_planting
        gc.collect()

        print(f"Processed and saved chunk {i} for {variable_name} to {output_geojson_path}")

    combined_gdf = gpd.GeoDataFrame()

    for i in range(0, len(gdf), chunk_size):
        chunk_path = os.path.join(output_folder, f"{variable_name}_chunk_{i}.geojson")
        chunk_gdf = gpd.read_file(chunk_path)
        combined_gdf = pd.concat([combined_gdf, chunk_gdf], ignore_index=True)

    combined_gdf.to_file(combined_output_path, driver="GeoJSON")
    print(f"Combined all chunks into {combined_output_path} for {variable_name}")

    for i in range(0, len(gdf), chunk_size):
        chunk_path = os.path.join(output_folder, f"{variable_name}_chunk_{i}.geojson")
        os.remove(chunk_path)
        print(f"Deleted chunk file: {chunk_path} for {variable_name}")
#####################################
###checking_polygons_axact_admin_area.ipynb 
#####################################
def geometries_almost_equal(geom1, geom2, tolerance=0.001):
    """
    Compare two geometries for near-equality within a tolerance.
    """
    return geom1.equals(geom2) or geom1.almost_equals(geom2, tolerance)

def process_chunk_gadm(chunk, gadm_data, layer_name):
    """
    Process a chunk of reforestation data to find exact matches with GADM geometries.
    """
    sindex = gadm_data.sindex
    exact_matches = []
    potential_matches_count = 0

    for idx, row in chunk.iterrows():
        possible_matches_index = list(sindex.intersection(row.geometry.bounds))
        if not possible_matches_index:
            continue

        potential_matches_count += 1
        possible_matches = gadm_data.iloc[possible_matches_index]

        for _, admin_row in possible_matches.iterrows():
            if geometries_almost_equal(row.geometry, admin_row.geometry):
                exact_matches.append(row)
                break

    print(f"Potential matches found in chunk for layer {layer_name}: {potential_matches_count}")
    print(f"Exact matches found in chunk for layer {layer_name}: {len(exact_matches)}")

    if exact_matches:
        return gpd.GeoDataFrame(exact_matches, crs=chunk.crs)
    else:
        return gpd.GeoDataFrame(columns=chunk.columns, crs=chunk.crs)
#####################################
###restor.ipynb
#####################################
# Extracting more data from the Restor website using the ids
def fetch_data(id):
    url = f"https://restor2-prod-1-api.restor.eco/sites/3/{id}"
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504], 
        allowed_methods=["HEAD", "GET", "OPTIONS"]  
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        response = session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)  # Timeout set to 10 seconds
        response.raise_for_status()

        if response.headers.get('content-type') == 'application/json':
            return response.json()
        else:
            print(f"API did not return JSON data for id {id}")
            return None

    except Exception as e:
        print(f"Error fetching data for id {id}: {e}")
        return None