

import pandas as pd
import json
import requests
import ast
from shapely.geometry import shape, Polygon, MultiPolygon,mapping
from shapely.ops import transform
import pyproj
import numpy as np
from scipy import stats
from shapely import wkt
import geopandas as gpd

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