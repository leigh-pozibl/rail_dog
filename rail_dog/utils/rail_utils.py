from pyhigh import get_elevation_batch
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


def add_elevation_to_gdf(gdf: gpd.GeoDataFrame, column_name: str = "elevation", continent: str = "Australia"):

    _gdf = gdf.to_crs(epsg=4326)

    if isinstance(_gdf.geometry[0], Point):
        coords = _gdf.geometry.coords[0][::-1]
    elif isinstance(_gdf.geometry[0], LineString):
        # for g in _gdf.geometry:
        #     print(g, g.centroid.coords[0][::1])
        coords = [g.centroid.coords[0][::-1] for g in _gdf.geometry]
    elif isinstance(_gdf.geometry[0], Polygon):
        coords = [g.centroid.coords[0][::-1] for g in _gdf.geometry]

    coords = [(c[0], c[1], continent) for c in coords]
    elevations = get_elevation_batch(coords)
    gdf[column_name] = elevations
