import logging

from pyhigh import get_elevation_batch
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

from rail_dog.configs.library import SECTIONS, get_section_metadata

from snappy_utils.general import clean_string


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


def find_hits(pt_geom: Point, segment_data, max_distance=100):
    tree, _ = find_hits
    hits = tree.query_nearest(pt_geom, max_distance=max_distance, all_matches=False)
    if hits:
        return hits
    return None
 

def align_to_common_cols(gdfs):
    cols = []
    for key, (_, gdf) in gdfs.items():
        cols.append(set(gdf.columns))

    common_cols = cols[0].copy()
    for s in cols[1:]:
        common_cols.intersection_update(s)

    for key, (_, gdf) in gdfs.items():
        gdf_cols = set(gdf.columns)
        logging.debug(f"In the dataset {key}, dropping unique columns:")
        for c in gdf_cols.difference(common_cols):
            logging.debug(f"  {c}")

        gdf = gdf[list(common_cols)]


def apply_corrections(gdf: gpd.GeoDataFrame, THOMAS_END: float, column_name_mapping: dict = {}):
    """
    Apply corrections to rail data using fully vectorized operations.

    :param gdf: Input gdf to modify in place
    :type gdf: gpd.GeoDataFrame
    :param THOMAS_END: Chainage threshold for Thomas line split
    :type THOMAS_END: float
    :param column_name_mapping: Optional column renaming dictionary
    :type column_name_mapping: dict
    """
    import numpy as np

    # Rename columns
    gdf.rename(columns={c: clean_string(c, custom_mapping=column_name_mapping) for c in gdf.columns}, inplace=True)

    # Adhoc string replacements (no apply needed)
    gdf["id"] = (gdf["id"]
                 .str.replace("Jones to CCK_MLE", "Jones to CCK_MLB", regex=False)
                 .str.replace("Jones to CCK", "Jones", regex=False))

    # Calculate chainage
    gdf["chainage_km"] = gdf["major"] + gdf["minor"] / 1000

    # Pre-split IDs once (split all underscores to get region and line_type)
    id_split = gdf["id"].str.split("_", n=1, expand=True)
    region = id_split[0]
    gdf["asset_name"] = id_split[1]

    # Populate the line_region
    gdf["line_region"] = region
    invalid_regions = ~region.isin(SECTIONS)
    if invalid_regions.any():
        for bad_id in gdf.loc[invalid_regions, "id"].unique():
            logging.warning(f"Unidentified region for id: {bad_id}")

    # Populate the line_code
    # conditions = [
    #     # Thomas MLW special case - early section
    #     (region == "Thomas") & (line_type == "MLW") & (gdf["chainage_km"] <= THOMAS_END),
    #     # Thomas MLW special case - late section
    #     (region == "Thomas") & (line_type == "MLW") & (gdf["chainage_km"] > THOMAS_END),
    #     # Firetail correction
    #     (region == "Firetail") & (line_type == "ELB"),
    #     # Future correction
    #     (region == "Future") & (line_type == "SLB"),
    #     # Main line identifiers (standard mapping)
    #     line_type.isin(MAIN_LINE_IDENTIFIERS.keys())
    # ]

    # choices = [
    #     "TL",  # Thomas early section
    #     "ML",  # Thomas late section
    #     "SL",  # Firetail
    #     "EL",  # Future
    #     line_type.map(MAIN_LINE_IDENTIFIERS)  # Regular mapping
    # ]

    # Apply conditions
    # gdf["line_code"] = np.select(conditions, choices, default="OTH")
    
    
    gdf[["line", "line_code", "line_class"]] = gdf["asset_name"].apply(get_section_metadata)
    
    # Handle collection date if available in the data. Format: YYYYMMDD## where first 8 digits are the date
    if "run_id" in gdf.columns:
        # Extract first 8 digits and convert to datetime
        gdf["collection_date"] = pd.to_datetime(
            gdf["run_id"].astype(str).str[:8],
            format='%Y%m%d',
            errors='coerce'
        )
    else:
        gdf["collection_date"] = None
