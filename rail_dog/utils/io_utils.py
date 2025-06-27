import os
import sys
import logging
import json
import toml
import yaml

import shapely
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
from pandas import concat
import duckdb

from rail_dog.configs.params import BaseConfiguration
from snappy_utils.params import Metadata, DBConnection, DBData, TABLES
from snappy_utils.io import upload_gdf, read_graph
from snappy_utils.db import query_table_sql, view_exists, create_view_from_boundary_intersection, insert_project_data


def load_config_file(file_path: str, root_path: str, metadata: Metadata, db_env: str = None):
    """
    Loads an input JSON/YAML configuration file and parses it into a BaseConfiguration dataclass instance.
    
    Args:
        file_path (str): The file path to the JSON file.

    Returns:
        A BaseConfiguration instance.
    """
    try:
        # Open and read the file
        file_extension = file_path.split('.')[-1].lower()

        with open(file_path, 'r') as file:
            if file_extension == 'json':
                configs_data = json.load(file)
            elif file_extension == 'toml':
                configs_data = toml.load(file)
            elif file_extension in ('yaml', 'yml'):
                configs_data = yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

        # Parse the JSON string using the previously defined function
        return parse_configs_to_dataclass(configs_data, root_path, metadata, db_env)
    
    except FileNotFoundError as e:
        logging.critical(f"File not found: {e}")
        raise
    except IOError as e:
        logging.critical(f"Error reading file: {e}")
        raise


def load_json_blob(json_blob: json, metadata: Metadata, db_env: str = None):
    """
    Loads an input json format blob and parses it into a BaseConfiguration dataclass instance.
    
    Args:
        json_input (json): The file path to the JSON file.

    Returns:
        A BaseConfiguration instance.
    """
    configs_data = json.loads(json_blob)
    return parse_configs_to_dataclass(configs_data, ".", metadata, db_env)


def parse_configs_to_dataclass(configs_data: str, root_path: str, metadata: Metadata, db_env: str = None) -> BaseConfiguration:
    """
    Parses a JSON string into a BaseConfiguration dataclass instance.  Then loads any
    GIS data.
    
    Args:
        configs_string (str): The input configs data.
        
    Returns:
        A BaseConfiguration instance.
    """
    def safe_load_wkb(val):
        return shapely.wkb.loads(val) if isinstance(val, (bytes, bytearray)) else None

    try:        
        # Validate the structure and convert it to a list of dataclass instances
        config = BaseConfiguration(**configs_data)
        config.root_path = root_path
        config.output_dir = os.path.join(root_path, config.output_dir)
        db_connection = None

        if db_env:
            db_connection = DBConnection(db_env)
            load_geospatial_data_from_db(config.base_data, root_path, metadata, db_connection)
            load_geospatial_data_from_db(config.controls, root_path, metadata, db_connection)

        else:
            # load all the gis data if we are using local file io
            load_geospatial_data_from_file(config.base_data, root_path)
            load_geospatial_data_from_file(config.controls, root_path)

        # read xlsx files
        if config.base_data.excel_files:
            xlsx = dict()
            for file, sheet_names in config.base_data.excel_files.items():
                file_path = os.path.join(root_path, file)
                sheets = list(sheet_names.keys())

                # excel_file = pd.ExcelFile(file_path)
                # sheet_names = excel_file.sheet_names
                # print(sheet_names)
                # exit(1)

                try:
                    xlsx = pd.read_excel(file_path, sheet_name=sheets)
                    logging.info(f"Loaded file: {file_path}")
                except Exception as e:
                    logging.critical(f"Error loading file {file_path} {e}")

            for sheet_name, sheet_alias in sheet_names.items():
                xlsx[sheet_alias] = xlsx.pop(sheet_name)
                # xlsx[sheet_alias].set_index("Chainage ID", inplace=True)

            config.base_data.update_dict("track_data", xlsx)
            
            # write the sheets to csv
            output_dir = os.path.dirname(file_path)
            for sheet_name, df in xlsx.items():
                csv_file = os.path.join(output_dir, f"{sheet_name}.csv")
                try:
                    df.to_csv(csv_file, index=False)
                    logging.info(f"Exported {sheet_name} to {csv_file}")
                except Exception as e:
                    logging.critical(f"Error exporting {sheet_name} to CSV: {e}")

        # read csv files
        if config.base_data.csv_files:
            csv = dict()
            for file in config.base_data.csv_files:
                file_path = os.path.join(root_path, file)
                name = os.path.splitext(os.path.basename(file))[0]
                try:
                    csv[name] = pd.read_csv(file_path)
                    # csv[name].set_index("Chainage ID", inplace=True)
                    logging.info(f"Loaded file: {file_path}")
                except Exception as e:
                    logging.critical(f"Error loading file {file_path} {e}")

            config.base_data.update_dict("track_data", csv)

        # read csv files
        con = None
        if config.base_data.rp_raw_data or config.base_data.tg_raw_data:
            duckdb_path = os.path.join(root_path, "ensco_data.duckdb")
            con = duckdb.connect(duckdb_path)
        elif config.base_data.ensco_db:
            duckdb_path = os.path.join(root_path, config.base_data.ensco_db)
            con = duckdb.connect(duckdb_path)
            results = con.execute("SELECT COUNT(*) FROM rp_data").fetchone()[0]
            logging.info(f"Loaded {results} rp_data records")
            results = con.execute("SELECT COUNT(*) FROM tg_data").fetchone()[0]
            logging.info(f"Loaded {results} tg_data records")

        if config.base_data.rp_raw_data:
            con.execute("DROP TABLE IF EXISTS rp_data")
            con.execute(f"""CREATE TABLE rp_data AS SELECT * FROM read_csv_auto('{config.base_data.rp_raw_data}/*.csv')""")
            results = con.execute("SELECT COUNT(*) FROM rp_data").fetchone()[0]
            logging.info(f"Loaded {results} rp_raw_data records")

        if config.base_data.tg_raw_data:
            con.execute("DROP TABLE IF EXISTS tg_data")
            con.execute(f"""CREATE TABLE tg_data AS SELECT * FROM read_csv_auto('{config.base_data.tg_raw_data}/*.csv')""")
            results = con.execute("SELECT COUNT(*) FROM tg_data").fetchone()[0]
            logging.info(f"Loaded {results} tg_raw_data records")

        if con:
            for table in ["rp_data", "tg_data"]:
                _data = con.execute(f"SELECT * FROM {table}").df()

                if "Longitude" in _data.columns:
                    _data["geometry"] = _data.apply(lambda x: Point(x["Longitude"], x["Latitude"]), axis=1)
                elif "geometry_wkb" in _data.columns:
                    _data['geometry'] = _data['geometry_wkb'].apply(shapely.wkt.loads)
                    _data.drop(columns=["geometry_wkb"], inplace=True)
                else:
                    logging.critical(f"No geometry column found in {table}")

                _data = gpd.GeoDataFrame(_data, geometry="geometry", crs="epsg:4326")
                config.base_data.set_data(table, _data)

        return config, db_connection

    except json.JSONDecodeError as e:
        logging.critical(f"Failed to decode JSON: {e}")
        raise
    except TypeError as e:
        logging.critical(f"TypeError during parsing: {e}")
        raise


def load_geospatial_data_from_file(
    data_container,
    root_path: str,
    named_layers: list = None
):
    """
    Loads all the geospatial files that are defined in the data_container. Supported formats
    include Shapefiles, GeoJSON, and other formats supported by GeoPandas/Fiona.

    If no named_layers, then all files are loaded.

    Data is attached to a data_container item.
    
    Args:
        data_container: A list of BaseData or ControlsData instances indicating the specific files to load.
        root_path: The root directory containing geospatial files.
        named_layers: To load a specific set of layers, format lke: [(layer_name: path_to_file), ...]
    """
    # Load each file into a GeoDataFrame
    if named_layers:
        items = named_layers
    else:
        items = data_container.active_layers.items()

    for layer_name, layer_files in items:
        if isinstance(layer_files, str):
            layer_files = [layer_files]

        elif isinstance(layer_files, DBData):
            raise RuntimeError("Trying to load a database source without specifying db-env input")
            sys.exit(1)

        layers_data = []
        for layer_file in layer_files:
            file_path = os.path.join(root_path, layer_file)
            try:
                gdf = gpd.read_file(file_path)
                gdf.geometry = gdf.geometry.set_precision(grid_size=1e-8)
                layers_data.append(gdf)
                combined = gpd.GeoDataFrame(concat(layers_data, ignore_index=True))
                data_container.set_data(layer_name, combined)
                logging.info(f"Loaded file: {file_path}")
            except Exception as e:
                logging.critical(f"Error loading file {file_path}: {e}")


def load_geospatial_data_from_db(
    data_container,
    root_path: str,
    metadata: Metadata,
    db: DBConnection
):
    """
    Pulls data specified in the data_config from database.

    Data is attached to a data_config item.
    
    Args:
        data_container: details of how to build a query for the data
        db_connection: A database connection instance 
    """
    # load the boundary data first as this may be used in subsequent queries
    boundary = data_container.active_layers.get("boundary")
    if boundary:
        if boundary.action == "push":
            load_geospatial_data_from_file(data_container, root_path, named_layers=[("boundary", boundary.source_file)])

            upload_gdf(
                data_container.boundary,
                db,
                "polygonfeature",
                label=boundary.label,
                metadata=metadata,
                expected_geom_types=TABLES["polygonfeature"]
            )

        elif boundary.action == "pull":
            sql = query_table_sql(boundary.table_name, label=None, metadata=metadata, filters=boundary.filters)
            gdf = gpd.read_postgis(sql, db.engine, geom_col="geometry")
            data_container.set_data("boundary", gdf)
            logging.info(f"Found {len(gdf)} {boundary.table_name} features for layer 'boundary'")

    for layer_name, query_data in data_container.active_layers.items():
        if layer_name == "boundary":
            continue

        layer_data = data_container.active_layers.get(layer_name)

        # where we just want to do standard file io
        if isinstance(query_data, str):
            load_geospatial_data_from_file(data_container, root_path, named_layers=[(layer_name, layer_data)])

        # optionally, read a file and then upload to the db
        elif query_data.action == "push":
            load_geospatial_data_from_file(data_container, root_path, named_layers=[(layer_name, layer_data.source_file)])

            table_name = data_container.table_lookup(layer_name)
            upload_gdf(
                data_container.layer_name,
                db,
                table_name,
                label=layer_name,
                metadata=metadata,
                expected_geom_types=TABLES[table_name]
            )

        # this is to pull data from an existing table
        elif query_data.action == "pull":
            
            # this is to pull data from openstreetmaps|openaddresses and then save a copy into a project
            if query_data.table_name in {"openstreetmaps", "openaddresses"}:
                view_name = f"view_{layer_name}_{metadata.project_id.replace('-','')}"
                table_name = data_container.table_lookup(layer_name)

                if not view_exists(db, view_name) or query_data.refresh:
                    logging.info(f"Creating view: {view_name}")
                    create_view_from_boundary_intersection(db, view_name, query_data)
                    logging.info(f"Inserting data into table: {table_name}")
                    insert_project_data(db, table_name, view_name, layer_name, metadata)

                sql = query_table_sql(table_name, label=layer_name, metadata=metadata)
                gdf = gpd.read_postgis(sql, db.engine, geom_col="geometry")
                data_container.set_data(layer_name, gdf)
                logging.info(f"Found {len(gdf)} {query_data.table_name} features for layer '{layer_name}'")

            # this is the case where we have just a source_id supplied - tells us all we need to know
            elif query_data.source_id:
                sql = query_table_sql(query_data.table_name, label=None, metadata=None, filters=query_data.filters)
                gdf = gpd.read_postgis(sql, db.engine, geom_col="geometry")
                data_container.set_data(layer_name, gdf)
                logging.info(f"Found {len(gdf)} {query_data.table_name} features for layer '{layer_name}'")

            else:
                sql = query_table_sql(query_data.table_name, label=None, metadata=metadata, filters=query_data.filters)
                gdf = gpd.read_postgis(sql, db.engine, geom_col="geometry")
                data_container.set_data(layer_name, gdf)
                logging.info(f"Found {len(gdf)} {query_data.table_name} features for layer '{layer_name}'")
