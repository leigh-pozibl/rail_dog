import logging
from typing import Type
from collections import defaultdict

import pandas as pd
import geopandas as gpd
from sqlmodel import Session, SQLModel, select
from sqlalchemy import inspect, text

from rail_dog.configs.schema import create_table, Asset, SAPRecord
from rail_dog.configs.thresholds import ASSET_WORK_ORDER_THRESHOLDS


def setup_table(engine, model_class: Type[SQLModel], action: str = "replace") -> bool:
    """
    Check if a table exists and handle creation/recreation.

    Args:
        engine: SQLAlchemy engine
        model_class: SQLModel class (e.g., Asset, RailSection, etc.)
        action: Behavior when table exists. Options:
            - "replace": Always drop and recreate the table
            - "skip": Skip insertion if table exists
            - "append": Allow insertion to existing table (returns True)

    Returns:
        bool: True if ready to insert data, False if should skip insertion
    """
    table_name = getattr(model_class, '__tablename__', model_class.__name__.lower())
    inspector = inspect(engine)
    table_exists = inspector.has_table(table_name)

    if table_exists:
        if action == "replace":
            logging.info(f"Dropping table '{table_name}'...")
            with Session(engine) as session:
                session.exec(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                session.commit()
            logging.info(f"Creating table '{table_name}'...")
            create_table(engine, model_class)
            return True
        elif action == "skip":
            logging.info(f"Table '{table_name}' already exists. Skipping insertion.")
            return False
        elif action == "append":
            logging.info(f"Table '{table_name}' already exists. Will append data.")
            return True
        else:
            raise ValueError(f"Invalid if_exists value: {action}. Must be 'replace', 'skip', or 'append'")
    else:
        # Table doesn't exist, create it
        logging.info(f"Creating table '{table_name}'...")
        create_table(engine, model_class)
        return True


def post_to_db(data: list[dict], model_class: Type[SQLModel], engine, action="replace"):
    """
    Post data to database table.

    Args:
        data: List of dictionaries containing record data
        model_class: SQLModel class (e.g., Asset, RailSection, etc.)
        engine: SQLAlchemy engine
        action: Behavior when table exists ("replace", "skip", "append")
    """
    from shapely.geometry.base import BaseGeometry

    # Get table name from the model class
    table_name = getattr(model_class, '__tablename__', model_class.__name__.lower())
    proceed = setup_table(engine, model_class, action=action)

    # Insert data to the db if we should proceed
    if proceed:
        logging.info(f"Inserting {len(data)} records into table: {table_name}...")
        with Session(engine) as session:
            for record_data in data:
                # Convert shapely geometry objects to WKT format for PostGIS
                if 'geometry' in record_data and isinstance(record_data['geometry'], BaseGeometry):
                    record_data['geometry'] = record_data['geometry'].wkt
                record = model_class(**record_data)
                session.add(record)
            session.commit()
        logging.info("Data successfully inserted into database")


def update_assets_with_sap_data(engine):
    """
    Update Asset records with aggregated SAP data.

    Iterates through all assets and for each asset:
    - Checks if SAP records exist for this asset_id
    - Counts total work orders
    - Aggregates work order counts by year
    - Updates Asset with sap_total_count, sap_year_count and sap_year_status

    Args:
        engine: SQLAlchemy engine
    """
    logging.info("Updating Asset records with SAP data...")
    
    thresholds = ASSET_WORK_ORDER_THRESHOLDS()

    with Session(engine) as session:
        # Get all assets
        assets_query = select(Asset)
        assets = session.exec(assets_query).all()

        updated_count = 0
        no_sap_count = 0

        for asset in assets:
            # Query SAP records for this asset_id
            sap_records_query = (
                select(SAPRecord)
                .where(SAPRecord.asset_id == asset.asset_id)
            )
            sap_records = list(session.exec(sap_records_query).all())
            
            if not sap_records:
                asset.sap_total_count = 0
                no_sap_count += 1
                session.add(asset)
                continue

            asset.sap_total_count = len(sap_records)

            # Aggregate counts by year
            if asset.type in thresholds.asset_types:
                year_counts = defaultdict(int)
                year_status = {}

                for record in sap_records:
                    year_counts[str(record.year)] += 1

                for year, year_count in year_counts.items():
                    year_status[year] = thresholds.get_status(asset.type, year_count)

                # SQLAlchemy JSON columns handle dicts automatically
                asset.sap_year_count = dict(year_counts)
                asset.sap_year_status = year_status
                asset.sap_max_threshold = thresholds.get_max_status(year_status)

                logging.debug(f"Updated {asset.asset_id}: {asset.sap_total_count} records, years: {list(year_counts.keys())}")

            session.add(asset)
            updated_count += 1

        try:
            session.commit()
            logging.info(f"Updated {updated_count} Asset records with SAP data")
            logging.info(f"{no_sap_count} assets have no SAP records")
        except Exception as e:
            session.rollback()
            logging.error(f"Error committing asset updates: {e}")
            raise


def get_table_data(engine, table_name: str = None, query: str = None, crs: str = None):
    if not query and not table_name:
        raise ValueError("Either table_name or query must be provided") 
        
    if not query:
        query = f"SELECT * FROM {table_name}"
        
    if crs:
        return gpd.read_postgis(
            query,
            engine,
            geom_col="geometry"
        ).to_crs(crs)
    else:
        return pd.read_sql(
            query,
            engine,
        )
    