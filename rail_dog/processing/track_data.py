import logging
from datetime import datetime

import pandas as pd

from shapely.geometry import Point

from snappy_utils.general import clean_string

from rail_dog.utils.db_utils import post_to_db, get_table_data
from rail_dog.configs.schema import (
    MoistureRecord, AggMoisture, GBFIRecord, AggGBFI,
    gbfi_column_mapping, moisture_column_mapping,
    ballast_column_mapping, BallastRecord, AggBallast
)
from rail_dog.configs.thresholds import GBFI_FOULED_THRESHOLDS


class TrackDataMixin:

    def process_gbfi_data(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Process the GBFI into standard format for database insertion
        """
        # Collect GBFI for all lines
        gbfi_list = []
        for _, line_code in self.line_name_to_code.items():
            gbfi_data = self.data.track_data.get(f"GBFI_{line_code}")
            if gbfi_data is not None:
                gbfi_data_clean = gbfi_data.rename(columns={c: clean_string(c, gbfi_column_mapping()) for c in gbfi_data.columns})
                gbfi_data_clean["line_code"] = line_code
                gbfi_list.append(gbfi_data_clean)

        if len(gbfi_list) == 0:
            logging.warning("No input GBFI data found")
            return

        gbfi_data = pd.concat(gbfi_list, ignore_index=True)

        # Fix chainage values with embedded spaces (e.g. "174. 323" → "174.323")
        def _fix_chainage(val):
            if pd.isna(val):
                return val
            cleaned = str(val).replace(" ", "")
            try:
                return float(cleaned)
            except ValueError:
                return val

        gbfi_data["start_chainage_km"] = gbfi_data["start_chainage_km"].apply(_fix_chainage)
        gbfi_data["end_chainage_km"] = gbfi_data["end_chainage_km"].apply(_fix_chainage)

        gbfi_records = []
        for _, row in gbfi_data.iterrows():

            if pd.isna(row["start_chainage_km"]):
                continue

            if float(row["end_chainage_km"]) <= self.THOMAS_END and row["line_code"] == "ML":
                line_code = "TL"
            else:
                line_code = row["line_code"]

            gbfi_record = {
                "collection_date": collection_date,
                "line_code": line_code,
                "chainage_start_km": float(row["start_chainage_km"]),
                "chainage_end_km": float(row["end_chainage_km"]),
                # "left": float(row["left"]),
                # "center": float(row["center"]),
                # "right": float(row["right"]),
                "avg": float(row["ave"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
                "geometry": Point(row["lng"], row["lat"])
            }
            gbfi_records.append(gbfi_record)

        # post data to the database
        post_to_db(gbfi_records, GBFIRecord, self.db.engine, action=db_action, upsert=True)

    def aggregate_gbfi_to_segments(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Aggregate GBFI data to rail segments
        """
        thresholds = GBFI_FOULED_THRESHOLDS()

        agg_gbfi = []
        for line_code, (_, segments_gdf) in self.segment_data.items():
            for _, row in segments_gdf.iterrows():
                chainage_id = row.chainage_id
                seg_chainage_start = row.chainage_start_km
                seg_chainage_end = row.chainage_end_km

                # Get all gbfi records for this segment
                query = f"""
                    SELECT *
                    FROM gbfi_records
                    WHERE chainage_start_km >= '{seg_chainage_start}'
                    AND chainage_end_km < '{seg_chainage_end}'
                    AND line_code = '{line_code}'
                    AND collection_date = '{collection_date}'
                """
                segment_gbfi = get_table_data(self.db.engine, query=query)

                if segment_gbfi.empty:
                    logging.debug(f"Did not find any GBFI records for line {line_code}, date {collection_date}, chainage {chainage_id}")
                    continue

                fouled_status = thresholds.get_status_counts(segment_gbfi)
                fouled_zone = True if fouled_status["reasonable"] + fouled_status["fouled"] + fouled_status["high"] > 0 else False
                agg_gbfi_data = {
                    "chainage_id": chainage_id,
                    "collection_date": collection_date,
                    "max_of_avg": float(segment_gbfi["avg"].max()),
                    "avg_of_avg": round(float(segment_gbfi["avg"].mean()), 1),
                    "reasonably_fouled": fouled_status["reasonable"],
                    "fouled": fouled_status["fouled"],
                    "highly_fouled": fouled_status["high"],
                    "fouled_zone": fouled_zone,
                }
                agg_gbfi.append(agg_gbfi_data)

        # self.agg_gbfi_records = pd.DataFrame(agg_gbfi)

        # post data to the database
        post_to_db(agg_gbfi, AggGBFI, self.db.engine, action=db_action)

    def process_moisture_data(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Process the Moisture data into standard format for database insertion
        """
        # Collect Moisture data for all lines
        moisture_list = []
        for _, line_code in self.line_name_to_code.items():
            moisture_data = self.data.track_data.get(f"MOI_{line_code}")
            if moisture_data is not None:
                moisture_data_clean = moisture_data.rename(columns={c: clean_string(c, moisture_column_mapping()) for c in moisture_data.columns})
                moisture_data_clean["line_code"] = line_code
                moisture_list.append(moisture_data_clean)

        if len(moisture_list) == 0:
            logging.warning("No input Moisture data found")
            return

        moisture_data = pd.concat(moisture_list, ignore_index=True)

        # Fix chainage values with embedded spaces (e.g. "174. 323" → "174.323")
        def _fix_chainage(val):
            if pd.isna(val):
                return val
            cleaned = str(val).replace(" ", "")
            try:
                return float(cleaned)
            except ValueError:
                return val

        moisture_data["start_chainage_km"] = moisture_data["start_chainage_km"].apply(_fix_chainage)
        moisture_data["end_chainage_km"] = moisture_data["end_chainage_km"].apply(_fix_chainage)

        # Compute avg/min/max from individual depth columns (e.g. cleaned "Depth-A-(m)" → starts with "depth")
        depth_cols = [c for c in moisture_data.columns if c.startswith("depth") and not c.startswith("deptha")]
        depth_vals = moisture_data[depth_cols].apply(pd.to_numeric, errors="coerce")
        moisture_data["_avg"] = depth_vals.mean(axis=1, skipna=True)
        moisture_data["_min"] = depth_vals.min(axis=1, skipna=True)
        moisture_data["_max"] = depth_vals.max(axis=1, skipna=True)

        moisture_records = []
        for _, row in moisture_data.iterrows():

            if pd.isna(row["start_chainage_km"]):
                continue

            if float(row["end_chainage_km"]) <= self.THOMAS_END and row["line_code"] == "ML":
                line_code = "TL"
            else:
                line_code = row["line_code"]

            moisture_record = {
                "collection_date": collection_date,
                "line_code": line_code,
                "chainage_start_km": float(row["start_chainage_km"]),
                "chainage_end_km": float(row["end_chainage_km"]),
                "avg": round(float(row["_avg"]), 2),
                "min": float(row["_min"]),
                "max": float(row["_max"]),
                "depth_a": float(row["deptham"]),  # surface value
            }
            moisture_records.append(moisture_record)

        # post data to the database
        post_to_db(moisture_records, MoistureRecord, self.db.engine, action=db_action, upsert=True)

    def aggregate_moisture_to_segments(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Aggregate Moisture data to rail segments
        """
        agg_moisture = []
        for line_code, (_, segments_gdf) in self.segment_data.items():
            for _, row in segments_gdf.iterrows():
                chainage_id = row.chainage_id
                seg_chainage_start = row.chainage_start_km
                seg_chainage_end = row.chainage_end_km

                # Get all moisture records for this segment
                query = f"""
                    SELECT *
                    FROM moisture_records
                    WHERE chainage_start_km >= '{seg_chainage_start}'
                    AND chainage_end_km < '{seg_chainage_end}'
                    AND line_code = '{line_code}'
                    AND collection_date = '{collection_date}'
                """
                segment_moisture = get_table_data(self.db.engine, query=query)

                if segment_moisture.empty:
                    logging.debug(f"Did not find any Moisture records for line {line_code}, date {collection_date}, chainage {chainage_id}")
                    continue

                agg_moisture_data = {
                    "chainage_id": chainage_id,
                    "collection_date": collection_date,
                    "max_of_avg": float(segment_moisture["avg"].max()),
                    "avg_of_avg": round(float(segment_moisture["avg"].mean()), 2),
                    "max_of_depth_a": float(segment_moisture["depth_a"].max()),
                    "avg_of_depth_a": round(float(segment_moisture["depth_a"].mean()), 2),
                }
                agg_moisture.append(agg_moisture_data)

        self.agg_moisture_records = pd.DataFrame(agg_moisture)

        # post data to the database
        post_to_db(agg_moisture, AggMoisture, self.db.engine, action=db_action, upsert=True)

    def process_ballast_data(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Process the Ballast into standard format for database insertion
        """
        # Collect Ballast for all lines
        ballast_list = []
        for _, line_code in self.line_name_to_code.items():
            ballast_data = self.data.track_data.get(f"BALL_{line_code}")
            if ballast_data is not None:
                ballast_data_clean = ballast_data.rename(columns={c: clean_string(c, ballast_column_mapping()) for c in ballast_data.columns})
                ballast_data_clean["line_code"] = line_code

                # Convert distance from "4+ 296" or "4. 296" format to 4.296 km
                def parse_chainage(x):
                    if not isinstance(x, str):
                        return None
                    if "+" in x:
                        parts = x.split("+")
                    elif "." in x:
                        parts = x.split(".")
                    else:
                        return None
                    try:
                        return float(parts[0].strip()) + float(parts[1].strip()) / 1000
                    except (ValueError, IndexError):
                        return None

                ballast_data_clean["start_chainage_km"] = ballast_data_clean["distance_str"].apply(parse_chainage)

                # Calculate center: use center value if valid float, otherwise average of left and right, otherwise NA
                def calculate_center(row):
                    try:
                        # Try to convert center to float
                        return float(row["center"])
                    except (ValueError, TypeError):
                        # If center is not valid, try average of left and right
                        try:
                            left = float(row["left"])
                            right = float(row["right"])
                            return (left + right) / 2
                        except (ValueError, TypeError, KeyError):
                            # If that fails, return NA
                            return pd.NA

                ballast_data_clean["center"] = ballast_data_clean.apply(calculate_center, axis=1)

                ballast_list.append(ballast_data_clean)

        if len(ballast_list) == 0:
            logging.warning("No input Ballast data found")
            return

        ballast_data = pd.concat(ballast_list, ignore_index=True)

        ballast_records = []
        for _, row in ballast_data.iterrows():

            if pd.isna(row["start_chainage_km"]):
                continue

            if float(row["start_chainage_km"] + 0.001) <= self.THOMAS_END and row["line_code"] == "ML":
                line_code = "TL"
            else:
                line_code = row["line_code"]

            ballast_record = {
                "collection_date": collection_date,
                "line_code": line_code,
                "chainage_start_km": float(row["start_chainage_km"]),
                "left": float(row["left"]),
                "center": float(row["center"]),
                "right": float(row["right"]),
                "geometry": Point(row["lng"], row["lat"])
            }
            ballast_records.append(ballast_record)

        # post data to the database
        post_to_db(ballast_records, BallastRecord, self.db.engine, action=db_action)

    def aggregate_ballast_to_segments(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Aggregate Ballast data to rail segments
        """
        agg_ballast = []
        for line_code, (_, segments_gdf) in self.segment_data.items():
            for _, row in segments_gdf.iterrows():
                chainage_id = row.chainage_id
                seg_chainage_start = row.chainage_start_km
                seg_chainage_end = row.chainage_end_km

                # Get all ballast records for this segment
                query = f"""
                    SELECT *
                    FROM ballast_records
                    WHERE chainage_start_km >= '{seg_chainage_start}'
                    AND chainage_start_km < '{seg_chainage_end}'
                    AND line_code = '{line_code}'
                    AND collection_date = '{collection_date}'
                """
                segment_ballast = get_table_data(self.db.engine, query=query)

                if segment_ballast.empty:
                    logging.debug(f"Did not find any Ballast records for line {line_code}, date {collection_date}, chainage {chainage_id}")
                    continue

                agg_ballast_data = {
                    "chainage_id": chainage_id,
                    "collection_date": collection_date,
                    "ballast_centre": segment_ballast["center"].mean() - 0.175,
                    "ballast_lt_250mm": segment_ballast["center"].mean() - 0.175 < 0.25,
                }
                agg_ballast.append(agg_ballast_data)

        # post data to the database
        post_to_db(agg_ballast, AggBallast, self.db.engine, action=db_action)
