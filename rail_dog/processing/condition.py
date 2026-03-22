import logging
from datetime import datetime

import pandas as pd

from snappy_utils.general import clean_string

from rail_dog.utils.db_utils import post_to_db, get_table_data
from rail_dog.configs.schema import TSRRecord, TQIRecord, AggTQI, DTRRecord, AggDTR
from rail_dog.configs.thresholds import TQI_THRESHOLDS
from rail_dog.configs.library import get_section_metadata


class ConditionMixin:

    def process_tsr_data(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Process the TSR into standard format for database insertion
        """
        # Collect TSR for all lines
        tsr_data_open = self.data.track_data.get("TSR_OPEN")
        tsr_data_complete = self.data.track_data.get("TSR_COMPLETE")
        tsr_data_all = self.data.track_data.get("TSR_ALL")

        legacy_format = False
        if tsr_data_open is not None and tsr_data_complete is not None:
            legacy_format = True
        elif tsr_data_all is None:
            logging.warning("No input TSR data found")
            return

        if legacy_format:
            if tsr_data_open is not None:
                tsr_data_open["status"] = "open"

            if tsr_data_complete is not None:
                tsr_data_complete["status"] = "complete"

            tsr_data = pd.concat([tsr_data_open, tsr_data_complete], ignore_index=True)
            tsr_data = tsr_data.rename(columns={c: clean_string(c) for c in tsr_data.columns})
        else:
            column_name_mapping = {
                "Imposed": "report_date",
                "Ch. Start": "start_chainage",
                "Ch. End": "end_chainage",
                "Speed": "speed",
                "Removed": "close_date",
                "Network": "line",
                "Days": "duration_days",
                "Comment": "comment",
            }
            logging.info("Applying corrections to raw TSR data")
            tsr_data = tsr_data_all.rename(columns={c: clean_string(c, custom_mapping=column_name_mapping) for c in tsr_data_all.columns})

            # filter/cleanup the data
            tsr_data["status"] = tsr_data["close_date"].apply(lambda x: "complete" if pd.notna(x) else "open")
            tsr_data = tsr_data[~(tsr_data["comment"].str.contains("WOLO", na=False) | tsr_data["comment"].isna() | (tsr_data["comment"].str.strip() == ""))]
            tsr_data = tsr_data[tsr_data["duration_days"] >= 1]
            tsr_data = tsr_data[tsr_data["line"].isin(["Mainline", "Solomon Mainline", "Eliwana Mainline", "Eastline"])]
            tsr_data["line"] = tsr_data["line"].replace({"Solomon Mainline": "Solomon", "Eliwana Mainline": "Eliwana"})

        tsr_records = []
        for _, row in tsr_data.iterrows():
            if pd.isna(row["start_chainage"]):
                continue

            if float(row["end_chainage"]) <= self.THOMAS_END and row["line"] == "Mainline":
                line_code = "TL"
            else:
                line_code = self.line_name_to_code.get(row["line"])

            if line_code is None:
                logging.debug(f"Skipping TSR record with line name: {row['line']}")
                continue

            report_date = pd.to_datetime(row["report_date"], format="%d/%m/%Y").round("s") if pd.notna(row["report_date"]) else None
            close_date = pd.to_datetime(row["close_date"], format="%d/%m/%Y").round("s") if pd.notna(row["close_date"]) else None

            tsr_record = {
                "report_date": report_date,
                "line_code": line_code,
                "status": row["status"],
                "chainage_start_km": float(row["start_chainage"]),
                "chainage_end_km": float(row["end_chainage"]),
                "speed": float(row["speed"]),
                "close_date": close_date,
                "duration_days": int(row["duration_days"]),
                "comment": row["comment"],
            }
            tsr_records.append(tsr_record)

        # Deduplicate: keep record with longest duration_days per (report_date, line_code, start_chainage_km)
        deduped = {}
        for r in tsr_records:
            key = (r["report_date"], r["line_code"], r["chainage_start_km"])
            if key not in deduped or r["duration_days"] > deduped[key]["duration_days"]:
                deduped[key] = r
        tsr_records = list(deduped.values())

        # post data to the database
        post_to_db(tsr_records, TSRRecord, self.db.engine, action=db_action, upsert=True)

    def process_tqi_data(self, collection_date: datetime, db_action: str = None) -> None:
        """
        Process ENSCO calculated TQI data into standard format for database insertion.

        Note, this is not calculating the TQI
        """
        # Collect TQI
        tqi_data = self.data.track_data.get("TQI")

        if tqi_data is None:
            logging.warning("No input TQI data found")
            return

        column_name_mapping = {
            "Asset Name": "asset_name",
            "Division": "network",
            "Subdivision": "line_section",
            "Segment ID": "segment_id",
            "Segment Length": "segment_len",
            "Start Location": "chainage_start_km",
            "End Location": "chainage_end_km",
            "TQI Geometry": "tqi",
            "Run ID TQI Geometry": "run_id",
        }

        tqi_data = tqi_data.rename(columns={c: clean_string(c, column_name_mapping) for c in tqi_data.columns})

        # Populate line, line_code, and line_class
        tqi_data[["line", "line_code", "line_class"]] = tqi_data["asset_name"].apply(get_section_metadata)

        tqi_data["chainage_mid_km"] = (tqi_data["chainage_start_km"] + tqi_data["chainage_end_km"]) / 2
        tqi_data["collection_date"] = collection_date

        # Filter out rows with missing critical data (including invalid collection dates)
        tqi_data = tqi_data.dropna(subset=['line_code', 'tqi', 'chainage_start_km'])

        # Select columns that match TGRecord schema
        columns_to_extract = ["line", "line_code", "line_class", "asset_name", "chainage_start_km",
                              "chainage_end_km", "chainage_mid_km", "tqi", "run_id", "collection_date"]

        # Convert to list of dictionaries
        tqi_records = tqi_data[columns_to_extract].to_dict('records')

        logging.info(f"Processing {len(tqi_records)} TQI records for database insertion")

        # Post data to the database
        post_to_db(tqi_records, TQIRecord, self.db.engine, action=db_action)

    def aggregate_tqi_to_segments(self, collection_date: datetime, db_action: str = None, line_class: str = "main") -> None:
        """
        Aggregate TQI data to rail segments. Take the segment chainage mid-point and find the closest TQI sample
        as the representative TQI for this segment.

        """
        # Load ALL TQI data once
        logging.info("Loading TQI data from database...")
        query = f"""
            SELECT * FROM tqi_records
            WHERE line_class = '{line_class}'
            AND collection_date = '{collection_date}'
        """

        tqi_data = get_table_data(self.db.engine, query=query)    # to do: add filter for collection date

        if tqi_data.empty:
            logging.warning("No TQI data found")
            return

        logging.info(f"Processing {len(tqi_data)} TQR records across segments...")

        thresholds = TQI_THRESHOLDS()

        agg_tqi = []
        for line_code, (_, segments_gdf) in self.segment_data.items():
            # Filter TQR data for this line code
            line_tqi = tqi_data[tqi_data["line_code"] == line_code]

            if line_tqi.empty:
                continue

            tqi_by_asset = {name: grp for name, grp in line_tqi.groupby("asset_name")}

            for _, segment in segments_gdf.iterrows():
                chainage_id = segment.chainage_id
                seg_chainage_start = segment.chainage_start_km
                seg_chainage_end = segment.chainage_end_km
                seg_chainage_mid = (seg_chainage_start + seg_chainage_end) / 2
                seg_asset_name = segment.asset_name

                # Find TQI record with closest chainage_mid_km to segment midpoint
                asset_tqi = tqi_by_asset.get(seg_asset_name)
                if asset_tqi is None:
                    continue
                closest_idx = (asset_tqi["chainage_mid_km"] - seg_chainage_mid).abs().idxmin()
                closest_tqi = asset_tqi.loc[closest_idx]

                agg_tqi_data = {
                    "chainage_id": chainage_id,
                    "tqi": closest_tqi.tqi,
                    "line_code": line_code,
                    "line_class": line_class,
                    "status": thresholds.get_status(closest_tqi.tqi),
                    "trend": None,  # populated post-insert by compute_trend()
                    "collection_date": closest_tqi.collection_date
                }

                agg_tqi.append(agg_tqi_data)

        logging.info(f"Aggregated TQI data for {len(agg_tqi)} segments")

        # post data to the database
        post_to_db(agg_tqi, AggTQI, self.db.engine, action=db_action)

    def process_dtr_data(self, collection_date: datetime, db_action: str = None):
        """
        Process ENSCO Dynamic Track Force data into standard format for database insertion.
        """
        # Collect DTR
        dtr_data = self.data.track_data.get("DTR")

        if dtr_data is None:
            logging.warning("No input DTR data found")
            return

        column_name_mapping = {
            "Approx Track location    (km +50m)": "chainage_km",
            "IRV Channel which measured location": "channel",
            "Track": "line",
            "Units of data": "units",
            "Recent Average": "recent_avg",
            "Getting Worse?": "getting_worse"
        }

        dtr_data = dtr_data.rename(columns={c: clean_string(c, column_name_mapping) for c in dtr_data.columns})

        # Populate line, line_code, and line_class
        dtr_data["line"] = dtr_data["line"].str.title()
        dtr_data.loc[dtr_data["chainage_km"] <= self.THOMAS_END, "line"] = "Thomas"
        dtr_data["line_code"] = dtr_data["line"].map(self.line_name_to_code)
        dtr_data["collection_date"] = collection_date

        # Convert getting_worse from "Yes"/"No" strings to boolean
        dtr_data["getting_worse"] = dtr_data["getting_worse"].str.lower() == "yes"

        # Filter out invalid rows
        dtr_data = dtr_data.dropna(subset=['line_code'])

        # Select columns that match TGRecord schema
        columns_to_extract = [
            "line_code", "chainage_km",
            "channel", "track_features", "units", "severity", "recent_max", "recent_avg", "getting_worse",
            "collection_date"
        ]

        # Convert to list of dictionaries
        dtr_records = dtr_data[columns_to_extract].to_dict('records')

        logging.info(f"Processing {len(dtr_records)} DTR records for database insertion")

        # Post data to the database
        post_to_db(dtr_records, DTRRecord, self.db.engine, action="append")

    def aggregate_dtr_to_segments(self, collection_date: datetime, db_action: str = None, line_class: str = "main") -> None:
        """
        Aggregate DTR data to rail segments.
        """
        # Load the DTR data
        logging.info("Loading DTR data from database...")
        dtr_data = get_table_data(
            self.db.engine,
            query=f"SELECT * FROM dtr_records WHERE collection_date = '{collection_date}'"
        )

        if dtr_data.empty:
            logging.warning("No DTR data found")
            return

        logging.info(f"Processing {len(dtr_data)} DTR records across segments...")

        agg_dtr = []
        for line_code, (_, segments_gdf) in self.segment_data.items():
            # Filter DTR data for this line code
            line_dtr = dtr_data[dtr_data["line_code"] == line_code]

            for _, segment in segments_gdf.iterrows():
                chainage_id = segment.chainage_id
                seg_chainage_start = segment.chainage_start_km
                seg_chainage_end = segment.chainage_end_km

                # Filter open DTR records for this segment
                segment_dtr = line_dtr[
                    (line_dtr["chainage_km"] >= seg_chainage_start) &
                    (line_dtr["chainage_km"] < seg_chainage_end)
                ]

                # Count by year
                if segment_dtr.empty:
                    sf_accel_w = sf_accel_e = suspension = rock = bounce = 0
                    dtf_s1 = dtf_s2 = dtf_s3 = getting_worse = 0
                    worst_dtf = ""
                else:
                    chan_counts = segment_dtr["channel"].value_counts()
                    sf_accel_w = chan_counts.get("SF Accel West", 0)
                    sf_accel_e = chan_counts.get("SF Accel East", 0)
                    suspension = chan_counts.get("Suspension Travel", 0)
                    rock = chan_counts.get("Rock", 0)
                    bounce = chan_counts.get("Bounce", 0)

                    severity_counts = segment_dtr["severity"].value_counts()
                    dtf_s1 = severity_counts.get(1, 0)
                    dtf_s2 = severity_counts.get(2, 0)
                    dtf_s3 = severity_counts.get(3, 0)

                    # Count rows where getting_worse is True (converted from "Yes")
                    getting_worse = segment_dtr["getting_worse"].sum()

                    worst_dtf = ""
                    if dtf_s1:
                        worst_dtf = "s1"
                    elif dtf_s2:
                        worst_dtf = "s2"
                    elif dtf_s3:
                        worst_dtf = "s3"

                agg_dtr_data = {
                    "chainage_id": chainage_id,
                    "line_code": line_code,
                    "line_class": "",  # raw data does not indicate the track, ie: main or bypass
                    "collection_date": collection_date,
                    "sf_accel_w": sf_accel_w,
                    "sf_accel_e": sf_accel_e,
                    "suspension": suspension,
                    "rock": rock,
                    "bounce": bounce,
                    "dtf_s1": dtf_s1,
                    "dtf_s2": dtf_s2,
                    "dtf_s3": dtf_s3,
                    "worst_dtf": worst_dtf,
                    "getting_worse": getting_worse,
                }
                agg_dtr.append(agg_dtr_data)

        logging.info(f"Aggregated DTR data for {len(agg_dtr)} segments")

        # post data to the database
        post_to_db(agg_dtr, AggDTR, self.db.engine, action="append")
