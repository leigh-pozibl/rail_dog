import logging

import geopandas as gpd  # noqa: F401 (used in type hint)

from shapely.geometry import Point
from shapely.strtree import STRtree

from snappy_utils.general import clean_string

from rail_dog.utils.db_utils import post_to_db, update_assets_with_sap_data, get_table_data
from rail_dog.configs.schema import Asset, AggAsset, SAPRecord, sap_column_mapping, RailSegmentAsset

import pandas as pd


class AssetsMixin:

    def create_assets(self) -> None:
        """
        Parse the assets input data ready for insertion into db
        """
        input_asset_data = self.data.track_data.get("ASSETS")

        if input_asset_data is None:
            logging.error("No input asset data found")

        def find_asset_segment(geom: Point, segment_data: tuple[STRtree, 'gpd.GeoDataFrame'], chainage: float, max_distance: float = 10, debug: bool = False):
            seg_tree, seg_gdf = segment_data
            # Fast bounding box query first
            candidates = seg_tree.query(geom.buffer(max_distance))
            if len(candidates) > 0:
                # Calculate exact distances, filter within max_distance, and sort
                distances = [(idx, geom.distance(seg_gdf.iloc[idx].geometry)) for idx in candidates]
                distances = [(idx, dist) for idx, dist in distances if dist <= max_distance]
                distances.sort(key=lambda x: x[1])
                hits = [idx for idx, _ in distances]
            else:
                hits = []

            if debug:
                logging.debug(f"Asset at chainage {chainage} has {len(hits)} hits within {max_distance}m")

            for hit in hits:
                hit_row = seg_gdf.loc[hit]
                if debug:
                    logging.debug(f"  Hit chainage: {hit_row['chainage_start_km']} to {hit_row['chainage_end_km']}")
                    logging.debug(f"{hit_row}")
                chainage_start = hit_row["chainage_start_km"]
                chainage_end = seg_gdf.iloc[hit]["chainage_end_km"]
                if chainage_start <= chainage <= chainage_end:
                    return seg_gdf.iloc[hit]["chainage_id"]

            return None

        assets = []
        segment_to_assets = []

        for _, row in input_asset_data.iterrows():
            line_code = self.line_name_to_code.get(row["Section"])

            if line_code is None:
                logging.debug(f"Skipping asset: {row['Functional Location']} as it is tagged with unknown Section: {row['Section']}")
                continue

            chainage = row["Chainage"]

            if line_code == "ML" and chainage <= self.THOMAS_END:
                line_code = "TL"

            coords = self.get_coords_from_chainage(line_code, chainage)
            geom = Point(coords["E"], coords["N"])
            geom_ll = Point(coords["lng"], coords["lat"])

            asset_data = {
                "asset_id": row["Functional Location"],
                "type": clean_string(row["Asset Type"]),
                "line": row["Section"],
                "line_code": line_code,
                "location_desc": row["Location Description"],
                "chainage": chainage,
                "coord_lng": coords["lng"],
                "coord_lat": coords["lat"],
                "geometry": geom_ll,  # as we are posting to db in epsg:4326
            }

            # Find which segment this asset belongs to
            debug = False
            # if row["Functional Location"] == "ROP-TA-APNN-MNLINE-TRACK0-ESTRL-IR002":
            #     debug = True

            chainage_id = find_asset_segment(geom, self.segment_data[line_code], chainage, max_distance=10, debug=debug)
            if chainage_id:
                rail_seg_to_asset_data = {
                    "chainage_id": chainage_id,
                    "asset_id": asset_data["asset_id"],
                }
                segment_to_assets.append(rail_seg_to_asset_data)
            else:
                logging.warning(f"Could not find segment for asset: {asset_data['asset_id']} at chainage: {chainage} on line: {line_code}")

            assets.append(asset_data)

        # post data to the database
        post_to_db(assets, Asset, self.db.engine, action="replace")

        # update assets with aggregated sap data
        update_assets_with_sap_data(self.db.engine)

        # post the mappings to the db
        post_to_db(segment_to_assets, RailSegmentAsset, self.db.engine, action="replace")

    def process_sap_records(self, db_action: str = None) -> None:
        """
        Process the SAP work order records into a suitable format
        """
        input_sap_data = self.data.track_data.get("SAP")

        if input_sap_data is None:
            return

        sap_data_clean = input_sap_data.rename(columns={c: clean_string(c, sap_column_mapping()) for c in input_sap_data.columns})

        sap_records = []
        seen_order_ids = set()
        for _, row in sap_data_clean.iterrows():
            asset_id = row["Functional Location"]
            report_date = pd.to_datetime(row["Start Date"], format="%d/%m/%Y") if pd.notna(row["Start Date"]) else None
            year = report_date.year if report_date else None

            if row["Order"] not in seen_order_ids:
                sap_record_data = {
                    "date": report_date,
                    "year": year,
                    "revision": row["Revision"],
                    "planner_grp": row["Maint Planner Group"],
                    "work_center": row["Maint Work Center"],
                    "user_status": row["User Status"],
                    "system_status": row["System Status"],
                    "priority": row["Priority Text"],
                    "order_id": row["Order"],
                    "description": row["Description"],
                    "asset_id": asset_id,
                    "iridium_code": row.get("Iridium Code", ""),
                }
                seen_order_ids.add(row["Order"])

                sap_records.append(sap_record_data)

        self.sap_records = pd.DataFrame(sap_records)

        # post data to the database
        post_to_db(sap_records, SAPRecord, self.db.engine, action=db_action)

    def aggregate_assets_to_segments(self, db_action: str = None) -> None:
        """
        Compile count of assets associated with each line segment
        """
        if self.assets is None or self.assets.empty:
            return

        agg_assets = []
        for _, segments_gdf in self.segment_data.values():
            for _, row in segments_gdf.iterrows():
                chainage_id = row.chainage_id

                # Get all assets for this segment
                query = f"""SELECT assets.*
                            FROM assets
                            JOIN rail_segment_assets ON assets.asset_id = rail_segment_assets.asset_id
                            WHERE rail_segment_assets.chainage_id = '{chainage_id}'"""
                segment_assets = get_table_data(self.db.engine, query=query)

                # Get counts of each asset type
                type_counts = segment_assets["type"].value_counts()

                level_crossing = int(type_counts.get("level_crossing", 0))
                irj = int(type_counts.get("irj", 0))
                turnout = int(type_counts.get("turnout", 0))
                bridge = int(type_counts.get("bridge", 0))

                # Count assets with sap_max_threshold flagged by type
                wz_level_crossing = 0
                wz_irj = 0
                wz_turnout = 0
                wz_bridge = 0

                for _, asset in segment_assets.iterrows():
                    if asset["sap_max_threshold"]:
                        asset_type = asset["type"]
                        if asset_type == "level_crossing":
                            wz_level_crossing += 1
                        elif asset_type == "irj":
                            wz_irj += 1
                        elif asset_type == "turnout":
                            wz_turnout += 1
                        elif asset_type == "bridge":
                            wz_bridge += 1

                agg_asset_data = {
                    "chainage_id": chainage_id,
                    "level_crossing": level_crossing,
                    "irj": irj,
                    "turnout": turnout,
                    "bridge": bridge,
                    "fixed_asset": (level_crossing + irj + turnout + bridge) > 0,
                    "wz_level_crossing": wz_level_crossing,
                    "wz_irj": wz_irj,
                    "wz_turnout": wz_turnout,
                    "wz_bridge": wz_bridge,
                    "wz_asset": (wz_level_crossing + wz_irj + wz_turnout + wz_bridge) > 0
                }
                agg_assets.append(agg_asset_data)

        # post data to the database
        post_to_db(agg_assets, AggAsset, self.db.engine, action=db_action)
