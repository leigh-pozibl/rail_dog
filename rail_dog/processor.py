import os
import logging
import copy
from collections import defaultdict
from datetime import datetime

import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj

from shapely import to_wkt
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.ops import substring
from intervaltree import Interval, IntervalTree

from snappy_utils.params import Metadata, DBConnection
from snappy_utils.geom_utils import (
    utm_crs_from_a_geom,
    move_points_onto_lines,
    split_lines_into_segments,
)
from snappy_utils.general import clean_string

from rail_dog.utils.db_utils import post_to_db, update_assets_with_sap_data, get_table_data
from rail_dog.utils.rail_utils import apply_corrections
from rail_dog.utils.report_utils import generate_excel_report, generate_powerbi_export, generate_condition_summary
from rail_dog.configs.params import BaseConfiguration
from rail_dog.configs.schema import (
    Asset, AggAsset, RailSegment, SAPRecord, RailSegmentAsset, RailSection, GBFIRecord, gbfi_column_mapping,
    ballast_column_mapping, AggGBFI, TSRRecord, AggTSR, TGRecord, TQIRecord, AggTQI, DTRRecord, AggDTR,
    BallastRecord, AggBallast
)
from rail_dog.configs.thresholds import RAIL_DEGRADATION_THRESHOLDS, GBFI_FOULED_THRESHOLDS, TQI_THRESHOLDS
from rail_dog.configs.library import get_station, get_section, get_section_metadata


class Processor():
    def __init__(self, params: BaseConfiguration, db: DBConnection, metadata: Metadata, output_dir: str):

        self.db = db
        self.data = params.base_data
        self.controls = params.controls
        self.params = params.parameters.preprocess
        self.input_fields = params.parameters.input_fields
        self.costs = params.parameters.costs
        self.get_filters = params.parameters.preprocess.get_filters
        self.metadata = metadata
        
        self.segment_method = self.params.segment_method
        self.segment_length = 100

        self.working_crs = self.get_working_crs(params.parameters.globals.working_crs)
        params.parameters.globals.working_crs = self.working_crs
        logging.info(f"Working CRS is: {self.working_crs}")

        # convert all data to a common working_crs
        self.use_centroid = None
        for layer in self.data.active_layers:
            layer_data = self.data.get_data(layer)
            layer_data.geometry = layer_data.geometry.force_2d()
            self.data.set_data(layer, layer_data.to_crs(self.working_crs))

        # some mappings
        self.index_to_node_id = dict()
        self.index_to_edge_id = dict()
        self.cand_id_to_solution_id = dict()
        self.trees = {}

        self.debug = True
        
        self.refresh_data = params.base_data.refresh_data

        src = pyproj.CRS(self.working_crs)
        dst = pyproj.CRS('epsg:4326')
        self.PROJ = pyproj.Transformer.from_crs(src, dst, always_xy=True)

        self.h3_resolution = 15

        self.output_dir = output_dir
        self.output_fmt = params.parameters.globals.output_fmt
        
        self.line_name_to_code = {
            "Mainline": "ML",
            "Thomas": "TL",
            "Eliwana": "EL",
            "Solomon": "SL",
        }
        self.line_code_to_name = {v: k for k, v in self.line_name_to_code.items()}

        # self.THOMAS_START = -3.7
        self.THOMAS_START = -3.7515
        self.THOMAS_END = 26.9
        self.SOLOMON_START = 174
        self.ELIWANA_START = 288.70
        self.START_OFFSET = {
            "TL": self.THOMAS_START,
            "ML": self.THOMAS_END,
            "SL": self.SOLOMON_START,
            "EL": self.ELIWANA_START,
        }

        self.THRESHOLDS = RAIL_DEGRADATION_THRESHOLDS()

    def get_working_crs(self, input_crs) -> str:
        """
        It is better to use a UTM crs that is localised to the area being processed.  
        nb: using the epsg:3857 crs will report erroneous lengths for linestrings
        """
        if input_crs.split(":")[0] == "epsg":
            return str(input_crs)

        elif input_crs == "utm":
            for layer in self.data.active_layers:
                try:
                    layer_data = self.data.get_data(layer)
                    sample_geom = layer_data["geometry"][0]
                    crs = utm_crs_from_a_geom(sample_geom)
                    return f"epsg:{crs.to_epsg()}"
                except:
                    pass
            logging.error("Couldn't determine a localised projection, using default epsg:3857")
            return "epsg:3857"
        
        else:
            logging.error("Couldn't determine a localised projection, using default epsg:3857")
            return "epsg:3857"

    def run(self):
        
        # process the rail geometry input and split into separate lines
        self.process_base_geometry()
        
        ############################
        logging.info("############################")
        if self.refresh_data("BASE_SEGMENTS"):
            logging.info(f"Segmenting into {self.segment_length}m chainage intervals")
            self.process_into_chainage_segments(self.segment_length)
        else:
            logging.info("Using existing BASE_SEGMENTS data from database")
            self.segment_data = {}
            for line_code in self.line_code_to_name.keys():
                segment_gdf = get_table_data(
                    self.db.engine,
                    crs=self.working_crs,
                    query=f"SELECT * FROM rail_segments WHERE line_code = '{line_code}'"
                )
                _geoms = list(segment_gdf.geometry)
                self.segment_data[line_code] = (STRtree(_geoms), segment_gdf)

        ############################
        logging.info("############################")
        logging.info("Processing SAP records")
        if self.refresh_data("SAP"):
            self.process_sap_records()
        else:
            logging.info("Using existing SAP data from database")
            # we only need sap data when refreshing assets
            
            # sap_date = self.data.analysis_dates.get("SAP")
            # if not sap_date:
            #     logging.error("An analysis date for SAP data not specified")
            #     exit()
            # self.sap_records = get_table_data(
            #     self.db.engine,
            #     query=f"SELECT * FROM sap_records WHERE collection_date = '{sap_date}"
            # )
            
        ############################
        logging.info("############################")
        logging.info("Processing assets")
        if self.refresh_data("ASSETS"):
            self.create_assets()
        else:
            logging.info("Using existing asset data from database")
        self.assets = get_table_data(self.db.engine, table_name="assets", crs=self.working_crs)

        ############################
        logging.info("############################")
        logging.info("Processing curves & tangent sections")
        if self.refresh_data("CURVE_SECTIONS"):
            self.create_curve_sections()
        else:
            logging.info("Using existing curves data from database")
        self.curve_sections = get_table_data(self.db.engine, table_name="rail_sections", crs=self.working_crs)
        
        ############################
        logging.info("############################")
        logging.info("Aggregating assets to chainage segments")
        if self.refresh_data("AGG_ASSETS"):
            self.aggregate_assets_to_segments()
        else:
            logging.info("Using existing aggregated asset data from database")
            
        ############################
        logging.info("############################")
        logging.info("Processing GBFI data")
        if self.refresh_data("GBFI"):
            collection_date = self.data.collection_dates.get("GBFI_ML")
            if collection_date is None:
                logging.error("Must specify a collection date for the GBFI dataset being loaded.")
                return
            self.process_gbfi_data(collection_date)
            self.aggregate_gbfi_to_segments(collection_date)
        else:
            logging.info("Using existing GBFI data from database")
            gbfi_date = self.data.analysis_dates.get("GBFI")
            if not gbfi_date:
                logging.error("An analysis date for GBFI data not specified")
                exit()
            self.agg_gbfi = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_gbfi WHERE collection_date = '{gbfi_date}'"
            )
            
        ############################
        logging.info("############################")
        logging.info("Processing Ballast data")
        if self.refresh_data("BALL"):
            collection_date = self.data.collection_dates.get("BALL_ML")
            if collection_date is None:
                logging.error("Must specify a collection date for the Ballast dataset being loaded.")
                return
            self.process_ballast_data(collection_date)
            self.aggregate_ballast_to_segments(collection_date)
        else:
            logging.info("Using existing Ballast data from database")
            ballast_date = self.data.analysis_dates.get("BALL")
            if not ballast_date:
                logging.error("An analysis date for Ballast data not specified")
                exit()
            self.agg_ballast = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_ballast WHERE collection_date = '{ballast_date}'"
            )
        
        ############################
        logging.info("############################")
        logging.info("Processing TSR data")
        if self.refresh_data("TSR"):
            self.process_tsr_data()
            self.aggregate_tsr_to_segments()
        else:
            logging.info("Using existing TSR data from database")
        
        ############################
        logging.info("############################")
        logging.info("Processing TQI data")
        if self.refresh_data("TQI"):
            collection_date = self.data.collection_dates.get("TQI")
            if collection_date is None:
                logging.error("Must specify a collection date for the TQI dataset being loaded.")
                return
            self.process_tqi_data(collection_date)
            self.aggregate_tqi_to_segments(collection_date, line_class='main')
        else:
            logging.info("Using existing TQI data from database")
            tqi_date = self.data.analysis_dates.get("TQI")
            if not tqi_date:
                logging.error("An analysis date for TQI data not specified")
                return
            self.agg_tqi = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_tqi WHERE collection_date = '{tqi_date}'"
            )
        
        ############################
        logging.info("############################")
        logging.info("Processing DTR data")
        if self.refresh_data("DTR"):
            collection_date = self.data.collection_dates.get("DTR")
            if collection_date is None:
                logging.error("Must specify a collection date for the DTR dataset being loaded.")
                return
            self.process_dtr_data(collection_date)
            self.aggregate_dtr_to_segments(collection_date, line_class='main')
        else:
            logging.info("Using existing DTR data from database")
            dtr_date = self.data.analysis_dates.get("DTR")
            if not dtr_date:
                logging.error("An analysis date for DTR data not specified")
                return
            self.agg_dtr = get_table_data(
                self.db.engine,
                query=f"SELECT * FROM agg_dtr WHERE collection_date = '{dtr_date}'"
            )
        
        ############################
        logging.info("############################")
        if self.data.rp_data is not None:
            logging.info("Processing RP data")
            self.process_rp_data_into_sections()
            self.calculate_rp_stats_per_section()
            
        ############################
        logging.info("############################")
        if self.refresh_data("TG"):
            if self.data.tg_data is not None:
                logging.info("Processing TG data")
                self.process_tg_data()
            else:
                logging.error("No input TG data provided")
        else:
            logging.info("Using existing TG data from database")
    
    def process_base_geometry(self) -> None:
        """
        Process the input path layer into the separate rail lines:
         - mainline
         - thomas
         - eliwana
         - solomon
         
        Apply some corrections and cleanup of data format.
        
        The resulting gdf are contained in self.main_lines and self.bypass_lines
        """
        # Point-of-Interest dataframe
        # eg: origin - 0km chainage point (source: Chainage Mainline.shp)
        poi_df = pd.DataFrame({
            "geometry": [
                Point(662669.066, 7746597.522),  # origin
                Point(664948.312, 7721405.294),  # mainline_start
            ],
            "name": [
                "origin",
                "mainline_start"
            ]
        })
        self.poi_gdf = gpd.GeoDataFrame(poi_df, crs=self.working_crs)
        self.poi_gdf = move_points_onto_lines(self.poi_gdf, self.data.path)
        
        # load in the base geometry for all line features
        base_features = {}
        for base_type in self.data.path["type"].unique().tolist():
            logging.info(f"Loading track types: {base_type}")
            base_features[base_type] = gpd.GeoDataFrame(self.data.path[self.data.path["type"] == base_type])

        # extract main lines and bypass lines
        self.main_lines = base_features["main"].set_index("line_code")
        self.bypass_lines = base_features["bypass"]

        logging.info(f"Found {len(self.main_lines)} main line features, total length: {round(sum(self.main_lines.geometry.length), 2)}")
        logging.info(f"Found {len(self.bypass_lines)} bypass line features, total length: {round(sum(self.bypass_lines.geometry.length), 2)}")
    
    def process_into_chainage_segments(self, segment_length: int) -> None:
        """
        Segment each line into equal length segments
        """
        self.segment_data = {}
        
        if self.segment_method == "ignore_curve_boundaries":
            logging.info("Segmenting Thomas")
            self.create_chainage_segments(self.lines["TL"], "TL", segment_length, chainage_end=self.THOMAS_END, reverse=True)
            logging.info("Segmenting Mainline")
            self.create_chainage_segments(self.lines["ML"], "ML", segment_length, chainage_start=self.THOMAS_END)
            logging.info("Segmenting Solomon")
            self.create_chainage_segments(self.lines["SL"], "SL", segment_length, chainage_start=self.SOLOMON_START)
            logging.info("Segmenting Eliwana")
            self.create_chainage_segments(self.lines["EL"], "EL", segment_length, chainage_start=self.ELIWANA_START)

            # Combine all segment data from all lines into a single list
            all_segments = []
            for _, (_, segments_gdf) in self.segment_data.items():
                all_segments.extend(segments_gdf.to_crs("epsg:4326").to_dict('records'))
                
            # Post all data to the database at once
            post_to_db(all_segments, RailSegment, self.db.engine, action="replace")
                
        elif self.segment_method == "respect_curve_boundaries":
            self.create_chainage_segments_repecting_curves(target_segment_length=segment_length)
    
    def create_chainage_segments(
        self,
        layer: gpd.GeoDataFrame,
        line_code: str,
        segment_length: int = 100,
        chainage_start: float = 0,
        chainage_end: float = 0,
        reverse: bool = False,
    ):
        segments = split_lines_into_segments(layer, segment_length, perserve_vertices=False, reverse=reverse)
        
        new_segments = []
        inc = segment_length / 1000

        track_data = self.data.track_data.get(line_code)
        if track_data is None:
            logging.critical(f"No track data found for line: {line_code}")
            return
        
        track_data.set_index("Chainage ID", inplace=True)

        if chainage_end:
            chainage_start = chainage_end - inc * len(segments)

        distance_from_start = 0
        start_km = chainage_start
        for _, row in segments.iterrows():
            end_km = round(start_km + inc, 2)

            chainage_id = f"CHAIN-{line_code}-S-{int(10*abs(start_km)):0>5}-E-{int(10*abs(end_km)):0>5}-MAINLINE"
            
            centroid_en = row.geometry.centroid.coords[0]
            centroid_ll = self.PROJ.transform(centroid_en[0], centroid_en[1])
            
            # bring in track data, but just the basic info
            _chainage_id = chainage_id
            if line_code == "TL":
                _chainage_id = chainage_id.replace("TL", "ML")

            track_segment_data = {
                clean_string(k): v for k, v in track_data.loc[_chainage_id].to_dict().items()
            }

            # Convert geometry from working CRS to EPSG:4326
            # geom = transform(self.PROJ.transform, row["geometry"])

            segment_data = {
                "chainage_id": chainage_id,
                "line": self.line_code_to_name[line_code],
                "line_code": line_code,
                "section": track_segment_data["section"],
                "section_name": track_segment_data["section_name"],
                "station": track_segment_data["station"],
                "chainage_start_km": track_segment_data["chainage_start_km"],
                "chainage_end_km": track_segment_data["chainage_end_km"],
                "curve_type": "",  # todo: populate
                "mid_coord_lng": centroid_ll[0],
                "mid_coord_lat": centroid_ll[1],
                "geometry": row["geometry"],
            }

            input_chainage_start = round(segment_data["chainage_start_km"], 1)
            assert abs(input_chainage_start - round(start_km, 1)) < 0.01
            segment_data["chainage_start_km"] = input_chainage_start

            input_chainage_end = round(segment_data["chainage_end_km"], 1)
            assert abs(input_chainage_end - round(end_km, 1)) < 0.01
            segment_data["chainage_end_km"] = input_chainage_end

            segment_data["asset_ids"] = set()

            start_km = end_km
            distance_from_start += inc

            new_segments.append(segment_data)

        segments_gdf = gpd.GeoDataFrame(new_segments, crs=self.working_crs)

        _geoms = list(segments_gdf.geometry)
        self.segment_data[line_code] = (STRtree(_geoms), segments_gdf)
    
    def create_chainage_segments_repecting_curves(self, target_segment_length: float = 100.0):
        """
        Create rail segments by splitting curve/tangent sections into equal-length segments.

        Uses CURVE_SECTIONS data (chainages) + self.main_lines (geometry) to create segments
        that don't span curve/tangent boundaries. These segments are stored in self.segment_data.

        Args:
            target_segment_length: Target length for each segment in meters (default 100m)
        """
        # Collect curve sections from all lines
        curve_sections_list = []
        for _, line_code in self.line_name_to_code.items():
            curve_data = self.data.track_data.get(f"CURVE_SECTIONS_{line_code}")
            if curve_data is not None:
                curve_data_copy, spiral_data = expand_curve_spirals(curve_data, line_code)
                curve_sections_list.append(curve_data_copy)
                curve_sections_list.append(spiral_data)

        if len(curve_sections_list) == 0:
            logging.warning("No input curve section data found")
            return

        curve_sections = pd.concat(curve_sections_list, ignore_index=True)
        curve_sections.rename(columns={c: clean_string(c) for c in curve_sections.columns}, inplace=True)
        curve_sections.rename(columns={"asset_name": "id"}, inplace=True)

        all_segments = []

        for idx, row in curve_sections.iterrows():
            section_id = row["id"]
            line_code = row["line_code"]
            
            chainage_start = float(row["start_chainage"])
            chainage_end = float(row["end_chainage"])
            curve_length_1 = float(row["curve_length"])
            
            # manual override for Thomas line
            if idx == 0:
                chainage_start = self.THOMAS_START
            
            # manual override for intersection between Thomas and Main lines
            if chainage_start == 26.902505:
                chainage_start = self.THOMAS_END
                curve_length_1 = 58.66
            if chainage_end == 26.902505:
                chainage_end = self.THOMAS_END
                curve_length_1 = 10007.805
            
            
            curve_length_2 = 1000 * abs(chainage_end - chainage_start)
            if abs(curve_length_1 - curve_length_2) > 0.01:
                logging.warning(f"{section_id}: curve length mismatch ({curve_length_1} vs {curve_length_2}), using {curve_length_2}")
            curve_length = curve_length_2

            if chainage_start == chainage_end or curve_length <= 0:
                logging.warning(f"{section_id}: zero length, skipping")
                continue
            
            # identify Thomas line segments
            if chainage_start < self.THOMAS_END:
                line_code = "TL"

            # Get the line geometry for this line_code
            line_geom = self.main_lines.loc[line_code].geometry

            # Account for line offset - chainage is relative to origin, line geometry starts at offset
            line_offset = self.START_OFFSET[line_code]
            start_distance = (chainage_start - line_offset) * 1000  # Convert km to meters
            end_distance = (chainage_end - line_offset) * 1000

            # Get normalized positions directly
            line_length = line_geom.length
            start_normalized = start_distance / line_length
            end_normalized = end_distance / line_length
            
            # print(line_length, start_distance, start_normalized, end_distance, end_normalized)
            
            # Extract the section geometry
            section_geom = substring(line_geom, start_normalized, end_normalized, normalized=True)

            # Calculate number of segments
            num_segments = max(1, round(curve_length / target_segment_length))
            actual_segment_length = curve_length / num_segments

            logging.debug(f"{section_id}: {curve_length}m → {num_segments} × {actual_segment_length:.1f}m")
            
            classification = clean_string(row["classification"])
            
            # Split into segments
            for seg_idx in range(num_segments):
                # Calculate chainage for this segment
                seg_start_km = chainage_start + (seg_idx * actual_segment_length / 1000)
                seg_end_km = chainage_start + ((seg_idx + 1) * actual_segment_length / 1000)
                seg_mid_km = (seg_start_km + seg_end_km) / 2
                seg_len = round(1000 * abs(seg_end_km - seg_start_km), 2)

                # Extract geometry for this segment
                start_norm = seg_idx / num_segments
                end_norm = (seg_idx + 1) / num_segments
                seg_geom = substring(section_geom, start_norm, end_norm, normalized=True)

                # Get midpoint coordinates
                seg_midpoint = seg_geom.interpolate(0.5, normalized=True)
                mid_coord_en = seg_midpoint.coords[0]
                mid_coord_ll = self.PROJ.transform(mid_coord_en[0], mid_coord_en[1])

                # Create chainage_id string
                chainage_id = f"CHAIN-{line_code}-S-{int(100*abs(seg_start_km)):0>5}-E-{int(100*abs(seg_end_km)):0>5}-MAINLINE"
                
                section_data = get_section(seg_start_km, line_code, line_class="main")
                if section_data:
                    if len(section_data) == 1:
                        section_name, section_id = section_data[0]
                    else:
                        logging.warning(
                            f"Found multiple track sections at chainage: {seg_start_km}, line_code: {line_code} \
                            and line_type: 'main'. Check the track section definitions."
                        )
                        section_name, section_id = section_data[0]
                else:
                    section_id, section_name = "", ""
                
                station = get_station(seg_start_km)
                
                segment_data = {
                    "chainage_id": chainage_id,
                    "line": self.line_code_to_name[line_code],
                    "line_code": line_code,
                    "section": section_id,
                    "section_name": section_name,
                    "station": station,
                    "chainage_start_km": round(seg_start_km, 6),
                    "chainage_end_km": round(seg_end_km, 6),
                    "chainage_mid_km": round(seg_mid_km, 6),
                    "segment_length_m": seg_len,
                    "mid_coord_lng": round(mid_coord_ll[0], 6),
                    "mid_coord_lat": round(mid_coord_ll[1], 6),
                    "curve_type": classification,
                    "max_speed": 0,  # todo
                    "geometry": seg_geom,
                }

                all_segments.append(segment_data)

        logging.info(f"Created {len(all_segments)} segments from {len(curve_sections)} curve/tangent sections")

        # Store as GeoDataFrame
        segments_gdf = gpd.GeoDataFrame(all_segments, crs=self.working_crs)

        # Replace self.segment_data - organize by line_code with STRtree
        self.segment_data = {}
        for line_code in self.line_code_to_name.keys():
            line_segments = segments_gdf[segments_gdf.line_code == line_code].reset_index(drop=True)
            if len(line_segments) > 0:
                _geoms = list(line_segments.geometry)
                self.segment_data[line_code] = (STRtree(_geoms), line_segments)
                logging.info(f"{line_code}: {len(line_segments)} segments")

        # Post to database
        post_to_db(all_segments, RailSegment, self.db.engine, action="replace")
    
    def get_coords_from_chainage(self, line_code: str, chainage: float) -> dict[str, float]:
        """
        Given a line and a chainage (in km), return the coordinates of the point on the line
        at the given chainage.

        Args:
            line_code: Line ID (TL, ML, SL, EL)
            chainage: Chainage in kilometers

        Returns:
            dict: Dictionary with keys 'E', 'N', 'lng', 'lat'
        """
        line_geom = self.main_lines.loc[line_code].geometry
        line_starting_chainage = self.START_OFFSET[line_code]

        # Calculate distance along the line from its start (in meters)
        distance_from_start = (chainage - line_starting_chainage) * 1000
        if chainage - line_starting_chainage < 0:
            logging.error(
                f"Negative distance_from_start value in function 'get_coords_from_chainage', \
                    line: {line_code}, chainage: {chainage}"
                )
            distance_from_start = 0
            
        # Get the point at the specified distance
        point = line_geom.interpolate(distance_from_start)

        # Get E/N coordinates
        coords_en = point.coords[0]

        # Transform to lat/lng
        coords_ll = self.PROJ.transform(coords_en[0], coords_en[1])

        return {
            'E': coords_en[0],
            'N': coords_en[1],
            'lng': coords_ll[0],
            'lat': coords_ll[1]
        }

    def create_assets(self) -> None:
        """
        Parse the assets input data ready for insertion into db
        """
        input_asset_data = self.data.track_data.get("ASSETS")
        
        if input_asset_data is None:
            logging.error("No input asset data found")
        
        def find_asset_segment(geom: Point, segment_data: tuple[STRtree, gpd.GeoDataFrame], chainage: float, max_distance: float = 10, debug: bool = False):
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
    
    def process_sap_records(self) -> None:
        """
        Process the SAP work order records into a suitable format
        """
        input_sap_data = self.data.track_data.get("SAP")
        
        if input_sap_data is None:
            return
        
        sap_records = []
        for _, row in input_sap_data.iterrows():
            asset_id = row["Functional Location"]
            
            sap_record_data = {
                "date": row["Start Date"],
                "year": row["Year"],
                "revision": row["Revision"],
                "planner_grp": row["Maint Planner Group"],
                "work_center": row["Maint Work Center"],
                "user_status": row["User Status"],
                "system_status": row["System Status"],
                "priority": row["Priority Text"],
                "order_id": row["Order"],
                "description": row["Description"],
                "asset_id": asset_id,
                "iridium_code": row["Iridium Code"],
            }
            
            sap_records.append(sap_record_data)

        self.sap_records = pd.DataFrame(sap_records)
        
        # post data to the database
        post_to_db(sap_records, SAPRecord, self.db.engine, action="replace")
    
    def process_gbfi_data(self, collection_date: datetime) -> None:
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
                
            # collection_date = None
            # gbfi_header = self.data.track_data.get(f"GBFI_{line_code}-HEADER")
            # if gbfi_header is not None:
            #     collection_date = extract_collection_date_from_header(gbfi_header)
            # elif self.data.collection_dates.get("GBFI"):
            #     collection_date = self.data.collection_dates["GBFI"]
            
        if len(gbfi_list) == 0:
            logging.warning("No input GBFI data found")
            return

        gbfi_data = pd.concat(gbfi_list, ignore_index=True)

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
        post_to_db(gbfi_records, GBFIRecord, self.db.engine, action="replace")
            
    def aggregate_gbfi_to_segments(self, collection_date: datetime) -> None:
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
        post_to_db(agg_gbfi, AggGBFI, self.db.engine, action="replace")
    
    def process_tsr_data(self) -> None:
        """
        Process the TSR into standard format for database insertion
        """
        # Collect TSR for all lines
        tsr_data_open = self.data.track_data.get("TSR_OPEN")
        tsr_data_complete = self.data.track_data.get("TSR_COMPLETE")
        
        if tsr_data_open is None and tsr_data_complete is None:
            logging.warning("No input TSR data found")
            return
        
        if tsr_data_open is not None:
            tsr_data_open["status"] = "open"
        
        if tsr_data_complete is not None:
            tsr_data_complete["status"] = "complete"
        
        tsr_data = pd.concat([tsr_data_open, tsr_data_complete], ignore_index=True)
        tsr_data = tsr_data.rename(columns={c: clean_string(c) for c in tsr_data.columns})
        
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
            
            report_date = pd.to_datetime(row["report_date"], format="%d/%m/%Y") if pd.notna(row["report_date"]) else None
            close_date = pd.to_datetime(row["close_date"], format="%d/%m/%Y") if pd.notna(row["close_date"]) else None

            tsr_record = {
                "report_date": report_date,
                "line_code": line_code,
                "status": row["status"],
                "chainage_start_km": float(row["start_chainage"]),
                "chainage_end_km": float(row["end_chainage"]),
                "speed": float(row["speed"]),
                "close_date": close_date,
            }
            tsr_records.append(tsr_record)
            
        # post data to the database
        post_to_db(tsr_records, TSRRecord, self.db.engine, action="replace")
    
    def aggregate_tsr_to_segments(self) -> None:
        """
        Aggregate TSR data to rail segments.
        
        rail segments:      A-----------------B-----------------C-----------------D
        tsr:                        X-------------------------------------Y
        
        associate the TSR with segments: AB, BC & CD
        """
        # Load ALL TSR data once (instead of querying for each segment)
        logging.info("Loading TSR data from database...")
        all_tsr_data = get_table_data(self.db.engine, table_name="tsr_records")

        if all_tsr_data.empty:
            logging.warning("No TSR data found")
            return

        # Convert report_date to datetime
        all_tsr_data["report_date"] = pd.to_datetime(all_tsr_data["report_date"], utc=True)

        # Extract year for faster filtering
        all_tsr_data["year"] = all_tsr_data["report_date"].dt.year

        # Split into open and complete TSR records
        open_tsr_data = all_tsr_data[all_tsr_data["status"] == "open"].copy()
        complete_tsr_data = all_tsr_data[all_tsr_data["status"] == "complete"].copy()

        logging.info(f"Processing {len(all_tsr_data)} TSR records across segments...")

        agg_tsr = []
        for line_code, (_, segments_gdf) in self.segment_data.items():
            # Filter TSR data for this line code once
            line_open_tsr = open_tsr_data[open_tsr_data["line_code"] == line_code]
            line_complete_tsr = complete_tsr_data[complete_tsr_data["line_code"] == line_code]

            for _, segment in segments_gdf.iterrows():
                chainage_id = segment.chainage_id
                seg_chainage_start = segment.chainage_start_km
                seg_chainage_end = segment.chainage_end_km

                # Filter open TSR records for this segment (in memory, fast)
                segment_open_tsr = line_open_tsr[
                    (line_open_tsr["chainage_end_km"] >= seg_chainage_start) &
                    (line_open_tsr["chainage_start_km"] < seg_chainage_end)
                ]
                # logging.debug(
                #     f"open: {chainage_id}, {seg_chainage_start}, {seg_chainage_end}, {len(segment_open_tsr)}"
                # )

                # Calculate days open
                days_open = 0
                if not segment_open_tsr.empty:
                    days_open = (pd.Timestamp.now(tz='UTC') - segment_open_tsr["report_date"]).dt.days.sum()

                # Filter complete TSR records for this segment (in memory, fast)
                segment_complete_tsr = line_complete_tsr[
                    (line_complete_tsr["chainage_end_km"] >= seg_chainage_start) &
                    (line_complete_tsr["chainage_start_km"] < seg_chainage_end)
                ]
                # logging.debug(
                #     f"complete: {chainage_id}, {seg_chainage_start}, {seg_chainage_end}, {len(segment_complete_tsr)}"
                # )

                # Count by year
                if segment_complete_tsr.empty:
                    cnt_2022 = cnt_2023 = cnt_2024 = cnt_2025 = cnt_2026 = 0
                else:
                    year_counts = segment_complete_tsr["year"].value_counts()
                    cnt_2022 = year_counts.get(2022, 0)
                    cnt_2023 = year_counts.get(2023, 0)
                    cnt_2024 = year_counts.get(2024, 0)
                    cnt_2025 = year_counts.get(2025, 0)
                    cnt_2026 = year_counts.get(2026, 0)

                agg_tsr_data = {
                    "chainage_id": chainage_id,
                    "open_tsr": not segment_open_tsr.empty,
                    "open_tsr_days": int(days_open),
                    "complete_tsr": not segment_complete_tsr.empty,
                    "cnt_2022": int(cnt_2022),
                    "cnt_2023": int(cnt_2023),
                    "cnt_2024": int(cnt_2024),
                    "cnt_2025": int(cnt_2025),
                    "cnt_2026": int(cnt_2026),
                }
                agg_tsr.append(agg_tsr_data)

        logging.info(f"Aggregated TSR data for {len(agg_tsr)} segments")

        # post data to the database
        post_to_db(agg_tsr, AggTSR, self.db.engine, action="replace")
    
    def create_curve_sections(self) -> None:
        def _get_segment(line_id: str, chainage: float):
            if line_id == "SL":
                line_code = "SL"
            elif line_id == "EL":
                line_code = "EL"
            else:
                if chainage <= self.THOMAS_END:
                    line_code = "TL"
                else:
                    line_code = "ML"
            gdf = self.segment_data[line_code][1]

            c1 = gdf["chainage_start_km"] <= chainage
            c2 = gdf["chainage_end_km"] >= chainage
            segment = gdf[c1 & c2]

            if len(segment) > 0:
                return segment, line_code
            else:
                return None, line_code

        def _get_curve_geom(line_code: str, curve_geom_start: Point, curve_geom_end: Point):
            line_geom = self.main_lines.loc[line_code].geometry
            proj1 = line_geom.project(curve_geom_start, normalized=True)
            proj2 = line_geom.project(curve_geom_end, normalized=True)
            
            # Ensure proj1 < proj2 for substring
            start, end = sorted([proj1, proj2])
            
            # Extract the segment between the two points
            segment = substring(line_geom, start, end, normalized=True)
            return segment

        new_curve_sections = []
        curve_intervals = defaultdict(list)

        # Collect curve sections from all lines
        curve_sections_list = []
        for _, line_code in self.line_name_to_code.items():
            # retrieve the input data
            curve_data = self.data.track_data.get(f"CURVE_SECTIONS_{line_code}")
            if curve_data is not None:
                curve_sections_list.append(curve_data)

        # Combine all curve sections
        if len(curve_sections_list) == 0:
            logging.warning("No input curve section data found")
            return

        self.curve_sections = pd.concat(curve_sections_list, ignore_index=True)
        self.curve_sections.rename(columns={c: clean_string(c) for c in self.curve_sections.columns}, inplace=True)
        self.curve_sections.rename(columns={"asset_name": "id"}, inplace=True)

        for idx, row in self.curve_sections.iterrows():
            curve_id = row["id"]
            chainage_start = float(row["start_chainage"])
            chainage_end = float(row["end_chainage"])
            
            # manual override for Thomas line
            if idx == 0 and chainage_start == 0:
                chainage_start = self.THOMAS_START
            
            if chainage_start == chainage_end:
                logging.warning(f"{curve_id} has zero length")
                continue
            
            section_seg_start, line_code_start = _get_segment(row["section"], chainage_start)
            curve_intervals[line_code_start].append(Interval(chainage_start, chainage_end, curve_id))
            
            if section_seg_start is not None:
                section_seg_start = section_seg_start.iloc[0]
                delta = chainage_start - section_seg_start["chainage_start_km"]
                coords_en = section_seg_start["geometry"].interpolate(1000 * delta).coords[-1]
                curve_geom_start = Point(coords_en)
            else:
                logging.warning(f"{line_code_start} {chainage_start}: start id=not found")

            section_seg_end, line_code_end = _get_segment(row["section"], chainage_end)
            if section_seg_end is not None:
                section_seg_end = section_seg_end.iloc[0]
                delta = chainage_end - section_seg_end["chainage_start_km"]
                coords_en = section_seg_end["geometry"].interpolate(1000 * delta).coords[-1]
                curve_geom_end = Point(coords_en)
            else:
                logging.warning(f"{line_code_end} {chainage_end}: end id=not found")

            #curve_geom = _get_curve_geom(prefix_start, curve_geom_start, curve_geom_end, chainage_start, chainage_end)
            curve_geom = _get_curve_geom(line_code_start, curve_geom_start, curve_geom_end)
            
            if isinstance(curve_geom, Point):
                logging.warning(f"Curve section {curve_id} resulted in a Point geometry, skipping")
                continue

            curve_centroid_en = curve_geom.centroid.coords[0]
            curve_centroid_ll = self.PROJ.transform(curve_centroid_en[0], curve_centroid_en[1])
            
            curve_type = clean_string(row["type"])
            curve_section_data = {
                "section_id": curve_id,
                "line": self.line_code_to_name[line_code_start],
                "line_code": line_code_start,
                "type": curve_type,
                "chainage_start_km": float(row["start_chainage"]),
                "chainage_end_km": float(row["end_chainage"]),
                "curve_length": float(row["curve_length"]),
                "classification": clean_string(row["classification"]),
                "mid_coord_lng": curve_centroid_ll[0],
                "mid_coord_lat": curve_centroid_ll[1],
                "geometry": curve_geom,
            }
                
            if curve_type == "curve":
                # only populated for 'curve' types
                extra_curve_section_data = {
                    "curve_id": row["curve_id"],
                    "ts": float(row["ts"]),
                    "sc": float(row["sc"]),
                    "cs": float(row["cs"]),
                    "st": float(row["st"]),
                    "radius": float(row["radius"]),
                    "hand": row["hand"],
                    "se_design": int(row["superelevation_design"]),
                    "gradient": float(row["track_gradient"]),
                }
            else:
                extra_curve_section_data = {
                    "curve_id": None,
                    "ts": None,
                    "sc": None,
                    "cs": None,
                    "st": None,
                    "radius": None,
                    "hand": None,
                    "se_design": None,
                    "gradient": None,
                }
                
            curve_section_data.update(extra_curve_section_data)

            # _split_section(prefix_start, curve_geom_start, curve_geom_end)
            new_curve_sections.append(curve_section_data)

        self.curve_sections = gpd.GeoDataFrame(new_curve_sections, crs=self.working_crs)

        # post data to the database
        post_to_db(new_curve_sections, RailSection, self.db.engine, action="replace")
        
        self.curve_interval_tree = {}
        for line_id, intervals in curve_intervals.items():
            self.curve_interval_tree[line_id] = IntervalTree(intervals)
            logging.info(f"Interals for line {line_id}:")
            logging.info(f"  num intervals: {len(intervals)}")
            logging.info(f"  start interval: {intervals[0]}")
            logging.info(f"  end interval: {intervals[-1]}")

    def aggregate_assets_to_segments(self) -> None:
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
        post_to_db(agg_assets, AggAsset, self.db.engine, action="replace")
    
    def process_rp_data_into_sections(self):
        """
        process the RP data into sections
        """
        def _apply_corrections(gdf: gpd.GeoDataFrame):
            gdf.rename(columns={"SF": "id"}, inplace=True)
            gdf.rename(columns={c: clean_string(c) for c in gdf.columns}, inplace=True)
            gdf["id"] = gdf.apply(lambda x: _clean_id(x), axis=1)
            gdf["line_id"] = gdf.apply(lambda x: _get_line_id(x), axis=1)
            gdf["line_region"] = gdf.apply(lambda x: _get_line_region(x), axis=1)
            gdf["chainage"] = gdf.apply(lambda x: x["major"] + x["minor"] / 1000, axis=1)
            gdf.to_crs(self.working_crs, inplace=True)

        def _clean_id(data):
            data['id'] = data['id'].replace("Jones to CCK_MLE", "Jones to CCK_MLB")
            data['id'] = data['id'].replace("Jones to CCK", "Jones")
            return data['id']
            
        def _get_section_id_old(data):
            # this relies on geometry alignment via the calculated field "len_from_line_start"
            line_id = data["line_id"]
            chainage = self.START_OFFSET[line_id] + data["len_from_line_start"] / 1000
            matches = self.curve_interval_tree[line_id][chainage]
            if len(matches) > 0:
                return matches.pop().data
            else:
                logging.warning(f"Could not locate RP data point {data['id']} at chainage {chainage} in a curve section")
                return None
        
        def _get_section_id(data: dict, d_type: str):
            line_id = data["line_id"]
            chainage = data["major"] + (data["minor"] / 1000)
            matches = self.curve_interval_tree[line_id][chainage]
            if len(matches) > 0:
                return matches.pop().data
            else:
                logging.debug(f"Could not locate {d_type} data point {data['id']} at chainage {chainage} in a curve section")
                return None
        
        def _get_line_id(data):
            pieces = data["id"].split("_")
            if pieces[1] in main_line_identifiers:
                # correction for the Thomas line
                if pieces[0] == "Thomas" and pieces[1] == "MLW":
                    chainage = data["major"] + (data["minor"] / 1000)
                    if chainage <= self.THOMAS_END:
                        return "TL"
                    else:
                        return "ML"
                # correction for the Firetail section
                elif pieces[0] == "Firetail" and pieces[1] == "ELB":
                    return "SL"
                # correction for the Future section
                elif pieces[0] == "Future" and pieces[1] == "SLB":
                    return "EL"
                else:
                    return main_line_identifiers[pieces[1]]
            else:
                other_set.add(pieces[1])
                return "OTH"

        def _get_line_region(data):
            pieces = data["id"].split("_")
            region = pieces[0]
            if not region in line_regions:
                logging.warning(f"Unidentified region for rp id: {data['id']}")
            return region

        # add some data attributes
        if "id" in self.data.rp_data.columns:
            rp_data = self.data.rp_data
            tg_data = self.data.tg_data
            logging.info("Using an already corrected dataset")
        else:
            logging.info("Applying corrections to dataset")
            _apply_corrections(self.data.rp_data)
            _apply_corrections(self.data.tg_data)
            # self.data.rp_data.rename(columns={"SF": "id"}, inplace=True)
            # self.data.rp_data.rename(columns={c: clean_string(c) for c in self.data.rp_data.columns}, inplace=True)
            # self.data.rp_data["id"] = self.data.rp_data.apply(lambda x: _clean_id(x), axis=1)
            # self.data.rp_data["line_id"] = self.data.rp_data.apply(lambda x: _get_line_id(x), axis=1)
            # self.data.rp_data["line_region"] = self.data.rp_data.apply(lambda x: _get_line_region(x), axis=1)
            # self.data.rp_data["chainage"] = self.data.rp_data.apply(lambda x: x["major"] + x["minor"] / 1000, axis=1)
            # self.data.rp_data.to_crs(self.working_crs, inplace=True)

            # filter the data to just the main lines
            rp_data = self.data.rp_data
            rp_data = copy.copy(rp_data[rp_data["line_id"] != "OTH"])
            
            tg_data = self.data.tg_data
            tg_data = copy.copy(tg_data[tg_data["line_id"] != "OTH"])

            """
            This method aligns the lat/long coords present in the rp data files to the
            underlying rail lines (as they are slightly offset).  This can be quite slow for large datasets
            
            The function move_points_onto_lines and appends the field "len_from_line_start" to the data. However this
            value is also provided in the rp data (attributes: Major and Minor).  Trialling both seems to work
            just as well so we will go for the simpler option.
            """
            # rp_data = move_points_onto_lines(rp_data, self.combined_lines, max_distance=150.0)

            write_db = True
            if write_db:
                import duckdb
                duckdb_path = os.path.join(self.output_dir, "ensco_data_corrected.duckdb")
                _tmp_rp_data = copy.copy(rp_data)
                _tmp_rp_data['geometry_wkb'] = _tmp_rp_data.geometry.apply(lambda geom: to_wkt(geom) if geom is not None else None)
                _tmp_rp_data = _tmp_rp_data.drop(columns="geometry")
                _tmp_tg_data = copy.copy(tg_data)
                _tmp_tg_data['geometry_wkb'] = _tmp_tg_data.geometry.apply(lambda geom: to_wkt(geom) if geom is not None else None)
                _tmp_tg_data = _tmp_tg_data.drop(columns="geometry")

                con = duckdb.connect(duckdb_path)
                con.execute("DROP TABLE IF EXISTS rp_data")
                con.execute(f"""CREATE TABLE rp_data AS SELECT * FROM _tmp_rp_data""")
                con.execute("DROP TABLE IF EXISTS tg_data")
                con.execute(f"""CREATE TABLE tg_data AS SELECT * FROM _tmp_tg_data""")

            logging.info(f"Other set: {other_set}")

        # find the curve section id
        rp_data["section_id"] = rp_data.apply(lambda x: _get_section_id(x, "rp"), axis=1)
        tg_data["section_id"] = tg_data.apply(lambda x: _get_section_id(x, "tg"), axis=1)

        # logging.info("Writing processed RP data")
        # path_out = os.path.join(self.output_dir, f"rp_data.{self.output_fmt}")
        # rp_data[rp_data["line_region"] == "Nunna-2"].to_crs('epsg:4326').to_file(path_out)

        self.rp_processed = rp_data
        self.tg_processed = tg_data

    def calculate_rp_stats_per_section(self):
        """
        Calculate various stats for each curve section
        """
        # 6 params, 3 for each track
        # WEST: relative_head_loss, vertical_wear, gauge_side_wear
        # EAST: relative_head_loss, vertical_wear, gauge_side_wear
        # O=ok, X=exceeds threshold
        metrics = {"avg", "p50", "p75", "p90"}

        def _update_status_string(status_string, track, param, colour):
            idx = 0 if track == "w" else 4

            if param == "rel_head_loss":
                idx += 0
            elif param == "vert_wear":
                idx += 1
            elif param == "side_wear":
                idx += 2

            return f"{status_string[:idx]}{colour}{status_string[idx+1:]}"

        def _status_score(status_string: str):
            """
            Score the status string, where:
            G = 0, A = 1, R = 3
            """
            score = 0
            for c in status_string:
                if c == "A":
                    score += 1
                elif c == "R":
                    score += 3
            return score

        def _status_level(status_string: str):
            if "R" in status_string:
                return "R"
            elif "A" in status_string:
                return "A"
            else:
                return "G"

        def _get_rp_stats(data):
            curve_id = data["section_id"].iloc[0]
            section_data = self.curve_sections[self.curve_sections["id"] == curve_id].iloc[0].to_dict()

            section_data["num_points"] = len(data)

            section_data["min_e_vert_wear"] = min(data["east_vertical_wear"])
            section_data["max_e_vert_wear"] = max(data["east_vertical_wear"])
            section_data["avg_e_vert_wear"] = np.mean(data["east_vertical_wear"])
            section_data["std_e_vert_wear"] = np.std(data["east_vertical_wear"])
            section_data["p50_e_vert_wear"] = np.percentile(data["east_vertical_wear"], 50)
            section_data["p75_e_vert_wear"] = np.percentile(data["east_vertical_wear"], 75)
            section_data["p90_e_vert_wear"] = np.percentile(data["east_vertical_wear"], 90)

            section_data["min_w_vert_wear"] = min(data["west_vertical_wear"])
            section_data["max_w_vert_wear"] = max(data["west_vertical_wear"])
            section_data["avg_w_vert_wear"] = np.mean(data["west_vertical_wear"])
            section_data["std_w_vert_wear"] = np.std(data["west_vertical_wear"])
            section_data["p50_w_vert_wear"] = np.percentile(data["west_vertical_wear"], 50)
            section_data["p75_w_vert_wear"] = np.percentile(data["west_vertical_wear"], 75)
            section_data["p90_w_vert_wear"] = np.percentile(data["west_vertical_wear"], 90)

            section_data["min_e_side_wear"] = min(data["east_gauge_side_wear"])
            section_data["max_e_side_wear"] = max(data["east_gauge_side_wear"])
            section_data["avg_e_side_wear"] = np.mean(data["east_gauge_side_wear"])
            section_data["std_e_side_wear"] = np.std(data["east_gauge_side_wear"])
            section_data["p50_e_side_wear"] = np.percentile(data["east_gauge_side_wear"], 50)
            section_data["p75_e_side_wear"] = np.percentile(data["east_gauge_side_wear"], 75)
            section_data["p90_e_side_wear"] = np.percentile(data["east_gauge_side_wear"], 90)

            section_data["min_w_side_wear"] = min(data["west_gauge_side_wear"])
            section_data["max_w_side_wear"] = max(data["west_gauge_side_wear"])
            section_data["avg_w_side_wear"] = np.mean(data["west_gauge_side_wear"])
            section_data["std_w_side_wear"] = np.std(data["west_gauge_side_wear"])
            section_data["p50_w_side_wear"] = np.percentile(data["west_gauge_side_wear"], 50)
            section_data["p75_w_side_wear"] = np.percentile(data["west_gauge_side_wear"], 75)
            section_data["p90_w_side_wear"] = np.percentile(data["west_gauge_side_wear"], 90)

            section_data["min_e_rel_head_loss"] = min(data["east_relative_head_loss"])
            section_data["max_e_rel_head_loss"] = max(data["east_relative_head_loss"])
            section_data["avg_e_rel_head_loss"] = np.mean(data["east_relative_head_loss"])
            section_data["std_e_rel_head_loss"] = np.std(data["east_relative_head_loss"])
            section_data["p50_e_rel_head_loss"] = np.percentile(data["east_relative_head_loss"], 50)
            section_data["p75_e_rel_head_loss"] = np.percentile(data["east_relative_head_loss"], 75)
            section_data["p90_e_rel_head_loss"] = np.percentile(data["east_relative_head_loss"], 90)

            section_data["min_w_rel_head_loss"] = min(data["west_relative_head_loss"])
            section_data["max_w_rel_head_loss"] = max(data["west_relative_head_loss"])
            section_data["avg_w_rel_head_loss"] = np.mean(data["west_relative_head_loss"])
            section_data["std_w_rel_head_loss"] = np.std(data["west_relative_head_loss"])
            section_data["p50_w_rel_head_loss"] = np.percentile(data["west_relative_head_loss"], 50)
            section_data["p75_w_rel_head_loss"] = np.percentile(data["west_relative_head_loss"], 75)
            section_data["p90_w_rel_head_loss"] = np.percentile(data["west_relative_head_loss"], 90)

            clas = section_data["classification"]      # Tangent, Mild Curve or Sharp Curve
            clas = clas.replace(" ", "_").lower()
            hand = section_data["hand"]                # LH or RH

            # check various metrics against the thresholds
            for metric in metrics:
                for track in ["e", "w"]:
                    for param in ["rel_head_loss", "vert_wear", "side_wear"]:
                        if clas == "tangent":
                            stat = f"{metric}_{track}_{param}"
                            if section_data[stat] > self.THRESHOLDS.get(clas, param, hand, track):
                                section_data[f"status_{metric}_{track}_{param}"] = "exceeds"
                                logging.info(f"{curve_id}, {clas}, {track}, {param}, {metric}: {section_data[stat]} exceeds threshold {self.THRESHOLDS.get(clas, param, hand, track)}")
                                # _update_status_string(metric, track, param)
                            else:
                                section_data[f"status_{metric}_{track}_{param}"] = ""

                        elif clas == "mild_curve":
                            stat = f"{metric}_{track}_{param}"
                            if section_data[stat] > self.THRESHOLDS.get(clas, param, hand, track):
                                section_data[f"status_{metric}_{track}_{param}"] = "exceeds"
                                logging.info(f"{curve_id}, {clas}, {track}, {param}, {metric}: {section_data[stat]} exceeds threshold {self.THRESHOLDS.get(clas, param, hand, track)}")
                                # _update_status_string(metric, track, param)
                            else:
                                section_data[f"status_{metric}_{track}_{param}"] = ""

                        elif clas == "sharp_curve":
                            stat = f"{metric}_{track}_{param}"
                            if section_data[stat] > self.THRESHOLDS.get(clas, param, hand, track):
                                section_data[f"status_{metric}_{track}_{param}"] = "exceeds"
                                logging.info(f"{curve_id}, {clas}, {track}, {param}, {metric}: {section_data[stat]} exceeds threshold {self.THRESHOLDS.get(clas, param, hand, track)}")
                                # _update_status_string(metric, track, param)
                            else:
                                section_data[f"status_{metric}_{track}_{param}"] = ""

            # set red/orange/green status
            m1 = "p90"      # this is the more relaxed metric
            m2 = "p75"      # this is the stricter metric
            status_string = "GGG-GGG"

            for track in ["e", "w"]:
                for param in ["rel_head_loss", "vert_wear", "side_wear"]:
                    status_1 = section_data[f"status_{m1}_{track}_{param}"]
                    status_2 = section_data[f"status_{m2}_{track}_{param}"]
                    if status_1 == "exceeds" and status_2 == "exceeds":
                        status_string = _update_status_string(status_string, track, param, colour="R")
                    elif status_1 == "exceeds" and status_2 == "":
                        status_string = _update_status_string(status_string, track, param, colour="A")
                    elif status_1 == "" and status_2 == "":
                        status_string = _update_status_string(status_string, track, param, colour="G")

            section_data["status_string"] = status_string
            section_data["status_level"] = _status_level(status_string)
            section_data["status_score"] = _status_score(status_string)
            return section_data

        def _get_tg_stats(data):
            curve_id = data["section_id"].iloc[0]
            section_data = self.curve_sections[self.curve_sections["id"] == curve_id].iloc[0].to_dict()

            section_data["num_points"] = len(data)
            section_data["min_speed"] = np.min(data["speed"])
            section_data["avg_speed"] = np.mean(data["speed"])
            section_data["max_speed"] = np.max(data["speed"])
            section_data["min_post_speed"] = np.min(data["post_speed"])
            section_data["avg_post_speed"] = np.mean(data["post_speed"])
            section_data["max_post_speed"] = np.max(data["post_speed"])
            return section_data

        # group by section and calculate stats
        tg_sections = self.tg_processed.groupby("section_id").apply(lambda x: _get_tg_stats(x))
        tg_sections = pd.DataFrame(list(tg_sections), index=tg_sections.index).reset_index()
        rp_sections = self.rp_processed.groupby("section_id").apply(lambda x: _get_rp_stats(x))
        rp_sections = pd.DataFrame(list(rp_sections), index=rp_sections.index).reset_index()
        
        cols_to_add = ['section_id', 'min_speed', 'avg_speed', 'max_speed', 'min_post_speed', 'avg_post_speed', 'max_post_speed']
        rp_sections = rp_sections.merge(tg_sections[cols_to_add], on='section_id', how='left')

        self.rp_sections = gpd.GeoDataFrame(rp_sections, geometry="geometry", crs=self.working_crs)
        self.tg_sections = gpd.GeoDataFrame(tg_sections, geometry="geometry", crs=self.working_crs)

    def process_tg_data(self):
        """
        Process track geometry (TG) data into standard format for database insertion
        """
        tg_data = self.data.tg_data
        if tg_data is None:
            logging.info("No TG data found")
            return
        
        column_name_mapping = {
            "SF": "id",
            "East Top 5m": "top_e_5",
            "East Top 10m": "top_e_10",
            "East Top 20m": "top_e_20",
            "West Top 5m": "top_w_5",
            "West Top 10m": "top_w_10",
            "West Top 20m": "top_w_20",
            "East Alignment 5m": "align_e_5",
            "East Alignment 10m": "align_e_10",
            "East Alignment 20m": "align_e_20",
            "West Alignment 5m": "align_w_5",
            "West Alignment 10m": "align_w_10",
            "West Alignment 20m": "align_w_20",
            "Gauge": "gauge",
            "Crosslevel": "crosslevel",
            "Crosslevel Rate": "crosslevel_rate",
            "Curve": "curve",
            "Curve Rate": "curve_rate",
            "Twist 2m": "twist_2",
            "Twist 7m": "twist_7",
            "Twist 14m": "twist_14",
            "Warp Medium": "warp",
            "Valid": "valid",
        }
        
        logging.info("applying corrections to raw data")
        apply_corrections(tg_data, self.THOMAS_END, column_name_mapping)
        tg_data = tg_data[tg_data["line_code"] != "OTH"].copy()
        logging.info("...done")

        # Filter out rows with missing critical data
        tg_data = tg_data.dropna(subset=['chainage_km'])

        # Select columns that match TGRecord schema
        columns_to_extract = [
            'chainage_km', 'speed', 'post_speed', 'trk_class',
            'top_e_5', 'top_e_10', 'top_e_20',
            'top_w_5', 'top_w_10', 'top_w_20',
            'align_e_5', 'align_e_10', 'align_e_20',
            'align_w_5', 'align_w_10', 'align_w_20',
            'gauge', 'crosslevel', 'crosslevel_rate',
            'curve', 'curve_rate',
            'twist_2', 'twist_7', 'twist_14',
            'warp', 'valid',
            'geometry', 'collection_date'
        ]

        # Convert to list of dictionaries
        tg_records = tg_data[columns_to_extract].to_dict('records')

        logging.info(f"Processing {len(tg_records)} TG records for database insertion")

        # Post data to the database
        post_to_db(tg_records, TGRecord, self.db.engine, action="replace")
    
    def process_tqi_data(self, collection_date: datetime) -> None:
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
        tqi_data = tqi_data.dropna(subset=['line_code', 'tqi'])

        # Select columns that match TGRecord schema
        columns_to_extract = ["line", "line_code", "line_class", "asset_name", "chainage_start_km",
                              "chainage_end_km", "chainage_mid_km", "tqi", "run_id", "collection_date"]

        # Convert to list of dictionaries
        tqi_records = tqi_data[columns_to_extract].to_dict('records')
        
        logging.info(f"Processing {len(tqi_records)} TQI records for database insertion")

        # Post data to the database
        post_to_db(tqi_records, TQIRecord, self.db.engine, action="replace")
      
    def aggregate_tqi_to_segments(self, collection_date: datetime, line_class: str = "main") -> None:
        """
        Aggregate TQI data to rail segments. Take the segment chainage mid-point and find the closest TQI sample
        as the representative TQI for this segment.
        
        """
        # Load ALL TQI data once
        logging.info("Loading TQI data from database...")
        query = f"""
            SELECT * FROM tqi_records
            WHERE line_class = '{line_class}'
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
            
            for _, segment in segments_gdf.iterrows():
                chainage_id = segment.chainage_id
                seg_chainage_start = segment.chainage_start_km
                seg_chainage_end = segment.chainage_end_km
                seg_chainage_mid = (seg_chainage_start + seg_chainage_end) / 2

                # Find TQI record with closest chainage_mid_km to segment midpoint
                closest_idx = (line_tqi["chainage_mid_km"] - seg_chainage_mid).abs().idxmin()
                closest_tqi = line_tqi.loc[closest_idx]

                agg_tqi_data = {
                    "chainage_id": chainage_id,
                    "tqi": closest_tqi.tqi,
                    "line_code": line_code,
                    "line_class": line_class,
                    "status": thresholds.get_status(closest_tqi.tqi),
                    "collection_date": closest_tqi.collection_date
                }

                agg_tqi.append(agg_tqi_data)

        logging.info(f"Aggregated TQI data for {len(agg_tqi)} segments")

        # post data to the database
        post_to_db(agg_tqi, AggTQI, self.db.engine, action="replace")

    def process_dtr_data(self, collection_date: datetime):
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
        post_to_db(dtr_records, DTRRecord, self.db.engine, action="replace")
    
    def aggregate_dtr_to_segments(self, collection_date: datetime, line_class: str = "main") -> None:
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
        post_to_db(agg_dtr, AggDTR, self.db.engine, action="replace")
    
    def process_ballast_data(self, collection_date: datetime) -> None:
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

                # Convert distance from "4+ 296" format to 4.296 km for example
                ballast_data_clean["start_chainage_km"] = ballast_data_clean["distance_str"].apply(
                    lambda x: float(x.split("+")[0].strip()) + float(x.split("+")[1].strip()) / 1000
                    if isinstance(x, str) and "+" in x else None
                )

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
        post_to_db(ballast_records, BallastRecord, self.db.engine, action="replace")
    
    def aggregate_ballast_to_segments(self, collection_date: datetime) -> None:
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
        post_to_db(agg_ballast, AggBallast, self.db.engine, action="replace")
        
    def write_outputs(self):
        logging.info("Writing outputs")
        path_out = os.path.join(self.output_dir, f"thomas_segments.{self.output_fmt}")
        gdf = self.segment_data["TL"][1]
        gdf.to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"mainline_segments.{self.output_fmt}")
        gdf = self.segment_data["ML"][1]
        self.segment_data["ML"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"solomon_segments.{self.output_fmt}")
        gdf = self.segment_data["SL"][1]
        self.segment_data["SL"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"eliwana_segments.{self.output_fmt}")
        gdf = self.segment_data["EL"][1]
        self.segment_data["EL"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"pois.{self.output_fmt}")
        self.poi_gdf.to_crs('epsg:4326').to_file(path_out)
        path_out = os.path.join(self.output_dir, f"assets.{self.output_fmt}")
        self.assets.to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"curve_sections.{self.output_fmt}")
        self.curve_sections.to_crs('epsg:4326').to_file(path_out)

        # path_out = os.path.join(self.output_dir, f"rp_data.{self.output_fmt}")
        # self.rp_processed.to_crs('epsg:4326').to_file(path_out)
        # path_out = os.path.join(self.output_dir, f"rp_data.csv")
        # self.rp_processed.to_crs('epsg:4326').to_csv(path_out)

    def generate_excel_report(self, filename: str = "rail_condition_report.xlsx") -> str:
        """
        Generate Excel report with one tab per line_code, joining all aggregated tables.

        Args:
            filename: Output filename (default: rail_condition_report.xlsx)

        Returns:
            Path to generated Excel file
        """
        output_path = os.path.join(self.output_dir, filename)
        return generate_excel_report(self.db.engine, output_path)

    def generate_powerbi_export(
        self,
        subdir: str = "powerbi",
        format: str = 'geojson'
    ) -> dict:
        """
        Generate Power BI compatible exports with geospatial data.

        Args:
            subdir: Subdirectory within output_dir for Power BI files
            format: Output format ('geojson', 'shapefile', 'geoparquet')

        Returns:
            Dictionary mapping file type to file path
        """
        output_dir = os.path.join(self.output_dir, subdir)
        return generate_powerbi_export(self.db.engine, output_dir, format)

    def generate_condition_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics of track conditions.

        Returns:
            DataFrame with aggregated statistics by line_code
        """
        summary = generate_condition_summary(self.db.engine)

        # Also save to CSV
        output_path = os.path.join(self.output_dir, "condition_summary.csv")
        summary.to_csv(output_path, index=False)
        logging.info(f"Condition summary saved to: {output_path}")

        return summary

        # path_out = os.path.join(self.output_dir, f"tg_data.{self.output_fmt}")
        # self.tg_processed.to_crs('epsg:4326').to_file(path_out)
        # path_out = os.path.join(self.output_dir, f"tg_data.csv")
        # self.tg_processed.to_crs('epsg:4326').to_csv(path_out)

        if self.data.rp_data is not None:
            path_out = os.path.join(self.output_dir, f"rp_sections.{self.output_fmt}")
            self.rp_sections.to_crs('epsg:4326').to_file(path_out)
            path_out = os.path.join(self.output_dir, f"rp_sections.csv")
            self.rp_sections.to_crs('epsg:4326').to_csv(path_out)

            path_out = os.path.join(self.output_dir, f"tg_sections.{self.output_fmt}")
            self.tg_sections.to_crs('epsg:4326').to_file(path_out)
            path_out = os.path.join(self.output_dir, f"tg_sections.csv")
            self.tg_sections.to_crs('epsg:4326').to_csv(path_out)

            # rp_section, tg_section - geom only
            path_out = os.path.join(self.output_dir, f"rp_sections_geoms.{self.output_fmt}")
            self.rp_sections[["section_id", "geometry"]].to_crs('epsg:4326').to_file(path_out)
            path_out = os.path.join(self.output_dir, f"tg_sections_geoms.{self.output_fmt}")
            self.tg_sections[["section_id", "geometry"]].to_crs('epsg:4326').to_file(path_out)
    
 
def expand_curve_spirals(curve_data: pd.DataFrame, line_code: str):
    """
    In the curve data, some curves have entry and exit spirals defined by SC and CS chainages.
    This function expands those curves into separate spiral entries and exits, and modifies
    the original curve to have its start and end chainages adjusted accordingly.
    
    TS-----SC========CS-----ST
    |      |         |      |
    |      |         |      +-- original curve end
    |      |         +--------- exit spiral start
    |      +------------------ entry spiral end
    +------------------------- original curve start
    Args:
        curve_data (pd.DataFrame): DataFrame containing curve data with SC and CS columns.
        line_code (str): The line code to assign to the new spiral entries.
    """
    data = curve_data.copy()
    data["line_code"] = line_code
    spiral_rows = []
    for idx, row in data.iterrows():
        if row["Type"] == "Curve" and not pd.isna(row["SC"]) and not pd.isna(row["CS"]):
            entry_spiral = row.copy()
            entry_spiral["Classification"] = "spiral_entry"
            entry_spiral["End Chainage"] = round(row["SC"] / 1000, 6)
            entry_spiral["Curve Length"] = 1000 * round(abs(entry_spiral["Start Chainage"] - entry_spiral["End Chainage"]), 6)
            spiral_rows.append(entry_spiral)
            
            exit_spiral = row.copy()
            exit_spiral["Classification"] = "spiral_exit"
            exit_spiral["Start Chainage"] = round(row["CS"] / 1000, 6)
            exit_spiral["Curve Length"] = 1000 * round(abs(exit_spiral["Start Chainage"] - exit_spiral["End Chainage"]), 6)
            spiral_rows.append(exit_spiral)
            
            # Modify the DataFrame directly (use .at for single-value assignment)
            new_start = round(row["SC"] / 1000, 6)
            new_end = round(row["CS"] / 1000, 6)
            data.at[idx, "Start Chainage"] = new_start
            data.at[idx, "End Chainage"] = new_end
            data.at[idx, "Curve Length"] = 1000 * round(abs(new_start - new_end), 6)
            
    spirals = pd.DataFrame(spiral_rows)
    return data, spirals