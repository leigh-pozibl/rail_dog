import logging

import geopandas as gpd
import pandas as pd

from shapely.geometry import Point
from shapely.ops import substring
from shapely.strtree import STRtree

from snappy_utils.geom_utils import (
    move_points_onto_lines,
    split_lines_into_segments,
)
from snappy_utils.general import clean_string

from rail_dog.utils.db_utils import post_to_db
from rail_dog.configs.schema import RailSegment
from rail_dog.configs.library import get_station, get_section
from rail_dog.processing.curves import expand_curve_spirals


class GeometryMixin:

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
                        section_name, section_id, asset_name = section_data[0]
                    else:
                        logging.warning(
                            f"Found multiple track sections at chainage: {seg_start_km}, line_code: {line_code} \
                            and line_type: 'main'. Check the track section definitions."
                        )
                        section_name, section_id, asset_name = section_data[0]
                else:
                    section_id, section_name, asset_name = "", "", ""

                station = get_station(seg_start_km)

                segment_data = {
                    "chainage_id": chainage_id,
                    "line": self.line_code_to_name[line_code],
                    "line_code": line_code,
                    "section": section_id,
                    "section_name": section_name,
                    "station": station,
                    "asset_name": asset_name,
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
