import os
import logging
import copy
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj

from shapely import to_wkt
from shapely.geometry import Point, LineString
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from intervaltree import Interval, IntervalTree

from snappy_utils.params import Metadata, DBConnection
from snappy_utils.geom_utils import (
    utm_crs_from_a_geom,
    split_lines_at_intersections,
    split_lines_at_points, split_lines_at_points_2,
    move_points_onto_lines,
    move_points_onto_lines_with_duckdb,
    split_lines_into_segments,
)
from snappy_utils.general import clean_string

from rail_dog.utils.rail_utils import add_elevation_to_gdf


class Processor():
    def __init__(self, params: dict, db: DBConnection, metadata: Metadata, output_dir: str):

        self.db = db
        self.data = params.base_data
        self.controls = params.controls
        self.params = params.parameters.preprocess
        self.input_fields = params.parameters.input_fields
        self.costs = params.parameters.costs
        self.get_filters = params.parameters.preprocess.get_filters
        self.metadata = metadata

        self.working_crs = self.get_working_crs(
            params.parameters.globals.working_crs,
            data=self.data
        )
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

        self.debug = True

        src = pyproj.CRS(self.working_crs)
        dst = pyproj.CRS('epsg:4326')
        self.PROJ = pyproj.Transformer.from_crs(src, dst, always_xy=True)

        self.h3_resolution = 15

        self.output_dir = output_dir
        self.output_fmt = params.parameters.globals.output_fmt

        self.THOMAS_START = -3.7
        self.THOMAS_END = 26.9
        self.SOLOMON_START = 174
        self.ELIWANA_START = 288.70
        self.START_OFFSET = {
            "TLX": self.THOMAS_START,
            "MLX": self.THOMAS_END,
            "SML": self.SOLOMON_START,
            "EML": self.ELIWANA_START,
        }
                
        class Thresholds:
            """
            Rail degradation thresholds.
            note: lo rail is the inside rail of a curve, hi rail is the outside rail
                A RH turn (as perceived when travelling away from the port) is a turn to the west -> the lo rail is the west rail
                A LH turn (as perceived when travelling away from the port) is a turn to the east -> the lo rail is the east rail
            """
            def __init__(self):
                self.tangent = {
                    "vert_wear": 16,
                    "side_wear": 4,
                    "rel_head_loss": {"lo": 40, "hi": 40},
                }
                self.mild_curve = {
                    "vert_wear": 14,
                    "side_wear": 6,
                    "rel_head_loss": {"lo": 32, "hi": 37},
                }
                self.sharp_curve = {
                    "vert_wear": 10,
                    "side_wear": 6,
                    "rel_head_loss": {"lo": 25, "hi": 25},
                }

            def get(self, clas, param, hand=None, track=None):
                d = getattr(self, clas)
                if param != "rel_head_loss":
                    if param in d:
                        return d[param]
                    else:
                        raise ValueError(f"Unrecognised param: {param} for class: {clas}")
                elif param == "rel_head_loss":
                    rail = self._get_rail(hand, track)
                    return d[param][rail]
                else:
                    raise ValueError(f"Unrecognised param: {param} for class: {clas}")

            def _get_rail(self, hand, track):
                if hand == "RH":
                    return "lo" if track == "w" else "hi"
                elif hand == "LH":
                    return "lo" if track == "e" else "hi"
                else:
                    if np.isnan(hand):
                        # these are tangent sections, don't need to distinguish between left and right hand
                        return "lo"
                    else:
                        raise ValueError(f"Unrecognised hand: {hand}")

        self.THRESHOLDS = Thresholds()

    def get_working_crs(self, input_crs, data):
        """
        It is better to use a UTM crs that is localised to the area being processed.  
        nb: using the epsg:3857 crs will report erroneous lengths for linestrings
        """
        if input_crs.split(":")[0] == "epsg":
            return input_crs

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

    def run(self):

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
        poi_gdf = gpd.GeoDataFrame(poi_df, crs=self.working_crs)
        poi_gdf = move_points_onto_lines(poi_gdf, self.data.path)

        # separate rail into the 4 main sections
        logging.info("Splitting the mainline")
        mainline = copy.copy(self.data.path[self.data.path.Layer == "Mainline"])
        solomon = copy.copy(self.data.path[self.data.path.Layer == "Solomon Spur"])
        solomon["line_id"] = "SML"
        eliwana = copy.copy(self.data.path[self.data.path.Layer == "Eliwana Mainline"])
        eliwana["line_id"] = "EML"

        # process the mainline section
        mainline = split_lines_at_points(mainline, poi_gdf[poi_gdf.name == "mainline_start"], buffer=1e-7)

        # deal with the loop at the end of the mainline
        geom_1 = mainline.loc[1, "geometry"]
        geom_2 = mainline.loc[2, "geometry"]
        geom = LineString(list(geom_1.coords[:-1]) + list(geom_2.coords[:-1]))
        mainline.loc[1, "geometry"] = geom
        mainline = mainline.drop(index=2)

        thomas = copy.copy(mainline.iloc[[0]])
        thomas["line_id"] = "TLX"
        mainline = copy.copy(mainline.iloc[[1]])
        mainline["line_id"] = "MLX"

        # clean up and standardise the column names
        self.lines = {"TLX": thomas, "MLX": mainline, "SML": solomon, "EML": eliwana}
        self.combined_lines = gpd.GeoDataFrame(pd.concat(self.lines.values()), crs=self.working_crs)
     
        self.attr_exclude_list = {"2022", "2023", "2024", "2025"}
        # for gdf in self.lines.values():
        #     gdf = gdf[[c for c in gdf.columns if c not in self.attr_exclude_list]]
        #     gdf = gdf.rename(columns={c: clean_string(c) for c in gdf.columns})
        # self.align_to_common_cols(self.lines)

        logging.info(f"Thomas section has {len(thomas)} features, total length: {round(sum(thomas.geometry.length), 2)}")
        logging.info(f"Mainline section has {len(mainline)} features, total length: {round(sum(mainline.geometry.length), 2)}")
        logging.info(f"Solomon section has {len(solomon)} features, total length: {round(sum(solomon.geometry.length), 2)}")
        logging.info(f"Eliwana section has {len(eliwana)} features, total length: {round(sum(eliwana.geometry.length), 2)}")

        logging.info("Segmenting into 100m chainage intervals")
        self.segment_data = {}
        logging.info("Thomas")
        self.create_chainage_segments(thomas, prefix="TLX", segment_length=100, chainage_end=self.THOMAS_END, reverse=True)
        logging.info("Mainline")
        self.create_chainage_segments(mainline, prefix="MLX", segment_length=100, chainage_start=self.THOMAS_END)
        logging.info("Solomon")
        self.create_chainage_segments(solomon, prefix="SML", segment_length=100, chainage_start=self.SOLOMON_START)
        logging.info("Eliwana")
        self.create_chainage_segments(eliwana, prefix="EML", segment_length=100, chainage_start=self.ELIWANA_START)

        self.align_to_common_cols(self.segment_data)

        logging.info("Creating switch layer")
        self.switch_data = self.create_switches(self.data.points)

        logging.info("Processing curves & tangent sections")
        self.curve_sections = self.data.track_data.get("CURVE_SECTIONS")
        self.curve_sections.rename(columns={c: clean_string(c) for c in self.curve_sections.columns}, inplace=True)
        self.curve_sections.rename(columns={"asset_name": "id"}, inplace=True)
        self.create_curve_sections()
        
        logging.info("Processing RP data")
        self.process_rp_data_into_sections()
        self.calculate_rp_stats_per_section()

        # # add some sample data
        # add_elevation_to_gdf(self.data.path, "elevation")
        # import random
        # self.data.path["track_condition"] = 1

        logging.info("Writing outputs")
        path_out = os.path.join(self.output_dir, f"thomas_segments.{self.output_fmt}")
        gdf = self.segment_data["TLX"][1]
        gdf["switch_ids"] = gdf.apply(lambda x: ','.join(x["switch_ids"]), axis=1)
        gdf.to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"mainline_segments.{self.output_fmt}")
        gdf = self.segment_data["MLX"][1]
        gdf["switch_ids"] = gdf.apply(lambda x: ','.join(x["switch_ids"]), axis=1)
        self.segment_data["MLX"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"solomon_segments.{self.output_fmt}")
        gdf = self.segment_data["SML"][1]
        gdf["switch_ids"] = gdf.apply(lambda x: ','.join(x["switch_ids"]), axis=1)
        self.segment_data["SML"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"eliwana_segments.{self.output_fmt}")
        gdf = self.segment_data["EML"][1]
        gdf["switch_ids"] = gdf.apply(lambda x: ','.join(x["switch_ids"]), axis=1)
        self.segment_data["EML"][1].to_crs('epsg:4326').to_file(path_out)

        path_out = os.path.join(self.output_dir, f"pois.{self.output_fmt}")
        poi_gdf.to_crs('epsg:4326').to_file(path_out)
        path_out = os.path.join(self.output_dir, f"switches.{self.output_fmt}")
        self.switch_data.to_crs('epsg:4326').to_file(path_out)

        # path_out = os.path.join(self.output_dir, f"rp_data.{self.output_fmt}")
        # self.rp_processed.to_crs('epsg:4326').to_file(path_out)
        # path_out = os.path.join(self.output_dir, f"rp_data.csv")
        # self.rp_processed.to_crs('epsg:4326').to_csv(path_out)

        # path_out = os.path.join(self.output_dir, f"tg_data.{self.output_fmt}")
        # self.tg_processed.to_crs('epsg:4326').to_file(path_out)
        # path_out = os.path.join(self.output_dir, f"tg_data.csv")
        # self.tg_processed.to_crs('epsg:4326').to_csv(path_out)

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

        exit(1)

    def create_chainage_segments(
        self,
        layer, prefix="MLX",
        segment_length=100,
        distance_start=0,
        chainage_start=None,
        chainage_end=None,
        reverse: bool = False,
        attr_exclude_list=set(),
    ):
        segments = split_lines_into_segments(layer, 100, perserve_vertices=False, reverse=reverse)
        
        new_segments = []
        inc = segment_length / 1000

        track_data = self.data.track_data.get(prefix)
        track_data.set_index("Chainage ID", inplace=True)

        if chainage_end:
            chainage_start = chainage_end - inc * len(segments)

        distance_from_start = 0
        start_km = chainage_start
        for idx, row in segments.iterrows():
            end_km = round(start_km + inc, 2)

            chainage_id = f"CHAIN-{prefix}-S-{int(10*abs(start_km)):0>5}-E-{int(10*abs(end_km)):0>5}-MAINLINE"

            segment_data = {}
            segment_data["geometry"] = row.geometry
            segment_data["chainage_id"] = chainage_id
            # segment_data["start_km"] = start_km
            # segment_data["end_km"] = end_km
            # segment_data["length"] = round(row.geometry.length, 2)

            centroid_en = row.geometry.centroid.coords[0]
            centroid_ll = self.PROJ.transform(centroid_en[0], centroid_en[1])

            # bring in track data
            if prefix == "TLX":
                chainage_id = chainage_id.replace("TLX", "MLX")

            track_segment_data = {
                clean_string(k): v for k, v in track_data.loc[chainage_id].to_dict().items()
                if k not in self.attr_exclude_list
            }

            segment_data.update(track_segment_data)
            segment_data["mid_coord_E"] = centroid_en[0]
            segment_data["mid_coord_N"] = centroid_en[1]
            segment_data["mid_coord_lng"] = centroid_ll[0]
            segment_data["mid_coord_lat"] = centroid_ll[1]

            input_chainage_start = round(segment_data["chainage_start_km"], 1)
            assert input_chainage_start == round(start_km, 1)
            segment_data["chainage_start_km"] = input_chainage_start

            input_chainage_end = round(segment_data["chainage_end_km"], 1)
            assert input_chainage_end == round(end_km, 1)
            segment_data["chainage_end_km"] = input_chainage_end

            segment_data["switch_ids"] = set()

            start_km = end_km
            distance_from_start += inc

            new_segments.append(segment_data)

        segments_gdf = gpd.GeoDataFrame(new_segments, crs=layer.crs)

        _geoms = list(segments_gdf.geometry)
        self.segment_data[prefix] = (STRtree(_geoms), segments_gdf)

    def create_switches(self, points):
        switches = []
        for i, row in points.iterrows():
            switch_data = {}
            switch_data["id"] = row.Name
            switch_data["offset"] = row.Offset
            switch_data["chainage"] = row["Chainage 12d (m)"]
            switch_data["geometry"] = row.geometry

            result = self.find_hits(row.geometry, max_distance=100)
            if result:
                key, idx = result
                data = self.segment_data[key][1].iloc[idx]
                switch_data["chainage_id"] = data["chainage_id"]
                data["switch_ids"].add(row.Name)
            else:
                switch_data["chainage_id"] = None

            switches.append(switch_data)

        return gpd.GeoDataFrame(switches, crs=points.crs)

    def find_hits(self, pt_geom: gpd.GeoDataFrame, max_distance=100):
        for key, (tree, gdf) in self.segment_data.items():
            hits = tree.query_nearest(pt_geom, max_distance=max_distance, all_matches=False)
            if hits:
                return key, hits[0]
        return None

    def align_to_common_cols(self, gdfs):
        cols = []
        for key, (_, gdf) in gdfs.items():
            cols.append(set(gdf.columns))

        common_cols = cols[0].copy()
        for s in cols[1:]:
            common_cols.intersection_update(s)

        for key, (_, gdf) in gdfs.items():
            gdf_cols = set(gdf.columns)
            logging.info(f"In the dataset {key}, dropping unique columns:")
            for c in gdf_cols.difference(common_cols):
                logging.info(f"  {c}")

            gdf = gdf[list(common_cols)]

    def create_curve_sections(self):
        def _get_segment(line_id: str, chainage: float):
            if line_id == "SL":
                prefix = "SML"
            elif line_id == "EL":
                prefix = "EML"
            else:
                if chainage <= self.THOMAS_END:
                    prefix = "TLX"
                else:
                    prefix = "MLX"
            gdf = self.segment_data[prefix][1]

            c1 = gdf["chainage_start_km"] <= chainage
            c2 = gdf["chainage_end_km"] >= chainage
            segment = gdf[c1 & c2]

            if len(segment) > 0:
                return segment, prefix
            else:
                # last_row = gdf.iloc[-1]
                return None, prefix

        def _build_line_vertices_trees():
            self.trees = {}
            for prefix, line_gdf in self.lines.items():
                geoms = line_gdf.geometry
                if len(geoms) != 1:
                    raise("Expected a single geometry for each line prefix")
                geom = geoms.iloc[0]

                if not isinstance(geom, LineString):
                        continue

                coords = list(geom.coords)
                tree = IntervalTree()
                dist = 0.0

                for i in range(len(coords) - 1):
                    pt1 = Point(coords[i])
                    pt2 = Point(coords[i + 1])
                    seg_len = round(pt1.distance(pt2), 6)
                    
                    # d1 = geom.project(pt1)
                    # d2 = geom.project(pt2)
                    # if i < 10:
                    #     logging.info(f"{d2-d1}, {seg_len}, {d2-d1-seg_len}")

                    if seg_len == 0:
                        logging.warning(f"Zero length segment found in line {prefix} near {pt1}")
                        continue

                    tree[dist: dist + seg_len] = (i, pt1, pt2)
                    if i < 10:
                        logging.info(f"{prefix} Adding segment distance: {pt1} to {pt2}, {dist} to {dist + seg_len}")
                    dist += seg_len

                self.trees[prefix] = tree

        def _get_curve_geom_old(prefix, curve_geom_start, curve_geom_end, start_chainage, end_chainage, method="slow"):
            if method == "straight":
                # Use the straight line between start and end points
                return LineString([curve_geom_start, curve_geom_end])

            line = self.lines[prefix].geometry.iloc[0]

            if method == "slow":
                d1 = round(line.project(curve_geom_start), 6)
                print(f"d1: {prefix} {d1}, {curve_geom_start}")
                d2 = round(line.project(curve_geom_end), 6)
                print(f"d2: {prefix} {d2}, {curve_geom_end}")
            else:
                d1 = round(1000 * (start_chainage - self.START_OFFSET[prefix]), 6)
                d2 = round(1000 * (end_chainage - self.START_OFFSET[prefix]), 6)

            d1, d2 = sorted([d1, d2])

            # logging.info(f"{prefix}: {round(d1 - _d1/1000, 3)}, {round(d2 - _d2/1000, 3)}")

            segments = sorted(self.trees[prefix][d1: d2], key=lambda iv: iv.begin)
            result = []
            
            if self.debug:
                print(curve_geom_start, curve_geom_end)
                print(d1, d2)
                for seg in segments:
                    print(seg.data)

            # Add start point
            start_point = line.interpolate(d1)
            result.append(start_point)

            # Add vertices of segments between start and end
            for seg in segments:
                _, p1, p2 = seg.data
                if p1 not in result:
                    result.append(p1)
                if p2 not in result:
                    result.append(p2)

            # Add end point
            end_point = line.interpolate(d2)
            if not end_point.equals(result[-1]):
                result.append(end_point)

            if len(result) > 1:
                return LineString(result)
            else:
                logging.warning("Curve section has only one point, returning a line between start and end points")
                return LineString([curve_geom_start, curve_geom_end])

        def _get_curve_geom(prefix: str, curve_geom_start: Point, curve_geom_end: Point):
            from shapely.ops import substring

            line = self.lines[prefix].geometry.iloc[0]
            proj1 = line.project(curve_geom_start, normalized=True)
            proj2 = line.project(curve_geom_end, normalized=True)
            
            # Ensure proj1 < proj2 for substring
            start, end = sorted([proj1, proj2])
            
            # Extract the segment between the two points
            segment = substring(line, start, end, normalized=True)
            return segment

        new_curve_sections = []
        curve_intervals = defaultdict(list)

        # line_trees = _build_line_vertices_trees()

        for idx, row in self.curve_sections.iterrows():
            if idx == 0:
                continue

            if row["section"] in {"nan", "LINE", "Section"}:
                continue

            curve_section_data = row.copy()

            curve_id = row["id"]
            chainage_start = float(row["start_chainage_km"])
            chainage_end = float(row["end_chainage_km"])
            
            # self.debug = False
            # #if curve_id == "CB_CV_071_183.868" or idx < 10:
            # if idx < 10:
            #     self.debug = True
            #     print(curve_id, chainage_start, chainage_end)
            
            if chainage_start == chainage_end:
                logging.warning(f"{curve_id} has zero length")
                continue

            if chainage_start > 0:
                section_seg_start, prefix_start = _get_segment(row["section"], chainage_start)
                row["prefix"] = prefix_start
                curve_intervals[prefix_start].append(Interval(chainage_start, chainage_end, curve_id))
                if section_seg_start is not None:
                    section_seg_start = section_seg_start.iloc[0]
                    delta = chainage_start - section_seg_start["chainage_start_km"]
                    coords_en = section_seg_start["geometry"].interpolate(1000 * delta).coords[-1]
                    # coords_ll = self.PROJ.transform(coords_en[0], coords_en[1])
                    # curve_section_data["start_coord_lng"] = coords_ll[0]
                    # curve_section_data["start_coord_lat"] = coords_ll[1]
                    curve_geom_start = Point(coords_en)
                else:
                    logging.error(f"{prefix_start} {chainage_start}: start id=not found")
                    curve_section_data["start_coord_lat"] = ""
                    curve_section_data["start_coord_lng"] = ""

                section_seg_end, prefix_end = _get_segment(row["section"], chainage_end)
                if section_seg_end is not None:
                    section_seg_end = section_seg_end.iloc[0]
                    delta = chainage_end - section_seg_end["chainage_start_km"]
                    coords_en = section_seg_end["geometry"].interpolate(1000 * delta).coords[-1]
                    # coords_ll = self.PROJ.transform(coords_en[0], coords_en[1])
                    # curve_section_data["end_coord_lng"] = coords_ll[0]
                    # curve_section_data["end_coord_lat"] = coords_ll[1]
                    curve_geom_end = Point(coords_en)
                else:
                    logging.error(f"{prefix_end} {chainage_end}: end id=not found")
                    # curve_section_data["end_coord_lat"] = ""
                    # curve_section_data["end_coord_lng"] = ""

                #curve_geom = _get_curve_geom(prefix_start, curve_geom_start, curve_geom_end, chainage_start, chainage_end)
                curve_geom = _get_curve_geom(prefix_start, curve_geom_start, curve_geom_end)
                curve_section_data["geometry"] = curve_geom

                curve_centroid_en = curve_geom.centroid.coords[0]
                curve_centroid_ll = self.PROJ.transform(curve_centroid_en[0], curve_centroid_en[1])
                curve_section_data["mid_coord_lng"] = curve_centroid_ll[0]
                curve_section_data["mid_coord_lat"] = curve_centroid_ll[1]

                # _split_section(prefix_start, curve_geom_start, curve_geom_end)
                new_curve_sections.append(curve_section_data)

        self.curve_sections = gpd.GeoDataFrame(new_curve_sections, crs=self.segment_data["MLX"][1].crs)

        self.curve_interval_tree = {}
        for line_id, intervals in curve_intervals.items():
            self.curve_interval_tree[line_id] = IntervalTree(intervals)
            logging.info(f"Interals for line {line_id}:")
            logging.info(f"  num intervals: {len(intervals)}")
            logging.info(f"  start interval: {intervals[0]}")
            logging.info(f"  end interval: {intervals[-1]}")

    def process_rp_data_into_sections(self):
        """
        process the RP data into sections
        """
        line_regions = {
            "MLB": "MLX",
            "MLW": "MLX",
            "CBM": "MLX",
            "Thomas": "TLX",
            "Barker": "MLX",
            "Canning": "MLX",
            "Chapman": "MLX",
            "Forrest-I": "MLX",
            "Forrest-II": "MLX",
            "Hillside": "MLX",
            "Gibb": "MLX",
            "Coonarie": "MLX",
            "Nunna": "MLX",
            "Nunna-2": "SML",
            "Hunter": "MLX",
            "Summit": "MLX",
            "Morgan": "MLX",
            "Maddina": "MLX",
            "CloudBreaker": "MLX",
            "Jones": "MLX",
            "Christmas Creek": "MLX",
            "Avon": "SML",
            "Bea Bea": "SML",
            "Bow": "SML",
            "Capel": "SML",
            "Firetail": "SML",
            "Future": "SML",
            "De Grey": "EML",
            "De Gray": "EML",
            "Duck": "EML",
            "Eliwana": "EML",
        }
        main_line_identifiers = {"MLB": "MLX", "MLW": "MLX", "SLB": "SML", "SLW": "SML", "SLE": "SML", "ELB": "EML"}
        other_line_identifiers = {"CBM", "MLE", "PTE", "PTW", "SPW", "SPE"}
        other_set = set()

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
                logging.warning(f"Could not locate {d_type} data point {data['id']} at chainage {chainage} in a curve section")
                return None
        
        def _get_line_id(data):
            pieces = data["id"].split("_")
            if pieces[1] in main_line_identifiers:
                # correction for the Thomas line
                if pieces[0] == "Thomas" and pieces[1] == "MLW":
                    chainage = data["major"] + (data["minor"] / 1000)
                    if chainage <= self.THOMAS_END:
                        return "TLX"
                    else:
                        return "MLX"
                # correction for the Firetail section
                elif pieces[0] == "Firetail" and pieces[1] == "ELB":
                    return "SML"
                # correction for the Future section
                elif pieces[0] == "Future" and pieces[1] == "SLB":
                    return "EML"
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
            print(len(self.data.tg_data))
            _apply_corrections(self.data.tg_data)
            print(len(self.data.tg_data))
            # self.data.rp_data.rename(columns={"SF": "id"}, inplace=True)
            # self.data.rp_data.rename(columns={c: clean_string(c) for c in self.data.rp_data.columns}, inplace=True)
            # self.data.rp_data["id"] = self.data.rp_data.apply(lambda x: _clean_id(x), axis=1)
            # self.data.rp_data["line_id"] = self.data.rp_data.apply(lambda x: _get_line_id(x), axis=1)
            # self.data.rp_data["line_region"] = self.data.rp_data.apply(lambda x: _get_line_region(x), axis=1)
            # self.data.rp_data["chainage"] = self.data.rp_data.apply(lambda x: x["major"] + x["minor"] / 1000, axis=1)
            # self.data.rp_data.to_crs(self.working_crs, inplace=True)

            # filter the data to just the main lines
            rp_data = self.data.rp_data
            print(len(rp_data))
            rp_data = copy.copy(rp_data[rp_data["line_id"] != "OTH"])
            print(len(rp_data))
            
            tg_data = self.data.tg_data
            print(len(tg_data))
            tg_data = copy.copy(tg_data[tg_data["line_id"] != "OTH"])
            print(len(tg_data))

            """
            This method aligns the lat/long corrds present in the rp data files to the
            underlying rail lines (as they are slightly offset).  This can be quite slow for large datasets
            
            The function move_points_onto_lines appends the field "len_from_line_start" to the data. Howver this
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
