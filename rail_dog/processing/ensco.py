import copy
import logging
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely import to_wkt

from snappy_utils.general import clean_string

from rail_dog.utils.db_utils import post_to_db
from rail_dog.configs.schema import AggTG
from rail_dog.utils.rail_utils import apply_corrections


class EnscoMixin:

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

    def process_tg_data(self, collection_date: datetime = None, db_action: str = None, line_class: str = "main"):
        """
        Aggregate track geometry (TG) data into 100m chainage segments.

        Rather than storing raw TG records, computes mean/min/max statistics across
        all TG readings that fall within each segment's chainage bounds.
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

        logging.info("Applying corrections to raw TG data")
        apply_corrections(tg_data, self.THOMAS_END, column_name_mapping)

        # Keep only valid, main-line records with a known chainage
        tg_data = tg_data[
            tg_data["line_code"].notna() &
            (tg_data["line_code"] != "OTH") &
            (tg_data["line_class"] == line_class) &
            tg_data["chainage_km"].notna()
        ].copy()
        logging.info(f"...{len(tg_data)} TG records after filtering")

        if collection_date is None and "collection_date" in tg_data.columns:
            valid_dates = tg_data["collection_date"].dropna()
            collection_date = valid_dates.mode().iloc[0] if not valid_dates.empty else None

        # Pre-group by line_code for efficient per-segment lookup
        tg_by_line = {lc: grp for lc, grp in tg_data.groupby("line_code")}

        # Speed columns: average only
        speed_cols = ["speed", "post_speed"]
        # Geometry columns: mean, min, max
        geom_cols = [
            "gauge", "crosslevel", "crosslevel_rate",
            "curve", "curve_rate",
            "twist_2", "twist_7", "twist_14",
            "warp",
            "top_e_10", "top_w_10",
            "align_e_10", "align_w_10",
        ]

        agg_tg = []
        for line_code, (_, segments_gdf) in self.segment_data.items():
            line_tg = tg_by_line.get(line_code)
            if line_tg is None or line_tg.empty:
                continue

            for _, segment in segments_gdf.iterrows():
                seg_tg = line_tg[
                    (line_tg["chainage_km"] >= segment.chainage_start_km) &
                    (line_tg["chainage_km"] < segment.chainage_end_km)
                ]

                if seg_tg.empty:
                    continue

                record = {
                    "chainage_id": segment.chainage_id,
                    "line_code": line_code,
                    "line_class": line_class,
                    "sample_count": len(seg_tg),
                    "collection_date": collection_date,
                }

                for col in speed_cols:
                    if col in seg_tg.columns:
                        col_data = seg_tg[col].dropna()
                        record[f"avg_{col}"] = float(col_data.mean()) if not col_data.empty else None

                for col in geom_cols:
                    if col in seg_tg.columns:
                        col_data = seg_tg[col].dropna()
                        if col_data.empty:
                            record[f"avg_{col}"] = record[f"min_{col}"] = record[f"max_{col}"] = None
                        else:
                            record[f"avg_{col}"] = float(col_data.mean())
                            record[f"min_{col}"] = float(col_data.min())
                            record[f"max_{col}"] = float(col_data.max())

                agg_tg.append(record)

        logging.info(f"Aggregated TG data for {len(agg_tg)} segments")

        if db_action:
            post_to_db(agg_tg, AggTG, self.db.engine, action=db_action)

        return agg_tg
