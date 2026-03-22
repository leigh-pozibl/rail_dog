import logging
from collections import defaultdict

import geopandas as gpd
import pandas as pd

from shapely.geometry import Point
from shapely.ops import substring
from intervaltree import Interval, IntervalTree

from snappy_utils.general import clean_string

from rail_dog.utils.db_utils import post_to_db
from rail_dog.configs.schema import RailSection


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


class CurvesMixin:

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
        post_to_db(new_curve_sections, RailSection, self.db.engine, action="append")

        self.curve_interval_tree = {}
        for line_id, intervals in curve_intervals.items():
            self.curve_interval_tree[line_id] = IntervalTree(intervals)
            logging.info(f"Interals for line {line_id}:")
            logging.info(f"  num intervals: {len(intervals)}")
            logging.info(f"  start interval: {intervals[0]}")
            logging.info(f"  end interval: {intervals[-1]}")
