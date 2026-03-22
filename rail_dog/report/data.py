"""
DataMixin: data loading, report generation, and problem-zone calculations for Report.
"""
import logging
from typing import Optional, List

import pandas as pd

from rail_dog.utils.db_utils import get_table_data, query_agg_tsr
from rail_dog.report.treatment import (
    find_plainline_treatment,
    find_lx_treatment,
    find_irj_treatment,
    find_turnout_treatment,
    find_bridge_treatment,
)


class DataMixin:
    def _date_query(self, table: str, key: str, cols: str = "*") -> str:
        """Build a SELECT query with an optional collection_date filter.

        - Specific date in analysis_dates: exact match on collection_date.
        - global_date fallback: most recent record per chainage_id up to and
          including global_date (DISTINCT ON keeps only the latest row).
        - No date configured: no filter, all rows returned.
        """
        if key in self.analysis_dates:
            date_str = str(self.analysis_dates[key])[:10]
            return f"SELECT {cols} FROM {table} WHERE collection_date = '{date_str}'"

        if self.global_date:
            date_str = str(self.global_date)[:10]
            return (
                f"SELECT {cols} FROM {table} "
                f"WHERE collection_date = ("
                f"SELECT MAX(collection_date) FROM {table} WHERE collection_date <= '{date_str}'"
                f")"
            )

        return f"SELECT {cols} FROM {table}"

    def _check_date_load(self, df: pd.DataFrame, table: str, key: str):
        """Raise if a date-filtered query returned no rows."""
        if not df.empty:
            return
        if key in self.analysis_dates:
            date_str = str(self.analysis_dates[key])[:10]
            raise ValueError(
                f"No data found in '{table}' for analysis date {date_str} (key '{key}'). "
                f"Check analysis_dates config or verify data has been loaded for this date."
            )
        if self.global_date:
            date_str = str(self.global_date)[:10]
            raise ValueError(
                f"No data found in '{table}' up to global_date {date_str}. "
                f"Verify data has been loaded for this date."
            )

    def generate_report(
        self,
        line_codes: Optional[List[str]] = None
    ):
        """
        Retrieve all relevant data required in order to generate the final xlsx report.set
        pack data into export_data dict
        """
        # If no line codes specified, get all unique line codes from segments
        if line_codes is None:
            segments_df = get_table_data(self.engine, table_name="rail_segments")
            line_codes = sorted(segments_df['line_code'].unique())
            
        self.line_codes = line_codes
        logging.info(f"Found line codes: {self.line_codes}")

        # Load all data tables once
        logging.info("Loading data from database...")
        segments_df = get_table_data(self.engine, table_name="rail_segments")
        agg_asset_df = get_table_data(self.engine, table_name="agg_assets")
        agg_gbfi_df = get_table_data(self.engine, query=self._date_query("agg_gbfi", "GBFI"))
        self._check_date_load(agg_gbfi_df, "agg_gbfi", "GBFI")
        agg_ballast_df = get_table_data(self.engine, query=self._date_query("agg_ballast", "BALL"))
        self._check_date_load(agg_ballast_df, "agg_ballast", "BALL")
        agg_moisture_df = get_table_data(self.engine, query=self._date_query("agg_moisture", "MOI"))
        self._check_date_load(agg_moisture_df, "agg_moisture", "MOI")
        agg_tsr_df = query_agg_tsr(self.engine, as_of_date=self.analysis_dates.get("TSR") or self.global_date)
        agg_tqi_df = get_table_data(self.engine, query=self._date_query("agg_tqi", "TQI"))
        self._check_date_load(agg_tqi_df, "agg_tqi", "TQI")
        agg_tqi_all_df = get_table_data(self.engine, table_name="agg_tqi")

        # Trend fields are stored directly in agg_tqi per collection_date.
        # Ensure columns are present in case TQI data has no trend yet.
        _trend_cols = ['trend', 'trend_slope', 'trend_r_squared',
                       'step_change_type', 'step_change_date', 'trend_segment_start']
        for _col in _trend_cols:
            if _col not in agg_tqi_df.columns:
                agg_tqi_df[_col] = None
            if _col not in agg_tqi_all_df.columns:
                agg_tqi_all_df[_col] = None
        agg_dtr_df = get_table_data(self.engine, query=self._date_query("agg_dtr", "DTR"))
        self._check_date_load(agg_dtr_df, "agg_dtr", "DTR")
        agg_tg_df = get_table_data(self.engine, query=self._date_query("agg_tg", "TG", cols="chainage_id, avg_speed"))
        self._check_date_load(agg_tg_df, "agg_tg", "TG")

        # Remove geometry and metadata columns from segments for Excel export
        self.export_data["segments"] = segments_df.drop(columns=['geometry', 'created_at', 'id'], errors='ignore')
        
        # Map avg_speed from agg_tg onto segments (column always present, populated when data available)
        self.export_data["segments"]['speed'] = None
        if not agg_tg_df.empty and 'avg_speed' in agg_tg_df.columns:
            speed_map = agg_tg_df.set_index('chainage_id')['avg_speed']
            self.export_data["segments"]['speed'] = self.export_data["segments"]['chainage_id'].map(speed_map)

        # Remove id and created_at from agg tables
        self.export_data["agg_asset"] = agg_asset_df.drop(columns=['id', 'created_at'], errors='ignore')
        # Also drop ballast_centre/ballast_lt_250mm - these should only come from agg_ballast
        self.export_data["agg_gbfi"] = agg_gbfi_df.drop(columns=['id', 'collection_date', 'created_at', 'ballast_centre', 'ballast_lt_250mm'], errors='ignore')
        self.export_data["agg_ballast"] = agg_ballast_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore')
        self.export_data["agg_moisture"] = agg_moisture_df[['chainage_id', 'max_of_avg', 'avg_of_avg']] if not agg_moisture_df.empty else agg_moisture_df
        self.export_data["agg_tsr"] = agg_tsr_df.drop(columns=['id', 'created_at'], errors='ignore')
        self.export_data["agg_tqi"] = agg_tqi_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore')
        # Explicit column order — formula in write_tqi_trend_tab assumes A=chainage_id, B=collection_date, C=tqi
        # Join lat/lng from segments for map support
        seg_coords = segments_df[['chainage_id', 'mid_coord_lat', 'mid_coord_lng']].drop_duplicates('chainage_id')
        self.export_data["agg_tqi_all"] = (
            agg_tqi_all_df[['chainage_id', 'collection_date', 'tqi', 'status',
                            'trend', 'trend_slope', 'trend_r_squared',
                            'step_change_type', 'step_change_date', 'trend_segment_start']]
            .merge(seg_coords, on='chainage_id', how='left')
            .sort_values(['chainage_id', 'collection_date'])
            .reset_index(drop=True)
        )
        self.export_data["agg_dtr"] = agg_dtr_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore')

        # OPTIMIZATION: Pre-merge all aggregated data with segments for faster rule evaluation
        logging.info("Merging all data tables...")
        merged_df = segments_df[['chainage_id']].copy()

        # Add asset fields
        merged_df = merged_df.merge(
            self.export_data["agg_asset"].add_prefix('agg_asset.'),
            left_on='chainage_id',
            right_on='agg_asset.chainage_id',
            how='left'
        ).drop(columns=['agg_asset.chainage_id'], errors='ignore')

        # Add GBFI fields
        merged_df = merged_df.merge(
            self.export_data["agg_gbfi"].add_prefix('agg_gbfi.'),
            left_on='chainage_id',
            right_on='agg_gbfi.chainage_id',
            how='left'
        ).drop(columns=['agg_gbfi.chainage_id'], errors='ignore')
        
        # Add ballast fields
        merged_df = merged_df.merge(
            self.export_data["agg_ballast"].add_prefix('agg_ballast.'),
            left_on='chainage_id',
            right_on='agg_ballast.chainage_id',
            how='left'
        ).drop(columns=['agg_ballast.chainage_id'], errors='ignore')

        # Add moisture fields
        if not self.export_data["agg_moisture"].empty:
            merged_df = merged_df.merge(
                self.export_data["agg_moisture"].add_prefix('agg_moisture.'),
                left_on='chainage_id',
                right_on='agg_moisture.chainage_id',
                how='left'
            ).drop(columns=['agg_moisture.chainage_id'], errors='ignore')

        # Add TSR fields
        merged_df = merged_df.merge(
            self.export_data["agg_tsr"].add_prefix('agg_tsr.'),
            left_on='chainage_id',
            right_on='agg_tsr.chainage_id',
            how='left'
        ).drop(columns=['agg_tsr.chainage_id'], errors='ignore')

        # Add TQI fields
        merged_df = merged_df.merge(
            self.export_data["agg_tqi"].add_prefix('agg_tqi.'),
            left_on='chainage_id',
            right_on='agg_tqi.chainage_id',
            how='left'
        ).drop(columns=['agg_tqi.chainage_id'], errors='ignore')

        # Add DTR fields
        merged_df = merged_df.merge(
            self.export_data["agg_dtr"].add_prefix('agg_dtr.'),
            left_on='chainage_id',
            right_on='agg_dtr.chainage_id',
            how='left'
        ).drop(columns=['agg_dtr.chainage_id'], errors='ignore')

        # Generate model recommendations using pre-merged data
        logging.info("Generating recommendations...")
        recommendations_df = pd.DataFrame({
            'chainage_id': segments_df['chainage_id'],
            'plainline_treatment_model': find_plainline_treatment(merged_df, self.export_data),
            'plainline_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'plainline_treatment_override': None,  # Empty column for manual input
            'plainline_treatment_final': None,  # Will be replaced with Excel formula
            'lx_treatment_model': find_lx_treatment(merged_df, self.export_data),
            'lx_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'lx_treatment_override': None,  # Empty column for manual input
            'lx_treatment_final': None,  # Will be replaced with Excel formula
            'irj_treatment_model': find_irj_treatment(merged_df, self.export_data),
            'irj_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'irj_treatment_override': None,  # Empty column for manual input
            'irj_treatment_final': None,  # Will be replaced with Excel formula
            'turnout_treatment_model': find_turnout_treatment(merged_df, self.export_data),
            'turnout_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'turnout_treatment_override': None,  # Empty column for manual input
            'turnout_treatment_final': None,  # Will be replaced with Excel formula
            'bridge_treatment_model': find_bridge_treatment(merged_df, self.export_data),
            'bridge_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'bridge_treatment_override': None,  # Empty column for manual input
            'bridge_treatment_final': None,  # Will be replaced with Excel formula
        })

        recommendations_df['problem_zone'] = None  # Will be replaced with Excel formula
        recommendations_df['problem_zone_group'] = None  # Will be replaced with Excel formula
        recommendations_df['problem_zone_length_m'] = None  # Will be replaced with Excel formula

        # Priority scoring columns - will be replaced with Excel formulas
        recommendations_df['priority_tsr'] = None
        recommendations_df['priority_dtf'] = None
        recommendations_df['priority_tqi_status'] = None
        recommendations_df['priority_tqi_trend'] = None
        recommendations_df['priority_speed'] = None
        recommendations_df['priority_curve'] = None
        recommendations_df['priority_score'] = None
        recommendations_df['priority_dtf_getting_worse'] = None
        recommendations_df['priority_fixed_asset'] = None
        recommendations_df['priority_lx_score'] = None
        recommendations_df['priority_irj_score'] = None
        recommendations_df['priority_turnout_score'] = None
        recommendations_df['priority_bridge_score'] = None
        recommendations_df['priority_fixed_asset_max'] = None

        self.export_data["recommendations"] = recommendations_df

    def calculate_problem_zone_group(self, recommendations_df: pd.DataFrame) -> pd.Series:
        """
        Calculate problem_zone_group based on consecutive problem zones.

        Excel formula: =IF(BM2=0, "", IF(COUNTA($BN$1:BN1)=0, 1, IF(BM1=1, BN1, MAXIFS($BN$1:BN1, $BM$1:BM1, 1)+1)))

        Logic:
        - If problem_zone=0 → None (empty)
        - If first problem zone → group 1
        - If previous row was also problem_zone=1 → same group number
        - If previous row was problem_zone=0 → new group number (increment)

        Args:
            recommendations_df: DataFrame with problem_zone column

        Returns:
            Series with group numbers for consecutive problem zones
            
        Example:
        Row | problem_zone | problem_zone_group | Description
        ----|--------------|-------------------|-------------
            1 |      0       |       None        | No problem
            2 |      1       |        1          | Start group 1
            3 |      1       |        1          | Continue group 1
            4 |      1       |        1          | Continue group 1
            5 |      0       |       None        | No problem
            6 |      0       |       None        | No problem
            7 |      1       |        2          | Start group 2 (gap after group 1)
            8 |      1       |        2          | Continue group 2
            9 |      0       |       None        | No problem
           10 |      1       |        3          | Start group 3 (gap after group 2)

        """
        groups = []
        current_group = 0

        for idx in range(len(recommendations_df)):
            problem_zone = recommendations_df.iloc[idx]['problem_zone']

            if problem_zone == 0:
                # No problem zone, no group
                groups.append(None)
            else:
                # problem_zone == 1
                if idx == 0:
                    # First row with problem_zone=1
                    current_group = 1
                    groups.append(current_group)
                else:
                    prev_problem_zone = recommendations_df.iloc[idx-1]['problem_zone']
                    if prev_problem_zone == 1:
                        # Continue previous group
                        groups.append(current_group)
                    else:
                        # Start new group (previous was 0, now is 1)
                        current_group += 1
                        groups.append(current_group)

        return pd.Series(groups, index=recommendations_df.index)

    def calculate_problem_zone_length(
        self,
        recommendations_df: pd.DataFrame,
        segments_df: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate total chainage length (in meters) for each problem_zone_group.
        Each row in a group gets the same total length value.

        Args:
            recommendations_df: DataFrame with problem_zone_group column
            segments_df: DataFrame with chainage_start_km and chainage_end_km

        Returns:
            Series with group total lengths in meters
            
        Example:
        Row | chainage_id | problem_zone | problem_zone_group | segment_length | problem_zone_length_m
        ----|-------------|--------------|-------------------|----------------|----------------------
            1 | CHAIN-001   |      0       |       None        |     100m       |         None
            2 | CHAIN-002   |      1       |        1          |     100m       |         300m  (total for group 1)
            3 | CHAIN-003   |      1       |        1          |     100m       |         300m  (total for group 1)
            4 | CHAIN-004   |      1       |        1          |     100m       |         300m  (total for group 1)
            5 | CHAIN-005   |      0       |       None        |     100m       |         None
            6 | CHAIN-006   |      1       |        2          |     100m       |         200m  (total for group 2)
            7 | CHAIN-007   |      1       |        2          |     100m       |         200m  (total for group 2)
        """
        # Merge to get chainage info
        merged = recommendations_df.merge(
            segments_df[['chainage_id', 'chainage_start_km', 'chainage_end_km']],
            on='chainage_id',
            how='left'
        )

        # Calculate individual segment length in meters
        merged['segment_length_m'] = (merged['chainage_end_km'] - merged['chainage_start_km']) * 1000

        # Calculate group totals
        group_lengths = merged.groupby('problem_zone_group')['segment_length_m'].sum()

        # Map group totals back to each row
        length_series = merged['problem_zone_group'].map(group_lengths)

        # Set None for rows without a problem_zone_group
        length_series = length_series.where(merged['problem_zone_group'].notna(), other=float('nan'))

        return length_series

