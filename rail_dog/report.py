import logging
import os
from typing import Optional, List, Any

import pandas as pd
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from sqlalchemy import Engine

from rail_dog.utils.db_utils import get_table_data
from rail_dog.configs.thresholds import PLAINLINE_TREATMENT_THRESHOLDS, LX_TREATMENT_THRESHOLDS, IRJ_TREATMENT_THRESHOLDS, TURNOUT_TREATMENT_THRESHOLDS, BRIDGE_TREATMENT_THRESHOLDS


class RuleEngine:
    """
    Rule engine for applying conditional logic to DataFrames using list-based rules.

    Rule format:
        rules = [
            [condition, result],
            [condition, result],
            ...
            default_result  # Optional default (can be string or None)
        ]

    Condition syntax:
        - Simple: (field, operator, value)
          Example: ("fixed_asset", "==", "Yes")

        - Qualified fields: (table.field, operator, value)
          Example: ("agg_asset.fixed_asset", "==", "Yes")
          Looks up value from export_data["agg_asset"] for the current chainage_id

        - AND: ["AND", condition1, condition2, ...]
          Example: ["AND", ("status", "==", "poor"), ("gbfi", ">=", 100)]

        - OR: ["OR", condition1, condition2, ...]
          Example: ["OR", ("status", "==", "poor"), ("status", "==", "critical")]

        - Nested: Combine AND/OR arbitrarily
          Example: ["AND",
                     ["OR", ("status", "==", "poor"), ("status", "==", "critical")],
                     ("gbfi", ">=", 100)]

    Operators:
        "==", "!=", ">", ">=", "<", "<=", "in", "not in", "is", "is not"

    Example usage:
        rules = [
            [("agg_asset.fixed_asset", "==", "Yes"), "No Treatment"],
            [["AND",
              ["OR", ("agg_tqi.status", "==", "poor"), ("agg_tqi.status", "==", "critical")],
              ("agg_gbfi.gbfi_avg", ">=", 100)
             ], "Treatment Required"],
            "No Treatment"  # Default
        ]

        engine = RuleEngine(export_data=export_data, thresholds={"min_gbfi": 100})
        result = engine.apply_rules(df, rules)
    """

    def __init__(self, export_data: Optional[dict] = None, thresholds: Optional[dict] = None):
        """
        Args:
            export_data: Optional dict of DataFrames (e.g., {"agg_asset": df, "agg_gbfi": df})
                        Used to resolve qualified field names like "agg_asset.fixed_asset"
            thresholds: Optional dict of threshold values that can be referenced in rules
        """
        self.export_data = export_data or {}
        self.thresholds = thresholds or {}

    def _get_field_value(self, row: pd.Series, field: str) -> Any:
        """
        Get field value from row, supporting qualified field names.

        OPTIMIZED: First checks if field exists directly in row (pre-merged data),
        then falls back to table lookup if needed.

        Args:
            row: Current row being evaluated
            field: Field name, either "field" or "table.field"

        Returns:
            Field value, or pd.NA if not found
        """
        # OPTIMIZATION: Check if field exists directly in row first (pre-merged data)
        if field in row.index:
            return row[field]

        # Check if field is qualified (contains a dot)
        if "." in field:
            table_name, field_name = field.split(".", 1)

            # Check if this table exists in export_data
            if table_name not in self.export_data:
                raise ValueError(f"Table '{table_name}' not found in export_data. Available: {list(self.export_data.keys())}")

            # Get chainage_id from current row
            if "chainage_id" not in row.index:
                raise ValueError(f"Row must have 'chainage_id' field to look up qualified field '{field}'")

            chainage_id = row["chainage_id"]
            source_df = self.export_data[table_name]

            # Look up the value from the source table
            if "chainage_id" not in source_df.columns:
                raise ValueError(f"Table '{table_name}' must have 'chainage_id' column")

            # Find matching row(s) in source table
            matching_rows = source_df[source_df["chainage_id"] == chainage_id]

            if matching_rows.empty:
                # No matching chainage_id in source table
                return pd.NA

            if len(matching_rows) > 1:
                logging.warning(f"Multiple rows found for chainage_id={chainage_id} in table '{table_name}', using first")

            # Get the field value
            if field_name not in matching_rows.columns:
                return pd.NA

            return matching_rows.iloc[0][field_name]

        else:
            # Simple field name - not found
            return pd.NA

    def _eval_condition(self, row: pd.Series, condition) -> bool:
        """Evaluate a single condition against a row."""
        if isinstance(condition, list):
            operator = condition[0]

            if operator == "AND":
                return all(self._eval_condition(row, c) for c in condition[1:])
            elif operator == "OR":
                return any(self._eval_condition(row, c) for c in condition[1:])
            else:
                raise ValueError(f"Unknown list operator: {operator}. Use 'AND' or 'OR'")

        elif isinstance(condition, tuple):
            if len(condition) != 3:
                raise ValueError(f"Condition tuple must have 3 elements: (field, op, value). Got: {condition}")

            field, op, value = condition

            # Handle threshold references
            if isinstance(value, str) and value.startswith("threshold."):
                threshold_key = value.split(".", 1)[1]
                if threshold_key not in self.thresholds:
                    raise ValueError(f"Threshold '{threshold_key}' not found in thresholds dict")
                value = self.thresholds[threshold_key]

            # Get field value using the qualified field resolver
            field_value = self._get_field_value(row, field)

            # Evaluate operator
            if op == "==":
                # Handle boolean comparisons - pandas/numpy bools need special handling
                if isinstance(value, bool):
                    # Convert field_value to Python bool for comparison
                    if pd.isna(field_value):
                        return False
                    # Use bool() to convert numpy.bool_ or other types to Python bool
                    return bool(field_value) == value
                return field_value == value
            elif op == "!=":
                # Handle boolean comparisons
                if isinstance(value, bool):
                    if pd.isna(field_value):
                        return True
                    return bool(field_value) != value
                return field_value != value
            elif op == ">":
                return pd.notna(field_value) and field_value > value
            elif op == ">=":
                return pd.notna(field_value) and field_value >= value
            elif op == "<":
                return pd.notna(field_value) and field_value < value
            elif op == "<=":
                return pd.notna(field_value) and field_value <= value
            elif op == "in":
                return field_value in value
            elif op == "not in":
                return field_value not in value
            elif op == "is":
                if value is None or value == "None":
                    return pd.isna(field_value)
                return field_value is value
            elif op == "is not":
                if value is None or value == "None":
                    return pd.notna(field_value)
                return field_value is not value
            else:
                raise ValueError(f"Unknown operator: {op}")

        else:
            raise ValueError(f"Condition must be a tuple or list. Got: {type(condition)}")

    def _apply_rules_to_row(self, row: pd.Series, rules: list) -> Any:
        """Apply rules to a single row and return the first matching result."""
        for rule in rules:
            # Check if this is the default (last item with no condition)
            if not isinstance(rule, (list, tuple)):
                return rule

            # Rule is [condition, result]
            if len(rule) != 2:
                raise ValueError(f"Rule must be [condition, result] or a default value. Got: {rule}")

            condition, result = rule

            if self._eval_condition(row, condition):
                return result

        # No rule matched and no default provided
        return None

    def apply_rules(self, df: pd.DataFrame, rules: list, debug: bool = False) -> pd.Series:
        """
        Apply rules to each row of a DataFrame.

        Args:
            df: DataFrame to apply rules to
            rules: List of [condition, result] pairs, optionally ending with a default value
            debug: If True, log which rules are matching (only for first row)

        Returns:
            Series with result for each row (indexed same as df)
        """
        if debug and len(df) > 0:
            logging.info("=== Rule Debug Mode (First Row) ===")
            first_row = df.iloc[0]
            logging.info(f"Evaluating chainage_id: {first_row.get('chainage_id', 'N/A')}")

            for i, rule in enumerate(rules):
                if not isinstance(rule, (list, tuple)):
                    logging.info(f"Rule {i+1}: DEFAULT -> {rule}")
                    break

                condition, result = rule
                matches = self._eval_condition(first_row, condition)
                logging.info(f"Rule {i+1}: {matches} -> {result}")
                if matches:
                    logging.info(f"  MATCHED! Returning: {result}")
                    break

        return df.apply(lambda row: self._apply_rules_to_row(row, rules), axis=1)

    def debug_row(self, row: pd.Series, rules: list) -> None:
        """
        Debug helper to show which rules match for a specific row.

        Args:
            row: Single row from DataFrame
            rules: List of rules to evaluate
        """
        print(f"\n=== Debugging Row: {row.get('chainage_id', 'N/A')} ===")

        for i, rule in enumerate(rules):
            if not isinstance(rule, (list, tuple)):
                print(f"Rule {i+1}: DEFAULT -> {rule}")
                break

            condition, result = rule
            try:
                matches = self._eval_condition(row, condition)
                print(f"Rule {i+1}: {'✓ MATCH' if matches else '✗ no match'} -> {result}")
                if matches:
                    print(f"  Result: {result}")
                    break
            except Exception as e:
                print(f"Rule {i+1}: ERROR - {e}")


class Report():
    def __init__(self, engine: Engine, output_path: str):
        """
        Args:
            engine: database engine
            output_path: Path to save the Excel file
        """
        self.engine = engine
        self.output_path = output_path
        
        self.export_data = {}
        
        self.generate_report()
    
    def find_plainline_treatment(self, segments_df: pd.DataFrame) -> pd.Series:
        """
        Apply plainline treatment recommendation rules to segment data.

        Uses qualified field names (table.field) to reference fields from different
        aggregated tables in self.export_data.

        Original Excel formula:
        =IF(
            M2="Yes",
            "No Treatment",
            IF(
                AND(
                    OR(AK2="poor", AK2="critical"),
                    S2>=Thresholds!$T$8
                ),
                "Local Formation Renewal (Bog Hole)",
                IF(
                    AND(
                        S2>=Thresholds!$T$9,
                        OR(AU2="s1", AU2="s2"),
                        AC2="Yes"
                    ),
                    "Local Formation Renewal (Bog Hole)",
                    IF(
                        AND(
                            OR(AK2="poor", AK2="critical"),
                            T2>=Thresholds!$V$10
                        ),
                        "Ballast Clean",
                        IF(
                            AND(
                                OR(AK2="poor", AK2="critical"),
                                T2<Thresholds!$V$11,
                                Y2<Thresholds!$Z$11
                            ),
                            "Lift & Tamp",
                            IF(
                                AND(
                                    OR(AK2="poor", AK2="critical"),
                                    T2<Thresholds!$V$12,
                                    Y2>=Thresholds!$Z$12
                                ),
                                "Tamp Only",
                                "No Treatment"
                            )
                        )
                    )
                )
            )
        )

        Column mappings:
        - M2 = fixed_asset (agg_asset.fixed_asset)
        - AK2 = TQI status (agg_tqi.status)
        - S2 = GBFI max_of_avg (agg_gbfi.max_of_avg)
        - T2 = GBFI avg_of_avg (agg_gbfi.avg_of_avg)
        - Y2 = ballast_centre (agg_ballast.ballast_centre)
        - AU2 = DTR worst_dtf (agg_dtr.worst_dtf)
        - AC2 = complete_tsr (agg_tsr.complete_tsr)

        Threshold mappings:
        - Thresholds!$T$8 = gbfi_bog_threshold_1 (40)
        - Thresholds!$T$9 = gbfi_bog_threshold_2 (40)
        - Thresholds!$V$10 = ballast_clean_threshold (20)
        - Thresholds!$V$11 = lift_tamp_gbfi_avg_threshold (20)
        - Thresholds!$Z$11 = lift_tamp_ballast_depth_threshold (0.25)
        - Thresholds!$V$12 = tamp_only_gbfi_avg_threshold (20)
        - Thresholds!$Z$12 = tamp_only_ballast_depth_threshold (0.25)

        Args:
            segments_df: DataFrame with chainage_id column (can be just segments or merged data)

        Returns:
            Series with treatment recommendation for each chainage_id
        """
        plainline_treatment_rules = [
            # Rule 1: IF fixed_asset=True, "No Treatment"
            [
                ("agg_asset.fixed_asset", "==", True),
                "No Treatment"
            ],

            # Rule 2: IF AND(OR(TQI status="poor"|"critical"), GBFI>=threshold), "Formation Renewal"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ("agg_gbfi.max_of_avg", ">=", "threshold.gbfi_bog_threshold_1")
                ],
                "Local Formation Renewal (Bog Hole)"
            ],

            # Rule 3: IF AND(GBFI>=threshold, OR(DTR severity), TSR history), "Formation Renewal"
            [
                ["AND",
                    ("agg_gbfi.max_of_avg", ">=", "threshold.gbfi_bog_threshold_2"),
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2")
                    ],
                    ("agg_tsr.complete_tsr", "==", True)
                ],
                "Local Formation Renewal (Bog Hole)"
            ],

            # Rule 4: IF AND(OR(TQI status="poor"|"critical"), GBFI Avg), "Ballast Clean"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ("agg_gbfi.avg_of_avg", ">=", "threshold.ballast_clean_threshold")
                ],
                "Ballast Clean"
            ],

            # Rule 5: IF AND(OR(TQI status="poor"|"critical"), GBFI Avg, Ballast Centre, "Lift & Tamp"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ("agg_gbfi.avg_of_avg", "<", "threshold.lift_tamp_gbfi_avg_threshold"),
                    ("agg_ballast.ballast_centre", "<", "threshold.lift_tamp_ballast_depth_threshold")
                ],
                "Lift & Tamp"
            ],

            # Rule 6: IF AND(OR(TQI status="poor"|"critical"), GBFI Avg, Ballast Cnetre), "Tamp Only"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ("agg_gbfi.avg_of_avg", "<", "threshold.tamp_only_gbfi_avg_threshold"),
                    ("agg_ballast.ballast_centre", ">=", "threshold.tamp_only_ballast_depth_threshold")
                ],
                "Tamp Only"
            ],

            # Default
            "No Treatment"
        ]

        # Apply rules using the RuleEngine with export_data for qualified field lookups
        engine = RuleEngine(export_data=self.export_data, thresholds=PLAINLINE_TREATMENT_THRESHOLDS)
        return engine.apply_rules(segments_df, plainline_treatment_rules)
    
    def find_lx_treatment(self, segments_df: pd.DataFrame) -> pd.Series:
        """
        Apply level crossing (LX) treatment recommendation rules to segment data.

        Original Excel formula:
        =IF(
            M2<>"Yes",
            "No treatment",
            IF(
                AND(
                    N2>0,
                    AC2="Yes",
                    OR(AK2="poor", AK2="critical"),
                    OR(S2>Thresholds!$T$14, T2>Thresholds!$V$14)
                ),
                N2,
                IF(
                    AND(
                        OR(AK2="poor", AK2="critical"),
                        OR(AU2="s1", AU2="s2", AU2="s3"),
                        T2>Thresholds!$V$18,
                        Y2>=Thresholds!$Z$18,
                        I2>0
                    ),
                    Thresholds!$Q$18,
                    "No treatment"
                )
            )
        )

        Column mappings:
        - M2 = fixed_asset (agg_asset.fixed_asset)
        - N2 = wz_level_crossing count (agg_asset.wz_level_crossing)
        - I2 = level_crossing count (agg_asset.level_crossing)
        - AC2 = complete_tsr (agg_tsr.complete_tsr)
        - AK2 = TQI status (agg_tqi.status)
        - S2 = GBFI max_of_avg (agg_gbfi.max_of_avg)
        - T2 = GBFI avg_of_avg (agg_gbfi.avg_of_avg)
        - Y2 = ballast_centre (agg_ballast.ballast_centre)
        - AU2 = DTR worst_dtf (agg_dtr.worst_dtf)
        - Thresholds!$T$14 = lx_renewal_gbfi_max_threshold (40)
        - Thresholds!$V$14 = lx_renewal_gbfi_avg_threshold (30)
        - Thresholds!$V$18 = tamp_only_gbfi_avg_threshold (30)
        - Thresholds!$Z$18 = tamp_only_ballast_depth_threshold (0.25)
        - Thresholds!$Q$18 = "Tamp Only" (result string)

        Args:
            segments_df: DataFrame with chainage_id column (can be just segments or merged data)

        Returns:
            Series with treatment recommendation for each chainage_id
            Note: Returns the wz_level_crossing count (numeric) when renewal conditions are met
        """
        lx_treatment_rules = [
            # Rule 1: IF fixed_asset != True, "No Treatment"
            [
                ("agg_asset.fixed_asset", "==", False),
                "No Treatment"
            ],

            # Rule 2: IF AND(WZ Level Crossing > 0, TSR History, TQI poor/critical, (GBFI max OR avg exceeds threshold)), return wz_level_crossing count
            [
                ["AND",
                    ("agg_asset.wz_level_crossing", ">", 0),
                    ("agg_tsr.complete_tsr", "==", True),
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ["OR",
                        ("agg_gbfi.max_of_avg", ">", "threshold.lx_renewal_gbfi_max_threshold"),
                        ("agg_gbfi.avg_of_avg", ">", "threshold.lx_renewal_gbfi_avg_threshold")
                    ]
                ],
                "__LX_COUNT__"  # Placeholder to be replaced with actual count
            ],

            # Rule 3: IF AND(TQI poor/critical, DTR S1/S2/S3, GBFI avg > threshold, ballast >= threshold, level_crossing > 0), "Tamp Only"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2"),
                        ("agg_dtr.worst_dtf", "==", "s3")
                    ],
                    ("agg_gbfi.avg_of_avg", ">", "threshold.tamp_only_gbfi_avg_threshold"),
                    ("agg_ballast.ballast_centre", ">=", "threshold.tamp_only_ballast_depth_threshold"),
                    ("agg_asset.level_crossing", ">", 0)
                ],
                "Tamp Only"
            ],

            # Default
            "No Treatment"
        ]

        # Apply rules using the RuleEngine with export_data for qualified field lookups
        engine = RuleEngine(export_data=self.export_data, thresholds=LX_TREATMENT_THRESHOLDS)
        result = engine.apply_rules(segments_df, lx_treatment_rules)

        # OPTIMIZED: Vectorized replacement of __LX_COUNT__ placeholder with actual wz_level_crossing count
        mask = result == "__LX_COUNT__"
        if mask.any() and "agg_asset.wz_level_crossing" in segments_df.columns:
            result[mask] = segments_df.loc[mask, "agg_asset.wz_level_crossing"]

        return result
    
    def find_irj_treatment(self, segments_df: pd.DataFrame) -> pd.Series:
        """
        Apply IRJ (Insulated Rail Joint) treatment recommendation rules to segment data.

        Original Excel formula:

        =IF(
            M2<>"Yes",
            "No treatment",
            IF(
                AND(
                    S2 > Thresholds!$T$15,
                    OR(AK2 = "Critical", AK2 = "Poor"),
                    OR(AU2 = "s1", AU2 = "s2"),
                    J2 > 0
                ),
                J2,
                IF(
                    AND(
                        OR(AK2 = "Critical", AK2 = "Poor"),
                        OR(AU2 = "s1", AU2 = "s2", AU2 = "s3"),
                        Y2 >= Thresholds!$Z$18,
                        J2 > 0,
                        T2>Thresholds!$V$18
                    ),
                    Thresholds!$Q$18,
                    "No treatment"
                )
            )
        )

        Column mappings:
        - M2 = fixed_asset (agg_asset.fixed_asset)
        - J2 = irj count (agg_asset.irj)
        - S2 = GBFI max_of_avg (agg_gbfi.max_of_avg)
        - T2 = GBFI avg_of_avg (agg_gbfi.avg_of_avg)
        - Y2 = ballast_centre (agg_ballast.ballast_centre)
        - AK2 = TQI status (agg_tqi.status)
        - AU2 = DTR worst_dtf (agg_dtr.worst_dtf)

        Args:
            segments_df: DataFrame with chainage_id column (can be just segments or merged data)

        Returns:
            Series with treatment recommendation for each chainage_id
            Note: Returns the IRJ count (numeric) when renewal conditions are met
        """
        irj_treatment_rules = [
            # Rule 1: IF fixed_asset != True, "No Treatment"
            [
                ("agg_asset.fixed_asset", "==", False),
                "No Treatment"
            ],

            # Rule 2: IF AND(GBFI max > threshold, TQI status poor/critical, DTR S1/S2, IRJ count > 0), return IRJ count
            [
                ["AND",
                    ("agg_gbfi.max_of_avg", ">", "threshold.irj_renewal_gbfi_max_threshold"),
                    ["OR",
                        ("agg_tqi.status", "==", "critical"),
                        ("agg_tqi.status", "==", "poor")
                    ],
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2")
                    ],
                    ("agg_asset.irj", ">", 0)
                ],
                "__IRJ_COUNT__"  # Placeholder to be replaced with actual count
            ],

            # Rule 3: IF AND(TQI status poor/critical, DTR S1/S2/S3, ballast >= threshold, IRJ > 0, GBFI avg > threshold), "Tamp Only"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "critical"),
                        ("agg_tqi.status", "==", "poor")
                    ],
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2"),
                        ("agg_dtr.worst_dtf", "==", "s3")
                    ],
                    ("agg_ballast.ballast_centre", ">=", "threshold.tamp_only_ballast_depth_threshold"),
                    ("agg_asset.irj", ">", 0),
                    ("agg_gbfi.avg_of_avg", ">", "threshold.tamp_only_gbfi_avg_threshold")
                ],
                "Tamp Only"
            ],

            # Default
            "No Treatment"
        ]

        # Apply rules using the RuleEngine with export_data for qualified field lookups
        engine = RuleEngine(export_data=self.export_data, thresholds=IRJ_TREATMENT_THRESHOLDS)
        result = engine.apply_rules(segments_df, irj_treatment_rules)

        # OPTIMIZED: Vectorized replacement of __IRJ_COUNT__ placeholder with actual IRJ count
        mask = result == "__IRJ_COUNT__"
        if mask.any() and "agg_asset.irj" in segments_df.columns:
            result[mask] = segments_df.loc[mask, "agg_asset.irj"]

        return result

    def find_turnout_treatment(self, segments_df: pd.DataFrame) -> pd.Series:
        """
        Apply turnout treatment recommendation rules to segment data.

        Original Excel formula:
        =IF(
            M2="Yes",
            IF(
                AND(
                    Y2 >= Thresholds!$Z$16,
                    OR(AU2 = "s1", AU2 = "s2", AU2 = "s3"),
                    AC2 = "Yes",
                    P2 > 0
                ),
                P2,
                IF(
                    AND(
                        OR(AK2 = "Poor", AK2 = "Critical"),
                        Y2 >= Thresholds!$Z$18,
                        OR(AU2 = "s1", AU2 = "s2", AU2 = "s3"),
                        T2 > Thresholds!$V$18
                    ),
                    Thresholds!$Q$18,
                    "No treatment"
                )
            ),
            "No treatment"
        )

        Column mappings:
        - M2 = fixed_asset (agg_asset.fixed_asset)
        - P2 = wz turnout count (agg_asset.wz_turnout)
        - Y2 = ballast_centre (agg_ballast.ballast_centre)
        - T2 = GBFI avg_of_avg (agg_gbfi.avg_of_avg)
        - AK2 = TQI status (agg_tqi.status)
        - AU2 = DTR worst_dtf (agg_dtr.worst_dtf)
        - AC2 = complete_tsr (agg_tsr.complete_tsr)

        Args:
            segments_df: DataFrame with chainage_id column (can be just segments or merged data)

        Returns:
            Series with treatment recommendation for each chainage_id
            Note: Returns the turnout count (numeric) when renewal conditions are met
        """
        turnout_treatment_rules = [
            # Rule 1: IF fixed_asset != True, "No Treatment"
            [
                ("agg_asset.fixed_asset", "==", False),
                "No Treatment"
            ],

            # Rule 2: IF AND(ballast >= threshold, DTR S1/S2/S3, TSR history, wz_turnout > 0), return turnout count
            [
                ["AND",
                    ("agg_ballast.ballast_centre", ">=", "threshold.turnout_renewal_ballast_depth_threshold"),
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2"),
                        ("agg_dtr.worst_dtf", "==", "s3")
                    ],
                    ("agg_tsr.complete_tsr", "==", True),
                    ("agg_asset.wz_turnout", ">", 0)
                ],
                "__TURNOUT_COUNT__"  # Placeholder to be replaced with actual count
            ],

            # Rule 3: IF AND(TQI poor/critical, ballast >= threshold, DTR S1/S2/S3, GBFI avg > threshold), "Tamp Only"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ("agg_ballast.ballast_centre", ">=", "threshold.tamp_only_ballast_depth_threshold"),
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2"),
                        ("agg_dtr.worst_dtf", "==", "s3")
                    ],
                    ("agg_gbfi.avg_of_avg", ">", "threshold.tamp_only_gbfi_avg_threshold")
                ],
                "Tamp Only"
            ],

            # Default
            "No Treatment"
        ]

        # Apply rules using the RuleEngine with export_data for qualified field lookups
        engine = RuleEngine(export_data=self.export_data, thresholds=TURNOUT_TREATMENT_THRESHOLDS)
        result = engine.apply_rules(segments_df, turnout_treatment_rules)

        # OPTIMIZED: Vectorized replacement of __TURNOUT_COUNT__ placeholder with actual turnout count
        mask = result == "__TURNOUT_COUNT__"
        if mask.any() and "agg_asset.wz_turnout" in segments_df.columns:
            result[mask] = segments_df.loc[mask, "agg_asset.wz_turnout"]

        return result

    def find_bridge_treatment(self, segments_df: pd.DataFrame) -> pd.Series:
        """
        Apply bridge treatment recommendation rules to segment data.

        Original Excel formula:
        =IF(
            M2<>"Yes",
            "No treatment",
            IF(
                AND(
                    Y2 >= Thresholds!$Z$17,
                    OR(AU2="s1", AU2="s2"),
                    T2 > Thresholds!$V$17,
                    AC2 = "Yes",
                    Q2 > 0
                ),
                Q2,
                IF(
                    AND(
                        OR(AK2="Poor", AK2="Critical"),
                        Y2 >= Thresholds!$Z$18,
                        OR(AU2="s1", AU2="s2", AU2="s3"),
                        T2 > Thresholds!$V$18
                    ),
                    Thresholds!$Q$18,
                    "No treatment"
                )
            )
        )

        Column mappings:
        - M2 = fixed_asset (agg_asset.fixed_asset)
        - Q2 = bridge count (agg_asset.wz_bridge)
        - Y2 = ballast_centre (agg_ballast.ballast_centre)
        - T2 = GBFI avg_of_avg (agg_gbfi.avg_of_avg)
        - AC2 = complete_tsr (agg_tsr.complete_tsr)
        - AU2 = DTR worst_dtf (agg_dtr.worst_dtf)
        - AK2 = TQI status (agg_tqi.status)

        Threshold mappings:
        - Thresholds!$Z$17 = bridge_renewal_ballast_depth_threshold (0.25)
        - Thresholds!$V$17 = bridge_renewal_gbfi_avg_threshold (30)
        - Thresholds!$V$18 = tamp_only_gbfi_avg_threshold (30)
        - Thresholds!$Z$18 = tamp_only_ballast_depth_threshold (0.25)
        - Thresholds!$Q$18 = "Tamp Only" (result string)

        Args:
            segments_df: DataFrame with chainage_id column (can be just segments or merged data)

        Returns:
            Series with treatment recommendation for each chainage_id
            Note: Returns the bridge count (numeric) when renewal conditions are met
        """
        bridge_treatment_rules = [
            # Rule 1: IF fixed_asset != True, "No Treatment"
            [
                ("agg_asset.fixed_asset", "==", False),
                "No Treatment"
            ],

            # Rule 2: IF AND(ballast >= threshold, DTR S1/S2, GBFI avg > threshold, TSR history, bridge > 0), return bridge count
            [
                ["AND",
                    ("agg_ballast.ballast_centre", ">=", "threshold.bridge_renewal_ballast_depth_threshold"),
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2")
                    ],
                    ("agg_gbfi.avg_of_avg", ">", "threshold.bridge_renewal_gbfi_avg_threshold"),
                    ("agg_tsr.complete_tsr", "==", True),
                    ("agg_asset.wz_bridge", ">", 0)
                ],
                "__BRIDGE_COUNT__"  # Placeholder to be replaced with actual count
            ],

            # Rule 3: IF AND(TQI poor/critical, ballast >= threshold, DTR S1/S2/S3, GBFI avg > threshold), "Tamp Only"
            [
                ["AND",
                    ["OR",
                        ("agg_tqi.status", "==", "poor"),
                        ("agg_tqi.status", "==", "critical")
                    ],
                    ("agg_ballast.ballast_centre", ">=", "threshold.tamp_only_ballast_depth_threshold"),
                    ["OR",
                        ("agg_dtr.worst_dtf", "==", "s1"),
                        ("agg_dtr.worst_dtf", "==", "s2"),
                        ("agg_dtr.worst_dtf", "==", "s3")
                    ],
                    ("agg_gbfi.avg_of_avg", ">", "threshold.tamp_only_gbfi_avg_threshold")
                ],
                "Tamp Only"
            ],

            # Default
            "No Treatment"
        ]

        # Apply rules using the RuleEngine with export_data for qualified field lookups
        engine = RuleEngine(export_data=self.export_data, thresholds=BRIDGE_TREATMENT_THRESHOLDS)
        result = engine.apply_rules(segments_df, bridge_treatment_rules)

        # OPTIMIZED: Vectorized replacement of __BRIDGE_COUNT__ placeholder with actual bridge count
        mask = result == "__BRIDGE_COUNT__"
        if mask.any() and "agg_asset.wz_bridge" in segments_df.columns:
            result[mask] = segments_df.loc[mask, "agg_asset.wz_bridge"]

        return result

    def generate_report(self):
        self.get_report_data()
    
    def get_report_data(
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
        agg_gbfi_df = get_table_data(self.engine, table_name="agg_gbfi")
        agg_ballast_df = get_table_data(self.engine, table_name="agg_ballast")
        agg_tsr_df = get_table_data(self.engine, table_name="agg_tsr")
        agg_tqi_df = get_table_data(self.engine, table_name="agg_tqi")
        agg_dtr_df = get_table_data(self.engine, table_name="agg_dtr")

        # Remove geometry and metadata columns from segments for Excel export
        self.export_data["segments"] = segments_df.drop(columns=['geometry', 'created_at', 'id'], errors='ignore')

        # Remove id and created_at from agg tables
        self.export_data["agg_asset"] = agg_asset_df.drop(columns=['id', 'created_at'], errors='ignore')
        self.export_data["agg_gbfi"] = agg_gbfi_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore')
        self.export_data["agg_ballast"] = agg_ballast_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore')
        self.export_data["agg_tsr"] = agg_tsr_df.drop(columns=['id', 'created_at'], errors='ignore')
        self.export_data["agg_tqi"] = agg_tqi_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore')
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
            'plainline_treatment_model': self.find_plainline_treatment(merged_df),
            'plainline_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'plainline_treatment_override': None,  # Empty column for manual input
            'plainline_treatment_final': None,  # Will be replaced with Excel formula
            'lx_treatment_model': self.find_lx_treatment(merged_df),
            'lx_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'lx_treatment_override': None,  # Empty column for manual input
            'lx_treatment_final': None,  # Will be replaced with Excel formula
            'irj_treatment_model': self.find_irj_treatment(merged_df),
            'irj_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'irj_treatment_override': None,  # Empty column for manual input
            'irj_treatment_final': None,  # Will be replaced with Excel formula
            'turnout_treatment_model': self.find_turnout_treatment(merged_df),
            'turnout_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'turnout_treatment_override': None,  # Empty column for manual input
            'turnout_treatment_final': None,  # Will be replaced with Excel formula
            'bridge_treatment_model': self.find_bridge_treatment(merged_df),
            'bridge_treatment_formula': None,  # TEMP: Excel formula for troubleshooting
            'bridge_treatment_override': None,  # Empty column for manual input
            'bridge_treatment_final': None,  # Will be replaced with Excel formula
        })

        # COMMENTED OUT: Calculate in Excel instead for dynamic updates when users override treatments
        # Calculate problem_zone: 1 if any treatment is not "No Treatment", 0 otherwise
        # Excel formula: =IF(OR(final_col1<>"No Treatment", final_col2<>"No Treatment", ...), 1, 0)
        # recommendations_df['problem_zone'] = (
        #     (recommendations_df['plainline_treatment_model'] != "No Treatment") |
        #     (recommendations_df['lx_treatment_model'] != "No Treatment") |
        #     (recommendations_df['irj_treatment_model'] != "No Treatment") |
        #     (recommendations_df['turnout_treatment_model'] != "No Treatment") |
        #     (recommendations_df['bridge_treatment_model'] != "No Treatment")
        # ).astype(int)
        recommendations_df['problem_zone'] = None  # Will be replaced with Excel formula

        # COMMENTED OUT: Calculate in Excel instead for dynamic updates
        # Calculate problem_zone_group: consecutive problem zones get the same group number
        # Excel formula: =IF(BM2=0, "", IF(COUNTA($BN$1:BN1)=0, 1, IF(BM1=1, BN1, MAXIFS($BN$1:BN1, $BM$1:BM1, 1)+1)))
        # recommendations_df['problem_zone_group'] = self.calculate_problem_zone_group(recommendations_df)
        recommendations_df['problem_zone_group'] = None  # Will be replaced with Excel formula

        # COMMENTED OUT: Calculate in Excel instead for dynamic updates
        # Calculate problem_zone_length_m: total chainage length for each group
        # Excel formula: =SUMIF($problem_zone_group_col, problem_zone_group, $segment_length_col)
        # recommendations_df['problem_zone_length_m'] = self.calculate_problem_zone_length(
        #     recommendations_df,
        #     segments_df
        # )
        recommendations_df['problem_zone_length_m'] = None  # Will be replaced with Excel formula

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
        length_series = length_series.where(merged['problem_zone_group'].notna(), None)

        return length_series

    def write_thresholds_tab(self, writer: pd.ExcelWriter):
        """
        Write all treatment thresholds to a 'Thresholds' tab in the Excel report.

        Args:
            writer: ExcelWriter object
        """
        # Compile all thresholds into a single DataFrame
        thresholds_data = []

        # Plainline thresholds
        for key, value in PLAINLINE_TREATMENT_THRESHOLDS.items():
            thresholds_data.append({
                'Treatment Type': 'Plainline',
                'Threshold Name': key,
                'Value': value,
                'Description': self._get_threshold_description(key)
            })

        # LX thresholds
        for key, value in LX_TREATMENT_THRESHOLDS.items():
            thresholds_data.append({
                'Treatment Type': 'Level Crossing',
                'Threshold Name': key,
                'Value': value,
                'Description': self._get_threshold_description(key)
            })

        # IRJ thresholds
        for key, value in IRJ_TREATMENT_THRESHOLDS.items():
            thresholds_data.append({
                'Treatment Type': 'IRJ',
                'Threshold Name': key,
                'Value': value,
                'Description': self._get_threshold_description(key)
            })

        # Turnout thresholds
        for key, value in TURNOUT_TREATMENT_THRESHOLDS.items():
            thresholds_data.append({
                'Treatment Type': 'Turnout',
                'Threshold Name': key,
                'Value': value,
                'Description': self._get_threshold_description(key)
            })

        # Bridge thresholds
        for key, value in BRIDGE_TREATMENT_THRESHOLDS.items():
            thresholds_data.append({
                'Treatment Type': 'Bridge',
                'Threshold Name': key,
                'Value': value,
                'Description': self._get_threshold_description(key)
            })

        # Create DataFrame and write to Excel
        thresholds_df = pd.DataFrame(thresholds_data)
        thresholds_df.to_excel(writer, sheet_name='Thresholds', index=False)

        # Format the Thresholds sheet
        worksheet = writer.sheets['Thresholds']

        # Set column widths
        worksheet.column_dimensions['A'].width = 20  # Treatment Type
        worksheet.column_dimensions['B'].width = 40  # Threshold Name
        worksheet.column_dimensions['C'].width = 15  # Value
        worksheet.column_dimensions['D'].width = 60  # Description

        # Style header row
        for col in ['A', 'B', 'C', 'D']:
            cell = worksheet[f'{col}1']
            cell.font = Font(bold=True, size=11)
            cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
            cell.alignment = Alignment(horizontal='center', vertical='center')

        logging.info("Written Thresholds tab")

    def _get_threshold_description(self, threshold_name: str) -> str:
        """Get human-readable description for threshold names."""
        descriptions = {
            'gbfi_bog_threshold_1': 'GBFI max threshold for bog hole (1st condition)',
            'gbfi_bog_threshold_2': 'GBFI max threshold for bog hole (2nd condition)',
            'ballast_clean_threshold': 'GBFI avg threshold for ballast clean',
            'lift_tamp_gbfi_avg_threshold': 'GBFI avg threshold for lift & tamp',
            'lift_tamp_ballast_depth_threshold': 'Ballast depth threshold for lift & tamp (m)',
            'tamp_only_gbfi_avg_threshold': 'GBFI avg threshold for tamp only',
            'tamp_only_ballast_depth_threshold': 'Ballast depth threshold for tamp only (m)',
            'lx_renewal_gbfi_max_threshold': 'GBFI max threshold for level crossing renewal',
            'lx_renewal_gbfi_avg_threshold': 'GBFI avg threshold for level crossing renewal',
            'irj_renewal_gbfi_max_threshold': 'GBFI max threshold for IRJ renewal',
            'turnout_renewal_ballast_depth_threshold': 'Ballast depth threshold for turnout renewal (m)',
            'bridge_renewal_ballast_depth_threshold': 'Ballast depth threshold for bridge renewal (m)',
            'bridge_renewal_gbfi_avg_threshold': 'GBFI avg threshold for bridge renewal',
        }
        return descriptions.get(threshold_name, '')

    def write_excel_report(self):
        """
        Generate an Excel workbook with one tab per line_code, joining all aggregated tables.

        """
        logging.info("Generating Excel report...")
        
        output_file = os.path.join(self.output_path, "fmg_rail_report.xlsx")
        
        # Create Excel writer
        all_line_segments = self.export_data["segments"]

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Write thresholds tab first
            self.write_thresholds_tab(writer)
            for line_code in self.line_codes:
                logging.info(f"Processing line_code: {line_code}")

                # Filter segments for this line_code
                line_segments = all_line_segments[all_line_segments['line_code'] == line_code].copy()

                if line_segments.empty:
                    logging.warning(f"No segments found for line_code: {line_code}")
                    continue

                # Track column sources for header row
                segment_cols = list(line_segments.columns)

                # Merge all aggregated data on chainage_id
                merged = line_segments.copy()

                # Left join each agg table and track new columns
                asset_cols = []
                if not self.export_data["agg_asset"].empty:
                    before_cols = set(merged.columns)
                    merged = merged.merge(
                        self.export_data["agg_asset"],
                        on='chainage_id',
                        how='left',
                        suffixes=('', '_asset')
                    )
                    asset_cols = [col for col in merged.columns if col not in before_cols]

                gbfi_cols = []
                if not self.export_data["agg_gbfi"].empty:
                    before_cols = set(merged.columns)
                    merged = merged.merge(
                        self.export_data["agg_gbfi"],
                        on='chainage_id',
                        how='left',
                        suffixes=('', '_gbfi')
                    )
                    gbfi_cols = [col for col in merged.columns if col not in before_cols]

                ballast_cols = []
                if not self.export_data["agg_ballast"].empty:
                    before_cols = set(merged.columns)
                    merged = merged.merge(
                        self.export_data["agg_ballast"],
                        on='chainage_id',
                        how='left',
                        suffixes=('', '_ballast')
                    )
                    ballast_cols = [col for col in merged.columns if col not in before_cols]

                tsr_cols = []
                if not self.export_data["agg_tsr"].empty:
                    before_cols = set(merged.columns)
                    merged = merged.merge(
                        self.export_data["agg_tsr"],
                        on='chainage_id',
                        how='left',
                        suffixes=('', '_tsr')
                    )
                    tsr_cols = [col for col in merged.columns if col not in before_cols]

                tqi_cols = []
                if not self.export_data["agg_tqi"].empty:
                    before_cols = set(merged.columns)
                    merged = merged.merge(
                        self.export_data["agg_tqi"],
                        on='chainage_id',
                        how='left',
                        suffixes=('', '_tqi')
                    )
                    tqi_cols = [col for col in merged.columns if col not in before_cols]
                
                dtr_cols = []
                if not self.export_data["agg_dtr"].empty:
                    before_cols = set(merged.columns)
                    merged = merged.merge(
                        self.export_data["agg_dtr"],
                        on='chainage_id',
                        how='left',
                        suffixes=('', '_dtr')
                    )
                    dtr_cols = [col for col in merged.columns if col not in before_cols]

                recommendation_cols = []
                if "recommendations" in self.export_data and not self.export_data["recommendations"].empty:
                    before_cols = set(merged.columns)
                    merged = merged.merge(
                        self.export_data["recommendations"],
                        on='chainage_id',
                        how='left',
                        suffixes=('', '_rec')
                    )
                    recommendation_cols = [col for col in merged.columns if col not in before_cols]

                # Sort by chainage_start_km
                merged = merged.sort_values('chainage_start_km')

                # Write to Excel tab (sheet name limited to 31 chars)
                sheet_name = f"{line_code}"[:31]
                merged.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)

                # Access the worksheet to add merged header row
                worksheet = writer.sheets[sheet_name]

                # Build column group mapping
                col_groups = []

                # Determine source for each column
                start_col = 1  # Excel columns are 1-indexed

                if segment_cols:
                    end_col = start_col + len(segment_cols) - 1
                    col_groups.append(('Rail Segments', start_col, end_col))
                    start_col = end_col + 1

                if asset_cols:
                    end_col = start_col + len(asset_cols) - 1
                    col_groups.append(('Assets', start_col, end_col))
                    start_col = end_col + 1

                if gbfi_cols:
                    end_col = start_col + len(gbfi_cols) - 1
                    col_groups.append(('Ground Ballast Fouling Index (GBFI)', start_col, end_col))
                    start_col = end_col + 1

                if ballast_cols:
                    end_col = start_col + len(ballast_cols) - 1
                    col_groups.append(('Ballast', start_col, end_col))
                    start_col = end_col + 1

                if tsr_cols:
                    end_col = start_col + len(tsr_cols) - 1
                    col_groups.append(('Temporary Speed Restriction (TSR)', start_col, end_col))
                    start_col = end_col + 1

                if tqi_cols:
                    end_col = start_col + len(tqi_cols) - 1
                    col_groups.append(('Track Quality Indicator (TQI)', start_col, end_col))
                    start_col = end_col + 1
                    
                if dtr_cols:
                    end_col = start_col + len(dtr_cols) - 1
                    col_groups.append(('Dynamic Track Force (DTR)', start_col, end_col))
                    start_col = end_col + 1

                if recommendation_cols:
                    end_col = start_col + len(recommendation_cols) - 1
                    col_groups.append(('Recommendations', start_col, end_col))

                add_merged_header_row(worksheet, col_groups)

                # Add Excel formulas to treatment final columns
                cols_list = list(merged.columns)

                # Helper function to add formulas for a treatment type
                def add_treatment_formulas(model_col, override_col, final_col):
                    if model_col in merged.columns and override_col in merged.columns and final_col in merged.columns:
                        # Find column indices (1-indexed for Excel)
                        model_col_idx = cols_list.index(model_col) + 1
                        override_col_idx = cols_list.index(override_col) + 1
                        final_col_idx = cols_list.index(final_col) + 1

                        # Convert to Excel column letters
                        model_col_letter = get_column_letter(model_col_idx)
                        override_col_letter = get_column_letter(override_col_idx)
                        final_col_letter = get_column_letter(final_col_idx)

                        # Write formula to each row (starting at row 3, since row 1 is merged header, row 2 is column names)
                        for row_num in range(3, len(merged) + 3):  # +3 because: 1 header row + 1 column name row + 1-indexed
                            formula = f'=IF(ISBLANK({override_col_letter}{row_num}),{model_col_letter}{row_num},{override_col_letter}{row_num})'
                            worksheet[f'{final_col_letter}{row_num}'] = formula

                # Add formulas for plainline treatment
                add_treatment_formulas('plainline_treatment_model', 'plainline_treatment_override', 'plainline_treatment_final')

                # Add formulas for lx treatment
                add_treatment_formulas('lx_treatment_model', 'lx_treatment_override', 'lx_treatment_final')
                
                # Add formulas for irj treatment
                add_treatment_formulas('irj_treatment_model', 'irj_treatment_override', 'irj_treatment_final')
                
                # Add formulas for turnout treatment
                add_treatment_formulas('turnout_treatment_model', 'turnout_treatment_override', 'turnout_treatment_final')
                
                # Add formulas for bridge treatment
                add_treatment_formulas('bridge_treatment_model', 'bridge_treatment_override', 'bridge_treatment_final')

                # TEMP: Add formula columns that reference Thresholds tab for troubleshooting
                self._add_treatment_formula_columns(worksheet, merged, cols_list)

                # Add formulas for problem_zone, problem_zone_group, problem_zone_length_m
                if all(col in merged.columns for col in ['plainline_treatment_final', 'lx_treatment_final', 'irj_treatment_final', 'turnout_treatment_final', 'bridge_treatment_final', 'problem_zone', 'problem_zone_group', 'problem_zone_length_m']):
                    plainline_idx = cols_list.index('plainline_treatment_final') + 1
                    lx_idx = cols_list.index('lx_treatment_final') + 1
                    irj_idx = cols_list.index('irj_treatment_final') + 1
                    turnout_idx = cols_list.index('turnout_treatment_final') + 1
                    bridge_idx = cols_list.index('bridge_treatment_final') + 1
                    problem_zone_idx = cols_list.index('problem_zone') + 1
                    problem_zone_group_idx = cols_list.index('problem_zone_group') + 1
                    problem_zone_length_idx = cols_list.index('problem_zone_length_m') + 1

                    # Get column letters
                    plainline_letter = get_column_letter(plainline_idx)
                    lx_letter = get_column_letter(lx_idx)
                    irj_letter = get_column_letter(irj_idx)
                    turnout_letter = get_column_letter(turnout_idx)
                    bridge_letter = get_column_letter(bridge_idx)
                    problem_zone_letter = get_column_letter(problem_zone_idx)
                    problem_zone_group_letter = get_column_letter(problem_zone_group_idx)
                    problem_zone_length_letter = get_column_letter(problem_zone_length_idx)

                    # Get chainage length columns if available
                    chainage_start_letter = None
                    chainage_end_letter = None
                    if 'chainage_start_km' in merged.columns and 'chainage_end_km' in merged.columns:
                        chainage_start_idx = cols_list.index('chainage_start_km') + 1
                        chainage_end_idx = cols_list.index('chainage_end_km') + 1
                        chainage_start_letter = get_column_letter(chainage_start_idx)
                        chainage_end_letter = get_column_letter(chainage_end_idx)

                    for row_num in range(3, len(merged) + 3):
                        # Formula for problem_zone: 1 if any final treatment is not "No Treatment", 0 otherwise
                        problem_zone_formula = f'=IF(OR({plainline_letter}{row_num}<>"No Treatment",{lx_letter}{row_num}<>"No Treatment",{irj_letter}{row_num}<>"No Treatment",{turnout_letter}{row_num}<>"No Treatment",{bridge_letter}{row_num}<>"No Treatment"),1,0)'
                        worksheet[f'{problem_zone_letter}{row_num}'] = problem_zone_formula

                        # Formula for problem_zone_group: consecutive problem zones get same group number
                        if row_num == 3:  # First data row
                            problem_zone_group_formula = f'=IF({problem_zone_letter}{row_num}=0,"",1)'
                        else:
                            prev_row = row_num - 1
                            problem_zone_group_formula = f'=IF({problem_zone_letter}{row_num}=0,"",IF({problem_zone_letter}{prev_row}=1,{problem_zone_group_letter}{prev_row},IF(COUNTA(${problem_zone_group_letter}$3:{problem_zone_group_letter}{prev_row})=0,1,MAX(${problem_zone_group_letter}$3:{problem_zone_group_letter}{prev_row})+1)))'
                        worksheet[f'{problem_zone_group_letter}{row_num}'] = problem_zone_group_formula

                        # Formula for problem_zone_length_m: total chainage length for the group
                        if chainage_start_letter and chainage_end_letter:
                            # Use SUMPRODUCT to calculate segment lengths and sum for the group
                            last_row = len(merged) + 2
                            problem_zone_length_formula = f'=IF({problem_zone_letter}{row_num}=0,"",SUMPRODUCT((${problem_zone_group_letter}$3:${problem_zone_group_letter}${last_row}={problem_zone_group_letter}{row_num})*(${chainage_end_letter}$3:${chainage_end_letter}${last_row}-${chainage_start_letter}$3:${chainage_start_letter}${last_row})*1000))'
                            worksheet[f'{problem_zone_length_letter}{row_num}'] = problem_zone_length_formula


                logging.info(f"  Written {len(merged)} rows to sheet '{sheet_name}'")

        logging.info(f"Excel report saved to: {self.output_path}")
    
def add_merged_header_row(worksheet, col_groups: list):
    # Insert merged cells in first row

    for source_name, start_idx, end_idx in col_groups:
        start_letter = get_column_letter(start_idx)
        end_letter = get_column_letter(end_idx)

        # Merge cells
        if start_idx == end_idx:
            cell_ref = f"{start_letter}1"
        else:
            cell_ref = f"{start_letter}1:{end_letter}1"
            worksheet.merge_cells(cell_ref)

        # Write source label
        cell = worksheet[f"{start_letter}1"]
        cell.value = source_name
        cell.font = Font(bold=True, size=12)
        cell.alignment = Alignment(horizontal='center', vertical='center')

        # Add background color based on source
        colors = {
            'Rail Segments': 'DDEBF7',  # Light blue
            'Assets': 'E2EFDA',    # Light green
            'Ground Ballast Fouling Index (GBFI)': 'FCE4D6',     # Light orange
            'Ballast': 'F8CBAD',    # Light peach
            'Temporary Speed Restriction (TSR)': 'FFF2CC',      # Light yellow
            'Track Quality Indicator (TQI)': 'F4B084',      # Light coral
            'Dynamic Track Force (DTR)': 'E4DFEC',      # Light purple
            'Recommendations': 'D5F4E6',      # Light mint green
        }
        if source_name in colors:
            cell.fill = PatternFill(start_color=colors[source_name],
                                end_color=colors[source_name],
                                fill_type='solid')

    # Set row height for header
    worksheet.row_dimensions[1].height = 20

