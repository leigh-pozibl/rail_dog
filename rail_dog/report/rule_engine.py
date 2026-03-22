import logging
from typing import Optional, Any

import pandas as pd


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

