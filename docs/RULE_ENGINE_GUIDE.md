# RuleEngine Guide

This guide explains how to use the list-based `RuleEngine` pattern to translate Excel formulas into Python rules.

## Quick Start

```python
from rail_dog.model import RuleEngine
import pandas as pd

# 1. Define your thresholds (optional)
thresholds = {
    "max_wear": 15.0,
    "min_quality": 80
}

# 2. Define your rules as a list
rules = [
    [("fixed_asset", "==", "Yes"), "No Treatment"],
    [("wear", ">=", "threshold.max_wear"), "Replace"],
    "Monitor"  # Default value
]

# 3. Apply to your DataFrame
engine = RuleEngine(thresholds=thresholds)
df['recommendation'] = engine.apply_rules(df, rules)
```

## Qualified Field Names (NEW!)

The RuleEngine supports **qualified field names** using dot notation (`table.field`) to reference fields from different source DataFrames. This is useful when you want to clearly specify which table a field comes from.

### Example with Qualified Fields

```python
# Your data is organized in multiple DataFrames
export_data = {
    "agg_asset": assets_df,      # contains: chainage_id, fixed_asset, turnout, etc.
    "agg_gbfi": gbfi_df,          # contains: chainage_id, max_of_avg, fouled_zone, etc.
    "agg_tqi": tqi_df,            # contains: chainage_id, tqi, status, etc.
}

# Define rules using qualified field names
rules = [
    [("agg_asset.fixed_asset", "==", True), "No Treatment"],
    [
        ["AND",
            ("agg_tqi.status", "==", "Poor"),
            ("agg_gbfi.max_of_avg", ">=", 100)
        ],
        "Formation Renewal"
    ],
    "Monitor"
]

# Pass export_data to the engine
engine = RuleEngine(export_data=export_data, thresholds=thresholds)

# Apply rules to a DataFrame with chainage_id column
# The engine will look up values from the appropriate source tables
segments_df = pd.DataFrame({'chainage_id': ['CH-001', 'CH-002', ...]})
segments_df['treatment'] = engine.apply_rules(segments_df, rules)
```

### How Qualified Fields Work

1. **Dot notation**: `"table.field"` splits into table name and field name
2. **Lookup**: Engine finds the table in `export_data`
3. **Match**: Looks up the row with matching `chainage_id`
4. **Return**: Gets the field value from that row

**Requirements:**
- All source DataFrames must have a `chainage_id` column
- The input DataFrame passed to `apply_rules()` must also have `chainage_id`
- Missing values return `pd.NA` if no match is found

### Backward Compatibility

Simple field names (without dots) still work and look up directly in the current row:

```python
# This works fine - looks for 'wear' in the current row
[("wear", ">=", 15), "Replace"]

# This looks up agg_asset table for the field 'fixed_asset'
[("agg_asset.fixed_asset", "==", True), "No Treatment"]
```

## Rule Syntax

### Basic Structure

```python
rules = [
    [condition, result],
    [condition, result],
    ...
    default_result  # Optional default (last item)
]
```

### Simple Conditions

Format: `(field_name, operator, value)`

```python
("wear", ">=", 15)
("status", "==", "Critical")
("rail_type", "in", ["A", "B", "C"])
```

**Supported operators:**
- `==`, `!=` - equality
- `>`, `>=`, `<`, `<=` - comparison
- `in`, `not in` - membership
- `is`, `is not` - identity (for None/null checks)

### AND Conditions

Format: `["AND", condition1, condition2, ...]`

```python
["AND",
    ("status", "==", "Poor"),
    ("age", ">", 10),
    ("wear", ">=", 5)
]
```

### OR Conditions

Format: `["OR", condition1, condition2, ...]`

```python
["OR",
    ("status", "==", "Poor"),
    ("status", "==", "Critical"),
    ("status", "==", "Failed")
]
```

### Nested Conditions

You can nest AND/OR to any depth:

```python
["AND",
    ["OR",
        ("type", "==", "Curve"),
        ("radius", "<", 500)
    ],
    ["OR",
        ("wear", ">", 10),
        ("defects", ">", 3)
    ]
]
```

## Threshold References

Instead of hardcoding values, reference thresholds using `"threshold.key_name"`:

```python
thresholds = {
    "severe_wear": 15.0,
    "moderate_wear": 8.0
}

rules = [
    [("wear", ">=", "threshold.severe_wear"), "Urgent"],
    [("wear", ">=", "threshold.moderate_wear"), "Scheduled"],
    "Normal"
]

engine = RuleEngine(thresholds=thresholds)
```

## Handling Missing Data

Missing/null values can be checked with `is None`:

```python
rules = [
    [("value", "is", None), "No Data"],
    [("value", ">", 100), "High"],
    "Normal"
]
```

## Complete Example: Translating Excel Formula

**Excel formula:**
```excel
=IF(fixed_asset="Yes",
    "No Treatment",
    IF(AND(OR(status="Poor", status="Critical"), gbfi>=100),
        "Formation Renewal",
        IF(AND(gbfi>=80, severity="S1"),
            "Formation Renewal",
            "No Treatment")))
```

**Python equivalent:**
```python
rules = [
    # First IF
    [
        ("fixed_asset", "==", "Yes"),
        "No Treatment"
    ],

    # Second IF: AND(OR(status), gbfi>=100)
    [
        ["AND",
            ["OR",
                ("status", "==", "Poor"),
                ("status", "==", "Critical")
            ],
            ("gbfi", ">=", 100)
        ],
        "Formation Renewal"
    ],

    # Third IF: AND(gbfi>=80, severity="S1")
    [
        ["AND",
            ("gbfi", ">=", 80),
            ("severity", "==", "S1")
        ],
        "Formation Renewal"
    ],

    # Default
    "No Treatment"
]

engine = RuleEngine()
result = engine.apply_rules(merged_df, rules)
```

## Integration with Model Class

In your `Model` class, create methods for each rule set:

```python
class Model:
    def plainline_treatment(self, merged_df: pd.DataFrame) -> pd.Series:
        """Apply plainline treatment rules."""
        thresholds = {
            "bog_threshold": 100,
            "ballast_threshold": 50,
        }

        rules = [
            [("fixed_asset", "==", "Yes"), "No Treatment"],
            # ... more rules
            "No Treatment"
        ]

        engine = RuleEngine(thresholds=thresholds)
        return engine.apply_rules(merged_df, rules)

    def generate_excel_report(self):
        """Generate report with rule-based columns."""
        # ... merge your data tables ...

        # Apply multiple rule sets
        merged['plainline_treatment'] = self.plainline_treatment(merged)
        merged['maintenance_priority'] = self.maintenance_priority(merged)
        merged['inspection_frequency'] = self.inspection_frequency(merged)

        # ... write to Excel ...
```

## Tips

1. **Order matters**: Rules are evaluated top-to-bottom, first match wins
2. **Always include a default**: Last item should be a simple value (string, None, etc.)
3. **Test incrementally**: Start with simple rules, add complexity gradually
4. **Use comments**: Document what each rule does
5. **Keep field names consistent**: Match your DataFrame column names exactly
6. **Validate thresholds**: Ensure threshold keys exist in your threshold dict

## Running Examples

See `rail_dog/rule_examples.py` for complete working examples:

```bash
cd /home/leigh/repos/rail_dog
python -m rail_dog.rule_examples
```

## Common Patterns

### Range checks
```python
# Between two values
["AND", ("value", ">=", 10), ("value", "<=", 20)]

# Outside range
["OR", ("value", "<", 10), ("value", ">", 20)]
```

### Multiple field checks
```python
["AND",
    ("wear", ">", 10),
    ("age", ">", 5),
    ("defects", ">", 0)
]
```

### Priority/cascade logic
```python
rules = [
    [("priority", "==", "P1"), "Urgent - 24hrs"],
    [("priority", "==", "P2"), "High - 1 week"],
    [("priority", "==", "P3"), "Medium - 1 month"],
    "Low - Next cycle"
]
```
