"""
Examples demonstrating the RuleEngine usage patterns for different scenarios.

This file shows how to translate Excel formulas to list-based rules.
"""

from rail_dog.report import RuleEngine
import pandas as pd


def example_simple_rules():
    """
    Simple sequential IF rules without complex logic.

    Excel equivalent:
        =IF(A2="High", "Priority 1",
           IF(A2="Medium", "Priority 2",
              IF(A2="Low", "Priority 3", "Unknown")))
    """
    rules = [
        [("priority", "==", "High"), "Priority 1"],
        [("priority", "==", "Medium"), "Priority 2"],
        [("priority", "==", "Low"), "Priority 3"],
        "Unknown"  # Default
    ]

    # Example usage
    df = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003', 'CH-004'],
        'priority': ['High', 'Medium', 'Low', None]
    })

    engine = RuleEngine()
    df['recommendation'] = engine.apply_rules(df, rules)
    return df


def example_and_or_rules():
    """
    Rules with AND/OR logic.

    Excel equivalent:
        =IF(AND(status="Critical", age>10), "Replace Immediately",
           IF(OR(status="Poor", AND(status="Fair", age>5)), "Schedule Replacement",
              "Monitor"))
    """
    rules = [
        # Critical AND old
        [
            ["AND",
                ("status", "==", "Critical"),
                ("age", ">", 10)
            ],
            "Replace Immediately"
        ],

        # Poor OR (Fair AND old)
        [
            ["OR",
                ("status", "==", "Poor"),
                ["AND",
                    ("status", "==", "Fair"),
                    ("age", ">", 5)
                ]
            ],
            "Schedule Replacement"
        ],

        "Monitor"  # Default
    ]

    df = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003'],
        'status': ['Critical', 'Poor', 'Fair'],
        'age': [12, 8, 3]
    })

    engine = RuleEngine()
    df['action'] = engine.apply_rules(df, rules)
    return df


def example_threshold_references():
    """
    Using threshold dictionary for configurable values.

    Excel equivalent:
        =IF(wear>=Thresholds!$A$1, "Severe Wear",
           IF(wear>=Thresholds!$A$2, "Moderate Wear",
              "Normal"))
    """
    thresholds = {
        "severe_wear": 15.0,
        "moderate_wear": 8.0,
        "max_temp": 45.0
    }

    rules = [
        [("wear_mm", ">=", "threshold.severe_wear"), "Severe Wear"],
        [("wear_mm", ">=", "threshold.moderate_wear"), "Moderate Wear"],
        "Normal"
    ]

    df = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003'],
        'wear_mm': [16.5, 10.2, 5.0]
    })

    engine = RuleEngine(thresholds=thresholds)
    df['wear_status'] = engine.apply_rules(df, rules)
    return df


def example_complex_nested_logic():
    """
    Complex nested AND/OR conditions.

    Excel equivalent:
        =IF(AND(OR(type="Curve", radius<500),
                OR(AND(wear>10, speed>80), defect_count>3)),
           "High Risk",
           IF(AND(wear>5, OR(speed>60, traffic="Heavy")),
              "Medium Risk",
              "Low Risk"))
    """
    rules = [
        # High Risk: (Curve OR tight radius) AND (high wear+speed OR many defects)
        [
            ["AND",
                ["OR",
                    ("section_type", "==", "Curve"),
                    ("radius", "<", 500)
                ],
                ["OR",
                    ["AND",
                        ("wear", ">", 10),
                        ("speed", ">", 80)
                    ],
                    ("defect_count", ">", 3)
                ]
            ],
            "High Risk"
        ],

        # Medium Risk: moderate wear AND (high speed OR heavy traffic)
        [
            ["AND",
                ("wear", ">", 5),
                ["OR",
                    ("speed", ">", 60),
                    ("traffic", "==", "Heavy")
                ]
            ],
            "Medium Risk"
        ],

        "Low Risk"  # Default
    ]

    df = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003'],
        'section_type': ['Curve', 'Tangent', 'Curve'],
        'radius': [400, 1000, 600],
        'wear': [12, 6, 4],
        'speed': [85, 70, 50],
        'defect_count': [2, 1, 5],
        'traffic': ['Light', 'Heavy', 'Medium']
    })

    engine = RuleEngine()
    df['risk_level'] = engine.apply_rules(df, rules)
    return df


def example_null_handling():
    """
    Handling missing/null values in rules.

    Excel equivalent:
        =IF(ISBLANK(A2), "No Data",
           IF(A2>100, "High", "Normal"))
    """
    rules = [
        # Check for missing data first
        [("value", "is", None), "No Data"],

        # Then check value
        [("value", ">", 100), "High"],

        "Normal"
    ]

    df = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003', 'CH-004'],
        'value': [150, 50, None, pd.NA]
    })

    engine = RuleEngine()
    df['status'] = engine.apply_rules(df, rules)
    return df


def example_in_operator():
    """
    Using 'in' operator for multiple value matching.

    Excel equivalent:
        =IF(OR(type="A", type="B", type="C"), "Group 1",
           IF(OR(type="D", type="E"), "Group 2", "Other"))
    """
    rules = [
        [("rail_type", "in", ["A", "B", "C"]), "Group 1"],
        [("rail_type", "in", ["D", "E"]), "Group 2"],
        "Other"
    ]

    df = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003', 'CH-004'],
        'rail_type': ['A', 'D', 'F', 'C']
    })

    engine = RuleEngine()
    df['group'] = engine.apply_rules(df, rules)
    return df


def example_multiple_rule_sets():
    """
    Applying multiple different rule sets to the same data.
    """
    df = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003'],
        'wear': [15, 8, 3],
        'defects': [5, 2, 0],
        'age': [12, 7, 2]
    })

    # Rule set 1: Maintenance priority
    priority_rules = [
        [("wear", ">=", 12), "Urgent"],
        [("wear", ">=", 6), "Scheduled"],
        "Normal"
    ]

    # Rule set 2: Inspection frequency
    inspection_rules = [
        [
            ["OR",
                ("defects", ">", 3),
                ("age", ">", 10)
            ],
            "Monthly"
        ],
        [("defects", ">", 0), "Quarterly"],
        "Annual"
    ]

    # Rule set 3: Treatment type
    treatment_rules = [
        [
            ["AND",
                ("wear", ">=", 10),
                ("defects", ">", 2)
            ],
            "Full Replacement"
        ],
        [("wear", ">=", 10), "Grinding"],
        [("defects", ">", 0), "Repair"],
        "None"
    ]

    engine = RuleEngine()
    df['priority'] = engine.apply_rules(df, priority_rules)
    df['inspection'] = engine.apply_rules(df, inspection_rules)
    df['treatment'] = engine.apply_rules(df, treatment_rules)

    return df


def example_qualified_fields():
    """
    Using qualified field names to reference fields from different source tables.

    This pattern is useful when you have data split across multiple DataFrames
    and want to avoid pre-merging them. The engine will look up values from
    the appropriate source table using chainage_id matching.
    """
    # Create source DataFrames (simulating agg_asset, agg_gbfi, agg_tqi tables)
    agg_asset = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003', 'CH-004'],
        'fixed_asset': [True, False, False, False],
        'turnout': [1, 0, 0, 2],
    })

    agg_gbfi = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003', 'CH-004'],
        'max_of_avg': [50, 105, 85, 30],
        'fouled_zone': [False, True, True, False],
        'ballast_lt_250mm': [False, True, False, False],
    })

    agg_tqi = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003', 'CH-004'],
        'tqi': [25, 18, 22, 30],
        'status': ['Good', 'Poor', 'Critical', 'Good'],
    })

    # Package into export_data dict
    export_data = {
        'agg_asset': agg_asset,
        'agg_gbfi': agg_gbfi,
        'agg_tqi': agg_tqi,
    }

    # Define thresholds
    thresholds = {
        'gbfi_threshold': 100,
        'tqi_threshold': 20,
    }

    # Define rules using qualified field names (table.field)
    rules = [
        # If it's a fixed asset, no treatment needed
        [
            ("agg_asset.fixed_asset", "==", True),
            "No Treatment"
        ],

        # High GBFI + Poor TQI + Fouled = Formation Renewal
        [
            ["AND",
                ("agg_gbfi.max_of_avg", ">=", "threshold.gbfi_threshold"),
                ["OR",
                    ("agg_tqi.status", "==", "Poor"),
                    ("agg_tqi.status", "==", "Critical")
                ],
                ("agg_gbfi.fouled_zone", "==", True)
            ],
            "Formation Renewal"
        ],

        # Poor TQI + Low ballast = Ballast Clean
        [
            ["AND",
                ["OR",
                    ("agg_tqi.status", "==", "Poor"),
                    ("agg_tqi.status", "==", "Critical")
                ],
                ("agg_gbfi.ballast_lt_250mm", "==", True)
            ],
            "Ballast Clean"
        ],

        # Poor TQI + Low TQI value = Tamp
        [
            ["AND",
                ["OR",
                    ("agg_tqi.status", "==", "Poor"),
                    ("agg_tqi.status", "==", "Critical")
                ],
                ("agg_tqi.tqi", "<", "threshold.tqi_threshold")
            ],
            "Tamp"
        ],

        "Monitor"  # Default
    ]

    # Create segments DataFrame (just chainage_ids)
    segments = pd.DataFrame({
        'chainage_id': ['CH-001', 'CH-002', 'CH-003', 'CH-004']
    })

    # Apply rules using qualified field names
    engine = RuleEngine(export_data=export_data, thresholds=thresholds)
    segments['treatment'] = engine.apply_rules(segments, rules)

    # Show the results with source data
    result = segments.copy()
    result['fixed_asset'] = agg_asset['fixed_asset'].values
    result['gbfi'] = agg_gbfi['max_of_avg'].values
    result['tqi_status'] = agg_tqi['status'].values
    result['fouled'] = agg_gbfi['fouled_zone'].values

    return result


if __name__ == "__main__":
    # Run examples
    print("=== Simple Rules ===")
    print(example_simple_rules())

    print("\n=== AND/OR Rules ===")
    print(example_and_or_rules())

    print("\n=== Threshold References ===")
    print(example_threshold_references())

    print("\n=== Complex Nested Logic ===")
    print(example_complex_nested_logic())

    print("\n=== Null Handling ===")
    print(example_null_handling())

    print("\n=== IN Operator ===")
    print(example_in_operator())

    print("\n=== Multiple Rule Sets ===")
    print(example_multiple_rule_sets())

    print("\n=== Qualified Field Names ===")
    print(example_qualified_fields())
