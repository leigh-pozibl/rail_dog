"""
Standalone treatment recommendation functions.

Each function takes a segments DataFrame and an export_data dict (keyed by table
name, values are DataFrames with a chainage_id column), and returns a Series of
treatment recommendations.
"""
import pandas as pd

from rail_dog.report.rule_engine import RuleEngine
from rail_dog.configs.thresholds import (
    PLAINLINE_TREATMENT_THRESHOLDS,
    LX_TREATMENT_THRESHOLDS,
    IRJ_TREATMENT_THRESHOLDS,
    TURNOUT_TREATMENT_THRESHOLDS,
    BRIDGE_TREATMENT_THRESHOLDS,
)

# Treatment hierarchy from most to least aggressive (used for windowed smoothing)
PLAINLINE_TREATMENT_HIERARCHY = [
    "Local Formation Renewal (Bog Hole)",
    "Ballast Clean",
    "Lift & Tamp",
    "Tamp Only",
    "No Treatment",
]


def find_plainline_treatment(segments_df: pd.DataFrame, export_data: dict) -> pd.Series:
    """
    Apply plainline treatment recommendation rules to segment data.

    Uses qualified field names (table.field) to reference fields from different
    aggregated tables in export_data.

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
    engine = RuleEngine(export_data=export_data, thresholds=PLAINLINE_TREATMENT_THRESHOLDS)
    return engine.apply_rules(segments_df, plainline_treatment_rules)
    
def find_lx_treatment(segments_df: pd.DataFrame, export_data: dict) -> pd.Series:
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
    engine = RuleEngine(export_data=export_data, thresholds=LX_TREATMENT_THRESHOLDS)
    result = engine.apply_rules(segments_df, lx_treatment_rules)

    # OPTIMIZED: Vectorized replacement of __LX_COUNT__ placeholder with actual wz_level_crossing count
    mask = result == "__LX_COUNT__"
    if mask.any() and "agg_asset.wz_level_crossing" in segments_df.columns:
        result[mask] = segments_df.loc[mask, "agg_asset.wz_level_crossing"]

    return result
    
def find_irj_treatment(segments_df: pd.DataFrame, export_data: dict) -> pd.Series:
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
    engine = RuleEngine(export_data=export_data, thresholds=IRJ_TREATMENT_THRESHOLDS)
    result = engine.apply_rules(segments_df, irj_treatment_rules)

    # OPTIMIZED: Vectorized replacement of __IRJ_COUNT__ placeholder with actual IRJ count
    mask = result == "__IRJ_COUNT__"
    if mask.any() and "agg_asset.irj" in segments_df.columns:
        result[mask] = segments_df.loc[mask, "agg_asset.irj"]

    return result

def find_turnout_treatment(segments_df: pd.DataFrame, export_data: dict) -> pd.Series:
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
    engine = RuleEngine(export_data=export_data, thresholds=TURNOUT_TREATMENT_THRESHOLDS)
    result = engine.apply_rules(segments_df, turnout_treatment_rules)

    # OPTIMIZED: Vectorized replacement of __TURNOUT_COUNT__ placeholder with actual turnout count
    mask = result == "__TURNOUT_COUNT__"
    if mask.any() and "agg_asset.wz_turnout" in segments_df.columns:
        result[mask] = segments_df.loc[mask, "agg_asset.wz_turnout"]

    return result

def find_bridge_treatment(segments_df: pd.DataFrame, export_data: dict) -> pd.Series:
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
    engine = RuleEngine(export_data=export_data, thresholds=BRIDGE_TREATMENT_THRESHOLDS)
    result = engine.apply_rules(segments_df, bridge_treatment_rules)

    # OPTIMIZED: Vectorized replacement of __BRIDGE_COUNT__ placeholder with actual bridge count
    mask = result == "__BRIDGE_COUNT__"
    if mask.any() and "agg_asset.wz_bridge" in segments_df.columns:
        result[mask] = segments_df.loc[mask, "agg_asset.wz_bridge"]

    return result
