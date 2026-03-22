"""
Quick debug script to test rule matching on your data.

Usage:
    python test_rules_debug.py
"""
import os
import logging
import pandas as pd
from dotenv import load_dotenv

from rail_dog.report import RuleEngine
from rail_dog.configs.thresholds import PLAINLINE_TREATMENT_THRESHOLDS
from rail_dog.utils.db_utils import get_table_data, query_agg_tsr

from snappy_utils.params import DBConnection

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_rules():
    """Test the plainline treatment rules on actual data."""

    # Get database connection
    db_env = os.environ.get("LOCAL_CONNECTION_STRING")
    db = DBConnection(db_env)
    engine = db.engine

    # Load data
    print("Loading data...")
    segments_df = get_table_data(engine, table_name="rail_segments")
    agg_asset_df = get_table_data(engine, table_name="agg_assets")
    agg_gbfi_df = get_table_data(engine, table_name="agg_gbfi")
    agg_tsr_df = query_agg_tsr(engine)
    agg_tqi_df = get_table_data(engine, table_name="agg_tqi")
    agg_dtr_df = get_table_data(engine, table_name="agg_dtr")

    # Package into export_data
    export_data = {
        "segments": segments_df,
        "agg_asset": agg_asset_df.drop(columns=['id', 'created_at'], errors='ignore'),
        "agg_gbfi": agg_gbfi_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore'),
        "agg_tsr": agg_tsr_df.drop(columns=['id', 'created_at'], errors='ignore'),
        "agg_tqi": agg_tqi_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore'),
        "agg_dtr": agg_dtr_df.drop(columns=['id', 'collection_date', 'created_at'], errors='ignore'),
    }

    print(f"\nLoaded {len(segments_df)} segments")
    print(f"Loaded {len(agg_asset_df)} asset records")
    print(f"Loaded {len(agg_tqi_df)} TQI records")
    print(f"Loaded {len(agg_gbfi_df)} GBFI records")

    # Define the rules
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
                    ("agg_dtr.worst_dtf", "==", "S1"),
                    ("agg_dtr.worst_dtf", "==", "S2")
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

        # Rule 5: IF AND(OR(TQI status="poor"|"critical"), GBFI Avg, Ballast Centre), "Lift & Tamp"
        [
            ["AND",
                ["OR",
                    ("agg_tqi.status", "==", "poor"),
                    ("agg_tqi.status", "==", "critical")
                ],
                ("agg_gbfi.avg_of_avg", "<", "threshold.lift_tamp_gbfi_avg_threshold"),
                ("agg_gbfi.ballast_centre", "<", "threshold.lift_tamp_ballast_depth_threshold")
            ],
            "Lift & Tamp"
        ],

        # Rule 6: IF AND(OR(TQI status="poor"|"critical"), GBFI Avg, Ballast Centre), "Tamp Only"
        [
            ["AND",
                ["OR",
                    ("agg_tqi.status", "==", "poor"),
                    ("agg_tqi.status", "==", "critical")
                ],
                ("agg_gbfi.avg_of_avg", "<", "threshold.tamp_only_gbfi_avg_threshold"),
                ("agg_gbfi.ballast_centre", ">=", "threshold.tamp_only_ballast_depth_threshold")
            ],
            "Tamp Only"
        ],

        # Default
        "Default"
    ]

    # Create RuleEngine
    engine_rule = RuleEngine(export_data=export_data, thresholds=PLAINLINE_TREATMENT_THRESHOLDS)

    # Target specific segment
    target_chainage = "CHAIN-SL-S-20100-E-20111-MAINLINE"

    # Find the target segment
    print("\n" + "="*80)
    print(f"DEBUGGING SPECIFIC SEGMENT: {target_chainage}")
    print("="*80)

    if target_chainage not in segments_df['chainage_id'].values:
        print(f"\n❌ ERROR: Segment '{target_chainage}' not found in database!")
        print(f"\nSearching for similar segments...")
        similar = segments_df[segments_df['chainage_id'].str.contains('19135', na=False)]
        if not similar.empty:
            print(f"Found {len(similar)} segments containing '19135':")
            for idx, row in similar.head().iterrows():
                print(f"  - {row['chainage_id']}")
        return

    target_row = segments_df[segments_df['chainage_id'] == target_chainage].iloc[0]

    # Show segment data
    print(f"\n📍 Segment Information:")
    print(f"  chainage_id: {target_row['chainage_id']}")
    print(f"  line_code: {target_row.get('line_code')}")
    print(f"  chainage_start_km: {target_row.get('chainage_start_km')}")
    print(f"  chainage_end_km: {target_row.get('chainage_end_km')}")
    print(f"  curve_type: {target_row.get('curve_type', 'N/A')}")

    # Show Asset data
    print(f"\n🏗️  Asset Data:")
    if target_chainage in export_data['agg_asset']['chainage_id'].values:
        asset_row = export_data['agg_asset'][export_data['agg_asset']['chainage_id'] == target_chainage].iloc[0]
        print(f"  fixed_asset: {asset_row['fixed_asset']} (type: {type(asset_row['fixed_asset']).__name__})")
        print(f"  wz_asset: {asset_row['wz_asset']}")
        print(f"  level_crossing: {asset_row.get('level_crossing', 0)}")
        print(f"  turnout: {asset_row.get('turnout', 0)}")
        print(f"  bridge: {asset_row.get('bridge', 0)}")
    else:
        print(f"  ❌ No asset data found")

    # Show TQI data
    print(f"\n📊 TQI Data:")
    if target_chainage in export_data['agg_tqi']['chainage_id'].values:
        tqi_row = export_data['agg_tqi'][export_data['agg_tqi']['chainage_id'] == target_chainage].iloc[0]
        print(f"  tqi: {tqi_row['tqi']}")
        print(f"  status: {tqi_row['status']}")
    else:
        print(f"  ❌ No TQI data found")

    # Show GBFI data
    print(f"\n🌍 GBFI Data:")
    if target_chainage in export_data['agg_gbfi']['chainage_id'].values:
        gbfi_row = export_data['agg_gbfi'][export_data['agg_gbfi']['chainage_id'] == target_chainage].iloc[0]
        print(f"  max_of_avg: {gbfi_row['max_of_avg']}")
        print(f"  avg_of_avg: {gbfi_row['avg_of_avg']}")
        print(f"  ballast_centre: {gbfi_row.get('ballast_centre', 'N/A')}")
        print(f"  fouled_zone: {gbfi_row.get('fouled_zone', 'N/A')}")
        print(f"  ballast_lt_250mm: {gbfi_row.get('ballast_lt_250mm', 'N/A')}")
    else:
        print(f"  ❌ No GBFI data found")

    # Show TSR data
    print(f"\n🚧 TSR Data:")
    if target_chainage in export_data['agg_tsr']['chainage_id'].values:
        tsr_row = export_data['agg_tsr'][export_data['agg_tsr']['chainage_id'] == target_chainage].iloc[0]
        print(f"  open_tsr: {tsr_row.get('open_tsr', 'N/A')}")
        print(f"  complete_tsr: {tsr_row.get('complete_tsr', 'N/A')}")
        print(f"  open_tsr_days: {tsr_row.get('open_tsr_days', 'N/A')}")
    else:
        print(f"  ❌ No TSR data found")

    # Show DTR data
    print(f"\n⚡ DTR Data:")
    if target_chainage in export_data['agg_dtr']['chainage_id'].values:
        dtr_row = export_data['agg_dtr'][export_data['agg_dtr']['chainage_id'] == target_chainage].iloc[0]
        print(f"  worst_dtf: {dtr_row.get('worst_dtf', 'N/A')}")
        print(f"  dtf_s1: {dtr_row.get('dtf_s1', 'N/A')}")
        print(f"  dtf_s2: {dtr_row.get('dtf_s2', 'N/A')}")
        print(f"  dtf_s3: {dtr_row.get('dtf_s3', 'N/A')}")
    else:
        print(f"  ❌ No DTR data found")

    # Show thresholds being used
    print(f"\n⚙️  Thresholds:")
    for key, value in PLAINLINE_TREATMENT_THRESHOLDS.items():
        print(f"  {key}: {value}")

    # Run debug on this specific row
    print(f"\n{'='*80}")
    print("🔍 RULE EVALUATION")
    print(f"{'='*80}")
    engine_rule.debug_row(target_row, plainline_treatment_rules)

    # Apply rule to this segment
    test_segment = segments_df[segments_df['chainage_id'] == target_chainage].copy()
    test_segment['treatment'] = engine_rule.apply_rules(test_segment, plainline_treatment_rules)

    print(f"\n{'='*80}")
    print("✅ FINAL RESULT")
    print(f"{'='*80}")
    print(f"Segment: {target_chainage}")
    print(f"Treatment: {test_segment.iloc[0]['treatment']}")

    print("\n" + "="*80)
    print("Debug complete!")
    print("="*80)


if __name__ == "__main__":
    test_rules()
