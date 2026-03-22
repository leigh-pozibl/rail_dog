"""Unit tests for treatment recommendation functions."""
import pytest
import pandas as pd

from rail_dog.report.treatment import (
    find_plainline_treatment,
    find_lx_treatment,
    find_irj_treatment,
    find_turnout_treatment,
    find_bridge_treatment,
    PLAINLINE_TREATMENT_HIERARCHY,
)


class TestTreatmentHierarchy:

    def test_most_aggressive_is_first(self):
        assert PLAINLINE_TREATMENT_HIERARCHY[0] == "Local Formation Renewal (Bog Hole)"

    def test_least_aggressive_is_last(self):
        assert PLAINLINE_TREATMENT_HIERARCHY[-1] == "No Treatment"

    def test_all_treatments_present(self):
        expected = {
            "Local Formation Renewal (Bog Hole)",
            "Ballast Clean",
            "Lift & Tamp",
            "Tamp Only",
            "No Treatment",
        }
        assert set(PLAINLINE_TREATMENT_HIERARCHY) == expected


def _make_segments(chainage_ids):
    """Minimal segments DataFrame for rule evaluation."""
    return pd.DataFrame({"chainage_id": chainage_ids})


def _make_export_data(chainage_ids, **table_overrides):
    """
    Build a minimal export_data dict for testing.
    Each table_override is a dict of {col: [values]} for the given chainage_ids.
    """
    export_data = {}
    # Provide sensible defaults for all tables the treatment rules use
    defaults = {
        "agg_asset": {
            "fixed_asset": [False] * len(chainage_ids),
            "level_crossing": [0] * len(chainage_ids),
            "irj": [0] * len(chainage_ids),
            "turnout": [0] * len(chainage_ids),
            "bridge": [0] * len(chainage_ids),
            "wz_level_crossing": [0] * len(chainage_ids),
            "wz_irj": [0] * len(chainage_ids),
            "wz_turnout": [0] * len(chainage_ids),
            "wz_bridge": [0] * len(chainage_ids),
        },
        "agg_gbfi": {
            "max_of_avg": [0.0] * len(chainage_ids),
            "avg_of_avg": [0.0] * len(chainage_ids),
        },
        "agg_tqi": {
            "status": ["good"] * len(chainage_ids),
        },
        "agg_dtr": {
            "worst_dtf": [""] * len(chainage_ids),
        },
        "agg_tsr": {
            "complete_tsr": [False] * len(chainage_ids),
        },
        "agg_ballast": {
            "ballast_centre": [0.3] * len(chainage_ids),
        },
    }
    for table, cols in defaults.items():
        overrides = table_overrides.get(table, {})
        merged_cols = {**cols, **overrides}
        export_data[table] = pd.DataFrame({"chainage_id": chainage_ids, **merged_cols})
    return export_data


class TestFindPlainlineTreatment:

    def test_fixed_asset_returns_no_treatment(self):
        ids = ["A"]
        export_data = _make_export_data(ids, agg_asset={"fixed_asset": [True]})
        segments = _make_segments(ids)
        result = find_plainline_treatment(segments, export_data)
        assert result.iloc[0] == "No Treatment"

    def test_poor_tqi_high_gbfi_returns_formation_renewal(self):
        ids = ["A"]
        export_data = _make_export_data(
            ids,
            agg_tqi={"status": ["poor"]},
            agg_gbfi={"max_of_avg": [45.0], "avg_of_avg": [45.0]},
        )
        segments = _make_segments(ids)
        result = find_plainline_treatment(segments, export_data)
        assert result.iloc[0] == "Local Formation Renewal (Bog Hole)"

    def test_good_conditions_returns_no_treatment(self):
        ids = ["A"]
        export_data = _make_export_data(ids)  # all defaults (good TQI, low GBFI)
        segments = _make_segments(ids)
        result = find_plainline_treatment(segments, export_data)
        assert result.iloc[0] == "No Treatment"


class TestFindLxTreatment:

    def test_non_fixed_asset_returns_no_treatment(self):
        ids = ["A"]
        export_data = _make_export_data(ids, agg_asset={"fixed_asset": [False]})
        segments = _make_segments(ids)
        result = find_lx_treatment(segments, export_data)
        assert result.iloc[0] == "No Treatment"
