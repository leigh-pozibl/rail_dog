"""Unit tests for RuleEngine."""
import pytest
import pandas as pd

from rail_dog.report.rule_engine import RuleEngine


class TestSimpleRules:

    def test_equality_match(self):
        engine = RuleEngine()
        rules = [
            [("status", "==", "poor"), "Treatment Required"],
            "No Treatment",
        ]
        df = pd.DataFrame({"chainage_id": ["A", "B"], "status": ["poor", "good"]})
        result = engine.apply_rules(df, rules)
        assert result.iloc[0] == "Treatment Required"
        assert result.iloc[1] == "No Treatment"

    def test_default_returned_when_no_match(self):
        engine = RuleEngine()
        rules = [
            [("status", "==", "critical"), "Critical Treatment"],
            "No Treatment",
        ]
        df = pd.DataFrame({"chainage_id": ["X"], "status": ["good"]})
        result = engine.apply_rules(df, rules)
        assert result.iloc[0] == "No Treatment"

    def test_first_matching_rule_wins(self):
        engine = RuleEngine()
        rules = [
            [("score", ">=", 5), "High"],
            [("score", ">=", 3), "Medium"],
            "Low",
        ]
        df = pd.DataFrame({"chainage_id": ["A", "B", "C"], "score": [7, 4, 1]})
        result = engine.apply_rules(df, rules)
        assert result.iloc[0] == "High"
        assert result.iloc[1] == "Medium"
        assert result.iloc[2] == "Low"


class TestAndOrConditions:

    def test_and_condition_both_true(self):
        engine = RuleEngine()
        rules = [
            [["AND", ("a", "==", 1), ("b", "==", 2)], "Match"],
            "No Match",
        ]
        df = pd.DataFrame({"chainage_id": ["X"], "a": [1], "b": [2]})
        assert engine.apply_rules(df, rules).iloc[0] == "Match"

    def test_and_condition_one_false(self):
        engine = RuleEngine()
        rules = [
            [["AND", ("a", "==", 1), ("b", "==", 2)], "Match"],
            "No Match",
        ]
        df = pd.DataFrame({"chainage_id": ["X"], "a": [1], "b": [99]})
        assert engine.apply_rules(df, rules).iloc[0] == "No Match"

    def test_or_condition(self):
        engine = RuleEngine()
        rules = [
            [["OR", ("status", "==", "poor"), ("status", "==", "critical")], "Needs Work"],
            "OK",
        ]
        df = pd.DataFrame({
            "chainage_id": ["X", "Y", "Z"],
            "status": ["poor", "critical", "good"]
        })
        result = engine.apply_rules(df, rules)
        assert result.iloc[0] == "Needs Work"
        assert result.iloc[1] == "Needs Work"
        assert result.iloc[2] == "OK"

    def test_nested_and_or(self):
        engine = RuleEngine()
        rules = [
            [["AND",
              ["OR", ("status", "==", "poor"), ("status", "==", "critical")],
              ("gbfi", ">=", 40)
             ], "High Priority"],
            "Normal",
        ]
        df = pd.DataFrame({
            "chainage_id": ["A", "B", "C"],
            "status": ["poor", "poor", "good"],
            "gbfi": [50, 30, 50],
        })
        result = engine.apply_rules(df, rules)
        assert result.iloc[0] == "High Priority"   # poor + gbfi>=40
        assert result.iloc[1] == "Normal"           # poor but gbfi<40
        assert result.iloc[2] == "Normal"           # good status


class TestThresholdReferences:

    def test_threshold_lookup(self):
        engine = RuleEngine(thresholds={"min_gbfi": 40})
        rules = [
            [("gbfi", ">=", "threshold.min_gbfi"), "Fouled"],
            "Clean",
        ]
        df = pd.DataFrame({"chainage_id": ["A", "B"], "gbfi": [50, 20]})
        result = engine.apply_rules(df, rules)
        assert result.iloc[0] == "Fouled"
        assert result.iloc[1] == "Clean"

    def test_missing_threshold_raises(self):
        engine = RuleEngine(thresholds={})
        rules = [[("x", ">=", "threshold.missing"), "Match"], "No"]
        df = pd.DataFrame({"chainage_id": ["A"], "x": [1]})
        with pytest.raises(ValueError, match="Threshold 'missing' not found"):
            engine.apply_rules(df, rules)


class TestBooleanComparisons:

    def test_boolean_true(self):
        engine = RuleEngine()
        rules = [
            [("fixed_asset", "==", True), "No Treatment"],
            "Treatment",
        ]
        df = pd.DataFrame({"chainage_id": ["A", "B"], "fixed_asset": [True, False]})
        result = engine.apply_rules(df, rules)
        assert result.iloc[0] == "No Treatment"
        assert result.iloc[1] == "Treatment"
