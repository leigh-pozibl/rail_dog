"""Unit tests for geometry processing functions."""
import pytest


class TestChainageId:
    """Tests for chainage ID formatting logic."""

    def test_chainage_id_format_mainline(self):
        line_code = "ML"
        start_km = 26.9
        end_km = 27.0
        chainage_id = f"CHAIN-{line_code}-S-{int(10*abs(start_km)):0>5}-E-{int(10*abs(end_km)):0>5}-MAINLINE"
        assert chainage_id == "CHAIN-ML-S-00269-E-00270-MAINLINE"

    def test_chainage_id_format_thomas(self):
        line_code = "TL"
        start_km = 3.7
        end_km = 3.8
        chainage_id = f"CHAIN-{line_code}-S-{int(10*abs(start_km)):0>5}-E-{int(10*abs(end_km)):0>5}-MAINLINE"
        assert chainage_id == "CHAIN-TL-S-00037-E-00038-MAINLINE"

    def test_chainage_id_format_solomon(self):
        line_code = "SL"
        start_km = 174.0
        end_km = 174.1
        chainage_id = f"CHAIN-{line_code}-S-{int(10*abs(start_km)):0>5}-E-{int(10*abs(end_km)):0>5}-MAINLINE"
        assert chainage_id == "CHAIN-SL-S-01740-E-01741-MAINLINE"


class TestExpandCurveSpirals:
    """Tests for expand_curve_spirals standalone function."""

    def test_import(self):
        from rail_dog.processing.curves import expand_curve_spirals
        assert callable(expand_curve_spirals)

    def test_no_spirals(self):
        """Curves without SC/CS should pass through unchanged."""
        import pandas as pd
        from rail_dog.processing.curves import expand_curve_spirals

        data = pd.DataFrame([{
            "Type": "Tangent",
            "Classification": "Tangent",
            "Start Chainage": 0.0,
            "End Chainage": 1.0,
            "Curve Length": 1000.0,
            "SC": None,
            "CS": None,
        }])

        result, spirals = expand_curve_spirals(data, "ML")
        assert len(result) == 1
        assert len(spirals) == 0
