"""
Report: thin orchestrator that composes DataMixin, ExcelMixin, and HtmlMixin.
"""
from typing import Optional

from sqlalchemy import Engine

from rail_dog.report.data import DataMixin
from rail_dog.report.excel import ExcelMixin
from rail_dog.report.html import HtmlMixin


class Report(DataMixin, ExcelMixin, HtmlMixin):
    def __init__(
        self,
        engine: Engine,
        output_path: str,
        analysis_dates: Optional[dict] = None,
        global_date=None,
        mapbox_token: str = "",
    ):
        """
        Args:
            engine: database engine
            output_path: Path to save the Excel file
            analysis_dates: Per-type dates keyed by data type, e.g.
                            {"GBFI": datetime(...), "TQI": datetime(...), "TG": datetime(...), ...}
                            Typically sourced from base_data.analysis_dates in the config.
            global_date: Fallback date used for any type not present in analysis_dates.
                         If neither is provided no date filter is applied.
        """
        self.engine = engine
        self.output_path = output_path
        self.analysis_dates = analysis_dates or {}
        self.global_date = global_date
        self.mapbox_token = mapbox_token

        self.export_data = {}

        self.cols_list = list()  # Will be set after merging data, used for column letter lookups in formulas

        self.generate_report()


