"""
HtmlMixin: HTML report generation for Report.
"""
import logging

import pandas as pd


class HtmlMixin:
    def write_tqi_html_report(self, excel_path: str):
        """Generate a self-contained HTML TQI trend report alongside the Excel file."""
        import json

        tqi_all = self.export_data.get("agg_tqi_all")
        if tqi_all is None or tqi_all.empty:
            logging.warning("No TQI history data — skipping HTML report")
            return

        def _line(cid: str) -> str:
            parts = str(cid).split("-")
            return parts[1] if len(parts) >= 2 else "?"

        df = tqi_all.copy()
        df["collection_date"] = pd.to_datetime(df["collection_date"]).dt.strftime("%Y-%m-%d")
        for _dc in ["step_change_date", "trend_segment_start"]:
            if _dc in df.columns:
                df[_dc] = df[_dc].apply(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None)
        df["line_code"] = df["chainage_id"].apply(_line)

        line_ids = {
            str(lc): sorted(grp["chainage_id"].dropna().unique().tolist())
            for lc, grp in df.groupby("line_code")
        }
        base_cols = ["chainage_id", "line_code", "collection_date",
                     "tqi", "status", "trend", "trend_slope", "trend_r_squared",
                     "step_change_type", "step_change_date", "trend_segment_start"]
        coord_cols = [c for c in ["mid_coord_lat", "mid_coord_lng"] if c in df.columns]
        records = (
            df[base_cols + coord_cols]
            .where(df.notna(), None)
            .to_dict("records")
        )

        data_json    = json.dumps(records,  ensure_ascii=False)
        line_id_json = json.dumps(line_ids, ensure_ascii=False)

        # Re-use the same HTML template from generate_tqi_html.py
        from rail_dog.utils.html_utils import tqi_trend_html
        html = tqi_trend_html(data_json, line_id_json, mapbox_token=self.mapbox_token)

        out_path = str(excel_path).replace(".xlsx", "_tqi_trend.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"TQI HTML report saved to: {out_path}")


