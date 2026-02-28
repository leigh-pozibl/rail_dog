"""
Trend computation utilities for rail condition metrics.

Each metric is described by a TrendConfig that points at a source DB table.
compute_trends() handles regression and DB write for any config, so adding a
new metric is a one-liner:

    BALLAST_TREND = TrendConfig(
        metric="ballast_centre",
        source_table="agg_ballast",
        value_col="ballast_centre",
        date_col="collection_date",
    )

Then register it in ALL_TRENDS and call compute_all_trends(engine).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlmodel import Session

from rail_dog.configs.schema import SegmentTrend, create_table
from rail_dog.utils.db_utils import setup_table


# ==============================================================================
# Trend configuration
# ==============================================================================

@dataclass
class TrendConfig:
    """Describes how to compute a trend for one metric."""
    metric: str             # written to segment_trends.metric
    source_table: str       # DB table to read from
    value_col: str          # column containing the numeric metric
    date_col: str           # column containing the sample date
    chainage_col: str = "chainage_id"
    slope_threshold: float = 0.5   # min |slope| (units/year) to register a trend
    min_r_squared: float = 0.3     # min r² below which trend is reported as "stable"
    metric_sense: str = "higher_is_better"  # "higher_is_better" or "lower_is_better"


# Register all metrics here
TQI_TREND = TrendConfig(
    metric="tqi",
    source_table="agg_tqi",
    value_col="tqi",
    date_col="collection_date",
    metric_sense="lower_is_better",  # higher TQI = better track quality
)

# Future metrics — uncomment and adjust as data becomes available:
# GBFI_TREND = TrendConfig(
#     metric="gbfi_avg",
#     source_table="agg_gbfi",
#     value_col="avg_of_avg",
#     date_col="collection_date",
# )
# BALLAST_TREND = TrendConfig(
#     metric="ballast_centre",
#     source_table="agg_ballast",
#     value_col="ballast_centre",
#     date_col="collection_date",
# )

ALL_TRENDS: list[TrendConfig] = [
    TQI_TREND,
    # GBFI_TREND,
    # BALLAST_TREND,
]


# ==============================================================================
# Regression helpers
# ==============================================================================

def _compute_slope(dates: list[datetime], values: list[float]) -> tuple[float | None, float | None]:
    """
    Fit a linear regression through metric values vs time.

    Returns:
        slope_per_year: metric units/year. Positive = improving, negative = degrading.
        r_squared: goodness of fit [0-1]. None if fewer than 3 samples.
    """
    if len(values) < 3:
        return None, None

    x = np.array([(d - dates[0]).days for d in dates], dtype=float)
    y = np.array(values, dtype=float)

    coeffs = np.polyfit(x, y, 1)
    slope_per_year = float(coeffs[0] * 365)

    predicted = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope_per_year, r_squared


def _classify_trend(slope: float | None, r_squared: float | None,
                    threshold: float, min_r_squared: float,
                    metric_sense: str = "higher_is_better") -> str:
    """
    Return 'improving', 'degrading', or 'stable'.

    metric_sense controls direction:
      'higher_is_better' — positive slope = improving (e.g. TQI)
      'lower_is_better'  — negative slope = improving (e.g. wear, fouling)
    """
    if slope is None or r_squared is None or r_squared < min_r_squared:
        return "stable"
    improving = slope > threshold if metric_sense == "higher_is_better" else slope < -threshold
    degrading = slope < -threshold if metric_sense == "higher_is_better" else slope > threshold
    if improving:
        return "improving"
    if degrading:
        return "degrading"
    return "stable"


# ==============================================================================
# Core computation
# ==============================================================================

def compute_trends(engine, config: TrendConfig, as_of_date: datetime) -> None:
    """
    Compute per-segment trends for one metric using only data up to as_of_date,
    and persist results to segment_trends keyed by (chainage_id, metric, computed_for_date).

    Each ingestion run appends a new snapshot so the full trend history is preserved.

    Args:
        engine: SQLAlchemy engine
        config: TrendConfig describing the source table and metric
        as_of_date: only source records on or before this date are used
    """
    logging.info(f"Computing trends for metric '{config.metric}' as of {as_of_date.date()}...")

    df = pd.read_sql(
        f"""
        SELECT {config.chainage_col}, {config.value_col}, {config.date_col}
        FROM {config.source_table}
        WHERE {config.date_col} <= '{as_of_date}'
        """,
        engine,
    )

    if df.empty:
        logging.warning(f"No records found in '{config.source_table}' up to {as_of_date.date()} — skipping")
        return

    df[config.date_col] = pd.to_datetime(df[config.date_col])

    # Deduplicate: keep mean value per (chainage_id, date) to handle repeated ingestion runs
    df = (
        df.groupby([config.chainage_col, config.date_col], as_index=False)[config.value_col]
        .mean()
    )
    df = df.sort_values(by=[config.chainage_col, config.date_col])

    records = []
    for chainage_id, group in df.groupby(config.chainage_col):
        dates = group[config.date_col].dt.to_pydatetime()
        dates = np.array(dates).tolist()
        values = group[config.value_col].tolist()

        slope, r_squared = _compute_slope(dates, values)
        trend_label = _classify_trend(slope, r_squared, config.slope_threshold, config.min_r_squared,
                                      config.metric_sense)

        records.append(SegmentTrend(
            chainage_id=str(chainage_id),
            metric=config.metric,
            computed_for_date=as_of_date,
            slope=slope,
            r_squared=r_squared,
            trend_label=trend_label,
            sample_count=len(values),
            date_range_start=dates[0],
            date_range_end=dates[-1],
            computed_at=datetime.now(timezone.utc),
        ))

    logging.info(f"Computed {len(records)} trend records for metric '{config.metric}' as of {as_of_date.date()}")

    setup_table(engine, SegmentTrend, action="append")
    with Session(engine) as session:
        # Remove any existing snapshot for this metric + date before inserting
        session.exec(  # type: ignore[call-overload]
            __import__("sqlalchemy", fromlist=["text"]).text(
                "DELETE FROM segment_trends WHERE metric = :metric AND computed_for_date = :date"
            ),
            params={"metric": config.metric, "date": as_of_date},
        )
        for record in records:
            session.add(record)
        session.commit()

    logging.info(f"Trend records for '{config.metric}' as of {as_of_date.date()} written to segment_trends")


def compute_all_trends(engine, as_of_date: datetime) -> None:
    """Compute and persist trends for every metric registered in ALL_TRENDS."""
    for config in ALL_TRENDS:
        compute_trends(engine, config, as_of_date)


def compute_trend(engine, metric: str, as_of_date: datetime) -> None:
    """Compute and persist trend for a specific metric."""
    config = next((c for c in ALL_TRENDS if c.metric == metric), None)
    if config is None:
        logging.error(f"No TrendConfig found for metric '{metric}'")
        return
    compute_trends(engine, config, as_of_date)


def update_tqi_trends(engine, collection_date: datetime) -> None:
    """
    Compute TQI trend per chainage_id using all agg_tqi data up to and including
    collection_date, then UPDATE the trend column for that collection date's rows.
    """
    from sqlalchemy import text

    logging.info(f"Updating TQI trends in agg_tqi for collection_date={collection_date.date()}...")

    df = pd.read_sql(
        f"SELECT chainage_id, tqi, collection_date FROM agg_tqi WHERE collection_date <= '{collection_date}'",
        engine,
    )

    if df.empty:
        logging.warning("No agg_tqi records found — skipping trend update")
        return

    df["collection_date"] = pd.to_datetime(df["collection_date"])
    df = df.groupby(["chainage_id", "collection_date"], as_index=False)["tqi"].mean()
    df = df.sort_values(by=["chainage_id", "collection_date"])

    updates = []
    for chainage_id, group in df.groupby("chainage_id"):
        dates = group["collection_date"].dt.to_pydatetime()
        dates = np.array(dates).tolist()
        values = group["tqi"].tolist()
        slope, r_squared = _compute_slope(dates, values)
        trend_label = _classify_trend(slope, r_squared, threshold=0.5, min_r_squared=0.3,
                                      metric_sense=TQI_TREND.metric_sense)
        updates.append({"chainage_id": chainage_id, "trend": trend_label,
                         "slope": slope, "r_squared": r_squared})

    with Session(engine) as session:
        for row in updates:
            session.exec(  # type: ignore[call-overload]
                text("""
                    UPDATE agg_tqi
                    SET trend = :trend, trend_slope = :slope, trend_r_squared = :r_squared
                    WHERE chainage_id = :chainage_id AND collection_date = :date
                """),
                params={"trend": row["trend"], "slope": row["slope"], "r_squared": row["r_squared"],
                        "chainage_id": row["chainage_id"], "date": collection_date},
            )
        session.commit()

    logging.info(f"Updated trends for {len(updates)} segments in agg_tqi")