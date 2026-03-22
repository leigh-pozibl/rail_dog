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

Trend storage
-------------
Trend results are stored directly in the source table (e.g. agg_tqi) for
each (chainage_id, collection_date) row. For each sample date, the trend is
computed using all data up to and including that date, giving a full history
of how the trend has evolved over time.

Step change detection
---------------------
Before computing a trend, each series is scanned for step changes using the
ruptures PELT algorithm (L2 cost, mean-shift model). Both upward and downward
shifts are detected.

For metrics where lower values are better (e.g. TQI):
  - Upward step  → "failure"  (track condition suddenly worsened)
  - Downward step → "repair"  (maintenance event, e.g. tamping)

For metrics where higher values are better the direction is reversed.

The trend window is then chosen based on the most recent step change:
  - Last step = repair   → trend from that repair date onwards
  - Last step = failure  → trend from the previous repair (or first datapoint)
  - No step changes      → trend uses all data

Since track condition can never improve on its own, any computed trend that
would otherwise be labelled "improving" is clamped to "stable" for metrics
where allow_improving=False (e.g. TQI).
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


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
    slope_threshold: float = 0.5    # min |slope| (units/year) to register a trend
    min_r_squared: float = 0.3      # min r² below which trend is reported as "stable"
    metric_sense: str = "higher_is_better"  # "higher_is_better" or "lower_is_better"
    allow_improving: bool = True    # set False when physical improvement is impossible
    step_penalty: float = 10.0      # ruptures PELT penalty; higher = fewer breakpoints
    min_segment_size: int = 3       # minimum samples per segment for ruptures


# Register all metrics here
TQI_TREND = TrendConfig(
    metric="tqi",
    source_table="agg_tqi",
    value_col="tqi",
    date_col="collection_date",
    metric_sense="lower_is_better",  # lower TQI value = better track quality
    allow_improving=False,           # track cannot heal itself between repairs
    # With L2 PELT, a spike of height h creates false breakpoints when h² > 2×penalty.
    # penalty=100 suppresses single outliers up to ~14 units while still catching
    # real tamping events (typically 20+ unit drops).
    step_penalty=100.0,
    # min_segment_size=3 (default) forces breakpoints up to 2 steps late because
    # each segment must contain at least min_size samples. Setting to 1 eliminates
    # the forced timestamp offset; the penalty alone guards against noise.
    min_segment_size=1,
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
# Step change detection
# ==============================================================================

def _detect_step_changes(
    values: list[float],
    penalty: float,
    min_size: int,
    metric_sense: str = "lower_is_better",
) -> list[dict]:
    """
    Detect step changes (mean shifts) in a series using ruptures PELT + L2 cost.

    Both upward and downward shifts are found. Direction labelling accounts for
    metric_sense so that "failure" always means the metric got worse.

    Returns a list of dicts (one per detected breakpoint, excluding endpoints):
        at_index  - index in the original array where the new segment starts
        delta     - new_segment_mean - prev_segment_mean
        type      - "failure" or "repair"
    """
    import ruptures as rpt

    if len(values) < min_size * 2:
        return []

    arr = np.array(values, dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(arr)
    breakpoints = algo.predict(pen=penalty)  # last entry is always len(values)

    if len(breakpoints) <= 1:
        return []

    breaks = [0] + breakpoints
    step_changes = []
    for i in range(1, len(breaks) - 1):
        prev_mean = float(np.mean(values[breaks[i - 1]:breaks[i]]))
        curr_mean = float(np.mean(values[breaks[i]:breaks[i + 1]]))
        delta = curr_mean - prev_mean

        if metric_sense == "lower_is_better":
            change_type = "failure" if delta > 0 else "repair"
        else:  # higher_is_better
            change_type = "repair" if delta > 0 else "failure"

        step_changes.append({
            "at_index": breaks[i],
            "delta": delta,
            "type": change_type,
        })

    return step_changes


def _get_trend_window(
    dates: list[datetime],
    values: list[float],
    step_changes: list[dict],
) -> tuple[list[datetime], list[float], str | None, datetime | None, datetime | None]:
    """
    Choose the trend window based on the most recent step change.

    Rules:
      - No step changes     → all data; step_change_type/date = None
      - Last step = repair  → data from that repair date onwards
      - Last step = failure → data from the previous repair (or first datapoint)

    Returns:
        (subset_dates, subset_values, step_change_type, step_change_date, trend_segment_start)
    """
    if not step_changes:
        return dates, values, None, None, dates[0] if dates else None

    last_change = step_changes[-1]
    step_change_date = dates[last_change["at_index"]]

    if last_change["type"] == "repair":
        idx = last_change["at_index"]
        return (
            dates[idx:], values[idx:],
            "repair", step_change_date, step_change_date,
        )

    # failure — find the most recent prior repair
    last_repair = next(
        (c for c in reversed(step_changes[:-1]) if c["type"] == "repair"),
        None,
    )
    if last_repair:
        idx = last_repair["at_index"]
        trend_start = dates[idx]
    else:
        idx = 0
        trend_start = dates[0]

    return (
        dates[idx:], values[idx:],
        "failure", step_change_date, trend_start,
    )


# ==============================================================================
# Regression helpers
# ==============================================================================

def _compute_slope(dates: list[datetime], values: list[float]) -> tuple[float | None, float | None]:
    """
    Fit a linear regression through metric values vs time.

    Returns:
        slope_per_year: metric units/year. Positive = metric value rising.
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


def _classify_trend(
    slope: float | None,
    r_squared: float | None,
    threshold: float,
    min_r_squared: float,
    metric_sense: str = "higher_is_better",
    allow_improving: bool = True,
) -> str:
    """
    Return 'improving', 'degrading', or 'stable'.

    metric_sense controls direction:
      'higher_is_better' — positive slope = improving
      'lower_is_better'  — negative slope = improving

    If allow_improving=False, any result that would be 'improving' is returned
    as 'stable' instead (used for metrics where physical improvement is impossible
    outside of a maintenance event, e.g. TQI).
    """
    if slope is None or r_squared is None or r_squared < min_r_squared:
        return "stable"

    improving = slope > threshold if metric_sense == "higher_is_better" else slope < -threshold
    degrading = slope < -threshold if metric_sense == "higher_is_better" else slope > threshold

    if improving:
        return "stable" if not allow_improving else "improving"
    if degrading:
        return "degrading"
    return "stable"


# ==============================================================================
# Core computation
# ==============================================================================

def compute_trends(engine, config: TrendConfig, collection_date: Optional[datetime] = None) -> None:
    """
    Compute per-segment trends for every collection_date in the source table,
    using only data up to that date, and update the trend fields in-place.

    For each (chainage_id, collection_date) row, trend columns are derived from
    the time-series of all previous samples for that segment, including the
    step-change logic described in the module docstring.

    Args:
        engine: SQLAlchemy engine
        config: TrendConfig describing the source table and metric
        collection_date: if provided, only update rows for this specific date
                         (all historical data is still used to compute the trend).
                         If None, all rows are updated.
    """
    if collection_date:
        logging.info(f"Computing trends for metric '{config.metric}' for date {collection_date.date() if hasattr(collection_date, 'date') else collection_date}...")
    else:
        logging.info(f"Computing trends for metric '{config.metric}' for all dates...")

    df = pd.read_sql(
        f"SELECT {config.chainage_col}, {config.value_col}, {config.date_col}"
        f" FROM {config.source_table}",
        engine,
    )

    if df.empty:
        logging.warning(f"No records in '{config.source_table}' — skipping")
        return

    df[config.date_col] = pd.to_datetime(df[config.date_col])

    # Deduplicate: keep mean value per (chainage_id, date) to handle repeated ingestion runs
    df = (
        df.groupby([config.chainage_col, config.date_col], as_index=False)[config.value_col]
        .mean()
    )
    df = df.sort_values(by=[config.chainage_col, config.date_col])

    updates = []
    for chainage_id, group in df.groupby(config.chainage_col):
        dates = group[config.date_col].dt.to_pydatetime()
        dates = np.array(dates).tolist()
        values = group[config.value_col].tolist()

        target_date = pd.Timestamp(collection_date).normalize() if collection_date is not None else None

        for i in range(len(dates)):
            row_date = pd.Timestamp(dates[i]).normalize()

            # When filtering to a specific date, skip rows after the target
            # (rows before it are still iterated so d_subset builds up correctly)
            if target_date is not None and row_date > target_date:
                break

            d_subset = dates[: i + 1]
            v_subset = values[: i + 1]

            # Only emit an update for this row if it matches the target date (or no filter)
            if target_date is not None and row_date != target_date:
                continue

            step_changes = _detect_step_changes(
                v_subset,
                penalty=config.step_penalty,
                min_size=config.min_segment_size,
                metric_sense=config.metric_sense,
            )
            trend_dates, trend_values, step_change_type, step_change_date, trend_segment_start = (
                _get_trend_window(d_subset, v_subset, step_changes)
            )
            slope, r_squared = _compute_slope(trend_dates, trend_values)
            trend_label = _classify_trend(
                slope, r_squared,
                config.slope_threshold, config.min_r_squared,
                config.metric_sense, config.allow_improving,
            )

            updates.append({
                "chainage_id": str(chainage_id),
                "collection_date": dates[i],
                "trend": trend_label,
                "trend_slope": slope,
                "trend_r_squared": r_squared,
                "step_change_type": step_change_type,
                "step_change_date": step_change_date,
                "trend_segment_start": trend_segment_start,
            })

    logging.info(f"Updating {len(updates)} rows in '{config.source_table}' with trend data...")

    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(
            text(f"""
                UPDATE {config.source_table} SET
                    trend               = :trend,
                    trend_slope         = :trend_slope,
                    trend_r_squared     = :trend_r_squared,
                    step_change_type    = :step_change_type,
                    step_change_date    = :step_change_date,
                    trend_segment_start = :trend_segment_start
                WHERE {config.chainage_col} = :chainage_id
                  AND {config.date_col}     = :collection_date
            """),
            updates,
        )

    logging.info(f"Trend fields updated in '{config.source_table}' for metric '{config.metric}'")


def compute_all_trends(engine, collection_date: Optional[datetime] = None) -> None:
    """Compute and persist trends for every metric registered in ALL_TRENDS."""
    for config in ALL_TRENDS:
        compute_trends(engine, config, collection_date=collection_date)


def compute_trend(engine, metric: str, collection_date: Optional[datetime] = None) -> None:
    """Compute and persist trend for a specific metric.

    Args:
        engine: SQLAlchemy engine
        metric: metric name (must match a registered TrendConfig)
        collection_date: if provided, only update rows for this date;
                         if None, update all dates
    """
    config = next((c for c in ALL_TRENDS if c.metric == metric), None)
    if config is None:
        logging.error(f"No TrendConfig found for metric '{metric}'")
        return
    compute_trends(engine, config, collection_date=collection_date)


