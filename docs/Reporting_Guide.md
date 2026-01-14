# Rail Dog Reporting & Visualization Guide

This guide explains how to generate reports and visualizations from the rail_dog database.

## Overview

Once you've processed your railway data and loaded it into the database, you can generate various types of reports and exports for analysis and visualization. The reporting system provides three main outputs:

1. **Excel Reports** - Spreadsheet with tabs per line, all metrics joined
2. **Power BI Exports** - Geospatial data optimized for Power BI dashboards
3. **Condition Summary** - Executive summary statistics by line

## Quick Start

### Generate All Reports

```python
from snappy_utils.params import DBConnection, Metadata
from rail_dog.configs.params import load_config
from rail_dog.processor import Processor

# Load configuration
config = load_config("data/config_fmg.yaml")

# Initialize database connection
db = DBConnection(db_env='local')
metadata = Metadata(project_name="FMG Rail Analysis")

# Create processor
processor = Processor(config, db, metadata, output_dir="output/reports")

# Generate reports
processor.generate_excel_report()
processor.generate_powerbi_export(format='geojson')
summary = processor.generate_condition_summary()
print(summary)
```

### Or use the standalone script

```bash
python examples/generate_reports.py
```

## Report Types

### 1. Excel Report

**File:** `rail_condition_report.xlsx`

**Structure:**
- One tab per line_code (ML, TL, SL, EL)
- Each row = one segment (chainage_id)
- Columns include all metrics from all `agg_*` tables

**Contains:**
- Segment metadata: chainage_id, line, station, start/end km, coordinates
- Asset data: level_crossing, irj, turnout, bridge counts
- GBFI data: fouling metrics, ballast condition flags
- TSR data: open TSRs, historical TSR counts by year
- TQI data: track quality indicator values

**Use cases:**
- Detailed segment-level analysis in Excel
- Pivot tables and charts
- Data validation and quality checking
- Ad-hoc filtering and sorting
- Export subsets for field crews

**Example - Find segments requiring attention:**
```
Filter: TQI < 75 AND fouled_zone = TRUE
Sort by: chainage_start_km
Result: Prioritized list of poor track with ballast issues
```

### 2. Power BI Export

**Directory:** `output/reports/powerbi/`

**Files generated:**

| File | Type | Contents |
|------|------|----------|
| `rail_segments.geojson` | Spatial | Segment geometries with metadata |
| `agg_assets.csv` | Tabular | Asset counts per segment |
| `agg_gbfi.csv` | Tabular | GBFI metrics per segment |
| `agg_tsr.csv` | Tabular | TSR data per segment |
| `agg_tqi.csv` | Tabular | TQI data per segment |
| `unified_rail_data.geojson` | Spatial | All metrics joined to geometries |
| `README_PowerBI.md` | Docs | Detailed Power BI import instructions |

**Import approaches:**

**Option A - Relational Model (Recommended for production):**
1. Import `rail_segments.geojson` as main table
2. Import each CSV as separate tables
3. Create relationships on `chainage_id`
4. Benefits: Smaller files, easier updates, better performance

**Option B - Unified Dataset (Quick analysis):**
1. Import `unified_rail_data.geojson` only
2. Benefits: Single table, simpler setup, faster to get started

**Format options:**
```python
# For best Power BI performance with large datasets:
processor.generate_powerbi_export(format='geoparquet')

# For compatibility and simplicity:
processor.generate_powerbi_export(format='geojson')

# For legacy GIS tools:
processor.generate_powerbi_export(format='shapefile')
```

### 3. Condition Summary

**File:** `condition_summary.csv`

**Columns:**
- `line_code`: Line identifier (ML, TL, SL, EL)
- `total_segments`: Count of segments
- `total_length_m`: Total track length in meters
- `fouled_segments`: Count with ballast fouling
- `segments_with_open_tsr`: Count with active speed restrictions
- `avg_tqi`: Average track quality indicator
- `min_tqi`: Minimum TQI (worst segment)
- `max_tqi`: Maximum TQI (best segment)
- `pct_fouled`: Percentage of segments fouled
- `pct_open_tsr`: Percentage with open TSRs

**Use cases:**
- Executive briefings
- Line-level comparison
- Trend monitoring over time
- KPI dashboards

**Example output:**
```
line_code  total_segments  total_length_m  fouled_segments  avg_tqi  pct_fouled
ML         1543           154300          89               78.5     5.8
TL         234            23400           12               82.1     5.1
SL         456            45600           23               75.3     5.0
EL         789            78900           45               71.2     5.7
```

## Database Schema

### Core Tables

**rail_segments** - Segment geometries and metadata
- Primary key: `chainage_id`
- Geometry: LineString (100m segments)
- Attributes: line_code, station, chainage bounds, coordinates, curve type

**agg_assets** - Asset counts aggregated to segments
- Foreign key: `chainage_id`
- Metrics: level_crossing, irj, turnout, bridge counts
- Flags: fixed_asset, wz_asset (work zone)

**agg_gbfi** - Ground Ballast Fouling Index
- Foreign key: `chainage_id`
- Metrics: avg_of_avg, fouled counts, ballast depth
- Flags: fouled_zone, ballast_lt_250mm

**agg_tsr** - Temporary Speed Restrictions
- Foreign key: `chainage_id`
- Metrics: open_tsr_days, yearly counts (2022-2026)
- Flags: open_tsr, complete_tsr

**agg_tqi** - Track Quality Indicator
- Foreign key: `chainage_id`
- Metrics: tqi value (0-100)
- Metadata: collection_date, line_class

## Visualization Examples

### Excel - Conditional Formatting

Create visual indicators for track condition:

```
TQI Column:
  < 50:  Red fill
  50-75: Yellow fill
  > 75:  Green fill

Fouled Zone Column:
  TRUE:  Red text
  FALSE: Green text
```

### Power BI - Track Condition Heatmap

1. Add Azure Maps visual
2. Location: `geometry` field
3. Color: `tqi` value
4. Gradient: Red (0) → Yellow (50) → Green (100)
5. Tooltip: chainage_id, station, TQI, fouled_zone

### Power BI - Multi-Metric Dashboard

**Page Layout:**
- Map (60% width): Segments colored by risk score
- KPI Cards (top right): Total km, % fouled, avg TQI
- Charts (bottom right):
  - TQI distribution histogram
  - TSR trend by year
  - Asset count by type

**Risk Score DAX:**
```dax
Risk Score =
VAR TQI_Risk = IF([tqi] < 50, 3, IF([tqi] < 75, 2, 1))
VAR Fouling_Risk = IF([fouled_zone] = TRUE, 3, 1)
VAR TSR_Risk = IF([open_tsr] = TRUE, 2, 1)
RETURN (TQI_Risk + Fouling_Risk + TSR_Risk) / 3
```

## Advanced Usage

### Custom Line Codes

Generate reports for specific lines only:

```python
from rail_dog.utils.report_utils import generate_excel_report

generate_excel_report(
    engine=db.engine,
    output_path="output/mainline_only.xlsx",
    line_codes=['ML']  # Only Mainline
)
```

### Automated Reporting

Set up scheduled report generation:

```python
# cron job or Windows Task Scheduler
# Run daily at 6 AM to generate fresh reports

import schedule
import time

def generate_daily_reports():
    processor = Processor(...)
    processor.generate_excel_report(
        filename=f"report_{datetime.now():%Y%m%d}.xlsx"
    )
    processor.generate_condition_summary()

schedule.every().day.at("06:00").do(generate_daily_reports)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Direct SQL Access

Query the database directly for custom analysis:

```python
import pandas as pd

query = """
SELECT
    rs.chainage_id,
    rs.line_code,
    rs.station,
    at.tqi,
    ag.fouled_zone,
    asr.open_tsr
FROM rail_segments rs
LEFT JOIN agg_tqi at ON rs.chainage_id = at.chainage_id
LEFT JOIN agg_gbfi ag ON rs.chainage_id = ag.chainage_id
LEFT JOIN agg_tsr asr ON rs.chainage_id = asr.chainage_id
WHERE at.tqi < 60
  AND ag.fouled_zone = TRUE
ORDER BY at.tqi ASC
LIMIT 50;
"""

critical_segments = pd.read_sql(query, db.engine)
print(critical_segments)
```

### Export to GIS

Convert to shapefile for QGIS/ArcGIS:

```python
import geopandas as gpd

# Read segments with geometry
segments_gdf = gpd.read_postgis(
    "SELECT * FROM rail_segments",
    db.engine,
    geom_col='geometry'
)

# Join all metrics
from rail_dog.utils.db_utils import get_table_data

agg_tqi = get_table_data(db.engine, table_name="agg_tqi")
merged = segments_gdf.merge(agg_tqi, on='chainage_id')

# Export to shapefile
merged.to_file("output/rail_segments_with_tqi.shp")
```

## Troubleshooting

### Excel file is too large
- Filter to specific line_codes
- Remove unnecessary columns
- Split into multiple files

### Power BI won't import GeoJSON
- Ensure CRS is EPSG:4326
- Try geoparquet format instead
- Check geometry validity

### Missing data in reports
- Verify all processing steps completed
- Check database for NULL values
- Ensure chainage_id matches across tables

### Performance issues with large datasets
- Use geoparquet instead of geojson
- Create database indexes on chainage_id
- Filter data by date ranges or line_codes
- Use DirectQuery instead of Import in Power BI

## See Also

- [Power BI Visualization Guide](PowerBI_Visualization_Guide.md) - Detailed Power BI setup and DAX formulas
- [CLAUDE.md](../CLAUDE.md) - Overall project architecture and processing
- [Schema Documentation](../rail_dog/schema.py) - Database table definitions

## Support

For issues or questions:
1. Check logs in `output/logs/`
2. Verify database connectivity
3. Ensure all processing steps completed successfully
4. Review example script: `examples/generate_reports.py`
