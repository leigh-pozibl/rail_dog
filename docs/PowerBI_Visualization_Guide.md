# Power BI Visualization Guide for Rail Dog

This guide provides strategies for visualizing railway track condition data in Power BI using geospatial features.

## Data Model

### Star Schema Approach (Recommended)

**Fact Table:** `rail_segments` (Geometry table)
- Primary Key: `chainage_id`
- Geometry: LineString (rail segment)
- Attributes: line_code, station, chainage_start_km, chainage_end_km, curve_type, etc.

**Dimension Tables:**
- `agg_assets`: Asset counts per segment
- `agg_gbfi`: Ballast fouling metrics
- `agg_tsr`: Temporary speed restrictions
- `agg_tqi`: Track quality indicators

**Relationships:** One-to-One on `chainage_id`

## Getting Data into Power BI

### Method 1: Import from Files (Simplest)
1. Run `generate_powerbi_export()` to create GeoJSON/Shapefile outputs
2. In Power BI Desktop:
   - Get Data → More → JSON (for GeoJSON)
   - Get Data → Text/CSV (for CSV tables)
3. Create relationships on `chainage_id` in Model view

### Method 2: Direct Database Connection (Best for Live Data)
1. Get Data → Database → PostgreSQL
2. Enter connection details
3. Select tables: `rail_segments`, `agg_*` tables
4. Power BI will auto-detect geometry columns in PostGIS

### Method 3: Unified Dataset (Quick Start)
1. Import `unified_rail_data.geojson` directly
2. Single table with all metrics joined
3. Trade-off: Larger file, but simpler model

## Map Visualizations

### Prerequisites
Enable the following visuals in Power BI:
- **Azure Maps** (built-in, best for simple maps)
- **ArcGIS Maps for Power BI** (advanced geospatial)
- **Mapbox Visual** (custom styling)

### Visualization 1: Track Quality Heatmap

**Visual Type:** Azure Maps / ArcGIS

**Data Setup:**
- Location: Use `geometry` field (WKT LineString)
- Color: `tqi` field
- Tooltip: `chainage_id`, `tqi`, `line_code`, `station`

**Color Scheme:**
```
TQI Value    Color       Meaning
< 50         Red         Poor condition
50-75        Orange      Fair condition
75-90        Yellow      Good condition
> 90         Green       Excellent condition
```

**DAX Measure for TQI Status:**
```dax
TQI Status =
SWITCH(
    TRUE(),
    rail_segments[tqi] < 50, "Poor",
    rail_segments[tqi] < 75, "Fair",
    rail_segments[tqi] < 90, "Good",
    "Excellent"
)
```

### Visualization 2: Ballast Fouling Map

**Visual Type:** ArcGIS Maps (for multi-layer support)

**Layer 1 - Fouled Zones:**
- Location: `geometry`
- Filter: `fouled_zone = TRUE`
- Color: Red
- Line Weight: 3

**Layer 2 - Normal Zones:**
- Location: `geometry`
- Filter: `fouled_zone = FALSE`
- Color: Green
- Line Weight: 1

**Tooltip Metrics:**
- `avg_of_avg`: Average GBFI across segment
- `highly_fouled`: Count of highly fouled spots
- `ballast_lt_250mm`: Boolean flag for thin ballast

**DAX Measure for Fouling Severity:**
```dax
Fouling Severity =
IF(
    agg_gbfi[highly_fouled] > 0, "Critical",
    IF(
        agg_gbfi[fouled] > 0, "High",
        IF(
            agg_gbfi[reasonably_fouled] > 0, "Moderate",
            "Good"
        )
    )
)
```

### Visualization 3: Multi-Metric Risk Map

Combine TQI, GBFI, and TSR into a composite risk score.

**DAX Measure - Composite Risk Score:**
```dax
Risk Score =
VAR TQI_Risk =
    SWITCH(
        TRUE(),
        rail_segments[tqi] < 50, 3,
        rail_segments[tqi] < 75, 2,
        1
    )
VAR Fouling_Risk =
    SWITCH(
        TRUE(),
        agg_gbfi[highly_fouled] > 0, 3,
        agg_gbfi[fouled] > 0, 2,
        1
    )
VAR TSR_Risk =
    IF(agg_tsr[open_tsr] = TRUE, 3, 1)
VAR Asset_Risk =
    IF(agg_assets[fixed_asset] = TRUE, 2, 1)

RETURN (TQI_Risk + Fouling_Risk + TSR_Risk + Asset_Risk) / 4
```

**Map Setup:**
- Color by: `Risk Score` (1=green, 2=yellow, 3=red)
- Size: `segment_length_m` (longer segments = thicker lines)
- Tooltip: All individual risk components

### Visualization 4: Asset Distribution Map

**Visual Type:** Azure Maps with Bubble Layer

**Data Setup:**
- Location: Use `mid_coord_lat`, `mid_coord_lng` (point at segment midpoint)
- Bubble Size: Total asset count
- Color: Asset type (level_crossing, turnout, bridge, irj)

**DAX Measure - Total Assets:**
```dax
Total Assets =
    agg_assets[level_crossing] +
    agg_assets[irj] +
    agg_assets[turnout] +
    agg_assets[bridge]
```

**Slicer:** Add slicer for `fixed_asset` to filter critical infrastructure

### Visualization 5: TSR Timeline with Spatial Context

**Visual Type:** Combo - Map + Timeline

**Map Panel:**
- Show segments colored by `open_tsr` status
- Size by `open_tsr_days` (larger = longer duration)

**Timeline Panel:**
- Line chart showing TSR counts by year
- X-axis: Year (use `cnt_2022`, `cnt_2023`, etc.)
- Y-axis: Count
- Color: `line_code`

**DAX for Unpivoting TSR Counts:**
```dax
TSR by Year =
UNION(
    SELECTCOLUMNS(agg_tsr, "chainage_id", [chainage_id], "Year", 2022, "Count", [cnt_2022]),
    SELECTCOLUMNS(agg_tsr, "chainage_id", [chainage_id], "Year", 2023, "Count", [cnt_2023]),
    SELECTCOLUMNS(agg_tsr, "chainage_id", [chainage_id], "Year", 2024, "Count", [cnt_2024]),
    SELECTCOLUMNS(agg_tsr, "chainage_id", [chainage_id], "Year", 2025, "Count", [cnt_2025]),
    SELECTCOLUMNS(agg_tsr, "chainage_id", [chainage_id], "Year", 2026, "Count", [cnt_2026])
)
```

## Dashboard Layouts

### Executive Dashboard

**Page 1 - Overview Map**
- Full-screen map showing all lines colored by composite risk score
- Slicers: Line Code, Station
- KPI Cards: Total segments, Total km, % Fouled, % with Open TSRs

**Page 2 - Line Detail**
- Filtered to single line_code
- Map on left (60% width)
- Right panel (40%):
  - TQI distribution histogram
  - Asset count by type (stacked bar)
  - TSR trend line chart

**Page 3 - Condition Analysis**
- Scatter plot: TQI vs GBFI (identify correlation)
- Table: Top 20 highest risk segments
- Map: Filtered to high-risk segments only

### Maintenance Planning Dashboard

**Page 1 - Ballast Renewal Priority**
- Map colored by `fouled_zone`
- Table showing segments with `ballast_lt_250mm = TRUE`
- Cost estimate based on segment length

**Page 2 - Rail Renewal Priority**
- Map colored by TQI
- Filter: TQI < 50
- Table with chainage_id, TQI, asset presence, TSR history

**Page 3 - TSR Hotspots**
- Map showing `open_tsr_days` as heatmap
- Time series of TSR openings
- Correlation with curve_type (tangent vs curves)

## Advanced Features

### 1. Drill-Through to Segment Detail

Create a detail page that shows all metrics for a single segment:
- Enable drill-through on `chainage_id`
- Show segment on zoomed map
- Display all metrics: TQI, GBFI, TSR, Assets
- Historical trend if time-series data available

### 2. What-If Analysis

Use Power BI's What-If parameters for scenario planning:
- "What if we improved TQI by 10 points?"
- "What if we renewed ballast on all fouled segments?"
- Show cost vs risk reduction

### 3. Mobile Layout

Optimize for Power BI Mobile:
- Use card visuals for KPIs at top
- Simplified map with fewer details
- Touch-friendly slicers

### 4. Report Subscriptions

Set up scheduled email reports:
- Weekly summary of new TSRs
- Monthly track condition report
- Quarterly executive dashboard

## Performance Optimization

### For Large Datasets (>10,000 segments)

1. **Aggregation Strategy:**
   - Pre-aggregate at 500m or 1km intervals instead of 100m
   - Use Power Query to group segments before import

2. **Geometry Simplification:**
   - Simplify LineString geometries (reduce vertices)
   - Use `ST_Simplify()` in PostGIS before export

3. **Incremental Refresh:**
   - Partition data by line_code
   - Set up incremental refresh policy on date fields

4. **Materialized Views:**
   - Create pre-joined view in PostgreSQL
   - Import view instead of individual tables

5. **Column Reduction:**
   - Only import columns actually used in visuals
   - Remove UUID fields, internal IDs

## Data Refresh Strategy

### Option 1: Scheduled Refresh (Power BI Service)
- Set up gateway to connect to PostgreSQL
- Schedule daily/weekly refresh
- Good for stable datasets

### Option 2: Manual Export + Import
- Run `generate_powerbi_export()` periodically
- Replace files in OneDrive/SharePoint
- Power BI auto-refreshes from file location

### Option 3: DirectQuery (Live Connection)
- Connect Power BI directly to PostgreSQL
- Always shows latest data
- Trade-off: Slower performance, requires gateway

## Sample DAX Measures Library

```dax
// Total Track Length (km)
Total Track Length =
SUMX(rail_segments, [segment_length_m]) / 1000

// Average TQI by Line
Avg TQI =
AVERAGE(agg_tqi[tqi])

// Segments Requiring Attention (%)
% Segments at Risk =
DIVIDE(
    CALCULATE(COUNTROWS(rail_segments), [Risk Score] >= 2.5),
    COUNTROWS(rail_segments),
    0
) * 100

// Cost Estimate for Ballast Renewal
Ballast Renewal Cost =
VAR CostPerMeter = 500  // $500/m assumption
RETURN
    SUMX(
        FILTER(agg_gbfi, [fouled_zone] = TRUE),
        RELATED(rail_segments[segment_length_m]) * CostPerMeter
    )

// Days Since Last GBFI Survey
Days Since Survey =
DATEDIFF(MAX(agg_gbfi[collection_date]), TODAY(), DAY)
```

## Export Checklist

Before importing to Power BI:
- [ ] Run `generate_powerbi_export()` with `format='geoparquet'` for best performance
- [ ] Verify CRS is EPSG:4326 (WGS84) for mapping visuals
- [ ] Check for NULL values in key fields
- [ ] Validate geometry is valid (no self-intersections)
- [ ] Ensure chainage_id is unique and present in all tables
- [ ] Document any custom calculations in source code

## Support Resources

- Power BI Mapping Documentation: https://docs.microsoft.com/power-bi/visuals/
- ArcGIS Maps for Power BI: https://www.esri.com/en-us/arcgis/products/maps-for-powerbi
- GeoJSON Specification: https://geojson.org/
- PostGIS to Power BI Guide: https://www.postgresonline.com/journal/archives/267-Using-PostgreSQL-PostGIS-data-in-Power-BI.html
