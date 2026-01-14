# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rail_dog is a railway analysis tool for processing and analyzing railway track data, particularly for FMG (Fortescue Metals Group) railway lines. The tool processes railway geometry, track conditions, and ENSCO rail measurement data to generate segmented outputs with condition analysis.

## Running the Application

### Installation
```bash
pip install -e .
```

### Running the Tool
The main entry point is via the `run_rail` command:
```bash
run_rail --config data/config_fmg.yaml --output-dir output
```

Parameters:
- `--config`: Path to YAML/JSON/TOML config file (required if no --json-input)
- `--json-input`: Alternative JSON blob input
- `--db-env`: Database environment ('local' or 'prod', or full connection string)
- `--project-id`: Project ID (default: all zeros UUID)
- `--output-dir`: Root output directory (default: "output")

### Running Tests
```bash
python -m pytest rail_dog/tests/test_processor.py
```

Or run individual tests:
```bash
python rail_dog/tests/test_processor.py
```

## Architecture

### Main Components

**Processor (`rail_dog/processor.py`)**
- Core processing engine that segments railway lines into 100m chainage intervals
- Processes four main rail sections: Thomas (TLX), Mainline (MLX), Solomon (SML), Eliwana (EML)
- Handles curve/tangent section classification and rail degradation analysis
- Uses interval trees for efficient spatial queries and section matching

**Configuration System (`rail_dog/configs/params.py`)**
- Pydantic dataclasses for type-safe configuration
- `BaseConfiguration`: Top-level config containing base_data, controls, parameters, execution
- `BaseData`: Geospatial layers (path, points), Excel/CSV track data, ENSCO RP/TG data
- `GlobalParams`: CRS, output format
- `PreprocessParams`: Processing tolerances and filters

**I/O Utils (`rail_dog/utils/io_utils.py`)**
- Loads YAML/JSON/TOML configs and parses into dataclass instances
- Handles geospatial file loading (shapefiles, GeoJSON)
- Database integration via snappy_utils for PostgreSQL/PostGIS
- Excel/CSV reading for track data
- DuckDB for efficient ENSCO data processing

**Output Writer (`rail_dog/output.py`)**
- Generates GIS outputs for fiber network design (though used for rail in this context)
- Creates fiber_devices, fiber_cables, ug_ducts layers

### Key Processing Steps

1. **Line Segmentation**: Railway lines are split at POIs (origin, mainline_start) and divided into TLX/MLX/SML/EML sections
2. **Chainage Creation**: Each section is segmented into 100m intervals with chainage IDs (format: `CHAIN-{PREFIX}-S-{START}-E-{END}-MAINLINE`)
3. **Switch Processing**: Point features (switches) are spatially matched to chainage segments
4. **Curve Section Processing**:
   - Curve/tangent sections from input data are geometrically matched to rail centerline
   - Uses IntervalTree for efficient chainage-based lookups
   - Sections classified as Tangent, Mild Curve, or Sharp Curve
5. **RP/TG Data Processing**:
   - Raw ENSCO Rail Profile (RP) and Track Geometry (TG) data loaded from CSV/DuckDB
   - Data points matched to curve sections by line_id and chainage
   - Statistics calculated per section: min/max/avg/p50/p75/p90 for wear metrics
6. **Degradation Analysis**: Rail wear metrics compared against thresholds based on section classification

### Rail Line Structure

The railway network is divided into 4 main sections with specific chainage offsets:
- **TLX (Thomas)**: -3.7 to 26.9 km
- **MLX (Mainline)**: 26.9 km onwards
- **SML (Solomon Spur)**: starts at 174 km
- **EML (Eliwana)**: starts at 288.70 km

### Coordinate Systems

- Default working CRS: EPSG:28350 (GDA2020 / MGA zone 50)
- Output typically in EPSG:4326 for GIS files
- UTM CRS auto-detection supported via `working_crs: "utm"`

### Degradation Thresholds

Rail condition thresholds are defined in `Processor.Thresholds` class:
- **Tangent sections**: vert_wear=16mm, side_wear=4mm, rel_head_loss=40%
- **Mild curves**: vert_wear=14mm, side_wear=6mm, rel_head_loss=32-37%
- **Sharp curves**: vert_wear=10mm, side_wear=6mm, rel_head_loss=25%

Thresholds vary by:
- Section classification (tangent/mild_curve/sharp_curve)
- Rail hand (LH/RH for curves)
- Track position (east/west rail)

### Data Processing Notes

**ENSCO Data Corrections**
- Line region mappings in `process_rp_data_into_sections()` map region names to line prefixes
- Special corrections for Thomas/Firetail/Future sections that cross line boundaries
- Data filtered to main lines only (excludes "OTH" line_id)
- Can use either raw CSV data or pre-processed DuckDB

**Status Scoring**
- Status string format: "GGG-GGG" representing [W_rel_head_loss, W_vert_wear, W_side_wear, E_rel_head_loss, E_vert_wear, E_side_wear]
- Colors: G=green (ok), A=amber (exceeds p90), R=red (exceeds p75)
- Status level: R if any red, A if any amber, G otherwise

## Dependencies

Key dependencies from `pyproject.toml`:
- **geopandas**: Geospatial data handling
- **shapely 2.0.6**: Geometry operations
- **pandas/numpy**: Data processing
- **pydantic**: Configuration validation
- **intervaltree** (not in pyproject but imported): Interval-based spatial queries
- **duckdb**: Fast CSV/data processing
- **click**: CLI interface
- **python-dotenv**: Environment config

External dependency:
- **snappy_utils**: Shared utilities for geometry, I/O, database (not in this repo)

## Config File Structure

Example config (YAML/JSON/TOML):
```yaml
base_data:
  path: "path/to/rail_centerline.geojson"
  points: "path/to/switches.geojson"
  csv_files:
    - "track_data/TLX.csv"
    - "track_data/CURVE_SECTIONS.csv"
  ensco_db: "process/ensco_data_corrected.duckdb"

parameters:
  globals:
    working_crs: "epsg:28350"
    output_fmt: "geojson"
    origin: [662669.066, 7746597.522]

execution:
  run_steps: ["pp"]
```

## Output Files

Generated in `{output_dir}/process/`:
- `thomas_segments.geojson`: TLX 100m segments with track data and switch_ids
- `mainline_segments.geojson`: MLX segments
- `solomon_segments.geojson`: SML segments
- `eliwana_segments.geojson`: EML segments
- `pois.geojson`: Points of interest
- `switches.geojson`: Switch locations with chainage_id
- `rp_sections.geojson/.csv`: Rail profile stats by curve section
- `tg_sections.geojson/.csv`: Track geometry stats by curve section

Segment fields include: chainage_id, chainage_start_km, chainage_end_km, mid_coord_E/N/lng/lat, switch_ids, plus all track data fields

## Important Implementation Details

**Chainage ID Format**: When working with TLX (Thomas) sections, the code replaces "TLX" with "MLX" when looking up track data, as the source data uses MLX naming for the entire mainline including Thomas section.

**Geometry Alignment**: The `create_curve_sections()` method uses shapely's `substring()` to extract curve geometries from the base rail line, preserving vertices for accurate representation.

**IntervalTree Usage**: Curve sections are stored in IntervalTree structures keyed by line_id for O(log n) chainage lookups when matching RP/TG data points to sections.

**DuckDB Integration**: Large ENSCO datasets are processed via DuckDB for performance. The code can either create a new DuckDB from raw CSVs or read from an existing corrected database.
