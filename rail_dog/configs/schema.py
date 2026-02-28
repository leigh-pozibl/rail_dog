"""
SQLModel schema for PostgreSQL database with PostGIS support.
"""
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import pandas as pd

from sqlmodel import SQLModel, Field, Relationship, Column, Numeric
from sqlalchemy import JSON, UniqueConstraint
from geoalchemy2 import Geometry


EPSG = 28350  # GDA2020 / MGA zone 50

# ==============================================================================
# Database Models (SQLModel - combines SQLAlchemy + Pydantic)
# ==============================================================================

class Project(SQLModel, table=True):
    """Project table - stores project metadata."""
    __tablename__ = "projects"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(max_length=255, index=True)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)}
    )

    # Relationships
    # rail_segments: list["RailSegment"] = Relationship(back_populates="project")


class RailSegmentAsset(SQLModel, table=True):
    """Junction table linking rail segments to assets (many-to-many)."""
    __tablename__ = "rail_segment_assets"

    chainage_id: str = Field(foreign_key="rail_segments.chainage_id", primary_key=True, max_length=100)
    asset_id: str = Field(foreign_key="assets.asset_id", primary_key=True, max_length=100)


class RailSegment(SQLModel, table=True):
    """Rail segment table - stores railway line segments."""
    __tablename__ = "rail_segments"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # project_id: UUID = Field(foreign_key="projects.id", index=True)

    # Basic attributes
    chainage_id: str = Field(max_length=100, unique=True, index=True)
    line: str = Field(max_length=10)
    line_code: str = Field(max_length=3)  # TL, ML, SL, EL
    section: str = Field(max_length=100)
    section_name: str = Field(max_length=100)
    station: str = Field(max_length=100)
    asset_name: str = Field(max_length=100)  # eg: "MLE_3.963_4.301"
    chainage_start_km: float
    chainage_end_km: float
    chainage_mid_km: float
    segment_length_m: float

    # Coordinates
    mid_coord_lng: float
    mid_coord_lat: float

    # Track classification
    curve_type: Optional[str] = Field(default=None, max_length=50)  # tangent, mild_curve, sharp_curve

    # Geometry - accepts shapely Point objects or WKT strings
    geometry: Any = Field(sa_column=Column(Geometry("LINESTRING", srid=EPSG)))

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    # project: Project = Relationship(back_populates="rail_segments")
    rail_profile_data: list["RailProfile"] = Relationship(back_populates="rail_segments")
    # track_geometry_data: list["TrackGeometry"] = Relationship(back_populates="rail_segments")
    assets: list["Asset"] = Relationship(back_populates="rail_segments", link_model=RailSegmentAsset)


class RailSection(SQLModel, table=True):
    """Rail section table - railway line sections correspond to curves and tangents"""
    __tablename__ = "rail_sections"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # project_id: UUID = Field(foreign_key="projects.id", index=True)

    # Basic attributes
    section_id: str = Field(max_length=100, unique=True, index=True)
    line: str = Field(max_length=10)
    line_code: str = Field(max_length=3)  # TL, ML, SL, EL
    type: str = Field(max_length=10)  # tangent or curve
    chainage_start_km: float
    chainage_end_km: float
    curve_length: float
    classification: str = Field(max_length=50)  # tangent, sharp_curve, mild_curve
    
    # only populated for curve types
    curve_id: Optional[str] = Field(default=None, max_length=10)
    ts: Optional[float]
    sc: Optional[float]
    cs: Optional[float]
    st: Optional[float]
    radius: Optional[float]
    hand: Optional[str] = Field(default=None, max_length=2)  # LH or RH
    se_design: Optional[int]
    gradient: Optional[float]
    
    # Track classification
    curve_type: Optional[str] = Field(default=None, max_length=50)  # tangent, mild_curve, sharp_curve

    # Geometry - accepts shapely Point objects or WKT strings
    geometry: Any = Field(sa_column=Column(Geometry("LINESTRING", srid=EPSG)))

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships


class RailProfile(SQLModel, table=True):
    """Rail profile measurements from ENSCO RP data."""
    __tablename__ = "rail_profile_data"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    rail_segment_id: UUID = Field(foreign_key="rail_segments.id", index=True)

    # Section info
    section_id: Optional[str] = Field(default=None, max_length=100)
    rail_hand: Optional[str] = Field(default=None, max_length=10)  # LH, RH

    # Measurements - West rail
    W_vert_wear_avg: Optional[float] = None
    W_vert_wear_max: Optional[float] = None
    W_vert_wear_p50: Optional[float] = None
    W_vert_wear_p75: Optional[float] = None
    W_vert_wear_p90: Optional[float] = None

    W_side_wear_avg: Optional[float] = None
    W_side_wear_max: Optional[float] = None
    W_side_wear_p50: Optional[float] = None
    W_side_wear_p75: Optional[float] = None
    W_side_wear_p90: Optional[float] = None

    W_rel_head_loss_avg: Optional[float] = None
    W_rel_head_loss_max: Optional[float] = None
    W_rel_head_loss_p50: Optional[float] = None
    W_rel_head_loss_p75: Optional[float] = None
    W_rel_head_loss_p90: Optional[float] = None

    # Measurements - East rail
    E_vert_wear_avg: Optional[float] = None
    E_vert_wear_max: Optional[float] = None
    E_vert_wear_p50: Optional[float] = None
    E_vert_wear_p75: Optional[float] = None
    E_vert_wear_p90: Optional[float] = None

    E_side_wear_avg: Optional[float] = None
    E_side_wear_max: Optional[float] = None
    E_side_wear_p50: Optional[float] = None
    E_side_wear_p75: Optional[float] = None
    E_side_wear_p90: Optional[float] = None

    E_rel_head_loss_avg: Optional[float] = None
    E_rel_head_loss_max: Optional[float] = None
    E_rel_head_loss_p50: Optional[float] = None
    E_rel_head_loss_p75: Optional[float] = None
    E_rel_head_loss_p90: Optional[float] = None

    # Status
    status_level: Optional[str] = Field(default=None, max_length=10)  # G, A, R
    status_string: Optional[str] = Field(default=None, max_length=50)  # GGG-GGG format

    # Geometry
    geometry: Optional[Any] = Field(
        default=None,
        sa_column=Column(Geometry("LINESTRING", srid=EPSG))
    )

    # Metadata
    measurement_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    rail_segments: RailSegment = Relationship(back_populates="rail_profile_data")


class TGRecord(SQLModel, table=True):
    """Track geometry measurements from ENSCO TG data."""
    __tablename__ = "tg_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    asset_name: str = Field(max_length=100)  # eg: "MLE_3.963_4.301"
    
    # Measurement info
    chainage_km: float
    speed: int
    post_speed: int
    trk_class: int
    
    top_e_5: float
    top_e_10: float
    top_e_20: float
    
    top_w_5: float
    top_w_10: float
    top_w_20: float
    
    align_e_5: float
    align_e_10: float
    align_e_20: float
    
    align_w_5: float
    align_w_10: float
    align_w_20: float
    
    gauge: float
    crosslevel: float
    crosslevel_rate: float
    curve: float
    curve_rate: float
    
    twist_2: float
    twist_7: float
    twist_14: float
    warp: float
    valid: float

    # Geometry - accepts shapely Point objects or WKT strings
    geometry: Any = Field(sa_column=Column(Geometry("POINT", srid=EPSG)))

    # Metadata
    collection_date: datetime = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    # rail_segments: RailSegment = Relationship(back_populates="track_geometry_data")


class AggTG(SQLModel, table=True):
    """Track geometry data aggregated to 100m chainage segments."""
    __tablename__ = "agg_tg"
    __table_args__ = (UniqueConstraint("chainage_id", "collection_date", name="uq_agg_tg_chainage_date"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    chainage_id: str = Field(max_length=100, index=True)
    line_code: str = Field(max_length=3, index=True)
    line_class: str = Field(max_length=10, index=True)
    sample_count: int

    # Speed
    avg_speed: Optional[float] = None
    avg_post_speed: Optional[float] = None

    # Gauge
    avg_gauge: Optional[float] = None
    min_gauge: Optional[float] = None
    max_gauge: Optional[float] = None

    # Crosslevel
    avg_crosslevel: Optional[float] = None
    min_crosslevel: Optional[float] = None
    max_crosslevel: Optional[float] = None
    avg_crosslevel_rate: Optional[float] = None
    min_crosslevel_rate: Optional[float] = None
    max_crosslevel_rate: Optional[float] = None

    # Curve
    avg_curve: Optional[float] = None
    min_curve: Optional[float] = None
    max_curve: Optional[float] = None
    avg_curve_rate: Optional[float] = None
    min_curve_rate: Optional[float] = None
    max_curve_rate: Optional[float] = None

    # Twist
    avg_twist_2: Optional[float] = None
    min_twist_2: Optional[float] = None
    max_twist_2: Optional[float] = None
    avg_twist_7: Optional[float] = None
    min_twist_7: Optional[float] = None
    max_twist_7: Optional[float] = None
    avg_twist_14: Optional[float] = None
    min_twist_14: Optional[float] = None
    max_twist_14: Optional[float] = None

    # Warp
    avg_warp: Optional[float] = None
    min_warp: Optional[float] = None
    max_warp: Optional[float] = None

    # Top (10m base)
    avg_top_e_10: Optional[float] = None
    min_top_e_10: Optional[float] = None
    max_top_e_10: Optional[float] = None
    avg_top_w_10: Optional[float] = None
    min_top_w_10: Optional[float] = None
    max_top_w_10: Optional[float] = None

    # Alignment (10m base)
    avg_align_e_10: Optional[float] = None
    min_align_e_10: Optional[float] = None
    max_align_e_10: Optional[float] = None
    avg_align_w_10: Optional[float] = None
    min_align_w_10: Optional[float] = None
    max_align_w_10: Optional[float] = None

    # Metadata
    collection_date: Optional[datetime] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Asset(SQLModel, table=True):
    """Railway assets (switches, signals, etc.)."""
    __tablename__ = "assets"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # project_id: UUID = Field(foreign_key="projects.id", index=True)

    # Asset info
    asset_id: str = Field(max_length=100, index=True, unique=True)
    type: str = Field(max_length=100, index=True)
    line: str = Field(max_length=10, index=True)
    line_code: str = Field(max_length=3, index=True)
    location_desc: str = Field(max_length=100)
    chainage: float = Field(sa_column=Column(Numeric(precision=6, scale=3)))  # 6 total digits, 3 after decimal

    # Location
    coord_lng: Optional[float] = None
    coord_lat: Optional[float] = None

    # Geometry - accepts shapely Point objects or WKT strings
    geometry: Any = Field(sa_column=Column(Geometry("POINT", srid=EPSG)))

    # SAP aggregate data
    sap_total_count: int = Field(default=0)
    sap_year_count: Optional[dict] = Field(default=None, sa_column=Column(JSON))   # {"2023": 5, "2024": 12}
    sap_year_status: Optional[dict] = Field(default=None, sa_column=Column(JSON))  # {"2023": "h", "2024": "vh"}
    sap_max_threshold: Optional[str] = Field(default="", max_length=10)            # "h" or "vh"

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    rail_segments: list["RailSegment"] = Relationship(back_populates="assets", link_model=RailSegmentAsset)


class AggAsset(SQLModel, table=True):
    """Railway asset counts aggregated to chainage segments."""
    __tablename__ = "agg_assets"
    __table_args__ = (UniqueConstraint("chainage_id", name="uq_agg_assets_chainage"),)
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Aggregated Asset info
    chainage_id: str = Field(max_length=100, index=True)
    level_crossing: int = Field(default=0)
    irj: int = Field(default=0)
    turnout: int = Field(default=0)
    bridge: int = Field(default=0)
    fixed_asset: bool
    wz_level_crossing: int = Field(default=0)
    wz_irj: int = Field(default=0)
    wz_turnout: int = Field(default=0)
    wz_bridge: int = Field(default=0)
    wz_asset: bool
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SAPRecord(SQLModel, table=True):
    """SAP work order records."""
    __tablename__ = "sap_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # project_id: UUID = Field(foreign_key="projects.id", index=True)

    # sap info
    date: datetime = Field(index=True)
    year: int = Field(index=True)
    revision: str = Field(max_length=10, index=True)
    planner_grp: str = Field(max_length=10)
    work_center: str = Field(max_length=10)
    user_status: str = Field(max_length=100)
    system_status: str = Field(max_length=100)
    priority: str = Field(max_length=10)
    order_id: str = Field(max_length=10, index=True)
    description: str = Field(max_length=254)
    asset_id: str = Field(max_length=100, index=True)
    iridium_code: str = Field(max_length=20)

    # Location
    # coord_lng: Optional[float] = None
    # coord_lat: Optional[float] = None

    # Geometry - accepts shapely Point objects or WKT strings
    # geometry: Any = Field(sa_column=Column(Geometry("POINT", srid=EPSG)))

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    # asset: Asset = Relationship(back_populates="asset")


def sap_column_mapping() -> dict:
    return {
        "FunctionalLocationLabel": "Functional Location",
        "BasicStartDate": "Start Date",
        "Revision": "Revision",
        "PlannerGroup": "Maint Planner Group",
        "MainWorkCentre": "Maint Work Center",
        "UserStatus": "User Status",
        "OrderSystemStatus": "System Status",
        "PriorityText": "Priority Text",
        "OrderNumber": "Order",
        "OrderDescription": "Description",
    }
          
class GBFIRecord(SQLModel, table=True):
    """Ground Penetrating Radar records."""
    __tablename__ = "gbfi_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # gbfi info
    collection_date: datetime = Field(index=True)
    chainage_start_km: float = Field(index=True)
    chainage_end_km: float = Field(index=True)
    # left: float
    # center: float
    # right: float
    avg: float
    min: float
    max: float
    
    # Location
    line_code: str = Field(max_length=3, index=True)
    # coord_lng: Optional[float] = None
    # coord_lat: Optional[float] = None

    # Geometry - accepts shapely Point objects or WKT strings
    geometry: Any = Field(sa_column=Column(Geometry("POINT", srid=EPSG)))
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def gbfi_column_mapping() -> dict:
    return {
        "From(km+m)": "start_chainage_km",
        "To(km+m)": "end_chainage_km",
        "Long_0": "lng",
        "Lat_0": "lat",
        "GBFI_Left2": "left",
        "GBFI_Center3": "center",
        "GBFI_Right4": "right",
    }


def extract_collection_date_from_header(gbfi_header: pd.DataFrame):
    # Extract collection date from header
    if len(gbfi_header) >= 4:
        data_collected_text = str(gbfi_header.iloc[3, 0])
        if "Data Collected:" in data_collected_text:
            collection_date_str = data_collected_text.split("Data Collected:")[1].strip()
            collection_date = pd.to_datetime(collection_date_str, format="%B %Y")
        else:
            collection_date = None
    else:
        collection_date = None
    return collection_date
        

class AggGBFI(SQLModel, table=True):
    """GBFI records aggregated to chainage segments."""
    __tablename__ = "agg_gbfi"
    __table_args__ = (UniqueConstraint("chainage_id", "collection_date", name="uq_agg_gbfi_chainage_date"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Aggregated GBFI info
    chainage_id: str = Field(max_length=100, index=True)
    collection_date: datetime = Field(index=True)
    max_of_avg: float
    avg_of_avg: float
    reasonably_fouled: int
    fouled: int
    highly_fouled: int
    fouled_zone: bool
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BallastRecord(SQLModel, table=True):
    """Ballast condition records."""
    __tablename__ = "ballast_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # ballast info
    collection_date: datetime = Field(index=True)
    chainage_start_km: float = Field(index=True)
    
    left: float
    center: float
    right: float
    # subballast_left: Optional[float] = None
    # subballast_center: Optional[float] = None
    # subballast_right: Optional[float] = None
    
    # Location
    line_code: str = Field(max_length=3, index=True)

    # Geometry - accepts shapely Point objects or WKT strings
    geometry: Any = Field(sa_column=Column(Geometry("POINT", srid=EPSG)))
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    
def ballast_column_mapping() -> dict:
    return {
        "Distance(km+m)": "distance_str",
        "Distance(km.m)": "distance_str",
        "WE(m)": "easting",
        "SN(m)": "northing",
        "Long_0": "lng",
        "Lat_0": "lat",
        "Ballast_Left (m)": "left",
        "Ballast_Center (m)": "center",
        "Ballast_Right (m)": "right",
        "Subballast_Left (m)": "subballast_left",
        "Subballast_Center (m)": "subballast_center",
        "Subballast_Right (m)": "subballast_right",
    }
    

class AggBallast(SQLModel, table=True):
    """Ballast records aggregated to chainage segments."""
    __tablename__ = "agg_ballast"
    __table_args__ = (UniqueConstraint("chainage_id", "collection_date", name="uq_agg_ballast_chainage_date"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Aggregated Ballast info
    chainage_id: str = Field(max_length=100, index=True)
    collection_date: datetime = Field(index=True)
    ballast_centre: float
    ballast_lt_250mm: bool
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    
class TSRRecord(SQLModel, table=True):
    """Temporary speed restriction records."""
    __tablename__ = "tsr_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # gbfi info
    report_date: datetime = Field(index=True)
    line_code: str = Field(max_length=3, index=True)
    status: str = Field(max_length=10, index=True)
    chainage_start_km: float = Field(index=True)
    chainage_end_km: float = Field(index=True)
    speed: float
    close_date: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AggTSR(SQLModel, table=True):
    """TSR records aggregated to chainage segments."""
    __tablename__ = "agg_tsr"
    __table_args__ = (UniqueConstraint("chainage_id", name="uq_agg_tsr_chainage"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Aggregated TSR info
    chainage_id: str = Field(max_length=100, index=True)
    open_tsr: bool
    open_tsr_days: int
    complete_tsr: bool
    cnt_2022: int
    cnt_2023: int
    cnt_2024: int
    cnt_2025: int
    cnt_2026: int
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    

class TQIRecord(SQLModel, table=True):
    """Track Quality Indicator records."""
    __tablename__ = "tqi_records"
    __table_args__ = (UniqueConstraint("asset_name", "chainage_start_km", "collection_date", name="uq_tqi_records_asset_chainage_date"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # tqi info
    line: str = Field(max_length=10, index=True)    # Thomas, Mainline, Solomon, Eliwana
    line_code: str = Field(max_length=3, index=True)    # ML, TH, SL, EL
    line_class: str = Field(max_length=10, index=True)  # main, bypass, loop
    
    asset_name: str     # eg: "MLE_3.963_4.301"
    chainage_start_km: float = Field(index=True)
    chainage_end_km: float = Field(index=True)
    chainage_mid_km: float = Field(index=True)
    tqi: float
    run_id: str    
    
    # Metadata
    collection_date: datetime = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AggTQI(SQLModel, table=True):
    """TQI records aggregated to chainage segments."""
    __tablename__ = "agg_tqi"
    __table_args__ = (UniqueConstraint("chainage_id", "collection_date", name="uq_agg_tqi_chainage_date"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Aggregated TQI info
    chainage_id: str = Field(max_length=100, index=True)
    # line_code: str = Field(max_length=3, index=True)    # ML, TH, SL, EL
    # line_class: str = Field(max_length=10, index=True)  # main, bypass, loop
    tqi: float
    status: str
    trend: Optional[str] = Field(default=None, max_length=20)      # improving / stable / degrading
    trend_slope: Optional[float] = None                             # TQI units/year
    trend_r_squared: Optional[float] = None                         # goodness of fit [0-1]
    
    # Metadata
    collection_date: datetime = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SegmentTrend(SQLModel, table=True):
    """Precomputed trend per chainage segment for a given metric (tqi, gbfi_avg, ballast_centre, etc.).

    One record per (chainage_id, metric, computed_for_date), so the full trend
    history is preserved as new data is ingested over time.
    """
    __tablename__ = "segment_trends"
    __table_args__ = (UniqueConstraint("chainage_id", "metric", "computed_for_date", name="uq_segment_trends_chainage_metric_date"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    chainage_id: str = Field(max_length=100, index=True)
    metric: str = Field(max_length=50, index=True)           # e.g. "tqi", "gbfi_avg", "ballast_centre"
    computed_for_date: datetime = Field(index=True)          # as-of date: only data up to this date was used
    slope: Optional[float] = None                            # metric units/year (positive = improving)
    r_squared: Optional[float] = None                        # goodness of fit [0-1]
    trend_label: Optional[str] = Field(default=None, max_length=20)  # improving / stable / degrading
    sample_count: int = Field(default=0)
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None

    # Metadata
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    

class DTRRecord(SQLModel, table=True):
    """Dynamic Track Force records."""
    __tablename__ = "dtr_records"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # dtr info
    line_code: str = Field(max_length=3, index=True)    # ML, TH, SL, EL
    # line_class: str = Field(max_length=10, index=True)  # main, bypass, loop
    chainage_km: float
    channel: str
    track_features: str
    units: str = Field(max_length=3)
    severity: int = Field(max_length=1)
    recent_max: float
    recent_avg: float
    getting_worse: int = Field(max_length=2)
    
    # Metadata
    collection_date: datetime = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AggDTR(SQLModel, table=True):
    """DTR records aggregated to chainage segments."""
    __tablename__ = "agg_dtr"
    __table_args__ = (UniqueConstraint("chainage_id", "collection_date", name="uq_agg_dtr_chainage_date"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    # Aggregated DTR info
    chainage_id: str = Field(max_length=100, index=True)
    # line_code: str = Field(max_length=3, index=True)    # ML, TH, SL, EL
    # line_class: str = Field(max_length=10, index=True)  # main, bypass, loop
    
    # counts of samples
    sf_accel_w: int = Field(max_length=3)
    sf_accel_e: int = Field(max_length=3)
    suspension: int  = Field(max_length=3)
    rock: int = Field(max_length=3)
    bounce: int = Field(max_length=3)
    
    dtf_s1: int = Field(max_length=1) # severity 1 count
    dtf_s2: int = Field(max_length=1) # severity 2 count
    dtf_s3: int = Field(max_length=1) # severity 3 count
    worst_dtf: str = Field(max_length=2) # s1, s2, s3 where s1 is worse
    getting_worse: int = Field(max_length=2) 
    
    # Metadata
    collection_date: datetime = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    
# ==============================================================================
# Helper Functions
# ==============================================================================

def create_tables(engine):
    """Create all tables in the database."""
    SQLModel.metadata.create_all(bind=engine)


def create_table(engine, model_class):
    """
    Create a single table in the database.

    Args:
        engine: SQLAlchemy engine
        model_class: SQLModel class (e.g., Asset, RailSegment, etc.)
    """
    model_class.metadata.create_all(bind=engine, tables=[model_class.__table__])


def drop_tables(engine):
    """Drop all tables from the database."""
    SQLModel.metadata.drop_all(bind=engine)
