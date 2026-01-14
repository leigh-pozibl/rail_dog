import logging
from typing import Dict, cast
from datetime import datetime

from pandas import DataFrame
from geopandas.geodataframe import GeoDataFrame
from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import List, Optional, Union

from rail_dog.configs import defaults
from snappy_utils.params import DBData
from snappy_utils.general import parse_collection_date


@dataclass
class BaseData:
    __pydantic_config__ = {"arbitrary_types_allowed": True}

    path: Optional[str | List[str] | DBData] = None
    points: Optional[str | List[str]] = None

    boundary: Optional[str | List[str] | DBData] = None

    excel_files: Optional[Dict] = None  # data is read and packed into track_data as a dict of DataFrames
    csv_files: Optional[List] = None  # data is read and packed into track_data as a dict of DataFrames
    
    track_data: Optional[Dict] = field(default_factory=lambda: {})
    
    """
    If the special field "COLLECTION_DATE" is included in the xlsx dict, then the date is added to this dict
    eg: 
        "excel_files": {
            "base_data/raw/tsr.xlsx": {
                "TSR Open": "TSR_OPEN",
                "TSR Complete": "TSR_COMPLETE",
                "COLLECTION_DATE": "2025-03-25",
            },
    """
    collection_dates: Optional[Dict] = field(default_factory=lambda: {})

    rp_raw_data: Optional[str] = None # point to a folder containing raw RP data
    tg_raw_data: Optional[str] = None # point to a folder containing raw TP data

    rp_data: Optional[GeoDataFrame] = None
    tg_data: Optional[GeoDataFrame] = None

    ensco_db: Optional[str] = None # points to a duckdb instance containing TP data
    
    actions: dict = field(default_factory=lambda: defaults.default_actions)  # whether to refesh db or use the existing data in db
    
    """
    Specify a dict indicating the collection dates to use for each data source, eg:
        "dates": {
            "SAP": "2024-09-01",
            "GBFI": "2024-09-01",
        }
    """
    analysis_dates: dict = field(default_factory=lambda: {})
    
    def __post_init__(self):
        self.active_layers = {
            l: getattr(self, l) for l in
            {"path", "points", "boundary"}
            if getattr(self, l) is not None
        }
        
        for key, val in defaults.default_actions.items():
            if key not in self.actions:
                self.actions[key] = val
                
        for key, val in self.analysis_dates.items():
            self.analysis_dates[key] = parse_collection_date(val)
        
    def set_data(self, layer_name: str, data: GeoDataFrame | DataFrame):
        setattr(self, layer_name, data)

    def update_dict(self, layer_name: str, data: Dict[str, GeoDataFrame | DataFrame]):
        dict_data = self.get_data(layer_name)
        for key, val in data.items():
            dict_data[key] = val

    def get_data(self, layer_name: str):
        return getattr(self, layer_name)

    def table_lookup(self, layer_name):
        if layer_name in {"path"}:
            return "linestringfeature"
        elif layer_name in {"points"}:
            return "pointfeature"
        elif layer_name in {"boundary"}:
            return "polygonfeature"
        else:
            raise f"Unrecognised layer name: {layer_name}"
        
    def refresh_data(self, layer_name: str) -> bool:
        """Check if data for a given layer should be refreshed from source or use existing data in db"""
        action = self.actions.get(layer_name)
        
        if action is None:
            logging.warning(f"No action specified for layer {layer_name}, defaulting to 'refresh'")
            return True
    
        if action == "refresh":
            return True
        elif action == "use_db":
            return False
        else:
            logging.error(f"Unrecognised action '{action}' for layer {layer_name}, defaulting to 'refresh'")
            return True
    
@dataclass
class ControlsData:
    __pydantic_config__ = {"arbitrary_types_allowed": True}

    hub_locations: str | List[str] | DBData = None
    blocks: str | List[str] | DBData = None

    def __post_init__(self):
        self.active_layers = {
            l: getattr(self, l) for l in {"hub_locations", "blocks"}
            if getattr(self, l) is not None
        }   

    def set_data(self, layer_name: str, data: GeoDataFrame):
        setattr(self, layer_name, data)

    def get_data(self, layer_name: str):
        return getattr(self, layer_name)

    def table_lookup(self, layer_name):
        if layer_name in {"blocks"}:
            return "linestringfeature"
        elif layer_name in {"hub_locations"}:
            return "pointfeature"
        # elif layer_name in {"boundary", "parcels"}:
        #     return "polygonfeature"
        else:
            raise f"Unrecognised layer name: {layer_name}"


@dataclass
class GlobalParams:
    working_crs: str
    output_fmt: str
    origin: List = None

    def __post_init__(self):
        pass


@dataclass
class PreprocessParams:
    segment_length: float = 0
    segment_method: str = "respect_curve_boundaries"  # options: "ignore_curve_boundaries", "respect_curve_boundaries"

    tag_filters: dict[str, Union[str, dict]] = field(default_factory=lambda: defaults.default_tag_filters)

    def __post_init__(self):
        pass

    def get_filters(self, nodes_or_edges: str, attr: str, set_or_str: str):
        if set_or_str == "string":
            if attr is not None:
                return self.tag_filters[nodes_or_edges][attr]
            else:
                return self.tag_filters[nodes_or_edges]
        elif set_or_str == "set":
            if attr is not None:
                return set(self.tag_filters[nodes_or_edges][attr].split(","))
            else:
                return set(self.tag_filters[nodes_or_edges].split(","))
        else:
            logging.error("choose set_or_str from 'set', 'string'")


@dataclass
class BaseParams:
    globals: GlobalParams = None
    preprocess: PreprocessParams = None
    # architecture: ArchParams = None
    # solve_configs: List[SolveConfig] = None
    input_fields: dict = field(default_factory=lambda: defaults.default_input_fields)
    costs: dict = field(default_factory=lambda: defaults.default_costs)


@dataclass
class ExecutionParams:
    run_steps: Optional[list] = field(default_factory=lambda: defaults.default_steps)
    solution_graph: Optional[str | DBData] = None
    base_graph: Optional[str | DBData] = None
    raw_solution: Optional[dict] = None


@dataclass
class BaseConfiguration:
    base_data: BaseData = None
    controls: ControlsData = None
    parameters: BaseParams = None
    root_path: str = None

    execution: ExecutionParams = None

    output_dir: str = "output"

    def __post_init__(self):
        pass
