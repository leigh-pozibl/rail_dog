import logging
from typing import Dict

from pandas import DataFrame
from geopandas.geodataframe import GeoDataFrame
from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import List, Optional, Union

from rail_dog.configs import defaults
from snappy_utils.params import DBData


@dataclass
class BaseData:
    path: str | List[str] | DBData = None
    points: str | List[str] = None

    boundary: str | List[str] | DBData = None

    excel_files: Dict = None  # data is read and packed into track_data as a dict of DataFrames
    csv_files: List = None  # data is read and packed into track_data as a dict of DataFrames
    track_data: Dict = field(default_factory=lambda: {})
    
    rp_raw_data: str = None # point to a folder containing raw RP data
    tg_raw_data: str = None # point to a folder containing raw TP data

    rp_raw: str = None
    tg_raw: str = None

    ensco_db: str = None # points to a duckdb instance containing TP data

    def __post_init__(self):
        self.active_layers = {
            l: getattr(self, l) for l in
            {"path", "points", "boundary"}
            if getattr(self, l) is not None
        }

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
        

@dataclass
class ControlsData:
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
    gap_tolerance: float = 0.1
    extend_tolerance: float = 0.1
    split_lines: bool = True
    split_lines_at_intersections: bool = False
    segment_ug_path_length: float = 0
    segment_ar_path_length: float = 0

    consolidate_leadins: bool = True
    demand_snapping_range: int = 1000

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
