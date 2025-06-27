import os
import copy
import logging
from collections import defaultdict
from itertools import count

from geonetworkx import GeoGraph, GeoMultiDiGraph
from geopandas.geodataframe import GeoDataFrame
import pyproj

from snappy_utils.io import write_gdf


class HybridHubConfig:
    def __init__(self, tags: set, spec: str):
        self.tags = tags
        self.spec = spec


class HybridHub:
    def __init__(self, cand_id: str, config: HybridHubConfig, nodes: list, demand: int):
        self.cand_id = cand_id
        self.nodes = nodes
        self.config = config
        self.placed = False
        self.id = ""
        self.demand = demand


class OutputWriter():
    def __init__(self, solution_graph: GeoMultiDiGraph, base_graph: GeoGraph, params: dict, output_dir: str):
        self.solution_graph = solution_graph
        self.base_graph = base_graph
        self.base_data = params.base_data
        self.params = params

        self.arch_params = params.parameters.architecture
        self.hub_configs = params.parameters.architecture.hub_configs
        self.cable_configs = params.parameters.architecture.cable_configs
        self.duct_configs = params.parameters.architecture.duct_configs
        self.solve_configs = params.parameters.solve_configs

        self.setup_output_configs()

        self.working_crs = params.parameters.globals.working_crs

        self.output_dir = output_dir
        self.output_fmt = params.parameters.globals.output_fmt

        self.ids = defaultdict(count)
        self.node_to_id = dict()  # map the solution_graph node id to its naming convention id
        self.edge_to_id = dict()  # map the solution_graph edge id to its naming convention id
        self.source_to_demands = dict()
        self.demands_to_source = dict()

        self.input_fields = params.parameters.input_fields

        self.add_path_stats = False
        self.pcst_path = True

        self.debug = True
        src = pyproj.CRS(self.working_crs)
        dst = pyproj.CRS('epsg:4326')
        self.PROJ = pyproj.Transformer.from_crs(src, dst, always_xy=True)

    def _set_db(self, db, metadata):
        self.db = db
        self.metadata = metadata

    def setup_output_configs(self):

        self.sol_tags = defaultdict(set)

        self.configs_dict = {}
        for sc in self.solve_configs:
            sol_hub_tag = sc.get_filters("nodes", "SOLUTION_HUB", "string")
            if sol_hub_tag:
                self.configs_dict[sol_hub_tag] = self.hub_configs[sol_hub_tag]

            sol_joint_tag = sc.get_filters("nodes", "SOLUTION_JOINT", "string")
            if sol_joint_tag:
                self.configs_dict[sol_joint_tag] = self.hub_configs[sol_joint_tag]

            sol_demand_tag = sc.get_filters("nodes", "DEMAND", "set")
            if sol_demand_tag:
                for tag in sol_demand_tag:
                    self.configs_dict[tag] = self.hub_configs[tag]

            sol_edge_tag = sc.get_filters("edges", "SOLUTION_EDGE", "string")
            if sol_edge_tag:
                self.configs_dict[sol_edge_tag] = self.cable_configs[sol_edge_tag]
                self.sol_tags["SOLUTION_EDGE"].add(sol_edge_tag)

            sol_assign_tag = sc.get_filters("edges", "SOLUTION_ASSIGN", "string")
            if sol_assign_tag:
                self.configs_dict[sol_assign_tag] = self.cable_configs[sol_assign_tag]
                self.sol_tags["SOLUTION_ASSIGN"].add(sol_assign_tag)

            sol_duct_tag = sc.get_filters("edges", "SOLUTION_DUCT", "string")
            if sol_duct_tag:
                self.configs_dict[sol_duct_tag] = self.duct_configs[sol_duct_tag]
                self.sol_tags["SOLUTION_DUCT"].add(sol_duct_tag)

        # make this configurable in future
        self.hybrid_hubs = []
        self.hybrid_hub_configs = [
            HybridHubConfig(
                tags={"ug-closure", "oh-closure"},
                spec="96-port",
            )
        ]

    def write_gis_output(self):
        """
        # optical elements
        fiber_devices - ie: hub devices such as cabinets, closures, terminals
        fiber_cables - ie: cables, slackloops
        fiber_equipment - ie: equipment that is placed in a fiber_device such as a splitter

        # physical elements
        ug_structures - ie: pits, chambers, hand holes, vaults, etc...
        ug_ducts - ie: conduits, microducts, etc
        aer_structures - ie: poles
        aer_wires - ie: messenger wires

        # other
        parcels/demand
        path - route where there is at least one cable placed
        assignments - misc tabular mappings
        """

        logging.info("Preparing the output dataframes")
        fiber_devices = list()
        fiber_cables = list()
        fiber_equipment = list()
        ug_structures = list()
        ug_ducts = list()
        aer_structures = list()
        aer_wires = list()
        demand = list()
        parcels = []
        path = list()
        assignments = list()

        # initialise id counters
        next(self.ids['fiber_devices'])
        next(self.ids['fiber_cables'])
        next(self.ids['fiber_equipment'])
        next(self.ids['ug_structures'])
        next(self.ids['ug_ducts'])
        next(self.ids['aer_structures'])
        next(self.ids['aer_wires'])
        next(self.ids['demand'])
        next(self.ids['parcels'])
        next(self.ids['path'])
        next(self.ids['assignments'])

        self.spec = {
            "oh-terminal": "8-port",
            "ug-closure": "96-port",
            "solution_hub": "fdh"
        }

        def get_hub_spec(node_data):
            spec = ""
            for t in node_data["tags"]:
                if t in self.configs_dict:
                    spec = self.arch_params.get_hub_attr(t, node_data.get("size", 0), "spec")
            return spec

        def get_cable_spec(edge_data):
            spec = ""
            for t in edge_data["tags"]:
                if t in self.configs_dict:
                    spec = self.arch_params.get_cable_attr(t, edge_data.get("size", 0), "spec")
            return spec

        def get_duct_spec(edge_data):
            spec = ""
            for t in edge_data["tags"]:
                if t in self.configs_dict:
                    spec = self.arch_params.get_duct_attr(t, edge_data.get("size", 0), "spec")
            return spec

        # get colocated hubs that can be consolidated into a single fibre device
        cand_id_to_hubs = defaultdict(list)
        for n, d in self.solution_graph.nodes(data=True):
            cand_id_to_hubs[d["cand_id"]].append((n, d["tags"], d["demand"]))

        for cand_id, tuples in cand_id_to_hubs.items():
            if len(tuples) == 1:
                continue
            colocated_hub_specs = set()
            for _, tags, _ in tuples:
                colocated_hub_specs = colocated_hub_specs.union(tags)
            for hhc in self.hybrid_hub_configs:
                if hhc.tags.issubset(colocated_hub_specs):
                    hh = HybridHub(
                        cand_id=cand_id,
                        config=hhc,
                        nodes=[n for n, tag, _ in tuples if tag.intersection(hhc.tags)],
                        demand=sum([d for _, _, d in tuples])
                    )
                    self.hybrid_hubs.append(hh)

        for n, d in self.solution_graph.nodes(data=True):
            new_node_data = {}

            if not d["tags"].intersection({"demand"}):

                hybrid_hub = [hh for hh in self.hybrid_hubs if n in hh.nodes]
                if hybrid_hub and hybrid_hub[0].placed:
                    self.node_to_id[n] = hybrid_hub[0].id
                    continue
                elif hybrid_hub:
                    spec = hybrid_hub[0].config.spec
                    demand = hybrid_hub[0].demand
                else:
                    spec = get_hub_spec(d)
                    demand = d["demand"]

                device_id = f"fd-{str(next(self.ids['fiber_devices'])).zfill(5)}"

                source_node = d.get("src", d["src"])
                source_id = self.node_to_id.get(source_node, source_node)
                source_of_source_node = self.demands_to_source.get(source_node, source_node)
                source_of_source_id = self.node_to_id.get(source_of_source_node, source_of_source_node)

                if d["tags"].intersection({"ug-closure", "oh-closure"}):
                    area_id = device_id
                elif d["tags"].intersection({"oh-terminal", "oh-joint"}):
                    area_id = source_id
                elif d["tags"].intersection({"access-pt"}):
                    area_id = source_id
                else:
                    area_id = source_of_source_id

                new_node_data["id"] = device_id
                new_node_data["input_id"] = d["input_id"]
                new_node_data["specification"] = spec
                new_node_data["source"] = source_id
                new_node_data["area"] = area_id
                new_node_data["demand"] = demand
                new_node_data["placement"] = d.get("placement", "")
                new_node_data["geometry"] = d["geometry"]
                new_node_data["leg"] = d.get("leg", "")
                if d.get("duct_info"):
                    new_node_data["duct_group"] = d["duct_info"].get("group_id", "")
                    new_node_data["duct_assignment"] = d["duct_info"].get("tube_ids", "")

                fiber_devices.append(new_node_data)
                self.node_to_id[n] = new_node_data["id"]
                if d.get("access_demand_ids"):
                    self.source_to_demands[n] = d["access_demand_ids"]
                    for dem_node in d["access_demand_ids"].split(","):
                        self.demands_to_source[dem_node] = n
                self.demands_to_source[n] = d["src"]

                # if d["tags"].intersection({"oh-terminal"}):
                #     if d["src"] != "":
                #         source = solution_to_cand_node_map[d["src"]]
                #         raw_oh_solution[source].append(solution_to_cand_node_map[n])
                #     else:
                #         raw_oh_solution[solution_to_cand_node_map[n]].append(solution_to_cand_node_map[n])
                # if d["tags"].intersection({"ug-closure", "oh-closure"}):
                #     source = solution_to_cand_node_map[d["src"]]
                #     raw_ug_solution[source][leg].append(solution_to_cand_node_map[n])

                if hybrid_hub:
                    hybrid_hub[0].placed = True
                    hybrid_hub[0].id = new_node_data["id"]

        base_parcels = None
        if self.base_data.parcels is not None:
            base_parcels = self.base_data.parcels.to_crs("epsg:4326")

        for n, d in self.solution_graph.nodes(data=True):
            new_node_data = {}

            if {"demand", "ug"}.issubset(d["tags"]):
                source_node = self.demands_to_source.get(n, "")
                source_id = self.node_to_id.get(source_node, "")

                source_of_source_node = self.demands_to_source.get(source_node, "")
                source_of_source_id = self.node_to_id.get(source_of_source_node, "")

                new_node_data["id"] = f"fd-{str(next(self.ids['fiber_devices'])).zfill(5)}"
                new_node_data["input_id"] = d["input_id"]
                new_node_data["specification"] = "niu"
                new_node_data["source"] = source_id
                new_node_data["area"] = source_of_source_id
                new_node_data["demand"] = d["demand"]
                new_node_data["placement"] = "ug"
                new_node_data["geometry"] = d["geometry"]
                new_node_data["leg"] = d.get("leg", "")
                if d.get("duct_info"):
                    new_node_data["duct_group"] = d["duct_info"].get("group_id", "")
                    new_node_data["duct_assignment"] = d["duct_info"].get("tube_ids", "")

            elif {"demand", "oh"}.issubset(d["tags"]):
                source_node = d.get("src", "")
                source_id = self.node_to_id.get(source_node, "")

                source_of_source_node = self.demands_to_source.get(source_node, "")
                source_of_source_id = self.node_to_id.get(source_of_source_node, "")

                new_node_data["id"] = f"fd-{str(next(self.ids['fiber_devices'])).zfill(5)}"
                new_node_data["input_id"] = d["input_id"]
                new_node_data["specification"] = "niu"
                new_node_data["source"] = source_id
                new_node_data["area"] = source_of_source_id
                new_node_data["demand"] = d["demand"]
                new_node_data["placement"] = "oh"
                new_node_data["geometry"] = d["geometry"]
                new_node_data["leg"] = d.get("leg", "")
                if d.get("duct_info"):
                    new_node_data["duct_group"] = d["duct_info"].get("group_id", "")
                    new_node_data["duct_assignment"] = d["duct_info"].get("tube_ids", "")

            if new_node_data:
                fiber_devices.append(new_node_data)
                self.node_to_id[n] = new_node_data["id"]

                if base_parcels is not None:
                    try:
                        parcel_geom = base_parcels.loc[base_parcels[self.input_fields["parcels"]["ID"]] == d["parcel_id"]]["geometry"].iloc[0]
                        # print(parcel_geom)
                        parcel_data = copy.copy(new_node_data)
                        parcel_data["id"] = f"p-{str(next(self.ids['parcels'])).zfill(5)}"
                        parcel_data["geometry"] = parcel_geom
                        parcel_data.pop("specification")
                        parcel_data.pop("placement")
                        parcels.append(parcel_data)
                    except:
                        pass

        # update any source attribute that wasn't initially populated.
        for fd in fiber_devices:
            if fd["source"] in self.node_to_id:
                fd["source"] = self.node_to_id[fd["source"]]
            if fd["area"] in self.node_to_id:
                fd["area"] = self.node_to_id[fd["area"]]

        if fiber_devices:
            fiber_devices = GeoDataFrame(fiber_devices, geometry='geometry', crs=self.working_crs)
            out_file = os.path.join(self.output_dir, f'fibre_devices.{self.output_fmt}')
            write_gdf(fiber_devices, out_file, self.db, "pointfeature", "out.fiber_devices", self.metadata)

        if base_parcels is not None:
            # the input parcel layer has been converted to 4326
            parcels = GeoDataFrame(parcels, geometry='geometry', crs="epsg:4326")
            out_file = os.path.join(self.output_dir, f'parcels.{self.output_fmt}')
            write_gdf(parcels, out_file, self.db, "polygonfeature", "out.parcels", self.metadata)

        path_arcs = set()
        path_arcs_count_ug_duct = defaultdict(int)
        path_arcs_group_ug_duct = defaultdict(set)
        path_arcs_count_ug_dist_cable = defaultdict(int)
        path_arcs_count_ug_feeder_cable = defaultdict(int)
        path_arcs_count_oh = defaultdict(int)

        for e in self.solution_graph.edges(data=True):
            start, end, d = e
            new_edge_data = {}
            if d["tags"].intersection(self.sol_tags["SOLUTION_ASSIGN"]):
                new_edge_data["id"] = f"a-{str(next(self.ids['assignments'])).zfill(5)}"
                new_edge_data["start_id"] = self.node_to_id.get(start, "")
                new_edge_data["end_id"] = self.node_to_id.get(end, "")
                new_edge_data["specification"] = "leadin"
                new_edge_data["source"] = self.node_to_id.get(d["src"], "")
                new_edge_data["placement"] = d.get("placement", "ug")
                new_edge_data["length"] = round(d["length"], 2)
                new_edge_data["geometry"] = d["geometry"]
                if d.get("duct_info"):
                    new_edge_data["duct_group"] = d["duct_info"].get("group_id", "")
                    new_edge_data["duct_assignment"] = d["duct_info"].get("tube_ids", "")

                assignments.append(new_edge_data)

            if d["tags"].intersection(self.sol_tags["SOLUTION_DUCT"]):
                spec = get_duct_spec(d)
                if spec:
                    new_edge_data["id"] = f"ug-duct-{str(next(self.ids['ug_ducts'])).zfill(5)}"
                    new_edge_data["start_id"] = self.node_to_id.get(start, "")
                    new_edge_data["end_id"] = self.node_to_id.get(end, "")
                    new_edge_data["specification"] = get_duct_spec(d)
                    new_edge_data["source"] = self.node_to_id.get(d["src"], "")
                    new_edge_data["placement"] = d.get("placement", "ug")
                    new_edge_data["length"] = round(d["length"], 2)
                    new_edge_data["geometry"] = d["geometry"]
                    new_edge_data["duct_group"] = d["duct_info"].get("group_id", "")
                    new_edge_data["duct_assignment"] = d["duct_info"].get("tube_ids", "")

                    ug_ducts.append(new_edge_data)

                    for arc in d.get("arcs", []):
                        arc = tuple(sorted(arc))
                        path_arcs.add(arc)
                        path_arcs_count_ug_duct[arc] += 1
                        path_arcs_group_ug_duct[arc].add(new_edge_data["duct_group"])

            if d["tags"].intersection(self.sol_tags["SOLUTION_EDGE"]):
                new_edge_data["id"] = f"fc-{str(next(self.ids['fiber_cables'])).zfill(5)}"
                new_edge_data["start_id"] = self.node_to_id.get(start, "")
                new_edge_data["end_id"] = self.node_to_id.get(end, "")
                new_edge_data["specification"] = get_cable_spec(d)
                new_edge_data["source"] = self.node_to_id.get(d["src"], "")
                new_edge_data["placement"] = d.get("placement", "ug")
                new_edge_data["length"] = round(d["length"], 2)
                new_edge_data["leg"] = d.get("leg", "")
                new_edge_data["geometry"] = d["geometry"]
                if d.get("duct_info"):
                    new_edge_data["duct_group"] = d["duct_info"].get("group_id", "")
                    new_edge_data["duct_assignment"] = d["duct_info"].get("tube_ids", "")

                fiber_cables.append(new_edge_data)

                if "oh-drop" in d["tags"]:
                    continue

                for arc in d.get("arcs", []):
                    arc = tuple(sorted(arc))
                    path_arcs.add(tuple(arc))
                    if "oh-dist" in d["tags"]:
                        path_arcs_count_oh[arc] += 1
                    elif "ug-dist" in d["tags"]:
                        path_arcs_count_ug_dist_cable[arc] += 1
                    elif "ug-feeder" in d["tags"]:
                        path_arcs_count_ug_feeder_cable[arc] += 1

        if self.pcst_path:
            # make the path out of the fibre_cable features
            for f in fiber_cables:
                path_data = {}
                path_data["id"] = f"path-{str(next(self.ids['path'])).zfill(5)}"
                path_data["start_id"] = f["start_id"]
                path_data["end_id"] = f["end_id"]
                path_data["placement"] = f["placement"]
                path_data["length"] = f["length"]
                path_data["geometry"] = f["geometry"]
                path.append(path_data)

        elif path_arcs:
            error = False
            error_cnt = 0
            for arc in path_arcs:
                try:
                    base_edge_data = self.base_graph.get_edge_data(*arc)

                    new_edge_data = {}
                    new_edge_data["id"] = f"path-{str(next(self.ids['path'])).zfill(5)}"

                    if "ug" in base_edge_data["tags"]:
                        new_edge_data["placement"] = "ug"
                    else:
                        new_edge_data["placement"] = "oh"

                    new_edge_data["length"] = round(base_edge_data["length"], 2)
                    new_edge_data["geometry"] = base_edge_data["geometry"]
                    if self.add_path_stats:
                        new_edge_data["dist_duct_bundle_cnt"] = len(path_arcs_group_ug_duct.get(arc, []))
                        new_edge_data["dist_duct_cnt"] = path_arcs_count_ug_duct.get(arc, 0)
                        new_edge_data["feeder_duct_cnt"] = path_arcs_count_ug_feeder_cable.get(arc, 0)
                        new_edge_data["oh_cable_cnt"] = path_arcs_count_oh.get(arc, 0)

                    path.append(new_edge_data)
                except:
                    error = True
                    error_cnt += 1
                    
            if error:
                logging.error("Problem with building arcs for the path layer - this is a known problem, need to fix building of arcs when rolling graphs")
                logging.error(f"Error count: {error_cnt}")

        if assignments:
            assignments = GeoDataFrame(assignments, geometry='geometry', crs=self.working_crs)
            out_file = os.path.join(self.output_dir, f'assignments.{self.output_fmt}')
            write_gdf(assignments, out_file, self.db, "linestringfeature", "out.assignments", self.metadata)

        if ug_ducts:
            ug_ducts = GeoDataFrame(ug_ducts, geometry='geometry', crs=self.working_crs)
            out_file = os.path.join(self.output_dir, f'ug_ducts.{self.output_fmt}')
            write_gdf(ug_ducts, out_file, self.db, "linestringfeature", "out.ug_ducts", self.metadata)

        if fiber_cables:
            fiber_cables = GeoDataFrame(fiber_cables, geometry='geometry', crs=self.working_crs)
            out_file = os.path.join(self.output_dir, f'fiber_cables.{self.output_fmt}')
            write_gdf(fiber_cables, out_file, self.db, "linestringfeature", "out.fiber_cables", self.metadata)

        if path:
            path = GeoDataFrame(path, geometry='geometry', crs=self.working_crs)
            out_file = os.path.join(self.output_dir, f'path.{self.output_fmt}')
            write_gdf(path, out_file, self.db, "linestringfeature", "out.path", self.metadata)

        # import json
        # with open(os.path.join(self.output_dir, 'ug_solution.json'), 'w') as file:
        #     json.dump(raw_ug_solution, file)

        # with open(os.path.join(self.output_dir, 'oh_solution.json'), 'w') as file:
        #     json.dump(raw_oh_solution, file)
