
## Processor param defaults

default_costs = {
    "nodes": {
        "default": 1.0,
    },
    "edges": {
        "default": 1.0,
    }
}

default_input_fields = {
    "demand": {
        "id": "ID",
        "address": "ADDRESS",
        "demand": "DEMAND",
    },
    "ug_path": {
        "id": "ID"
    },
    "ug_points": {
        "id": "ID"
    }
}

# use these to specify filters to apply to nodes and edges a graph, based on their tags.
# note: the keys are defined terms used in the code, so shouldn't be changed
default_tag_filters = {
    "base": "ug",
    "filter_demands_to_base": True,
    "nodes": {
        "DEMAND": "demand",
        "AGGDEMAND": "access",
    },
    "edges": {
        "LEADIN": "leadin",
        "CROSSING": "crossing",
    }
}

## Solve param defaults
weight_field = "length"
cost_field = "cost"
demand_field = "demand"

default_steps = ["pp", "name_of_first_solve"]

default_solver_specific_params = {
    "pcst": {
        "num_clusters": 4,
        "pruning": "strong",
        "verbosity_level": 0,
    }
}
