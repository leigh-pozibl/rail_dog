import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString

from snappy_cat.data_loader import DataLoader
from snappy_cat.processor import Processor


def test_create_base_graph_1():
    """
    B---------------C
    |               |
    |               |
    |               |
    |               |
    A               D

    Test logical snapping
    Expect a graph with 4 nodes and 3 edges
    """
    data = DataLoader()
    p = Processor(data, params={"working_crs": "epsg:3857"}, output_dir="")

    edges = []
    edges.append({"geometry": LineString([(0, 0), (0, 1)]), "id": "AB", "start_id": "A", "end_id": "B", "tags": "test"})
    edges.append({"geometry": LineString([(0, 1), (1, 1)]), "id": "BC", "start_id": "B", "end_id": "C", "tags": "test"})
    edges.append({"geometry": LineString([(1, 1), (1, 0)]), "id": "CD", "start_id": "C", "end_id": "D", "tags": "test"})

    paths = gpd.GeoDataFrame(
        pd.DataFrame(edges),
        geometry="geometry",
        crs='epsg:3857'
    )

    p.create_base_graph_II([paths], points=[])

    assert p.base_graph.number_of_nodes() == 4
    assert p.base_graph.number_of_edges() == 3

    # all nodes are inferred
    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"inferred"}]) == 4

    # all edges are from the input data
    assert len([a for a, b, d in p.base_graph.edges(data=True) if d["tags"] == {"test"}]) == 3


def test_create_base_graph_2():
    """
                                  
    B---------------C
    |               |
    |               |
    |               |
    |               |
    A               D

    Test logical snapping
    Expect a graph with 4 nodes and 3 edges
    """
    data = DataLoader()
    p = Processor(data, params={"working_crs": "epsg:3857"}, output_dir="")

    edges = []
    edges.append({"geometry": LineString([(0, 0), (0, 1)]), "id": "AB", "start_id": "A", "end_id": "B", "tags": "test"})
    edges.append({"geometry": LineString([(0, 1), (1, 1)]), "id": "BC", "start_id": "B", "end_id": "C", "tags": "test"})
    edges.append({"geometry": LineString([(1, 1), (1, 0)]), "id": "CD", "start_id": "C", "end_id": "D", "tags": "test"})

    nodes = []
    nodes.append({"geometry": Point([(0, 0)]), "id": "A", "tags": "test,node"})
    nodes.append({"geometry": Point([(0, 1)]), "id": "B", "tags": "test,node"})
    nodes.append({"geometry": Point([(1, 1)]), "id": "C", "tags": "test,node"})
    nodes.append({"geometry": Point([(1, 0)]), "id": "D", "tags": "test,node"})

    paths = gpd.GeoDataFrame(
        pd.DataFrame(edges),
        geometry="geometry",
        crs='epsg:3857'
    )
    points = gpd.GeoDataFrame(
        pd.DataFrame(nodes),
        geometry="geometry",
        crs='epsg:3857'
    )

    # use logical snapping
    p.create_base_graph_II([paths], [points], snap_radius=0, id_fields={"node": "id", "edge_start": "start_id", "edge_end": "end_id"})

    assert p.base_graph.number_of_nodes() == 4
    assert p.base_graph.number_of_edges() == 3

    # all nodes are NOT inferred
    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"inferred"}]) == 0
    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"test", "node"}]) == 4

    # all edges are from the input data
    assert len([a for a, b, d in p.base_graph.edges(data=True) if d["tags"] == {"test"}]) == 3


def test_create_base_graph_3():
    """

    C---------------B---------------A
                                  
    I               H               G
    |               |               |
    |               |               |
    |               |               |
    |               |               |
    D               E               F

    A->G, B->H, C->I distances < 0.07m
    A->F, B->E, C->D distances > 0.07m, < 1.0m

    In this case we differentiate between the nodes C-I, B-H and A-G even though they are very closely
    located.  We achieve this by setting snap_radius=0.01.  The next test will combine these nodes

    """
    data = DataLoader()
    p = Processor(data, params={"working_crs": "epsg:3857"}, output_dir="")

    A_coords = (-119.28888597775532787, 35.49722214067361392)
    B_coords = (-119.28868441001461065, 35.49722267675803522)
    C_coords = (-119.2890875454960451, 35.49722240871582812)
    D_coords = (-119.28908741147496642, 35.49721617673448293)
    E_coords = (-119.28888581022899018, 35.49721557363952229)
    F_coords = (-119.28868427599351776, 35.4972149035340081)
    G_coords = (-119.28868440163830655, 35.49722216580259726)
    H_coords = (-119.28888598613174565, 35.49722170091689577)
    I_coords = (-119.28908754130806358, 35.49722198571174658)

    edges = []
    edges.append({"geometry": LineString([A_coords, B_coords]), "id": "AB", "start_id": "A", "end_id": "B", "tags": "test"})
    edges.append({"geometry": LineString([B_coords, C_coords]), "id": "BC", "start_id": "B", "end_id": "C", "tags": "test"})

    # edges.append({"geometry": LineString([C_coords, D_coords]), "id": "CD", "start_id": "C", "end_id": "D", "tags": "test"})
    # edges.append({"geometry": LineString([B_coords, E_coords]), "id": "BE", "start_id": "B", "end_id": "E", "tags": "test"})
    # edges.append({"geometry": LineString([A_coords, F_coords]), "id": "AF", "start_id": "A", "end_id": "F", "tags": "test"})

    edges.append({"geometry": LineString([D_coords, I_coords]), "id": "DI", "start_id": "D", "end_id": "I", "tags": "test"})
    edges.append({"geometry": LineString([E_coords, H_coords]), "id": "EH", "start_id": "E", "end_id": "H", "tags": "test"})
    edges.append({"geometry": LineString([F_coords, G_coords]), "id": "FG", "start_id": "F", "end_id": "G", "tags": "test"})

    nodes = []
    nodes.append({"geometry": Point([A_coords]), "id": "A", "tags": "test,node"})
    nodes.append({"geometry": Point([B_coords]), "id": "B", "tags": "test,node"})
    nodes.append({"geometry": Point([C_coords]), "id": "C", "tags": "test,node"})

    nodes.append({"geometry": Point([D_coords]), "id": "D", "tags": "test,node"})
    nodes.append({"geometry": Point([E_coords]), "id": "E", "tags": "test,node"})
    nodes.append({"geometry": Point([F_coords]), "id": "F", "tags": "test,node"})

    nodes.append({"geometry": Point([G_coords]), "id": "G", "tags": "test,node"})
    nodes.append({"geometry": Point([H_coords]), "id": "H", "tags": "test,node"})
    nodes.append({"geometry": Point([I_coords]), "id": "I", "tags": "test,node"})

    paths = gpd.GeoDataFrame(
        pd.DataFrame(edges),
        geometry="geometry",
        crs='epsg:4326'
    ).to_crs(p.working_crs)

    points = gpd.GeoDataFrame(
        pd.DataFrame(nodes),
        geometry="geometry",
        crs='epsg:4326'
    ).to_crs(p.working_crs)

    p.create_base_graph_II([paths], [points], snap_radius=0.01, multigraph=False)

    assert p.base_graph.number_of_nodes() == 9
    assert p.base_graph.number_of_edges() == 5

    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"inferred"}]) == 0
    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"test", "node"}]) == 9

    # all edges are from the input data
    assert len([a for a, b, d in p.base_graph.edges(data=True) if d["tags"] == {"test"}]) == 5


def test_create_base_graph_4():
    """

    C---------------B---------------A
    |               |               |
    I               H               G
    |               |               |
    |               |               |
    |               |               |
    |               |               |
    D               E               F

    A->G, B->H, C->I distances < 0.07m
    A->F, B->E, C->D distances > 0.07m, < 1.0m

    In this case we snap the lines DI, EH, FG to terminate at nodes C, B, A respectively.
    We achieve this by setting snap_radius = 0.1m

    """
    data = DataLoader()
    p = Processor(data, params={"working_crs": "epsg:3857"}, output_dir="")

    A_coords = (-119.28888597775532787, 35.49722214067361392)
    B_coords = (-119.28868441001461065, 35.49722267675803522)
    C_coords = (-119.2890875454960451, 35.49722240871582812)
    D_coords = (-119.28908741147496642, 35.49721617673448293)
    E_coords = (-119.28888581022899018, 35.49721557363952229)
    F_coords = (-119.28868427599351776, 35.4972149035340081)
    G_coords = (-119.28868440163830655, 35.49722216580259726)
    H_coords = (-119.28888598613174565, 35.49722170091689577)
    I_coords = (-119.28908754130806358, 35.49722198571174658)

    edges = []
    edges.append({"geometry": LineString([A_coords, B_coords]), "id": "AB", "start_id": "A", "end_id": "B", "tags": "test"})
    edges.append({"geometry": LineString([B_coords, C_coords]), "id": "BC", "start_id": "B", "end_id": "C", "tags": "test"})

    edges.append({"geometry": LineString([D_coords, I_coords]), "id": "DI", "start_id": "D", "end_id": "I", "tags": "test"})
    edges.append({"geometry": LineString([E_coords, H_coords]), "id": "EH", "start_id": "E", "end_id": "H", "tags": "test"})
    edges.append({"geometry": LineString([F_coords, G_coords]), "id": "FG", "start_id": "F", "end_id": "G", "tags": "test"})

    nodes = []
    nodes.append({"geometry": Point([A_coords]), "id": "A", "tags": "test,node"})
    nodes.append({"geometry": Point([B_coords]), "id": "B", "tags": "test,node"})
    nodes.append({"geometry": Point([C_coords]), "id": "C", "tags": "test,node"})

    nodes.append({"geometry": Point([D_coords]), "id": "D", "tags": "test,node"})
    nodes.append({"geometry": Point([E_coords]), "id": "E", "tags": "test,node"})
    nodes.append({"geometry": Point([F_coords]), "id": "F", "tags": "test,node"})

    paths = gpd.GeoDataFrame(
        pd.DataFrame(edges),
        geometry="geometry",
        crs='epsg:4326'
    ).to_crs(p.working_crs)

    points = gpd.GeoDataFrame(
        pd.DataFrame(nodes),
        geometry="geometry",
        crs='epsg:4326'
    ).to_crs(p.working_crs)

    p.create_base_graph_II([paths], [points], snap_radius=0.1, multigraph=False)

    assert p.base_graph.number_of_nodes() == 6
    assert p.base_graph.number_of_edges() == 5

    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"inferred"}]) == 0
    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"test", "node"}]) == 6

    # all edges are from the input data
    assert len([a for a, b, d in p.base_graph.edges(data=True) if d["tags"] == {"test"}]) == 5

    # check the geoms have been aligned
    node_dict = {n: d for n, d in p.base_graph.nodes(data=True)}
    for a, b, d in p.base_graph.edges(data=True):
        a_node_geom = node_dict[a]["geometry"]
        b_node_geom = node_dict[b]["geometry"]
        a_edge_geom = Point(d["geometry"].coords[0])
        b_edge_geom = Point(d["geometry"].coords[-1])

        try:
            assert a_node_geom.equals(a_edge_geom)
            assert b_node_geom.equals(b_edge_geom)
        except AssertionError:
            assert a_node_geom.equals(b_edge_geom)
            assert b_node_geom.equals(a_edge_geom)


def test_create_base_graph_5():
    """
    C---------------B---------------A
    |               |               |
    I               H               G
    |               |               |
    |               |               |
    |               |               |
    |               |               |
    D               E               F

    A->G, B->H, C->I distances < 0.07m
    A->F, B->E, C->D distances > 0.07m, < 1.0m

    In this case we don't do any snapping (snap_radius = 0.01m)

    """
    data = DataLoader()
    p = Processor(data, params={"working_crs": "epsg:3857"}, output_dir="")

    A_coords = (-119.28888597775532787, 35.49722214067361392)
    B_coords = (-119.28868441001461065, 35.49722267675803522)
    C_coords = (-119.2890875454960451, 35.49722240871582812)
    D_coords = (-119.28908741147496642, 35.49721617673448293)
    E_coords = (-119.28888581022899018, 35.49721557363952229)
    F_coords = (-119.28868427599351776, 35.4972149035340081)
    G_coords = (-119.28868440163830655, 35.49722216580259726)
    H_coords = (-119.28888598613174565, 35.49722170091689577)
    I_coords = (-119.28908754130806358, 35.49722198571174658)

    edges = []
    edges.append({"geometry": LineString([A_coords, B_coords]), "id": "AB", "start_id": "A", "end_id": "B", "tags": "test"})
    edges.append({"geometry": LineString([B_coords, C_coords]), "id": "BC", "start_id": "B", "end_id": "C", "tags": "test"})

    edges.append({"geometry": LineString([C_coords, I_coords]), "id": "CI", "start_id": "C", "end_id": "I", "tags": "test"})
    edges.append({"geometry": LineString([B_coords, H_coords]), "id": "BH", "start_id": "B", "end_id": "H", "tags": "test"})
    edges.append({"geometry": LineString([A_coords, G_coords]), "id": "AG", "start_id": "A", "end_id": "G", "tags": "test"})

    edges.append({"geometry": LineString([D_coords, I_coords]), "id": "DI", "start_id": "D", "end_id": "I", "tags": "test"})
    edges.append({"geometry": LineString([E_coords, H_coords]), "id": "EH", "start_id": "E", "end_id": "H", "tags": "test"})
    edges.append({"geometry": LineString([F_coords, G_coords]), "id": "FG", "start_id": "F", "end_id": "G", "tags": "test"})

    nodes = []
    nodes.append({"geometry": Point([A_coords]), "id": "A", "tags": "test,node"})
    nodes.append({"geometry": Point([B_coords]), "id": "B", "tags": "test,node"})
    nodes.append({"geometry": Point([C_coords]), "id": "C", "tags": "test,node"})

    nodes.append({"geometry": Point([D_coords]), "id": "D", "tags": "test,node"})
    nodes.append({"geometry": Point([E_coords]), "id": "E", "tags": "test,node"})
    nodes.append({"geometry": Point([F_coords]), "id": "F", "tags": "test,node"})

    nodes.append({"geometry": Point([G_coords]), "id": "G", "tags": "test,node"})
    nodes.append({"geometry": Point([H_coords]), "id": "H", "tags": "test,node"})
    nodes.append({"geometry": Point([I_coords]), "id": "I", "tags": "test,node"})

    paths = gpd.GeoDataFrame(
        pd.DataFrame(edges),
        geometry="geometry",
        crs='epsg:4326'
    ).to_crs(p.working_crs)

    points = gpd.GeoDataFrame(
        pd.DataFrame(nodes),
        geometry="geometry",
        crs='epsg:4326'
    ).to_crs(p.working_crs)

    p.create_base_graph_II([paths], [points], snap_radius=0.01, multigraph=False)

    assert p.base_graph.number_of_nodes() == 9
    assert p.base_graph.number_of_edges() == 8

    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"inferred"}]) == 0
    assert len([n for n, d in p.base_graph.nodes(data=True) if d["tags"] == {"test", "node"}]) == 9

    # all edges are from the input data
    assert len([a for a, b, d in p.base_graph.edges(data=True) if d["tags"] == {"test"}]) == 8


if __name__ == "__main__":
    test_create_base_graph_1()
    test_create_base_graph_2()
    test_create_base_graph_3()
    test_create_base_graph_4()
    test_create_base_graph_5()
