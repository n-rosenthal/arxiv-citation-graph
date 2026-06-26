# src/graph/metrics.py

import networkx as nx


def compute_graph_metrics(g):
    """Métricas estruturais básicas do grafo."""

    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()

    if num_nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0,
            "avg_clustering": 0,
            "giant_component_ratio": 0,
        }

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "density": nx.density(g),
        "avg_clustering": nx.average_clustering(g.to_undirected()),
        "giant_component_ratio": compute_giant_component_ratio(g),
    }


def compute_giant_component_ratio(g):
    """Fração dos nós pertencentes ao maior componente conectado."""

    num_nodes = g.number_of_nodes()

    if num_nodes == 0:
        return 0

    components = nx.connected_components(
        g.to_undirected()
    )

    largest = max(
        components,
        key=len,
    )

    return len(largest) / num_nodes