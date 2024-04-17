from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def generate_network_from_edges(edges: List[dict]) -> nx.Graph:
    """Generate a network from edges.

    Args:
        edges: List[dict]. A list of edges. Each edge is a dictionary with the following keys:
            - "source": str. The source node id.
            - "target": str. The target node id.
            - "distance": float. The distance between the source and target nodes.

    Returns:
        G: nx.Graph. A networkx graph.
    """

    G = nx.Graph()
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        weight = edge["distance"]
        if source not in G.nodes:
            G.add_node(source)
        if target not in G.nodes:
            G.add_node(target)
        G.add_edge(source, target, weight=1 / weight)
    return G


def plot_network(G: nx.Graph, save_path: str = None):
    """Plot the network.

    Args:
        G: nx.Graph. A networkx graph.
        save_path: str. The path to save the plot.

    Returns:
        None
    """
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_edge_distribution(distances: List[dict]):
    """Plot the distribution of edge weights.

    Args:
        distances: List[dict]. A list of distances. Each element is a dictionary with the following keys:
            - "id": str. A cluster identifier for the node.
            - "distance": dict. A dictionary of distances to other nodes. The key is the node id and the value is the distance.

    Returns:
        None
    """

    edge_weights = []
    for distance in distances:
        for key, value in distance["distance"].items():
            edge_weights.append(value)

    sns.histplot(edge_weights, kde=True)
    plt.title("Distribution of edge weights")
    plt.xlabel("Edge weight")
    plt.ylabel("Frequency")
    plt.show()
