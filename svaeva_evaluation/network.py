from typing import Dict, List, Union

import dotenv
import networkx as nx
import numpy as np

dotenv.load_dotenv()
from svaeva_redux.schemas.redis import UserModel


def threshold_distances(distances: List[dict], threshold: float) -> List[dict]:
    edges = []
    for distance_info in distances:
        current_id = distance_info["id"]
        for other_id, distance in distance_info["distance"].items():
            if distance <= threshold:
                edge = {"source": current_id, "target": other_id, "distance": distance}
                edges.append(edge)
    return edges


def calculate_distances(nodes: List[dict], distance_metric: str = "euclidean") -> List[dict]:
    """Calculate distances between nodes.

    Args:
        nodes: List[dict]. A list of nodes. Each node is a dictionary with the following keys:
            - "id": str. A unique identifier for the node.
            - "cluster_id": int. A cluster identifier for the node.
            - "embedding": np.ndarray. An embedding vector for the node.
        distance_metric: str. A distance metric to use. It can be "euclidean" or "cosine".

    Returns:
        distances: List[dict]. A list of distances. Each element is a dictionary with the following keys:
            - "id": str. A cluster identifier for the node.
            - "distance": dict. A dictionary of distances to other nodes. The key is the node id and the value is the distance.
    """
    distances = []
    for node_x in nodes:
        vector_x = node_x["embedding"]
        current_id = node_x["id"]
        distance_info = {"id": current_id, "distance": {}, "cluster_id": node_x["cluster_id"]}
        for node_y in nodes:
            if node_x["id"] == node_y["id"]:
                continue
            vector_y = node_y["embedding"]
            other_id = node_y["id"]

            if distance_metric == "euclidean":
                distance = np.linalg.norm(vector_x - vector_y)
            elif distance_metric == "cosine":
                distance = 1 - np.dot(vector_x, vector_y) / (np.linalg.norm(vector_x) * np.linalg.norm(vector_y))

            distance_info["distance"][str(other_id)] = distance
        distances.append(distance_info)

    return distances


def calculate_distances_between_users(users: UserModel, cluster_ids: Union[Dict[str, int], None] = None):
    """Calculate distances between users.

    Args:
        users: List[UserModel]. A list of UserModel objects.
        cluster_ids: Dict[str, int]. A dictionary of cluster ids. The key is the user id and the value is the cluster id.

    Returns:
        distances: List[dict]. A list of distances. Each element is a dictionary with the following keys:
            - "id": str. A cluster identifier for the node.
            - "distance": dict. A dictionary of distances to other nodes. The key is the node id and the value is the distance.
    """
    nodes = []
    for user in users:
        if cluster_ids is not None:
            cluster_id = cluster_ids[user.id]
        else:
            cluster_id = 0
        nodes.append({"id": user.id, "embedding": np.array(user.conversation_embedding), "cluster_id": cluster_id})

    distances = calculate_distances(nodes)

    return distances


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


def select_best_connected_nodes(edges: List[dict], num: int = 5) -> List[str]:
    """Select the best connected node.

    Args:
        edges: List[dict]. A list of edges. Each edge is a dictionary with the following keys
            - "source": str. The source node id.
            - "target": str. The target node id.
            - "distance": float. The distance between the source and target nodes.
    Returns:
        best_node: str. The best connected node.
    """
    G = generate_network_from_edges(edges)
    best_nodes = sorted(G.nodes, key=lambda x: len(list(nx.all_neighbors(G, x))), reverse=True)[:num]
    # best_nodes = max(G.nodes, key=lambda x: len(list(nx.all_neighbors(G, x))))[:5]
    return best_nodes
