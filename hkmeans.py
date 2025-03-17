#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Clustering with Centroid Removal and Disconnected Component Handling

@author: wtonr
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import json
from kmeans import perform_kmeans, graph_to_adjacency_matrix
import cupy as cp

import warnings
warnings.filterwarnings("ignore")


def generate_weighted_graph(n=20, p=0.3):
    """Generate an undirected weighted graph with edge weights between 0.2 and 1.0."""
    G = nx.erdos_renyi_graph(n, p)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(2.0, 10.0), 2)
    return G


def hierarchical_clustering(graph, max_k=32):
    """Recursively apply K-Means clustering with centroid removal and disconnected component handling."""
    hierarchy = {}
    hierarchy_tree = nx.Graph()

    def recursive_clustering(subgraph, cluster_id, parent=None):
        """Modified recursive clustering with centroid removal and component handling."""
        if len(subgraph.nodes) == 0:
            return {}

        if len(subgraph.nodes) == 1:
            node = list(subgraph.nodes)[0]
            hierarchy[node] = cluster_id
            hierarchy_tree.add_node(node)
            if parent and parent != node:
                weight = graph[parent][node]["weight"] if graph.has_edge(parent, node) else 2.0
                hierarchy_tree.add_edge(parent, node, weight=weight)
            return {node: cluster_id}

        local_max_k = min(max_k, len(subgraph.nodes))
        cluster_assignments, representatives = perform_kmeans(subgraph, local_max_k)

        cluster_map = {}
        for cluster_label in representatives:
            representative = representatives[cluster_label]
            sub_nodes = [node for node, lbl in cluster_assignments.items() if lbl == cluster_label]

            # Add representative to hierarchy and link to parent
            new_cluster_id = f"{cluster_id}-{cluster_label}"
            hierarchy_tree.add_node(representative)
            if parent and parent != representative:
                weight = graph[parent][representative]["weight"] if graph.has_edge(parent, representative) else 2.0
                hierarchy_tree.add_edge(parent, representative, weight=weight)

            # Remove representative from subsequent processing
            sub_nodes_without_rep = [n for n in sub_nodes if n != representative]
            
            if sub_nodes_without_rep:
                # Handle disconnected components in remaining subgraph
                remaining_subgraph = subgraph.subgraph(sub_nodes_without_rep)
                connected_components = list(nx.connected_components(remaining_subgraph))

                # Recursively process each component
                for comp_id, component in enumerate(connected_components):
                    comp_subgraph = remaining_subgraph.subgraph(component)
                    comp_cluster_id = f"{new_cluster_id}-comp{comp_id}"
                    cluster_map.update(
                        recursive_clustering(
                            comp_subgraph,
                            comp_cluster_id,
                            parent=representative
                        )
                    )

        return cluster_map

    # Initialize with full graph
    adj_matrix = graph_to_adjacency_matrix(graph)
    centroid = cp.mean(adj_matrix, axis=0)
    distances = cp.linalg.norm(adj_matrix - centroid, axis=1)
    min_idx = cp.argmin(distances).item()
    root_node = list(graph.nodes())[min_idx]

    hierarchy = recursive_clustering(graph, cluster_id="root", parent=root_node)
    nx.set_node_attributes(graph, hierarchy, name="cluster")
    return hierarchy, hierarchy_tree, graph


def plot_hierarchy_tree(hierarchy_tree):
    """Plot the hierarchical clustering tree."""
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(hierarchy_tree, seed=42, k=0.5)
    labels = nx.get_edge_attributes(hierarchy_tree, "weight")

    nx.draw(hierarchy_tree, pos, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=1200, font_size=10, font_weight="bold", alpha=0.8)
    nx.draw_networkx_edge_labels(hierarchy_tree, pos, edge_labels=labels, font_size=8)

    plt.title("Hierarchical Clustering Tree with Centroid Removal", fontsize=14)
    plt.show()


def save_hierarchical_tree(hierarchy_tree, filename="hierarchical_tree.json"):
    """Save the hierarchical tree to JSON."""
    data = {
        "nodes": list(hierarchy_tree.nodes()),
        "edges": [(u, v, hierarchy_tree[u][v]["weight"]) for u, v in hierarchy_tree.edges() if u != v]
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Hierarchical tree saved to {filename}")


if __name__ == "__main__":
    G = generate_weighted_graph(n=20, p=0.3)
    hierarchy, hierarchy_tree, clustered_graph = hierarchical_clustering(G, max_k=5)
    
    print("Hierarchical Clusters:", hierarchy)
    plot_hierarchy_tree(hierarchy_tree)
    save_hierarchical_tree(hierarchy_tree)
