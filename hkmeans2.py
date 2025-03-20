#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Hierarchical Clustering with support for incremental tree building.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import json
from kmeans import perform_kmeans, graph_to_adjacency_matrix
import cupy as cp

def generate_weighted_graph(n=20, p=0.3):
    """Generate weighted undirected graph"""
    G = nx.erdos_renyi_graph(n, p)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(2.0, 10.0), 2)
    return G

def hierarchical_clustering(graph, max_k=32, parent_node=None, existing_tree=None, existing_hierarchy=None):
    """Build hierarchical tree, extending an existing tree if provided."""
    hierarchy_tree = existing_tree if existing_tree is not None else nx.DiGraph()
    hierarchy = existing_hierarchy.copy() if existing_hierarchy is not None else {}

    if len(graph.nodes) == 0:
        return hierarchy, hierarchy_tree, graph

    adj_matrix = graph_to_adjacency_matrix(graph)
    nodes_order = list(graph.nodes())
    global_center = adj_matrix.mean(axis=0)
    distances = cp.linalg.norm(adj_matrix - global_center, axis=1)
    root_node = nodes_order[cp.argmin(distances).item()]

    if parent_node is not None:
        if parent_node not in hierarchy_tree:
            hierarchy_tree.add_node(parent_node)
        weight = graph[parent_node][root_node]['weight'] if graph.has_edge(parent_node, root_node) else 2.0
        hierarchy_tree.add_edge(parent_node, root_node, weight=weight)

    def recursive_clustering(subgraph, cluster_id, parent, hierarchy_tree, hierarchy):
        if len(subgraph.nodes) == 0:
            return {}
        if len(subgraph.nodes) == 1:
            node = list(subgraph.nodes)[0]
            hierarchy[node] = cluster_id
            hierarchy_tree.add_node(node)
            if parent != node:
                weight = subgraph[parent][node]['weight'] if subgraph.has_edge(parent, node) else 2.0
                hierarchy_tree.add_edge(parent, node, weight=weight)
            return {node: cluster_id}

        local_max_k = min(max_k, len(subgraph.nodes))
        cluster_assignment, cluster_to_rep = perform_kmeans(subgraph, local_max_k)
        cluster_map = {}

        for label, rep in cluster_to_rep.items():
            sub_nodes = [n for n, lbl in cluster_assignment.items() if lbl == label]
            hierarchy_tree.add_node(rep)
            if parent != rep:
                weight = subgraph[parent][rep]['weight'] if subgraph.has_edge(parent, rep) else 2.0
                hierarchy_tree.add_edge(parent, rep, weight=weight)
            remaining = [n for n in sub_nodes if n != rep]
            if not remaining:
                cluster_map[rep] = f"{cluster_id}-{label}"
                continue

            remaining_subgraph = subgraph.subgraph(remaining)
            components = list(nx.connected_components(remaining_subgraph))
            for idx, comp in enumerate(components):
                comp_subgraph = remaining_subgraph.subgraph(comp)
                new_cluster_id = f"{cluster_id}-{label}-cc{idx}"
                comp_map = recursive_clustering(comp_subgraph, new_cluster_id, rep, hierarchy_tree, hierarchy)
                cluster_map.update(comp_map)
        return cluster_map

    cluster_map = recursive_clustering(graph, "0", root_node, hierarchy_tree, hierarchy)
    hierarchy.update(cluster_map)
    nx.set_node_attributes(graph, hierarchy, name="cluster")
    return hierarchy, hierarchy_tree, graph

def plot_hierarchy_tree(hierarchy_tree):
    """Plot the hierarchical clustering tree"""
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(hierarchy_tree, seed=42, k=0.5)
    labels = nx.get_edge_attributes(hierarchy_tree, "weight")

    nx.draw(hierarchy_tree, pos, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=1200, font_size=10, font_weight="bold", alpha=0.8)
    nx.draw_networkx_edge_labels(hierarchy_tree, pos, edge_labels=labels, font_size=8)

    plt.title("Hierarchical Clustering Tree with Centroid Removal", fontsize=14)
    plt.show()

def save_hierarchical_tree(hierarchy_tree, filename="hierarchical_tree.json"):
    """Save hierarchical tree to JSON"""
    data = {
        "nodes": list(hierarchy_tree.nodes()),
        "edges": [(u, v, hierarchy_tree[u][v]["weight"]) for u, v in hierarchy_tree.edges() if u != v]
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Hierarchical tree saved to {filename}")

# Remaining functions (plot_hierarchy_tree, save_hierarchical_tree) remain the same

# Example Usage
if __name__ == "__main__":
    G = generate_weighted_graph(n=40, p=0.3)
    hierarchy, hierarchy_tree, clustered_graph = hierarchical_clustering(G, max_k=5)
    print("Hierarchical Clusters:", hierarchy)
    plot_hierarchy_tree(hierarchy_tree)
    save_hierarchical_tree(hierarchy_tree)
