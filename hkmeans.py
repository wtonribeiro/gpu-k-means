#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Clustering with Centroid Removal and Disconnected Subgraph Processing
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

def hierarchical_clustering(graph, max_k=32):
    """Build hierarchical clustering tree with centroid removal"""
    hierarchy = {}
    hierarchy_tree = nx.Graph()

    # Find root node as global centroid
    adj_matrix = graph_to_adjacency_matrix(graph)
    global_center = adj_matrix.mean(axis=0)
    nodes_order = list(graph.nodes())
    distances = cp.linalg.norm(adj_matrix - global_center, axis=1)
    root_node = nodes_order[cp.argmin(distances).item()]

    def recursive_clustering(subgraph, cluster_id, parent=None):
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
        cluster_assignment, cluster_to_rep = perform_kmeans(subgraph, local_max_k)
        cluster_map = {}

        for label, representative in cluster_to_rep.items():
            # Get nodes in current cluster
            sub_nodes = [n for n, lbl in cluster_assignment.items() if lbl == label]
            
            # Add representative to hierarchy
            hierarchy_tree.add_node(representative)
            if parent and parent != representative:
                weight = graph[parent][representative]["weight"] if graph.has_edge(parent, representative) else 2.0
                hierarchy_tree.add_edge(parent, representative, weight=weight)
            
            # Remove representative and process remaining nodes
            remaining_nodes = [n for n in sub_nodes if n != representative]
            if not remaining_nodes:
                cluster_map[representative] = f"{cluster_id}-{label}"
                continue

            # Process disconnected components
            remaining_subgraph = subgraph.subgraph(remaining_nodes)
            connected_components = list(nx.connected_components(remaining_subgraph))
            
            for cc_idx, component in enumerate(connected_components):
                component_subgraph = remaining_subgraph.subgraph(component)
                new_cluster_id = f"{cluster_id}-{label}-cc{cc_idx}"
                
                # Recursively process each component
                component_map = recursive_clustering(
                    component_subgraph, 
                    new_cluster_id, 
                    parent=representative
                )
                cluster_map.update(component_map)

        return cluster_map

    hierarchy = recursive_clustering(graph, "root", root_node)
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

if __name__ == "__main__":
    G = generate_weighted_graph(n=40, p=0.3)
    hierarchy, hierarchy_tree, clustered_graph = hierarchical_clustering(G, max_k=5)
    print("Hierarchical Clusters:", hierarchy)
    plot_hierarchy_tree(hierarchy_tree)
    save_hierarchical_tree(hierarchy_tree)