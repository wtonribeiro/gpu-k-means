#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Clustering with Tree Saving (No Self-Loops) and PageRank Selection

@author: wtonr
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import json
from kmeans import perform_kmeans  # Import K-Means module


def generate_weighted_graph(n=20, p=0.3):
    """Generate an undirected weighted graph with edge weights between 0.2 and 1.0."""
    G = nx.erdos_renyi_graph(n, p)  # Create random undirected graph
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(2.0, 10.0), 2)  # Assign random weights
    return G


def hierarchical_clustering(graph, max_k=10):
    """Recursively apply K-Means clustering and build a hierarchy tree."""
    hierarchy = {}
    hierarchy_tree = nx.Graph()  # Hierarchical undirected graph (tree structure)

    def get_representative_node(subgraph):
        """Select the node with the highest PageRank as the representative."""
        page_ranks = nx.pagerank(subgraph)  # Compute PageRank scores
        return max(page_ranks, key=page_ranks.get)  # Return node with highest score

    def recursive_clustering(subgraph, cluster_id, parent=None):
        """Recursive clustering function."""
        if len(subgraph.nodes) == 1:
            node = list(subgraph.nodes)[0]
            hierarchy[node] = cluster_id
            hierarchy_tree.add_node(node)  # Add original node to the tree
            if parent and parent != node:  # Avoid self-loops
                weight = graph[parent][node]["weight"] if graph.has_edge(parent, node) else 2.0
                hierarchy_tree.add_edge(parent, node, weight=weight)
            return {node: cluster_id}

        local_max_k = min(max_k, len(subgraph.nodes))
        clusters = perform_kmeans(subgraph, local_max_k)

        if not clusters or len(set(clusters.values())) == 1:
            cluster_map = {node: f"{cluster_id}-{i}" for i, node in enumerate(subgraph.nodes)}
        else:
            cluster_map = {}
            unique_labels = set(clusters.values())

            representatives = {}  # Store representative nodes per cluster

            for cluster_label in unique_labels:
                sub_nodes = [node for node, label in clusters.items() if label == cluster_label]
                if not sub_nodes or len(sub_nodes) == len(subgraph.nodes):
                    cluster_map = {node: f"{cluster_id}-{i}" for i, node in enumerate(subgraph.nodes)}
                    break

                sub_cluster = subgraph.subgraph(sub_nodes)
                representative = get_representative_node(sub_cluster)  # Select best node
                representatives[cluster_label] = representative

                new_cluster_id = f"{cluster_id}-{cluster_label}"
                hierarchy_tree.add_node(representative)

                if parent and parent != representative:  # Avoid self-loops
                    weight = graph[parent][representative]["weight"] if graph.has_edge(parent, representative) else 2.0
                    hierarchy_tree.add_edge(parent, representative, weight=weight)

                cluster_map.update(recursive_clustering(sub_cluster, new_cluster_id, parent=representative))

        return cluster_map

    # Start clustering process
    root_node = get_representative_node(graph)  # Root representative
    hierarchy = recursive_clustering(graph, cluster_id="root", parent=root_node)

    # Assign hierarchy labels as node attributes in the original graph
    nx.set_node_attributes(graph, hierarchy, name="cluster")

    return hierarchy, hierarchy_tree, graph


def plot_hierarchy_tree(hierarchy_tree):
    """Plot the hierarchical clustering tree (without self-loops)."""
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(hierarchy_tree, seed=42, k=0.5)
    labels = nx.get_edge_attributes(hierarchy_tree, "weight")

    nx.draw(hierarchy_tree, pos, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=1200, font_size=10, font_weight="bold", alpha=0.8)
    nx.draw_networkx_edge_labels(hierarchy_tree, pos, edge_labels=labels, font_size=8)

    plt.title("Hierarchical Clustering Tree", fontsize=14)
    plt.show()


def save_hierarchical_tree(hierarchy_tree, filename="hierarchical_tree.json"):
    """Save the hierarchical clustering tree (without self-loops) to a JSON file."""
    data = {
        "nodes": list(hierarchy_tree.nodes()),
        "edges": [(u, v, hierarchy_tree[u][v]["weight"]) for u, v in hierarchy_tree.edges() if u != v]  # No self-loops
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Hierarchical clustering tree saved to {filename}")


# Example usage
if __name__ == "__main__":
    G = generate_weighted_graph(n=20, p=0.3)  # Generate a weighted graph
    hierarchy, hierarchy_tree, clustered_graph = hierarchical_clustering(G, max_k=5)

    print("Hierarchical Clusters:", hierarchy)

    # Plot tree
    plot_hierarchy_tree(hierarchy_tree)  

    # Save hierarchical tree (without self-loops)
    save_hierarchical_tree(hierarchy_tree)