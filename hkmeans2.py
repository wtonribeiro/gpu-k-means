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

import warnings
warnings.filterwarnings("ignore")

def generate_weighted_graph(n=20, p=0.3):
    """Generate weighted undirected graph"""
    G = nx.erdos_renyi_graph(n, p)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(0.01, 10.0), 2)
    return G


def hierarchical_clustering(graph, max_k=32, parent_node=None, existing_tree=None, 
                           existing_hierarchy=None, original_graph=None):
    
    """Build hierarchical tree, extending an existing tree if provided."""
    hierarchy_tree = existing_tree if existing_tree is not None else nx.DiGraph()
    hierarchy = existing_hierarchy.copy() if existing_hierarchy is not None else {}
    original_graph = original_graph if original_graph is not None else graph  # Fallback

    if len(graph.nodes) == 0:
        return hierarchy, hierarchy_tree, graph

    adj_matrix = graph_to_adjacency_matrix(graph)
    nodes_order = list(graph.nodes())
    global_center = adj_matrix.mean(axis=0)
    distances = cp.linalg.norm(adj_matrix - global_center, axis=1)
    root_original = nodes_order[cp.argmin(distances).item()]
    
    # Generate synthetic root ID
    if parent_node is None:
        root_node = "root_0"
    else:
        root_node = f"{parent_node}_child_{root_original}"
        
        print(root_node)
    
    hierarchy_tree.add_node(root_node, original_id=root_original, label=root_original)

    # Connect to parent (if exists) using original_graph for edge weights
    if parent_node is not None:
        parent_original = hierarchy_tree.nodes[parent_node].get('original_id', parent_node)
        # Use original_graph instead of graph for cross-partition edges
        weight = original_graph[parent_original][root_original]['weight'] if original_graph.has_edge(parent_original, root_original) else 0.01
        hierarchy_tree.add_edge(parent_node, root_node, weight=weight)

    def recursive_clustering(subgraph, parent_synthetic_id, hierarchy_tree, hierarchy):
        if len(subgraph.nodes) == 0:
            return {}
        
        if len(subgraph.nodes) == 1:
            
            
            original_node = list(subgraph.nodes)[0]
            synthetic_id = f"{parent_synthetic_id}_leaf_{original_node}"
            
            print(synthetic_id)
            
            hierarchy_tree.add_node(synthetic_id, original_id=original_node, label=original_node)
            if parent_synthetic_id != synthetic_id:
                parent_original = hierarchy_tree.nodes[parent_synthetic_id].get('original_id', parent_synthetic_id)
                # Use original_graph instead of subgraph for cross-partition edges
                weight = original_graph[parent_original][original_node]['weight'] if original_graph.has_edge(parent_original, original_node) else 0.01
                hierarchy_tree.add_edge(parent_synthetic_id, synthetic_id, weight=weight)
            return {synthetic_id: original_node}

        local_max_k = min(max_k, len(subgraph.nodes))
        cluster_assignment, cluster_to_rep = perform_kmeans(subgraph, local_max_k)
        
        for label, rep_original in cluster_to_rep.items():
            
            rep_synthetic_id = f"{parent_synthetic_id}_cluster_{label}"
            
            print(rep_synthetic_id)
            
            hierarchy_tree.add_node(rep_synthetic_id, original_id=rep_original, label=rep_original)
            parent_original = hierarchy_tree.nodes[parent_synthetic_id].get('original_id', parent_synthetic_id)
            # Use original_graph instead of graph for cross-partition edges
            weight = original_graph[parent_original][rep_original]['weight'] if original_graph.has_edge(parent_original, rep_original) else 0.01
            hierarchy_tree.add_edge(parent_synthetic_id, rep_synthetic_id, weight=weight)
            
            remaining = [n for n, lbl in cluster_assignment.items() if lbl == label and n != rep_original]
            remaining_subgraph = subgraph.subgraph(remaining)
            components = list(nx.connected_components(remaining_subgraph))
            
            for comp in components:
                comp_subgraph = remaining_subgraph.subgraph(comp)
                recursive_clustering(comp_subgraph, rep_synthetic_id, hierarchy_tree, hierarchy)

        return hierarchy

    hierarchy = recursive_clustering(graph, root_node, hierarchy_tree, hierarchy)
    return hierarchy, hierarchy_tree, graph


def collapse_duplicate_labels(hierarchy_tree, original_graph):
    """Collapse nodes with same label connected by synthetic edges (weight=0.01)."""
    collapsed_tree = hierarchy_tree.copy()
    changed = True
    EPSILON = 1e-6  # For floating-point comparison
    
    while changed:
        changed = False
        nodes_to_process = list(collapsed_tree.nodes())
        
        for node in nodes_to_process:
            if not collapsed_tree.has_node(node):
                continue
            
            predecessors = list(collapsed_tree.predecessors(node))
            if len(predecessors) != 1:
                continue  # Only process nodes with one parent
            
            parent = predecessors[0]
            node_label = collapsed_tree.nodes[node]['label']
            parent_label = collapsed_tree.nodes[parent]['label']
            edge_weight = collapsed_tree[parent][node]['weight']
            
            # Condition: Same label and synthetic edge (weight ≈ 0.01)
            if (node_label == parent_label) and (abs(edge_weight - 0.01) < EPSILON):
                # Reconnect children to grandparent
                children = list(collapsed_tree.successors(node))
                for child in children:
                    child_weight = collapsed_tree[node][child]['weight']
                    # Preserve the child's edge weight
                    collapsed_tree.add_edge(parent, child, weight=child_weight)
                    collapsed_tree.remove_edge(node, child)
                
                # Remove the redundant node
                collapsed_tree.remove_node(node)
                changed = True
                break  # Restart iteration
        
    return collapsed_tree



def plot_hierarchy_tree(hierarchy_tree):
    """Plot the tree using original labels instead of synthetic IDs"""
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(hierarchy_tree, seed=42, k=0.5)
    labels = nx.get_node_attributes(hierarchy_tree, "label")  # Use "label" attribute
    edge_labels = nx.get_edge_attributes(hierarchy_tree, "weight")

    nx.draw(hierarchy_tree, pos, labels=labels, node_color="lightblue", edge_color="gray",
            node_size=1200, font_size=10, font_weight="bold", alpha=0.8)
    nx.draw_networkx_edge_labels(hierarchy_tree, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Hierarchical Clustering Tree (Original Labels)", fontsize=14)
    plt.show()

def save_hierarchical_tree(hierarchy_tree, filename="hierarchical_tree.json"):
    """Save tree with both synthetic IDs and original labels"""
    data = {
        "nodes": [
            {"id": node, "label": hierarchy_tree.nodes[node]["label"]} 
            for node in hierarchy_tree.nodes()
        ],
        "edges": [
            {"from": u, "to": v, "weight": hierarchy_tree[u][v]["weight"]} 
            for u, v in hierarchy_tree.edges() if u != v
        ]
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
