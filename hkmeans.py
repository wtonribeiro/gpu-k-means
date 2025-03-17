#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Hierarchical Clustering with Head-Body-Tail Support and Saving
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import json
import copy
from collections import defaultdict
from kmeans import perform_kmeans, graph_to_adjacency_matrix
import cupy as cp

import warnings
warnings.filterwarnings("ignore")

# Global for visualization
original_partitions = None

def generate_weighted_graph(n=20, p=0.3):
    """Generate weighted graph with head-body-tail partitions"""
    G = nx.erdos_renyi_graph(n, p)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(2.0, 10.0), 2)
    
    # Create partitions
    nodes = list(G.nodes())
    partitions = {
        'head': nodes[:len(nodes)//3],
        'body': nodes[len(nodes)//3:2*len(nodes)//3],
        'tail': nodes[2*len(nodes)//3:]
    }
    G.graph['partitions'] = partitions
    return G

def hierarchical_clustering(graph, partitions):
    """Main clustering function with three-tier processing"""
    hierarchy_tree = nx.Graph()
    node_registry = defaultdict(int)

    def get_unique_id(node_label):
        """Generate unique node ID with occurrence suffix"""
        node_registry[node_label] += 1
        return f"{node_label}-{node_registry[node_label]}"

    def process_partition(base_node, current_group, next_group):
        """Process partition extensions with component handling"""
        connected_nodes = [
            n for n in partitions[next_group]
            if graph.has_edge(base_node, n)
        ]
        if not connected_nodes:
            return

        extension_subgraph = graph.subgraph(connected_nodes)
        components = list(nx.connected_components(extension_subgraph))

        for comp_id, component in enumerate(components):
            comp_subgraph = graph.subgraph(component)
            _, comp_tree, _ = build_hierarchy(
                comp_subgraph, 
                parent=base_node,
                group=next_group,
                comp_id=comp_id
            )
            hierarchy_tree.add_edges_from(comp_tree.edges(data=True))
            hierarchy_tree.add_nodes_from(comp_tree.nodes())

    def build_hierarchy(subgraph, parent=None, group='head', comp_id=0):
        """Recursive hierarchy builder"""
        local_tree = nx.Graph()
        node_map = {}

        if len(subgraph.nodes) == 1:
            node = list(subgraph.nodes)[0]
            unique_id = get_unique_id(node)
            local_tree.add_node(unique_id, label=node)
            node_map[node] = unique_id
            
            if parent:
                weight = graph[parent][node]['weight'] if graph.has_edge(parent, node) else 2.0
                local_tree.add_edge(parent, unique_id, weight=weight)
            
            return {node: unique_id}, local_tree, node_map

        cluster_assignments, representatives = perform_kmeans(subgraph, min(32, len(subgraph.nodes)))
        
        for cluster_label, rep_node in representatives.items():
            unique_rep_id = get_unique_id(rep_node)
            local_tree.add_node(unique_rep_id, label=rep_node)
            node_map[rep_node] = unique_rep_id

            if parent:
                weight = graph[parent][rep_node]['weight'] if graph.has_edge(parent, rep_node) else 2.0
                local_tree.add_edge(parent, unique_rep_id, weight=weight)

            sub_nodes = [n for n, lbl in cluster_assignments.items() if lbl == cluster_label]
            subgraph_cluster = subgraph.subgraph(sub_nodes)
            
            _, sub_tree, sub_map = build_hierarchy(
                subgraph_cluster,
                parent=unique_rep_id,
                group=group
            )
            
            local_tree = nx.compose(local_tree, sub_tree)
            node_map.update(sub_map)

        return node_map, local_tree, node_map

    # Process head partition
    head_subgraph = graph.subgraph(partitions['head'])
    head_map, head_tree, _ = build_hierarchy(head_subgraph)
    hierarchy_tree = nx.compose(hierarchy_tree, head_tree)

    # Extend to body
    for head_node in partitions['head']:
        if head_node in head_map:
            process_partition(head_map[head_node], 'head', 'body')

    # Extend to tail
    for body_node in partitions['body']:
        process_partition(body_node, 'body', 'tail')

    return hierarchy_tree

def visualize_hierarchy(hierarchy_tree):
    """Color-coded visualization by partition"""
    global original_partitions
    plt.figure(figsize=(15, 10))
    pos = nx.nx_agraph.graphviz_layout(hierarchy_tree, prog='dot')
    
    node_colors = []
    for node in hierarchy_tree.nodes():
        label = hierarchy_tree.nodes[node]['label']
        if label in original_partitions['head']:
            node_colors.append('red')
        elif label in original_partitions['body']:
            node_colors.append('blue')
        else:
            node_colors.append('green')
    
    nx.draw(hierarchy_tree, pos, with_labels=True, 
            labels={n: hierarchy_tree.nodes[n]['label'] for n in hierarchy_tree.nodes()},
            node_color=node_colors, edge_color='gray',
            node_size=800, font_size=8)
    
    edge_labels = nx.get_edge_attributes(hierarchy_tree, 'weight')
    nx.draw_networkx_edge_labels(hierarchy_tree, pos, edge_labels=edge_labels)
    plt.title("Three-Tier Hierarchical Clustering", fontsize=14)
    plt.show()

def save_hierarchical_tree(hierarchy_tree, partitions, filename="hierarchical_tree.json"):
    """Save full hierarchy to JSON"""
    tree_data = {
        "nodes": [{"id": n, "label": hierarchy_tree.nodes[n]['label']} for n in hierarchy_tree.nodes()],
        "edges": [{"source": u, "target": v, "weight": w} 
                 for u, v, w in hierarchy_tree.edges(data='weight')],
        "partitions": {
            "head": [int(n) for n in partitions['head']],
            "body": [int(n) for n in partitions['body']],
            "tail": [int(n) for n in partitions['tail']]
        }
    }

    with open(filename, 'w') as f:
        json.dump(tree_data, f, indent=2)
    print(f"Hierarchy saved to {filename}")

if __name__ == "__main__":
    # Generate and process graph
    G = generate_weighted_graph(n=30)
    partitions = G.graph['partitions']
    global original_partitions
    original_partitions = copy.deepcopy(partitions)
    
    # Build hierarchy
    hierarchy_tree = hierarchical_clustering(G, partitions)
    
    # Visualize and save
    visualize_hierarchy(hierarchy_tree)
    save_hierarchical_tree(hierarchy_tree, original_partitions)
