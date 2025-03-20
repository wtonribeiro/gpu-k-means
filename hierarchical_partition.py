#!/usr/bin/env python3
import networkx as nx
import random
from hkmeans2 import hierarchical_clustering, plot_hierarchy_tree, save_hierarchical_tree, collapse_duplicate_labels

import warnings
warnings.filterwarnings("ignore")

def partition_nodes(graph):
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    n = len(nodes)
    return (nodes[:int(0.2*n)],
            nodes[int(0.2*n):int(0.8*n)],
            nodes[int(0.8*n):])

def get_leaves(tree):
    return [node for node in tree.nodes() if tree.out_degree(node) == 0]


def h_tree(G, head, body, tail):
    # Step 1: Cluster head (pass original_graph=G)
    head_sub = G.subgraph(head)
    hierarchy, tree, _ = hierarchical_clustering(head_sub, max_k=5, original_graph=G)

    # Step 2: Process body for each head leaf (pass original_graph=G)
    leaves = get_leaves(tree)
    for leaf in leaves:
        original_leaf_id = tree.nodes[leaf].get('original_id', leaf)
        connected = [nbr for nbr in G.neighbors(original_leaf_id) if nbr in body]
        if not connected:
            continue
        body_sub = G.subgraph(connected)
        hierarchy, tree, _ = hierarchical_clustering(
            body_sub, max_k=5, parent_node=leaf, 
            existing_tree=tree, existing_hierarchy=hierarchy, 
            original_graph=G  # <-- Pass original graph
        )
    
    # Step 3: Process tail similarly (pass original_graph=G)
    leaves = get_leaves(tree)
    for leaf in leaves:
        original_leaf_id = tree.nodes[leaf].get('original_id', leaf)
        connected = [nbr for nbr in G.neighbors(original_leaf_id) if nbr in tail]
        if not connected:
            continue
        tail_sub = G.subgraph(connected)
        hierarchy, tree, _ = hierarchical_clustering(
            tail_sub, max_k=5, parent_node=leaf, 
            existing_tree=tree, existing_hierarchy=hierarchy,
            original_graph=G  # <-- Pass original graph
        )
    
    
    collapsed_tree = collapse_duplicate_labels(tree, G)
    
    plot_hierarchy_tree(collapsed_tree)
    save_hierarchical_tree(collapsed_tree)


if __name__ == "__main__":
    # Generate or load your graph
    G = nx.erdos_renyi_graph(25, 0.3)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(0.01, 10.0), 2)

    head, body, tail = partition_nodes(G)
    
    
    h_tree(G, head, body, tail)
