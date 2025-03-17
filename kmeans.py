#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means Clustering with Edge-Weight-Aware Distance Metric (GPU-Accelerated)
"""

import networkx as nx
import numpy as np
import cupy as cp
import cudf
from cuml.cluster import KMeans

def graph_to_distance_matrix(graph):
    """Convert edge weights to distance matrix (higher weights = shorter distance)"""
    adj_matrix = nx.to_numpy_array(graph, weight='weight', nonedge=0.0)
    
    # Invert weights (weight=10 → distance=0.1, weight=2 → distance=0.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        distance_matrix = np.divide(1.0, adj_matrix, where=adj_matrix!=0)
    
    # Set non-edges to max distance (10x largest edge-based distance)
    max_edge_distance = 0.5  # 1/min_weight (min_weight=2 in your case)
    distance_matrix[adj_matrix == 0] = max_edge_distance * 10
    
    return cp.array(distance_matrix)

def compute_davies_bouldin(graph, labels):
    """Edge-weight-aware DBI using original graph connectivity"""
    nodes = list(graph.nodes())
    labels = labels.get()  # Convert from CuPy array
    n_clusters = len(np.unique(labels))
    
    # Precompute adjacency dictionary for faster lookups
    adj_dict = {node: dict(graph[node]) for node in nodes}
    
    # Intra-cluster compactness
    intra_scores = []
    for cluster_id in range(n_clusters):
        cluster_nodes = [nodes[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        
        total_weight = 0.0
        valid_pairs = 0
        for i, u in enumerate(cluster_nodes):
            for v in cluster_nodes[i+1:]:
                if v in adj_dict[u]:
                    total_weight += adj_dict[u][v]['weight']
                    valid_pairs += 1
        
        avg_weight = total_weight / valid_pairs if valid_pairs > 0 else 0.0
        intra_scores.append(1.0 / avg_weight if avg_weight > 0 else float('inf'))
    
    # Inter-cluster separation
    dbi_scores = []
    for i in range(n_clusters):
        max_ratio = -np.inf
        cluster_i = [nodes[idx] for idx, lbl in enumerate(labels) if lbl == i]
        
        for j in range(n_clusters):
            if i == j:
                continue
                
            cluster_j = [nodes[idx] for idx, lbl in enumerate(labels) if lbl == j]
            
            total_inter = 0.0
            valid_pairs = 0
            for u in cluster_i:
                for v in cluster_j:
                    if v in adj_dict[u]:
                        total_inter += adj_dict[u][v]['weight']
                        valid_pairs += 1
            
            avg_inter = total_inter / valid_pairs if valid_pairs > 0 else 0.0
            separation = 1.0 / avg_inter if avg_inter > 0 else float('inf')
            ratio = (intra_scores[i] + intra_scores[j]) / separation
            max_ratio = max(max_ratio, ratio)
        
        dbi_scores.append(max_ratio)
    
    return np.mean(dbi_scores)

def find_optimal_k(graph, max_k=32):
    """Optimal k search using edge-aware DBI"""
    distance_matrix = graph_to_distance_matrix(graph)
    adj_df = cudf.DataFrame(cp.asnumpy(distance_matrix))
    
    dbi_scores = []
    k_range = range(2, max_k+1)
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(adj_df)
            dbi = compute_davies_bouldin(graph, clusters.to_cupy())
            dbi_scores.append(dbi)
        except:
            continue
    
    return k_range[np.argmin(dbi_scores)]

def perform_kmeans(graph, max_k=32):
    """Full clustering workflow with edge-weight awareness"""
    distance_matrix = graph_to_distance_matrix(graph)
    optimal_k = find_optimal_k(graph, max_k)
    
    adj_df = cudf.DataFrame(cp.asnumpy(distance_matrix))
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(adj_df)
    
    # Find representatives closest to centroids
    nodes = list(graph.nodes())
    clusters_cp = clusters.to_cupy()
    reps = {}
    for lbl in range(optimal_k):
        mask = (clusters_cp == lbl)
        indices = cp.where(mask)[0]
        if len(indices) == 0:
            continue
            
        # Get the node closest to cluster centroid
        centroid = kmeans.cluster_centers_[lbl]
        distances = cp.linalg.norm(distance_matrix[indices] - centroid, axis=1)
        min_idx = cp.argmin(distances)
        reps[lbl] = nodes[indices[min_idx].item()]
    
    return {n: int(clusters[i]) for i, n in enumerate(nodes)}, reps


# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.erdos_renyi_graph(n=100, p=0.05)  # Random graph with 100 nodes
    clusters, reps = perform_kmeans(G, max_k=10)
    print("Cluster Assignments:", clusters)
    print("Representative Nodes:", reps)
