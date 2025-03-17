#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated K-Means with Manual Silhouette Score
"""

import networkx as nx
import numpy as np
import cupy as cp
import cudf
from cuml.cluster import KMeans

def graph_to_distance_matrix(graph):
    """Convert edge weights to distance matrix (higher weights = shorter distance)"""
    adj_matrix = nx.to_numpy_array(graph, weight='weight', nonedge=0.0)
    
    # Convert weights to distances with inversion
    with np.errstate(divide='ignore', invalid='ignore'):
        distance_matrix = np.divide(1.0, adj_matrix, where=adj_matrix!=0)
    
    # Handle non-edges with maximum distance (10x minimum edge distance)
    max_distance = 10.0  # Corresponds to minimum weight of 0.1
    distance_matrix[adj_matrix == 0] = max_distance
    
    return cp.array(distance_matrix)

def compute_silhouette(distance_matrix, labels):
    """GPU-accelerated Silhouette Score implementation"""
    labels = cp.asarray(labels)
    unique_labels = cp.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = distance_matrix.shape[0]

    if n_clusters == 1:
        return 0.0  # All points in one cluster

    # Precompute masks for all clusters
    masks = [labels == lbl for lbl in unique_labels]
    
    # Intra-cluster distances (a_i)
    intra_dists = cp.zeros(n_samples)
    for i, mask in enumerate(masks):
        cluster_size = cp.sum(mask)
        if cluster_size <= 1:
            intra_dists[mask] = 0
            continue
        
        # Exclude self-distance in intra-cluster calculation
        cluster_distances = distance_matrix[mask][:, mask]
        intra_dists[mask] = (cp.sum(cluster_distances, axis=1) - cp.diag(cluster_distances)) / (cluster_size - 1)

    # Inter-cluster distances (b_i)
    inter_dists = cp.full(n_samples, cp.inf)
    for i, mask_i in enumerate(masks):
        for j, mask_j in enumerate(masks[i+1:], start=i+1):
            # Calculate mean distances between clusters
            inter_means = cp.mean(distance_matrix[mask_i][:, mask_j], axis=1)
            inter_dists[mask_i] = cp.minimum(inter_dists[mask_i], inter_means)
            
            # Symmetric update for the other cluster
            inter_means = cp.mean(distance_matrix[mask_j][:, mask_i], axis=1)
            inter_dists[mask_j] = cp.minimum(inter_dists[mask_j], inter_means)

    # Compute final silhouette scores
    sil_scores = (inter_dists - intra_dists) / cp.maximum(intra_dists, inter_dists)
    sil_scores = cp.nan_to_num(sil_scores, nan=0.0)
    return float(cp.mean(sil_scores))

def find_optimal_k(graph, max_k=32):
    """Find optimal cluster count using Silhouette Score"""
    distance_matrix = graph_to_distance_matrix(graph)
    adj_df = cudf.DataFrame(distance_matrix.get())

    best_score = -1.0
    best_k = 2

    for k in range(2, max_k+1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(adj_df)
            
            score = compute_silhouette(distance_matrix, clusters.to_cupy())
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            continue

    return best_k

def perform_kmeans(graph, max_k=32):
    """Complete K-Means workflow with Silhouette optimization"""
    distance_matrix = graph_to_distance_matrix(graph)
    optimal_k = find_optimal_k(graph, max_k)
    
    adj_df = cudf.DataFrame(distance_matrix.get())
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(adj_df)
    
    # Find representatives closest to centroids
    nodes = list(graph.nodes())
    clusters_cp = clusters.to_cupy()
    reps = {}
    for lbl in range(optimal_k):
        mask = clusters_cp == lbl
        indices = cp.where(mask)[0]
        if len(indices) == 0:
            continue
            
        centroid = kmeans.cluster_centers_[lbl]
        distances = cp.linalg.norm(distance_matrix[indices] - centroid, axis=1)
        min_idx = cp.argmin(distances)
        reps[lbl] = nodes[indices[min_idx].item()]
    
    return {n: int(clusters[i]) for i, n in enumerate(nodes)}, reps

# Example usage
if __name__ == "__main__":
    # Create sample graph
    G = nx.erdos_renyi_graph(n=100, p=0.1)
    for u, v in G.edges():
        G[u][v]['weight'] = round(np.random.uniform(2.0, 10.0), 2)
    
    # Perform clustering
    clusters, representatives = perform_kmeans(G)
    print("Cluster assignments:", clusters)
    print("Representative nodes:", representatives)
