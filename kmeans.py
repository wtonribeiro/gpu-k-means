#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated K-Means with Explicit Data Handling
"""

import networkx as nx
import cupy as cp
import cudf
from cuml.cluster import KMeans

def graph_to_adjacency_matrix(graph):
    """Convert graph to normalized distance matrix using edge weights"""
    nodes = list(graph.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize distance matrix with inverse weights
    dist_matrix = cp.full((n, n), cp.inf, dtype=cp.float32)
    cp.fill_diagonal(dist_matrix, 0)
    
    for u, v, data in graph.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        dist = 1 / data['weight']  # Higher weight = shorter distance
        dist_matrix[i, j] = dist_matrix[j, i] = cp.minimum(dist_matrix[i, j], dist)
    
    # Floyd-Warshall shortest paths
    for k in range(n):
        dist_matrix = cp.minimum(dist_matrix, dist_matrix[:, k:k+1] + dist_matrix[k:k+1, :])
    
    # Normalize
    return (dist_matrix - cp.mean(dist_matrix)) / cp.std(dist_matrix)

def compute_silhouette_score(X, labels):
    """GPU-accelerated silhouette score with explicit data handling"""
    labels = cp.asarray(labels)
    unique_labels = cp.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0.0

    intra_dists = cp.zeros(X.shape[0], dtype=cp.float32)
    inter_dists = cp.full(X.shape[0], cp.inf, dtype=cp.float32)
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]
        
        # Intra-cluster distances
        if cluster_points.shape[0] > 1:
            intra_dists[mask] = cp.mean(cluster_points[:, mask], axis=1)
        
        # Inter-cluster distances
        for other_label in unique_labels:
            if other_label != label and cp.sum(labels == other_label) > 0:
                inter_dists[mask] = cp.minimum(
                    inter_dists[mask],
                    cp.mean(cluster_points[:, labels == other_label], axis=1)
                )

    sil_scores = (inter_dists - intra_dists) / cp.maximum(intra_dists, inter_dists)
    return cp.nanmean(sil_scores).item()

def find_optimal_k(adj_matrix, max_k=32):
    """Find optimal clusters with explicit GPU data"""
    k_range = list(range(2, max_k + 1))
    silhouette_scores = []
    
    # Create cuDF DataFrame directly from cuPy array
    adj_df = cudf.DataFrame(adj_matrix.astype(cp.float32))
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(adj_df)
            score = compute_silhouette_score(adj_matrix, clusters.values)
            silhouette_scores.append(score)
        except:
            silhouette_scores.append(-1)
    
    best_idx = cp.nanargmax(cp.array(silhouette_scores)).item()
    return k_range[best_idx]

def perform_kmeans(graph, max_k=32):
    """K-Means with explicit GPU data handling"""
    adj_matrix = graph_to_adjacency_matrix(graph)
    optimal_k = find_optimal_k(adj_matrix, max_k)

    # Create cuDF DataFrame directly from cuPy array
    adj_df = cudf.DataFrame(adj_matrix.astype(cp.float32))
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(adj_df)
    
    # Convert cluster centers to cuPy array explicitly
    cluster_centers = cp.asarray(kmeans.cluster_centers_.values)  # Critical fix
    nodes = list(graph.nodes())
    cluster_to_rep = {}
    
    for label in range(optimal_k):
        mask = clusters.values == label
        if cp.sum(mask) == 0:
            continue
            
        # GPU-based indexing
        indices = cp.where(mask)[0]
        cluster_points = adj_matrix[indices]
        
        # Reshape using cupy
        centroid = cluster_centers[label].reshape(1, -1)  # Now works
        
        # Distance calculation on GPU
        distances = cp.linalg.norm(cluster_points - centroid, axis=1)
        rep_idx = cp.argmin(distances).item()
        cluster_to_rep[label] = nodes[indices[rep_idx].item()]
    
    return {node: int(clusters.iloc[i]) for i, node in enumerate(nodes)}, cluster_to_rep

import random

def generate_weighted_graph(n=20, p=0.3):
    """Generate weighted undirected graph"""
    G = nx.erdos_renyi_graph(n, p)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(2.0, 10.0), 2)
    return G


# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    G = generate_weighted_graph(n=100, p=0.3)
    clusters, reps = perform_kmeans(G, max_k=10)
    print("Cluster Assignments:", clusters)
    print("Cluster Representatives:", reps)
