#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified K-Means with Silhouette Score for optimal cluster selection
"""

import networkx as nx
import cupy as cp
import cudf
from cuml.cluster import KMeans

def graph_to_adjacency_matrix(graph):
    """Convert NetworkX graph to cuPy adjacency matrix"""
    adj_matrix = nx.to_numpy_array(graph, weight='weight')
    return cp.array(adj_matrix)

def compute_silhouette_score(X, labels):
    """Compute Silhouette Score manually on GPU"""
    n_samples = X.shape[0]
    labels = cp.asarray(labels)  # Ensure labels are Cupy array
    unique_labels = cp.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0.0

    # Vectorized pairwise distance calculation
    pairwise_dists = cp.linalg.norm(X[:, None] - X, axis=2)

    # Precompute cluster masks
    cluster_masks = {label.item(): (labels == label) for label in unique_labels}

    silhouette_scores = cp.zeros(n_samples)
    
    for i in range(n_samples):
        label_i = labels[i].item()  # Convert to Python int
        mask_a = cluster_masks[label_i]
        
        # Intra-cluster distance (a_i)
        a_i = cp.mean(pairwise_dists[i, mask_a])
        
        # Inter-cluster distances (b_i)
        b_values = []
        for label_j in cluster_masks:
            if label_j != label_i:
                b_values.append(cp.mean(pairwise_dists[i, cluster_masks[label_j]]))
        
        b_i = cp.min(cp.array(b_values)) if b_values else 0.0
        silhouette_scores[i] = (b_i - a_i) / cp.maximum(a_i, b_i)

    return cp.nanmean(silhouette_scores).item()  # Return Python float

def find_optimal_k(adj_matrix, max_k=32):
    """Find optimal clusters using Silhouette Score"""
    silhouette_scores = []
    k_range = list(range(2, max_k + 1))  # Convert to list for safe indexing
    
    adj_df = cudf.DataFrame(cp.asnumpy(adj_matrix))

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(adj_df)
            score = compute_silhouette_score(adj_matrix, clusters.to_cupy())
            silhouette_scores.append(score)
        except Exception as e:
            silhouette_scores.append(-1)  # Invalid score

    # Find k with highest silhouette score
    valid_scores = cp.array([s if s != -1 else -cp.inf for s in silhouette_scores])
    best_idx = cp.argmax(valid_scores).item()  # Convert to Python int
    best_k = k_range[best_idx]
    
    return best_k

def perform_kmeans(graph, max_k=32):
    """Perform K-Means clustering with Silhouette-optimized K"""
    adj_matrix = graph_to_adjacency_matrix(graph)
    optimal_k = find_optimal_k(adj_matrix, max_k)

    adj_df = cudf.DataFrame(cp.asnumpy(adj_matrix))
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(adj_df)

    # Representative selection logic (same as previous version)
    clusters_cp = cp.asarray(clusters.to_numpy())
    nodes_order = list(graph.nodes())
    cluster_to_rep = {}
    
    cluster_centers_cp = cp.asarray(kmeans.cluster_centers_.values)

    for label in range(optimal_k):
        mask = clusters_cp == label
        indices = cp.where(mask)[0]
        if len(indices) == 0:
            continue

        cluster_rows = adj_matrix[indices]
        center = cluster_centers_cp[label].reshape(1, -1)
        distances = cp.linalg.norm(cluster_rows - center, axis=1)
        min_idx = cp.argmin(distances).item()
        cluster_to_rep[label] = nodes_order[indices[min_idx].item()]

    cluster_assignment = {node: int(clusters.iloc[i]) for i, node in enumerate(nodes_order)}
    return cluster_assignment, cluster_to_rep


# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.erdos_renyi_graph(n=100, p=0.05)  # Random graph with 100 nodes
    clusters, reps = perform_kmeans(G, max_k=10)
    print("Cluster Assignments:", clusters)
    print("Cluster Representatives:", reps)
