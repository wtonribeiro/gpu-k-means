#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 08:19:03 2025

@author: washigtonsegundo
"""

# Required Libraries (Install with the following commands if not installed):
# !pip install networkx numpy cupy rapidsai-cuml cudf matplotlib

import networkx as nx
import numpy as np
import cupy as cp  # For GPU acceleration
import cudf  # GPU-accelerated dataframe
from cuml.cluster import KMeans  # GPU-based K-Means
#import matplotlib.pyplot as plt


def graph_to_adjacency_matrix(graph):
    """Convert a NetworkX graph to a CuPy-based adjacency matrix considering weights"""
    adj_matrix = nx.to_numpy_array(graph, weight='weight')  # Preserve edge weights
    return cp.array(adj_matrix)  # Convert to CuPy array for GPU acceleration


def compute_davies_bouldin(X, labels):
    """Compute Davies-Bouldin Index manually on the GPU"""
    n_clusters = len(cp.unique(labels))
    cluster_centers = cp.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    intra_cluster_distances = cp.array([
        cp.linalg.norm(X[labels == i] - cluster_centers[i], axis=1).mean()
        for i in range(n_clusters)
    ])

    dbi_scores = []
    for i in range(n_clusters):
        scores = []
        for j in range(n_clusters):
            if i != j:
                inter_cluster_dist = cp.linalg.norm(cluster_centers[i] - cluster_centers[j])
                score = (intra_cluster_distances[i] + intra_cluster_distances[j]) / inter_cluster_dist
                scores.append(score)
        dbi_scores.append(max(scores))
    
    return cp.mean(cp.array(dbi_scores))


def find_optimal_k(adj_matrix, max_k=32):
    """Find the optimal number of clusters using the Elbow and DBI methods"""
    distortions = []
    dbi_scores = []
    k_range = range(2, max_k + 1)

    # Ensure k_range is valid
    if not k_range:
        return 2  # Default minimum

    # Convert CuPy array to cuDF DataFrame for cuML
    adj_df = cudf.DataFrame(cp.asnumpy(adj_matrix))

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(adj_df)

            distortions.append(kmeans.inertia_)
            dbi_scores.append(compute_davies_bouldin(adj_matrix, clusters).get())

        except Exception as e:
            pass
            

    if not dbi_scores:  # Prevent empty max() call
        return 2

    best_k = k_range[np.argmin(dbi_scores)]
    return best_k


def perform_kmeans(graph, max_k=32):
    """Perform K-Means clustering on a graph using GPU acceleration and return cluster assignments and representatives."""
    adj_matrix = graph_to_adjacency_matrix(graph)
    optimal_k = find_optimal_k(adj_matrix, max_k)

    # Convert CuPy array to cuDF DataFrame for cuML processing
    adj_df = cudf.DataFrame(cp.asnumpy(adj_matrix))

    # Run final K-Means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(adj_df)

    # Get cluster centers and convert to CuPy
    cluster_centers = cp.asarray(kmeans.cluster_centers_)
    clusters_cupy = clusters.to_cupy()  # Convert to CuPy array

    nodes = list(graph.nodes())
    representatives = {}

    for label in range(optimal_k):
        # Get indices of nodes in the current cluster
        mask = (clusters_cupy == label)
        indices = cp.where(mask)[0]
        if len(indices) == 0:
            continue  # Skip empty clusters
        
        # Extract rows from adjacency matrix for the cluster
        cluster_data = adj_matrix[indices]
        centroid = cluster_centers[label]
        
        # Compute distances and find closest node
        distances = cp.linalg.norm(cluster_data - centroid, axis=1)
        min_idx = cp.argmin(distances)
        representative_node = nodes[indices[min_idx].item()]
        representatives[label] = representative_node

    # Map nodes to their clusters
    cluster_assignment = {node: int(clusters[i]) for i, node in enumerate(nodes)}

    return cluster_assignment, representatives


# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.erdos_renyi_graph(n=100, p=0.05)  # Random graph with 100 nodes
    clusters, reps = perform_kmeans(G, max_k=10)
    print("Cluster Assignments:", clusters)
    print("Representative Nodes:", reps)
