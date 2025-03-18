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
    adj_matrix = graph_to_adjacency_matrix(graph)  # Returns cuPy array
    optimal_k = find_optimal_k(adj_matrix, max_k)

    # Convert cuPy array to cuDF DataFrame via numpy (temporary workaround)
    adj_df = cudf.DataFrame(cp.asnumpy(adj_matrix))

    # Run K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(adj_df)

    # Get cluster centers (already in CPU memory due to .asnumpy())
    cluster_centers = kmeans.cluster_centers_.values  # Get as numpy array
    cluster_centers_cp = cp.asarray(cluster_centers)  # Convert to cuPy array

    # Get node order and initialize representatives
    nodes_order = list(graph.nodes())
    cluster_to_rep = {}
    
    # Convert clusters to cuPy array
    clusters_cp = cp.asarray(clusters.to_numpy())  # Explicit conversion

    for label in range(optimal_k):
        # Get indices of nodes in this cluster
        mask = clusters_cp == label
        indices = cp.where(mask)[0]

        if len(indices) == 0:
            continue

        # Get rows from original adjacency matrix (cuPy array)
        cluster_rows = adj_matrix[indices]

        # Get cluster center and reshape for broadcasting
        center = cluster_centers_cp[label].reshape(1, -1)

        # Compute distances (all GPU operations)
        distances = cp.linalg.norm(cluster_rows - center, axis=1)

        # Find representative node
        min_idx = cp.argmin(distances).item()
        representative_node = nodes_order[indices[min_idx].item()]
        cluster_to_rep[label] = representative_node

    # Convert clusters to dictionary
    cluster_assignment = {
        node: int(clusters.iloc[i])  # Directly use cuDF Series index
        for i, node in enumerate(nodes_order)
    }

    return cluster_assignment, cluster_to_rep

# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.erdos_renyi_graph(n=100, p=0.05)  # Random graph with 100 nodes
    clusters, reps = perform_kmeans(G, max_k=10)
    print("Cluster Assignments:", clusters)
    print("Cluster Representatives:", reps)