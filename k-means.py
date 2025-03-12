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
from cuml.metrics import silhouette_score  # GPU-based Silhouette Score
import matplotlib.pyplot as plt

def graph_to_adjacency_matrix(graph):
    """Convert a NetworkX graph to a CuPy-based adjacency matrix"""
    adj_matrix = nx.to_numpy_array(graph)
    return cp.array(adj_matrix)  # Convert to CuPy array for GPU acceleration

def find_optimal_k(adj_matrix, max_k=10):
    """Find the optimal number of clusters using the Elbow and Silhouette methods"""
    distortions = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    # Convert CuPy array to cuDF DataFrame for cuML
    adj_df = cudf.DataFrame(cp.asnumpy(adj_matrix))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(adj_df)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(adj_df, clusters))
    
    # Plot the Elbow Method
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, distortions, marker='o', label='WCSS (Elbow)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.title('Elbow Method')
    plt.legend()
    plt.show()
    
    # Find the best k from the silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal K found: {best_k}")
    return best_k

def perform_kmeans(graph, max_k=10):
    """Perform K-Means clustering on a graph using GPU acceleration"""
    adj_matrix = graph_to_adjacency_matrix(graph)
    optimal_k = find_optimal_k(adj_matrix, max_k)
    
    # Convert CuPy array to cuDF DataFrame for cuML processing
    adj_df = cudf.DataFrame(cp.asnumpy(adj_matrix))
    
    # Run final K-Means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(adj_df)
    
    # Convert results to dictionary {node: cluster}
    cluster_assignment = {node: int(clusters[i]) for i, node in enumerate(graph.nodes())}
    return cluster_assignment

# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.erdos_renyi_graph(n=100, p=0.05)  # Random graph with 100 nodes
    clusters = perform_kmeans(G, max_k=10)
    print("Cluster Assignments:", clusters)
