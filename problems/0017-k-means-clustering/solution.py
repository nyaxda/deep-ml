import numpy as np
from sklearn.cluster import KMeans

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	km = KMeans(n_clusters=k, init=initial_centroids, max_iter=max_iterations)
	km.fit(points)
	final_centroids = [tuple(r) for r in km.cluster_centers_]
	return final_centroids