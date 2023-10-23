import umap
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

import config as cfg


def umap_reduction(emnist, kmnist, n_components=cfg.REDUCTION_DIM):
    umap_model = umap.UMAP(n_components=n_components)
    emnist = umap_model.fit_transform(emnist)
    kmnist = umap_model.fit_transform(kmnist)
    return emnist, kmnist

def k_means(emnist, kmnist):
    kmeans = KMeans(n_clusters=47)
    emnist_clusters = kmeans.fit_predict(emnist)
    kmnist_clusters = kmeans.fit_predict(kmnist)
    return emnist_clusters, kmnist_clusters


def spectral_embedding(emnist, kmnist, sigma=1):
    spectral_embedding = SpectralEmbedding(n_components=cfg.REDUCTION_DIM, affinity='precomputed')

    distance_matrix = pairwise_distances(emnist, metric='euclidean')
    affinity_matrix = np.exp(-distance_matrix / (2 * sigma ** 2))
    emnist = spectral_embedding.fit_transform(affinity_matrix)

    distance_matrix = pairwise_distances(kmnist, metric='euclidean')
    affinity_matrix = np.exp(-distance_matrix / (2 * sigma ** 2))
    kmnist = spectral_embedding.fit_transform(affinity_matrix)

    return emnist, kmnist


def umap_and_spectral(emnist, kmnist):
    emnist, kmnist = umap_reduction(emnist, kmnist, n_components=cfg.REDUCTION_DIM * 4)
    emnist, kmnist = spectral_embedding(emnist, kmnist)
    return emnist, kmnist


reduction_models = {
    'spectral_embedding': spectral_embedding,
    'UMAP': umap_reduction,
    'UMAP_and_spectral': umap_and_spectral
}
reduction_model = reduction_models.get(cfg.REDUCTION_MODEL, None)
if reduction_model is None:
    raise ValueError(f"Invalid reduction model: {cfg.REDUCTION_MODEL}")


clustering_models = {
    'k_means': k_means,
}
clustering_model = clustering_models.get(cfg.CLUSTERING_MODEL, None)
if clustering_model is None:
    raise ValueError(f"Invalid clustering model: {cfg.CLUSTERING_MODEL}")

