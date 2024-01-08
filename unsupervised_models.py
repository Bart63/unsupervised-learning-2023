import umap
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding

import config as cfg


# Clustering

def k_means(arrays, n_clusters, random_state=0):
    clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    clusters = clustering.fit_predict(arrays)
    return clusters, clustering

def spectral_clustering(arrays, n_clusters, random_state=0):
    clustering = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
    clusters = clustering.fit_predict(arrays)
    return clusters, clustering

def dbscan(arrays, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clustering.fit_predict(arrays)
    return clusters, clustering

def gaussian_mixture(arrays, n_components=1, random_state=0):
    clustering = GaussianMixture(n_components, random_state=random_state)
    clusters = clustering.fit_predict(arrays)
    return clusters, clustering


# Dimensionality reduction

def pca_reduction(arrays, n_components=2, random_state=0):
    dim_red = PCA(n_components=n_components, random_state=random_state)
    reduced = dim_red.fit_transform(arrays)
    return reduced, dim_red

# Too long... do not use
def mds_reduction(arrays, n_components=2, random_state=0):
    dim_red = MDS(n_components=n_components, eps=0.1, max_iter=100, n_init=2, n_jobs=-1, random_state=random_state)
    reduced = dim_red.fit_transform(arrays)
    return reduced, dim_red

# No transform method
def tsne_reduction(arrays, n_components=2, random_state=0):
    dim_red = TSNE(n_components=n_components, random_state=random_state)
    reduced = dim_red.fit_transform(arrays)
    return reduced, dim_red

def umap_reduction(arrays, n_components=2, random_state=0):
    dim_red = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced = dim_red.fit_transform(arrays)
    return reduced, dim_red

# Too much data
# ValueError: Error in determining null-space with ARPACK. Error message: 'Factor is exactly singular'. Note that eigen_solver='arpack' can fail when the weight matrix is singular or otherwise ill-behaved. In that case, eigen_solver='dense' is recommended. See online documentation for more information
def local_linear_embedding(arrays, n_components=2, random_state=0):
    dim_red = LocallyLinearEmbedding(n_components=n_components, random_state=random_state, eigen_solver='dense')
    reduced = dim_red.fit_transform(arrays)
    return reduced, dim_red

# Too much data, eigensolver cannot ARPACK will not work
def isomap_reduction(arrays, n_components=2, random_state=0):
    dim_red = Isomap(n_components=n_components, random_state=random_state, eigen_solver='dense')
    reduced = dim_red.fit_transform(arrays)
    return reduced, dim_red

# reduction_models = {
#     'UMAP': umap_reduction,
#     'UMAP_and_spectral': umap_and_spectral
# }
# reduction_model = reduction_models.get(cfg.REDUCTION_MODEL, None)
# if reduction_model is None:
#     raise ValueError(f"Invalid reduction model: {cfg.REDUCTION_MODEL}")


# clustering_models = {
#     'k_means': k_means,
# }
# clustering_model = clustering_models.get(cfg.CLUSTERING_MODEL, None)
# if clustering_model is None:
#     raise ValueError(f"Invalid clustering model: {cfg.CLUSTERING_MODEL}")

