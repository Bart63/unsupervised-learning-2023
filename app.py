import os
import cv2
import torch
from ml import ConvAutoEncoder
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

DATASET_FOLDER = 'dataset'
MODELS_FOLDER = 'models'

class EncoderModel:
    def __init__(self, ds='emnist'):
        self.ds = ds
        self.load_models()

    def load_models(self, type='cae'):
        sd_path = os.path.join(MODELS_FOLDER, f'{self.ds}_{type}.pth')
        device = torch.device('cpu')
        #self.model:ConvAutoEncoder = ConvAutoEncoder()
        self.model = ConvAutoEncoder().to(device)
        #self.model.load_state_dict(torch.load(sd_path))
        state_dict = torch.load(sd_path, map_location=device)
        self.model.load_state_dict(state_dict)
    
    def encode(self, img):
        return self.model.encoder(img) 


class PageProcessor:
    def __init__(self) -> None:
        self.emnist_model = EncoderModel('emnist')
        self.kmnist_model = EncoderModel('kmnist')
        self.load_pages()

    def load_pages(self):
        emnist_path = os.path.join(DATASET_FOLDER, 'emnist_page.png')
        self.emnist_page = cv2.imread(emnist_path, cv2.IMREAD_GRAYSCALE)
        kmnist_path = os.path.join(DATASET_FOLDER, 'kmnist_page.png')
        self.kmnist_page = cv2.imread(kmnist_path, cv2.IMREAD_GRAYSCALE)
    
    def encode_pages(self):
        encoded_emnist_letters = []
        encoded_kmnist_letters = []
        size = 32
        rows = 114
        columns = 80
        for row in range(rows):
            for column in range(columns):
                emnist_letter = self.emnist_page[size*row:size*(row+1), size*column:size*(column+1)]
                kmnist_letter = self.kmnist_page[size*row:size*(row+1), size*column:size*(column+1)]

                tensor = torch.from_numpy(emnist_letter.reshape(1, 1, size, size)).float()
                with torch.no_grad():
                    rep = self.emnist_model.encode(tensor).numpy()
                    rep = rep.reshape((32))
                    encoded_emnist_letters.append(rep)
                tensor = torch.from_numpy(kmnist_letter.reshape(1, 1, size, size)).float()
                with torch.no_grad():
                    rep = self.kmnist_model.encode(tensor).numpy()
                    rep = rep.reshape((32))
                    encoded_kmnist_letters.append(rep)
        self.encoded_emnist_letters = np.array(encoded_emnist_letters)
        self.encoded_kmnist_letters = np.array(encoded_kmnist_letters)

    def reduce_dimensions(self, n_components=2):
        umap_model = umap.UMAP(n_components=n_components)
        self.encoded_emnist_letters = umap_model.fit_transform(self.encoded_emnist_letters)
        self.encoded_kmnist_letters = umap_model.fit_transform(self.encoded_kmnist_letters)

    def emnist_encode(self, img):
        return self.emnist_model.encode(img)

    def clustering(self, sigma=1, num_clusters=48):
        distance_matrix = pairwise_distances(self.encoded_emnist_letters, metric='euclidean')
        affinity_matrix = np.exp(-distance_matrix / (2 * sigma ** 2))
        spectral_embedding = SpectralEmbedding(n_components=num_clusters, affinity='precomputed')
        reduced_data = spectral_embedding.fit_transform(affinity_matrix)
        kmeans = KMeans(n_clusters=num_clusters)
        emnist_clusters = kmeans.fit_predict(reduced_data)

        distance_matrix = pairwise_distances(self.encoded_kmnist_letters, metric='euclidean')
        affinity_matrix = np.exp(-distance_matrix / (2 * sigma ** 2))
        spectral_embedding = SpectralEmbedding(n_components=num_clusters, affinity='precomputed')
        reduced_data = spectral_embedding.fit_transform(affinity_matrix)
        kmeans = KMeans(n_clusters=num_clusters)
        kmnist_clusters = kmeans.fit_predict(reduced_data)
        self.emnist_clusters = emnist_clusters
        self.kmnist_clusters = kmnist_clusters
        return emnist_clusters, kmnist_clusters

    def reassign_classes(self):  # reassign classes based on cardinality
        class_counts = {}
        for class_val in self.emnist_clusters:
            if class_val in class_counts:
                class_counts[class_val] += 1
            else:
                class_counts[class_val] = 1
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_mapping = {class_val: idx for idx, (class_val, _) in enumerate(sorted_counts)}
        self.emnist_clusters = [class_mapping[class_val] for class_val in self.emnist_clusters]

        class_counts = {}
        for class_val in self.kmnist_clusters:
            if class_val in class_counts:
                class_counts[class_val] += 1
            else:
                class_counts[class_val] = 1
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_mapping = {class_val: idx for idx, (class_val, _) in enumerate(sorted_counts)}
        self.kmnist_clusters = [class_mapping[class_val] for class_val in self.kmnist_clusters]




if __name__ == '__main__':
    page_processor = PageProcessor()
    page_processor.encode_pages()
    page_processor.reduce_dimensions()
    umap_result = page_processor.encoded_emnist_letters
    emnist_clusters, kmnist_clusters = page_processor.clustering()
    page_processor.reassign_classes()

    # visualize clusters in 2D
    plt.figure(figsize=(8, 6))
    colors = plt.cm.Spectral(np.linspace(0, 1, 48))
    for cluster_label in range(48):
        plt.scatter(umap_result[emnist_clusters == cluster_label, 0], umap_result[emnist_clusters == cluster_label, 1],
                    c=colors[cluster_label],
                    label=f'Cluster {cluster_label}')

    plt.title('Spectral Clustering Results')
    plt.legend()
    plt.show()

    # calculate accuracy of model
    correct_predictions = [1 if true == pred else 0 for true, pred in zip(page_processor.kmnist_clusters,
                                                                          page_processor.emnist_clusters)]
    accuracy = sum(correct_predictions) / len(correct_predictions)
    print(accuracy)
