# %%
import os
import cv2
import torch
import glob
from helpers import is_whitespace
from ml import ConvAutoEncoder, VarConvAutoEncoder
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from sklearn.metrics import silhouette_score 
from PIL import Image
import sklearn

from unsupervised_models import *
# %%

DATASET_FOLDER = os.path.join('dataset', '1char_spl')
MODELS_FOLDER = 'models/ae'

# %%
class EncoderModel:
    def __init__(self, ds='emnist', base='cae', type='bce', device='cuda'):
        self.ds = ds
        self.base = base
        self.type = type
        self.fname = f'{self.ds}_{self.base}_{self.type}.pth'
        self.device = device
        self.load_models()

    def load_models(self):
        sd_path = os.path.join(MODELS_FOLDER, self.fname)
        
        if self.base == 'cae':
            self.model:ConvAutoEncoder = ConvAutoEncoder().to(self.device)
        elif self.base == 'vcae':
            self.model:VarConvAutoEncoder = VarConvAutoEncoder().to(self.device)
        state_dict = torch.load(sd_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def encode(self, img):
        with torch.no_grad():
            res = self.model.encoder(img)
        if self.base == 'cae':
            return np.array([res.cpu().numpy().reshape(8)])
        return np.array([res[0].cpu().numpy().reshape(8), res[1].cpu().numpy().reshape(8)])
        
    def decode(self, args):
        if self.base == 'cae':
            assert len(args) == 1
            
            tensor = torch.from_numpy(args.reshape(1, 1, -1)).float().to(self.device)
            with torch.no_grad():
                rep = self.model.decoder(tensor)

        elif self.base == 'vcae':
            assert len(args) == 2

            mu, sigma = args
            mu = torch.from_numpy(mu.reshape(1, 1, -1)).float()
            sigma = torch.from_numpy(sigma.reshape(1, 1, -1)).float()
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            z = z.to(self.device)
            with torch.no_grad():
                rep = self.model.decoder(z)
        if 'cuda' in self.device:
            rep = rep.detach().cpu()
        rep = rep.numpy().reshape((-1))
        return rep

# %%
    
class PageProcessor:
    def __init__(self, base='cae', type='bce') -> None:
        self.base = base
        self.emnist_model = EncoderModel('emnist', base, type)
        self.kmnist_model = EncoderModel('kmnist', base, type)
        self.load_pages()
        self._divide_characters()

    def _load_pages(self, type='emnist'):
        transform = ToTensor()
        path_wildcard = os.path.join(DATASET_FOLDER, f'{type}_page_*.png')
        paths = sorted(glob.glob(path_wildcard))
        pages = [transform(Image.open(path)).numpy()[0] for path in paths]
        return pages
    
    def _divide_characters(self, size=32, rows=114, columns=80):
        self.emnist_letter_imgs = []
        self.kmnist_letter_imgs = []
        for emnist_page, kmnist_page in zip(self.emnist_pages, self.kmnist_pages):
            for row in range(rows):
                for column in range(columns):
                    emnist_letter = emnist_page[size*row:size*(row+1), size*column:size*(column+1)]
                    kmnist_letter = kmnist_page[size*row:size*(row+1), size*column:size*(column+1)]

                    if is_whitespace(emnist_letter) or is_whitespace(kmnist_letter):
                        continue

                    self.emnist_letter_imgs.append(emnist_letter)
                    self.kmnist_letter_imgs.append(kmnist_letter)

    def load_pages(self):
        self.emnist_pages = self._load_pages('emnist')
        self.kmnist_pages = self._load_pages('kmnist')
    
    def encode_pages(self, size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded_emnist_letters = []
        encoded_kmnist_letters = []
        
        for emnist_letter, kmnist_letter in zip(self.emnist_letter_imgs, self.kmnist_letter_imgs):
            tensor = torch.from_numpy(emnist_letter.reshape(1, 1, size, size)).float().to(device)
            with torch.no_grad():
                rep = self.emnist_model.encode(tensor)
            encoded_emnist_letters.append(rep)

            tensor = torch.from_numpy(kmnist_letter.reshape(1, 1, size, size)).float().to(device)
            with torch.no_grad():
                rep = self.kmnist_model.encode(tensor)
            encoded_kmnist_letters.append(rep)
        self.encoded_emnist_letters = np.array(encoded_emnist_letters)
        self.encoded_kmnist_letters = np.array(encoded_kmnist_letters)

    def reduce_dimensions(self, reduction_function):
        self.emnist_reduced_letters, self.emnist_reduction_model = reduction_function(self.encoded_emnist_letters[:, 0, :])
        self.kmnist_reduced_letters, self.kmnist_reduction_model = reduction_function(self.encoded_kmnist_letters[:, 0, :])

    def emnist_encode(self, img):
        return self.emnist_model.encode(img)

    def clustering(self, clustering_model):
        self.emnist_clusters, self.kmnist_clusters = clustering_model(self.encoded_emnist_letters, self.encoded_kmnist_letters)
        return self.emnist_clusters, self.kmnist_clusters

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

# %% Cosine sim
from scipy.spatial.distance import cosine
# %% Ładowanie modeli i stron z obu alfabetów
page_processor = PageProcessor(base='cae', type='bce')
# %% Kodowanie znaków z usuwaniem znaków białych
page_processor.encode_pages()

# %% Trenowanie metod klasteryzujących: K-Means, Spectral Clustering, Gaussian Mixture, DBSCAN 
from unsupervised_models import *

# %%
encoded_emnist_letters = page_processor.encoded_emnist_letters[:, 0, :]
encoded_kmnist_letters = page_processor.encoded_kmnist_letters[:, 0, :]
# %%
# Determinacja liczby klastrów jest problematyczna
# c1, a1 = dbscan(encoded_emnist_letters, 0.3)
# c2, a2 = dbscan(encoded_kmnist_letters, 0.3)
# %%
res = gaussian_mixture(encoded_emnist_letters, 47)[0]
# %%
import joblib
# %%
for base, type in [['cae','bce'], ['cae','mse'], ['cae', 'ssim'], ['vcae', 'bce'], ['vcae', 'mse']]:
    print(f'Base: {base}, type: {type}')
    page_processor = PageProcessor(base=base, type=type)
    page_processor.encode_pages()
    encoded_emnist_letters = page_processor.encoded_emnist_letters[:, 0, :]
    encoded_kmnist_letters = page_processor.encoded_kmnist_letters[:, 0, :]
    
    results = {
    'k_means': {},
    'gaussian_mixture': {},
    'nb_clusters': list(range(40, 51))
    }
    for cluster_name, cluster_fn in zip(['k_means', 'gaussian_mixture'], [k_means, gaussian_mixture]):
        emnist_sil_scores = []
        kmnist_sil_scores = []
        emnist_models = []
        kmnist_models = []
        emnist_labels = []
        kmnist_labels = []
        for nb_clusters in range(40, 51):
            c1, a1 = cluster_fn(encoded_emnist_letters, nb_clusters)
            c2, a2 = cluster_fn(encoded_kmnist_letters, nb_clusters)
            silhouette_avg_c1 = silhouette_score(encoded_emnist_letters, c1, random_state=0)
            silhouette_avg_c2 = silhouette_score(encoded_emnist_letters, c2, random_state=0)

            emnist_sil_scores.append(silhouette_avg_c1)
            kmnist_sil_scores.append(silhouette_avg_c2)
            emnist_models.append(a1)
            kmnist_models.append(a2)
            emnist_labels.append(c1)
            kmnist_labels.append(c2)

        results[cluster_name] = {
            'emnist_sil_scores': emnist_sil_scores,
            'kmnist_sil_scores': kmnist_sil_scores,
            'emnist_models': emnist_models,
            'kmnist_models': kmnist_models,
            'emnist_labels': emnist_labels,
            'kmnist_labels': kmnist_labels
        }

    joblib.dump(results, f'results_{base}_{type}_spl.joblib')
# %%
DATASET_FOLDER = os.path.join('dataset', '1char_spl')
for base, type in [['cae','bce'], ['cae','mse'], ['cae', 'ssim'], ['vcae', 'bce'], ['vcae', 'mse']]:
    print(f'Base: {base}, type: {type}')
    page_processor = PageProcessor(base=base, type=type)
    page_processor.encode_pages()
    encoded_emnist_letters = page_processor.encoded_emnist_letters[:, 0, :]
    encoded_kmnist_letters = page_processor.encoded_kmnist_letters[:, 0, :]
    
    results = {
    'k_means': {},
    'gaussian_mixture': {},
    'nb_clusters': list(range(40, 51))
    }
    for cluster_name, cluster_fn in zip(['k_means', 'gaussian_mixture'], [k_means, gaussian_mixture]):
        emnist_sil_scores = []
        kmnist_sil_scores = []
        emnist_models = []
        kmnist_models = []
        emnist_labels = []
        kmnist_labels = []
        for nb_clusters in range(40, 51):
            c1, a1 = cluster_fn(encoded_emnist_letters, nb_clusters)
            c2, a2 = cluster_fn(encoded_kmnist_letters, nb_clusters)
            silhouette_avg_c1 = silhouette_score(encoded_emnist_letters, c1, random_state=0)
            silhouette_avg_c2 = silhouette_score(encoded_emnist_letters, c2, random_state=0)

            emnist_sil_scores.append(silhouette_avg_c1)
            kmnist_sil_scores.append(silhouette_avg_c2)
            emnist_models.append(a1)
            kmnist_models.append(a2)
            emnist_labels.append(c1)
            kmnist_labels.append(c2)

        results[cluster_name] = {
            'emnist_sil_scores': emnist_sil_scores,
            'kmnist_sil_scores': kmnist_sil_scores,
            'emnist_models': emnist_models,
            'kmnist_models': kmnist_models,
            'emnist_labels': emnist_labels,
            'kmnist_labels': kmnist_labels
        }

    joblib.dump(results, f'results_{base}_{type}_spl.joblib')
# %%
from collections import Counter
# %%
# Match clusters c1 and c2 based on number of counts in Counter
# Create mapping from c1 to c2
def match_clusters(c1, c2):
    counts_c1 = Counter(c1)
    counts_c2 = Counter(c2)
    c1_to_c2 = {}

    counts_c1 = sorted(counts_c1, key=counts_c1.get)
    counts_c2 = sorted(counts_c2, key=counts_c2.get)
    # sort both counters by values. Match the biggest one with the biggest
    for key_c1, key_c2 in zip(counts_c1, counts_c2):
        c1_to_c2[key_c1] = key_c2
    return c1_to_c2
# %%
import json
import joblib

dataset_type = ['1char_all', '1char_spl', '1char_none']
for dstype in dataset_type:
    DATASET_FOLDER = os.path.join('dataset', dstype)
    ress = glob.glob(f'results*_{dstype.split("_")[-1]}.joblib')

    for r in ress:
        print(r)
        base_name = '_'.join(r.split('.')[0].split('_')[1:-1])
        base_model = base_name.split('_')[0]
        loss_name = base_name.split('_')[1]
        page_processor = PageProcessor(base=base_model, type=loss_name)
        page_processor.encode_pages()
        encoded_emnist_letters = page_processor.encoded_emnist_letters[:, 0, :]
        encoded_kmnist_letters = page_processor.encoded_kmnist_letters[:, 0, :]
        
        d = joblib.load(r)
        
        k_means_cum_score = [
            s1*s2
            for s1, s2 in zip(d['k_means']['kmnist_sil_scores'], d['k_means']['emnist_sil_scores'])
        ]
        gaussian_mixture_cum_score = [
            s1*s2
            for s1, s2 in zip(d['gaussian_mixture']['kmnist_sil_scores'], d['gaussian_mixture']['emnist_sil_scores'])
        ]
        k_means_best = np.argmax(k_means_cum_score)
        print(40+k_means_best)
        gaussian_mixture_best = np.argmax(gaussian_mixture_cum_score)
        print(40+gaussian_mixture_best)
        print()

        k_means_best_lables_kmnist = d['k_means']['kmnist_labels'][k_means_best]
        k_means_best_lables_emnist = d['k_means']['emnist_labels'][k_means_best]
        
        k_means_kmnist_to_emnist = match_clusters(k_means_best_lables_kmnist, k_means_best_lables_emnist)
        k_means_best_lables_emnist = [k_means_kmnist_to_emnist[cluster] for cluster in k_means_best_lables_kmnist]
        
        k_means_medoids_emnist = {}
        for cluster_label in np.unique(k_means_best_lables_emnist):
            cluster_data = encoded_emnist_letters[k_means_best_lables_emnist == cluster_label]
            distance_matrix = np.linalg.norm(cluster_data[:, np.newaxis] - cluster_data, axis=-1)
            medoid_index = np.argmin(np.sum(distance_matrix, axis=1))
            medoid = cluster_data[medoid_index]
            k_means_medoids_emnist[cluster_label] = medoid

        k_means_medoids_kmnist = {}
        for cluster_label in np.unique(k_means_best_lables_kmnist):
            cluster_data = encoded_kmnist_letters[k_means_best_lables_kmnist == cluster_label]
            distance_matrix = np.linalg.norm(cluster_data[:, np.newaxis] - cluster_data, axis=-1)
            medoid_index = np.argmin(np.sum(distance_matrix, axis=1))
            medoid = cluster_data[medoid_index]
            k_means_medoids_kmnist[cluster_label] = medoid
        
        k_means_out = {
            'medoids_emnist': k_means_medoids_emnist,
            'medoids_kmnist': k_means_medoids_kmnist,
            'kmnist_to_emnist': k_means_kmnist_to_emnist,
            'lables_kmnist': k_means_best_lables_kmnist,
            'lables_emnist': k_means_best_lables_emnist,
            'model_kmnist': d['k_means']['kmnist_models'][k_means_best],
            'model_emnist': d['k_means']['emnist_models'][k_means_best],
        }

        gaussian_mixture_best_lables_kmnist = d['gaussian_mixture']['kmnist_labels'][gaussian_mixture_best]
        gaussian_mixture_best_lables_emnist = d['gaussian_mixture']['emnist_labels'][gaussian_mixture_best]

        gaussian_mixture_kmnist_to_emnist = match_clusters(gaussian_mixture_best_lables_kmnist, gaussian_mixture_best_lables_emnist)
        gaussian_mixture_best_lables_emnist = [gaussian_mixture_kmnist_to_emnist[cluster] for cluster in gaussian_mixture_best_lables_kmnist]

        gaussian_mixture_medoids_emnist = {}
        for cluster_label in np.unique(gaussian_mixture_best_lables_emnist):
            cluster_data = encoded_emnist_letters[gaussian_mixture_best_lables_emnist == cluster_label]
            distance_matrix = np.linalg.norm(cluster_data[:, np.newaxis] - cluster_data, axis=-1)
            medoid_index = np.argmin(np.sum(distance_matrix, axis=1))
            medoid = cluster_data[medoid_index]
            gaussian_mixture_medoids_emnist[cluster_label] = medoid

        gaussian_mixture_medoids_kmnist = {}
        for cluster_label in np.unique(gaussian_mixture_best_lables_kmnist):
            cluster_data = encoded_kmnist_letters[gaussian_mixture_best_lables_kmnist == cluster_label]
            distance_matrix = np.linalg.norm(cluster_data[:, np.newaxis] - cluster_data, axis=-1)
            medoid_index = np.argmin(np.sum(distance_matrix, axis=1))
            medoid = cluster_data[medoid_index]
            gaussian_mixture_medoids_kmnist[cluster_label] = medoid
        
        reduction_models = {}

        nb_clusters_kmeans = len(np.unique(k_means_best_lables_kmnist))
        nb_clusters_gaussian = len(np.unique(gaussian_mixture_best_lables_kmnist))

        colors_kmeans = plt.colormaps['nipy_spectral'](np.linspace(0,1,nb_clusters_kmeans))
        colors_gaussian = plt.colormaps['nipy_spectral'](np.linspace(0,1,nb_clusters_gaussian))
        
        for reduction_name, reduction_fn in zip(['umap', 'pca'], [umap_reduction, pca_reduction]):
            page_processor.reduce_dimensions(reduction_fn)
            reduction_models[reduction_name] = {
                'kmnist': page_processor.kmnist_reduction_model,
                'emnist': page_processor.emnist_reduction_model
            }

            # K-Means
            to_plot = []
            for cluster_label in sorted(np.unique(k_means_best_lables_emnist)):
                cluster_data = page_processor.emnist_reduced_letters[k_means_best_lables_emnist == cluster_label]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors_kmeans[cluster_label])
                med = k_means_medoids_emnist[cluster_label]
                med_red = page_processor.emnist_reduction_model.transform(med.reshape(1, -1))
                # encoded_emnist_letters_cluster = encoded_emnist_letters[k_means_best_lables_emnist == cluster_label]
                # med_arg = np.argmin((encoded_emnist_letters_cluster-med)**2, axis=0)
                to_plot.append(
                    [med_red[0, 0], med_red[0, 1], colors_kmeans[cluster_label]]
                )
            for to_plot_data in to_plot:
                plt.scatter(to_plot_data[0], to_plot_data[1], color=to_plot_data[2], s=40, linewidths=0.5, edgecolor='white')
            plt.savefig(f'EMNIST_kmeans_{reduction_name}_{base_model}_{loss_name}_{dstype.split("_")[-1]}.png')
            plt.close()

            to_plot = []
            for cluster_label in sorted(np.unique(k_means_best_lables_kmnist)):
                cluster_data = page_processor.kmnist_reduced_letters[k_means_best_lables_kmnist == cluster_label]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors_kmeans[cluster_label])
                med = k_means_medoids_kmnist[cluster_label]
                med_red = page_processor.kmnist_reduction_model.transform(med.reshape(1, -1))
                # encoded_kmnist_letters_cluster = encoded_kmnist_letters[k_means_best_lables_kmnist == cluster_label]
                # med_arg = np.argmin((encoded_kmnist_letters_cluster-med)**2, axis=0)
                to_plot.append(
                    [med_red[0, 0], med_red[0, 1], colors_kmeans[cluster_label]]
                )
            for to_plot_data in to_plot:
                plt.scatter(to_plot_data[0], to_plot_data[1], color=to_plot_data[2], s=40, linewidths=0.5, edgecolor='white')
            plt.savefig(f'KMNIST_kmeans_{reduction_name}_{base_model}_{loss_name}_{dstype.split("_")[-1]}.png')
            plt.close()

            # Gaussian
            to_plot = []
            for cluster_label in sorted(np.unique(gaussian_mixture_best_lables_emnist)):
                cluster_data = page_processor.emnist_reduced_letters[gaussian_mixture_best_lables_emnist == cluster_label]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors_gaussian[cluster_label])
                med = gaussian_mixture_medoids_emnist[cluster_label]
                med_red = page_processor.emnist_reduction_model.transform(med.reshape(1, -1))
                # encoded_emnist_letters_cluster = encoded_emnist_letters[gaussian_mixture_best_lables_emnist == cluster_label]
                # med_arg = np.argmin((encoded_emnist_letters_cluster-med)**2, axis=0)
                to_plot.append(
                    [med_red[0,0], med_red[0,1], colors_gaussian[cluster_label]]
                )
            for to_plot_data in to_plot:
                plt.scatter(to_plot_data[0], to_plot_data[1], color=to_plot_data[2], s=40, linewidths=0.5, edgecolor='white')
            plt.savefig(f'EMNIST_gaussian_{reduction_name}_{base_model}_{loss_name}_{dstype.split("_")[-1]}.png')
            plt.close()

            to_plot = []
            for cluster_label in sorted(np.unique(gaussian_mixture_best_lables_kmnist)):
                cluster_data = page_processor.kmnist_reduced_letters[gaussian_mixture_best_lables_kmnist == cluster_label]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors_gaussian[cluster_label])
                med = gaussian_mixture_medoids_kmnist[cluster_label]
                med_red = page_processor.kmnist_reduction_model.transform(med.reshape(1, -1))
                # encoded_kmnist_letters_cluster = encoded_kmnist_letters[gaussian_mixture_best_lables_kmnist == cluster_label]
                # med_arg = np.argmin((encoded_kmnist_letters_cluster-med)**2, axis=0)
                to_plot.append(
                    [med_red[0,0], med_red[0, 1], colors_gaussian[cluster_label]]
                )
            for to_plot_data in to_plot:
                plt.scatter(to_plot_data[0], to_plot_data[1], color=to_plot_data[2], s=40, linewidths=0.5, edgecolor='white')
            plt.savefig(f'KMNIST_gaussian_{reduction_name}_{base_model}_{loss_name}_{dstype.split("_")[-1]}.png')
            plt.close()

        gaussian_mixture_out = {
            'medoids_emnist': gaussian_mixture_medoids_emnist,
            'medoids_kmnist': gaussian_mixture_medoids_kmnist,
            
            'kmnist_to_emnist': gaussian_mixture_kmnist_to_emnist,
            
            'lables_kmnist': gaussian_mixture_best_lables_kmnist,
            'lables_emnist': gaussian_mixture_best_lables_emnist,
            
            'reduction_emnist':page_processor.emnist_reduction_model,
            'reduction_kmnist':page_processor.kmnist_reduction_model,
            
            'cluster_kmnist': d['gaussian_mixture']['kmnist_models'][gaussian_mixture_best],
            'cluster_emnist': d['gaussian_mixture']['emnist_models'][gaussian_mixture_best],
        }

        full_res = {
            'k_means': k_means_out,
            'k_means_best': k_means_best,
            'gaussian_mixture': gaussian_mixture_out,
            'gaussian_mixture_best': gaussian_mixture_best,
            'reduction_models': reduction_models
        }

        joblib.dump(full_res, f'{base_name}_{dstype.split("_")[-1]}.joblib')
# %%


# %%
for i in range(10):
    plt.scatter([i], [2], c=plt.cm.get_cmap('gist_ncar')(0.1))
    plt.scatter([i], [-1], c=plt.cm.get_cmap('gist_ncar')(0.9))
    plt.scatter([i], [0], c=plt.cm.get_cmap('gist_ncar')(0.4))
    plt.scatter([i], [1], c=plt.cm.get_cmap('gist_ncar')(0.4))
plt.show()
# %%
gaussian_mixture_cum_score = [
    
    
]
# %%
import joblib
joblib.dump(results, 'results_cae_bce.joblib')
# %%
d = joblib.load('results_cae_bce.joblib')
d['gaussian_mixture']['emnist_models']
# %%
results['gaussian_mixture']['kmnist_sil_scores']
# %%
dataset_type = ['1char_all', '1char_spl', '1char_none']
for dstype in dataset_type:
    DATASET_FOLDER = os.path.join('dataset', dstype)
    for base, type in [['cae','bce'], ['cae','mse'], ['cae', 'ssim'], ['vcae', 'bce'], ['vcae', 'mse']]:
        page_processor = PageProcessor(base=base, type=type)
        page_processor.encode_pages()

        for reduction_name, reduction_fn in zip(['pca', 'tsne', 'umap'], [pca_reduction, tsne_reduction, umap_reduction]):
            page_processor.reduce_dimensions(reduction_fn)
            plt.scatter(page_processor.emnist_reduced_letters[:, 0], page_processor.emnist_reduced_letters[:, 1])
            plt.savefig(f'EMNIST_{reduction_name}_{base}_{type}_{dstype.split("_")[-1]}.png')
            plt.close()

            plt.scatter(page_processor.kmnist_reduced_letters[:, 0], page_processor.kmnist_reduced_letters[:, 1])
            plt.savefig(f'KMNIST_{reduction_name}_{base}_{type}_{dstype.split("_")[-1]}.png')
            plt.close()

# %%
page_processor.encoded_emnist_letters[0]
# %%
decoded_img = page_processor.emnist_model.decode(page_processor.encoded_emnist_letters[3])
plt.figure(figsize=(8, 6))
plt.gray()
# plt.imshow(page_processor.emnist_letter_imgs[3], cmap='gray')
plt.imshow(decoded_img.reshape(32, 32))
plt.plot()
# %%
cosine(page_processor.encoded_emnist_letters[1][0], page_processor.encoded_emnist_letters[3][0])
# %%
# %%
c2, a2 = gaussian_mixture(encoded_kmnist_letters, 37)

# %%
c1_to_c2 = match_clusters(c1, c2)
# %%
c2_gen = [c1_to_c2[cluster] for cluster in c1]