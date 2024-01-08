from collections import Counter
import json
import os
import glob

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from helpers import is_whitespace
from ml import ConvAutoEncoder, VarConvAutoEncoder


DATASET_FOLDER = 'dataset'
MODELS_FOLDER = 'models/ae'

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


class PageProcessor:
    def __init__(self, base='cae', type='bce') -> None:
        self.base = base
        self.emnist_model = EncoderModel('emnist', base, type)
        self.kmnist_model = EncoderModel('kmnist', base, type)
        self.character_indices = []
        self.labels = []
        
        self.size = 32
        self.rows = 114
        self.columns = 80

        self.load_pages()
        self._load_labels()
        self._divide_characters()

    def _load_pages(self, type='emnist'):
        transform = ToTensor()
        path_wildcard = os.path.join(DATASET_FOLDER, f'{type}_page_*.png')
        paths = sorted(glob.glob(path_wildcard), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        pages = [transform(Image.open(path)).numpy()[0] for path in paths]
        return pages

    def _load_labels(self):
        labels_path = os.path.join(DATASET_FOLDER, 'characters.json')
        self.all_labels = json.load(open(labels_path))

    def _divide_characters(self):
        size, rows, columns = self.size, self.rows, self.columns
        self.emnist_letter_imgs = []
        self.kmnist_letter_imgs = []
        for page_nb, (emnist_page, kmnist_page) in enumerate(zip(self.emnist_pages, self.kmnist_pages)):
            for row in range(rows):
                for column in range(columns):
                    idx = page_nb * rows * columns + row * columns + column
                    emnist_letter = emnist_page[size*row:size*(row+1), size*column:size*(column+1)]
                    kmnist_letter = kmnist_page[size*row:size*(row+1), size*column:size*(column+1)]

                    if is_whitespace(emnist_letter) or is_whitespace(kmnist_letter):
                        continue

                    self.emnist_letter_imgs.append(emnist_letter)
                    self.kmnist_letter_imgs.append(kmnist_letter)
                    self.character_indices.append((page_nb, row, column))
                    self.labels.append(self.all_labels[idx])

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

    def decode_pages(self, medoids_dict, labels, ds='emnist'):
        size, rows, columns = self.size, self.rows, self.columns
        pages = max([page_nb for page_nb, _, _ in self.character_indices]) + 1
        blank_pages = np.zeros((pages, rows*size, columns*size))

        model = self.emnist_model if ds == 'emnist' else self.kmnist_model

        for label_idx, (page_nb, row, column) in enumerate(self.character_indices):
            medoid = medoids_dict[labels[label_idx]]
            
            if self.base == 'cae':
                dec_img = model.decode(np.array([medoid]))
            elif self.base == 'vcae':
                dec_img = model.decode(np.array([
                    medoid,
                    np.zeros_like(medoid)
                ]))
            blank_pages[page_nb, size*row:size*(row+1), size*column:size*(column+1)] = dec_img.reshape(32,32)
        self.decoded_pages = blank_pages
        return blank_pages

    def calc_accuracy(self, labels):
        unique_labels = np.unique(labels)
        self.labels = np.array(self.labels)
        right_guesses = 0
        for l in unique_labels:
            indices = np.where(labels == l)[0]
            most_common = Counter(self.labels[indices]).most_common(1)
            common_count = most_common[0][1]
            right_guesses += common_count
        return right_guesses / len(labels)
