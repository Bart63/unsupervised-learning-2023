import os
import numpy as np

from scipy.io import loadmat
from typing import Tuple, Dict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def loadmatfile(file='emnist-bymerge.mat'):
    data_file = os.path.join(BASE_DIR, file)
    return loadmat(data_file)

def extract(nb_images=3, seed=0, logger=None, only_mapping=False) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Extracts a specified number of images and their labels from a dataset.
    Parameters:
        nb_images (int): The number of images to extract for each label. Default is 3.
        seed (int): The seed value for the random number generator. Default is 0.
    Returns:
        Tuple[Dict[int, np.ndarray], np.ndarray]: A tuple containing two elements:
            - chosen_images (Dict[int, np.ndarray]): A dictionary where the keys are the labels and the values are arrays of extracted images.
            - mapping (np.ndarray): An array of labels corresponding to ascii characters.
    """
    logger = print if logger is None else logger.info

    np.random.seed(seed)
    dataset = loadmatfile()['dataset']
    training, testing, mapping = dataset[0][0]
    characters = np.vectorize(chr)(mapping[:, -1])
    mapping = np.hstack((mapping, characters.reshape(-1, 1)), dtype=object)
    if only_mapping: return mapping

    training, testing = training[0], testing[0]
    
    logger('Extracting labels')
    labels = np.concatenate((training['labels'][0], testing['labels'][0])).reshape(-1)
    unique_labels = sorted(np.unique(labels).tolist())

    logger(f'Choosing {nb_images} images')
    chosen_positions = {}
    for label in unique_labels:
        logger(f'For {label}: ')
        positions = np.where(labels == label)[0]
        positions = np.random.choice(positions, size=nb_images, replace=False).tolist()
        logger(positions)
        chosen_positions[label] = positions
    
    logger('Extracting images')
    images = np.concatenate((training['images'][0], testing['images'][0]))
    chosen_images = {}
    for label, positions in chosen_positions.items():
        chosen_images[label] = np.transpose(
            images[positions].reshape(-1, 28, 28), (0, 2, 1))
    return chosen_images, mapping
