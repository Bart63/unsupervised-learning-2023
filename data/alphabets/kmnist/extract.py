import os
import numpy as np
import pandas as pd

from typing import Tuple, Dict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = [
    'k49-train-imgs.npz', 'k49-train-labels.npz', 
    'k49-test-imgs.npz', 'k49-test-labels.npz',
    'k49_classmap.csv'
]
FILES = [os.path.join(BASE_DIR, f) for f in FILES]

def loadnpzfile(file):
    return np.load(file)['arr_0']

def loadcsvfile(file):
    return pd.read_csv(file)

def extract(nb_images=3, nb_remove=1, seed=0) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Extracts a specified number of images and labels from the FILES directory.
    Args:
        nb_images (int): The number of images to extract for each label. Defaults to 5.
        nb_remove (int): The number of labels to remove from the extracted labels. Defaults to 1.
        seed (int): The seed value for the random number generator. Default is 0.
    Returns:
        Tuple[Dict[int, np.ndarray], np.ndarray]: A tuple containing the chosen images and the mapping of labels.
            - chosen_images (Dict[int, np.ndarray]): A dictionary where the keys are labels and the values are arrays of chosen images.
            - mapping (np.ndarray): An array containing the mapping of labels to unicode values and characters.
    """
    np.random.seed(seed)
    print('Extracting labels')
    labels = np.array([], dtype=int)
    for file in [FILES[1]]:
        print('Extracting {}'.format(os.path.basename(file)))
        labels = np.append(labels, loadnpzfile(file))
    
    print(f'Discarding {nb_remove+1} characters')
    unique_labels = sorted(np.unique(labels))
    chosen_labels = np.random.choice(unique_labels[:-1], replace=False, \
                                     size=len(unique_labels[:-1])-nb_remove
                                    ).tolist() # Remove iteration label with [:-1]
    chosen_labels = sorted(chosen_labels)
    print(f'Discarded characters: {set(unique_labels).difference(chosen_labels)}')

    print(f'Choosing {nb_images} images')
    chosen_positions = {}
    for label in chosen_labels:
        print(f'For {label}: ', end='')
        positions = np.where(labels == label)[0]
        positions = np.random.choice(positions, size=nb_images, replace=False).tolist()
        print(positions)
        chosen_positions[label] = positions
    
    print('Extracting images')
    images = None
    for file in [FILES[0]]:
        print('Extracting {}'.format(os.path.basename(file)))
        images = loadnpzfile(file) if images is None \
            else np.concatenate((images, loadnpzfile(file)))

    chosen_images = {}
    for label, positions in chosen_positions.items():
        chosen_images[label] = images[positions]
    
    mapping = loadcsvfile(FILES[-1]).to_numpy()
    return chosen_images, mapping
