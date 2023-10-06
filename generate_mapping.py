import argparse
import numpy as np
import cv2
import os

from logger import setup_logger, get_logger
from data import emnist_extract, kmnist_extract


BASE_PATH = 'mapping'

def generate_mapping(emnist_seed=3, kmnist_seed=3, seed=0, logger=None):
    # Extract all data
    e_images_dict, e_mapping = emnist_extract(seed=emnist_seed, logger=logger)
    k_images_dict, k_mapping = kmnist_extract(seed=kmnist_seed, logger=logger)

    # Save e_mapping to csv
    e_mapping_path = os.path.join(BASE_PATH, f'e_mapping.csv')
    if not os.path.exists(e_mapping_path):
        np.savetxt(e_mapping_path, e_mapping, fmt='%s')

    # Save k_mapping to csv
    k_mapping_path = os.path.join(BASE_PATH, f'k_mapping.csv')
    if not os.path.exists(k_mapping_path):
        np.savetxt(k_mapping_path, k_mapping, fmt='%s')

    np.random.seed(seed)

    # Choose random mapping
    e_keys = list(e_images_dict.keys())
    k_keys = list(k_images_dict.keys())
    np.random.shuffle(e_keys)
    np.random.shuffle(k_keys)
    mapping = np.column_stack([e_keys, k_keys])

    # Save mapping to csv
    mapping_path = os.path.join(BASE_PATH, f'mapping_e{emnist_seed}_k{kmnist_seed}_m{seed}.csv')
    np.savetxt(mapping_path, mapping, fmt='%s')

    # Create directories for each alphabet
    emnist_path = os.path.join(BASE_PATH, f'emnist_{emnist_seed}')
    os.mkdir(emnist_path)
    kmnist_path = os.path.join(BASE_PATH, f'kmnist_{kmnist_seed}')
    os.mkdir(kmnist_path)

    # Iterate through mapping rows
    # Save images separately for each alphabet
    for e_key, k_key in mapping:
        e_character = e_mapping[e_mapping[:, 0] == e_key][0][-1]
        k_character = k_mapping[k_mapping[:, 0] == k_key][0][-1]

        print('Saving: ', e_character, ' <-> ', k_character)
        # Use emnist key as a label for both
        emnist_images_path = os.path.join(emnist_path, str(e_key))
        os.mkdir(emnist_images_path)
        kmnist_images_path = os.path.join(kmnist_path, str(e_key))
        os.mkdir(kmnist_images_path)

        for i, (e_img, k_img) in enumerate(zip(e_images_dict[e_key], k_images_dict[k_key])):
            cv2.imwrite(os.path.join(emnist_images_path, f'{i}.png'), e_img)
            cv2.imwrite(os.path.join(kmnist_images_path, f'{i}.png'), k_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmnist_seed', type=int, default=0)
    parser.add_argument('--emnist_seed', type=int, default=0)
    args = parser.parse_args()

    setup_logger('logger')
    logger = get_logger('logger')
    os.makedirs(BASE_PATH, exist_ok=True)
    generate_mapping(kmnist_seed=args.kmnist_seed, emnist_seed=args.emnist_seed, logger=logger)


if __name__ == '__main__':
    main()
