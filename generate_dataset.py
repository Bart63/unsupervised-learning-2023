from data import emnist_extract, kmnist_extract
import numpy as np
import cv2
import os


BASE_PATH = 'dataset'

def generate_dataset(emnist_seed=3, kmnist_seed=3, seed=0):
    # Extract all data
    e_images_dict, e_mapping = emnist_extract(seed=emnist_seed)
    k_images_dict, k_mapping = kmnist_extract(seed=kmnist_seed)

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
    for seed in range(20):
        generate_dataset(kmnist_seed=seed, emnist_seed=seed)


if __name__ == '__main__':
    main()