import numpy as np
import pandas as pd
import cv2
import os
import glob
import argparse

from data import winnie_extract
from globals import CHAR_MAP

BASE_PATH = 'dataset'
MAPPING_BASE = 'mapping'


def map_text(text, mapping):
    mapped_text = []
    for char in text:
        mapped_text.append(
            mapping[char] \
            if char in mapping else char
        )
    return mapped_text


def generate_dataset(start_line, nb_cols, nb_rows, img_height=32, img_width=32, 
                     seed=0, mapping_fname='mapping_e0_k0_m0.csv'):
    np.random.seed(seed)

    # read e_mapping csv
    print('Loading mappings...')
    e_mapping_fname = os.path.join(MAPPING_BASE, 'e_mapping.csv')
    e_mapping_df = pd.read_csv(e_mapping_fname, delimiter=' ', names=['label', 'ascii', 'char'], \
                               dtype=[('label', int), ('ascii', int), ('char', str)])
    
    # read mapping csv - not necessary, labels are the same for both datasets
    # mapping_fname = os.path.join(MAPPING_BASE, mapping_fname)
    # mapping = np.loadtxt(mapping_fname, delimiter=' ', dtype=int)

    e_num, k_num = mapping_fname.split('_')[1:3]
    e_num, k_num = e_num[1:], k_num[1:]

    # Text for dataset generation
    print('Loading and processing text...')
    text = winnie_extract(start_line=start_line, nb_chars=nb_cols*nb_rows)
    text = ''.join(map_text(text, CHAR_MAP))

    # Generate list of labels
    char2label = dict(zip(e_mapping_df['char'], e_mapping_df['label']))
    labels = map_text(text, char2label)

    # Paths to directory of images
    emnist_dir = os.path.join(MAPPING_BASE, f'emnist_{e_num}')
    kmnist_dir = os.path.join(MAPPING_BASE, f'kmnist_{k_num}')

    get_img_path = lambda l, ds_dir: np.random.choice(
        glob.glob(os.path.join(ds_dir, str(l), '*.png')))

    # Create pages for EMNIST and KMNIST
    print('Generating pages for emnist and kmnist...')
    emnist_page = np.zeros((nb_rows * img_height, nb_cols * img_width), dtype=np.uint8)
    kmnist_page = np.zeros((nb_rows * img_height, nb_cols * img_width), dtype=np.uint8)
    for row in range(nb_rows):
        for col in range(nb_cols):
            label = labels[row * nb_cols + col]
            emnist_path = get_img_path(label, emnist_dir)
            kmnist_path = get_img_path(label, kmnist_dir)

            # Load EMNIST image
            emnist_image = cv2.imread(emnist_path, cv2.IMREAD_GRAYSCALE)

            # Load KMNIST image
            kmnist_image = cv2.imread(kmnist_path, cv2.IMREAD_GRAYSCALE)

            # Resize images if not already the same size
            if emnist_image.shape[:2] != (img_height, img_width):
                emnist_image = cv2.resize(emnist_image, (img_width, img_height))
            
            if kmnist_image.shape[:2] != (img_height, img_width):
                kmnist_image = cv2.resize(kmnist_image, (img_width, img_height))

            # TODO: Distort images as in the instruction

            # Calculate the position to paste the image
            x = col * img_width
            y = row * img_height

            # Paste the EMNIST image onto the canvas
            emnist_page[y:y+img_height, x:x+img_width] = emnist_image

            # Paste the KMNIST image onto the canvas
            kmnist_page[y:y+img_height, x:x+img_width] = kmnist_image

    # Save EMNIST page as PNG
    print('Saving...')
    emnist_fname = os.path.join(BASE_PATH, 'emnist_page.png')
    cv2.imwrite(emnist_fname, emnist_page)

    # Save KMNIST page as PNG
    kmnist_fname = os.path.join(BASE_PATH, 'kmnist_page.png')
    cv2.imwrite(kmnist_fname, kmnist_page)

    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_line', type=int, default=0)
    parser.add_argument('--nb_cols', type=int, default=80)
    parser.add_argument('--nb_rows', type=int, default=114)
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(BASE_PATH, exist_ok=True)
    generate_dataset(args.start_line, args.nb_cols, args.nb_rows, 
                     args.img_height, args.img_width, args.seed)


if __name__ == '__main__':
    main()
