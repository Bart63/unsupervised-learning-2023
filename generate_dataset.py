import numpy as np
import pandas as pd
import cv2
import os
import glob
import shutil
import argparse
import transformations

from data import winnie_extract
from globals import CHAR_MAP
from helpers import is_whitespace

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


def generate_dataset(start_line, nb_cols, nb_rows, nb_pages, img_height=32, img_width=32, 
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
    text = winnie_extract(start_line=start_line, nb_chars=nb_cols*nb_rows*nb_pages)
    text = ''.join(map_text(text, CHAR_MAP))

    # Generate list of labels
    char2label = dict(zip(e_mapping_df['char'], e_mapping_df['label']))
    labels = map_text(text, char2label)

    # Paths to directory of images
    emnist_dir = os.path.join(MAPPING_BASE, f'emnist_{e_num}')
    kmnist_dir = os.path.join(MAPPING_BASE, f'kmnist_{k_num}')
    
    pairs_dir = os.path.join(BASE_PATH, f'pairs_e{e_num}_k{k_num}_s{seed}')
    if os.path.exists(pairs_dir):
        shutil.rmtree(pairs_dir)
    os.mkdir(pairs_dir)

    get_img_path = lambda l, ds_dir: np.random.choice(
        glob.glob(os.path.join(ds_dir, str(l), '*.png')))

    # Create pages for EMNIST and KMNIST
    print('Generating pages for emnist and kmnist...')

    for nb_page in range(nb_pages):
        emnist_page = np.zeros((nb_rows * img_height, nb_cols * img_width), dtype=np.uint8)
        kmnist_page = np.zeros((nb_rows * img_height, nb_cols * img_width), dtype=np.uint8)
        for row in range(nb_rows):
            for col in range(nb_cols):
                idx = nb_page * nb_cols * nb_rows + row * nb_cols + col
                label = labels[idx]

                if label == '\n':
                    # Replace new line with whitespaces to the end of a row
                    whitespaces_to_add = nb_cols - col
                    labels.pop(idx)
                    for _ in range(whitespaces_to_add):
                        labels.insert(idx, ' ')
                    label = labels[idx]
                
                if label == ' ':
                    # Load EMNIST image
                    emnist_image = np.zeros((img_height, img_width))

                    # Load KMNIST image
                    kmnist_image = np.zeros((img_height, img_width))
                else:
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

                # Rotate the image with given probability
                emnist_image = transformations.rotate(emnist_image)
                kmnist_image = transformations.rotate(kmnist_image)

                # Shrink/stretch the image with given probability
                emnist_image = transformations.scale(emnist_image)
                kmnist_image = transformations.scale(kmnist_image)

                # Transform image to binary representation
                emnist_image = transformations.one_hot(emnist_image)
                kmnist_image = transformations.one_hot(kmnist_image)
                # Calculate the position to paste the image
                x = col * img_width
                y = row * img_height

                # Paste the EMNIST image onto the canvas
                emnist_page[y:y+img_height, x:x+img_width] = emnist_image

                # Paste the KMNIST image onto the canvas
                kmnist_page[y:y+img_height, x:x+img_width] = kmnist_image

        # Apply salt and pepper noise on image
        noise_probability = 0.01  # apply each type of noise to 1% of all pixels
        emnist_page = transformations.salt_and_pepper_noise(emnist_page, p=noise_probability)
        kmnist_page = transformations.salt_and_pepper_noise(kmnist_page, p=noise_probability)

        # Apply paper folding noise
        num_lines = 10
        emnist_page = transformations.folding_lines(emnist_page, num_lines=num_lines)
        kmnist_page = transformations.folding_lines(kmnist_page, num_lines=num_lines)

        # Save EMNIST page as PNG
        print(f'Saving page {nb_page}...')
        emnist_fname = os.path.join(BASE_PATH, f'emnist_page_{nb_page}.png')
        cv2.imwrite(emnist_fname, emnist_page)

        # Save KMNIST page as PNG
        kmnist_fname = os.path.join(BASE_PATH, f'kmnist_page_{nb_page}.png')
        cv2.imwrite(kmnist_fname, kmnist_page)

        # Go thru pages emnist and kmnist and save pairs 
        for row in range(nb_rows):
            for col in range(nb_cols):
                pair_idx = nb_page * nb_cols * nb_rows + row * nb_cols + col
                idx = row * nb_cols + col
                
                # Get images of a characters from pages
                row_slice = slice(row*img_height, (row+1)*img_height)
                col_slice = slice(col*img_width, (col+1)*img_width)
                emnist_image = emnist_page[row_slice, col_slice]
                kmnist_image = kmnist_page[row_slice, col_slice]

                if is_whitespace(emnist_image) or is_whitespace(kmnist_image):
                    continue

                # Concat emnist_image and kmnist_image to one image
                pair_img = np.concatenate((emnist_image, kmnist_image), axis=1)

                # Save pair (EMNIST, KMNIST)
                pair_fname = os.path.join(pairs_dir, f'{pair_idx}.png')
                cv2.imwrite(pair_fname, pair_img)

    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_line', type=int, default=202)
    parser.add_argument('--nb_cols', type=int, default=80)
    parser.add_argument('--nb_rows', type=int, default=114)
    parser.add_argument('--nb_pages', type=int, default=5)
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mapping', type=str, default='mapping_e22_k2_m0.csv')
    args = parser.parse_args()

    os.makedirs(BASE_PATH, exist_ok=True)
    generate_dataset(args.start_line, args.nb_cols, args.nb_rows,
                     args.nb_pages, args.img_height, args.img_width, 
                     args.seed, mapping_fname=args.mapping)


if __name__ == '__main__':
    main()
