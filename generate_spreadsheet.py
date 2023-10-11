import numpy as np
import cv2
import os
import argparse

from data import emnist_extract, kmnist_extract
from logger import setup_logger, get_logger


BASE_PATH = 'spreadsheets'

def preview(extract, label, seed=0, nb_images=2, logger=None):
    print(f'Generating {label} spreadsheets...')

    images_dict, mapping = extract(seed=seed, logger=logger, nb_images=nb_images)
    spreadsheet = images_dict[list(images_dict)[0]].reshape(-1, 28)

    for key in list(images_dict.keys())[1:]:
        spreadsheet = np.append(spreadsheet, images_dict[key].reshape(-1, 28), axis=1)
    fname = f'{label}_s{seed}.png'
    save_path = os.path.join(BASE_PATH, fname)
    cv2.imwrite(save_path, spreadsheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=25)
    parser.add_argument('--nb_from', type=int, default=0)
    parser.add_argument('--nb_imgs', type=int, default=2)
    args = parser.parse_args()

    setup_logger('logger')
    logger = get_logger('logger')
    os.makedirs(BASE_PATH, exist_ok=True)
    for seed in range(args.nb_from, args.nb_from + args.num_seeds):
        preview(kmnist_extract, 'kmnist', seed=seed, logger=logger, nb_images=args.nb_imgs)
        preview(emnist_extract, 'emnist', seed=seed, logger=logger, nb_images=args.nb_imgs)


if __name__ == '__main__':
    main()
