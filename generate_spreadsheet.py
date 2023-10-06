import numpy as np
from data import emnist_extract, kmnist_extract
import cv2
import os
import argparse


BASE_PATH = 'spreadsheets'

def preview(extract, label, seed=0):
    print(f'Generating {label} spreadsheets...')

    images_dict, mapping = extract(seed=seed)
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
    args = parser.parse_args()

    os.makedirs('spreadsheets', exist_ok=True)
    for seed in range(args.nb_from, args.nb_from + args.num_seeds):
        preview(kmnist_extract, 'kmnist', seed=seed)
        preview(emnist_extract, 'emnist', seed=seed)


if __name__ == '__main__':
    main()
