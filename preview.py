import argparse
import cv2

from data import emnist_extract, kmnist_extract
from logger import setup_logger, get_logger


def preview(extract, label, seed=0, logger=None):
    print(f'Previewing {label} dataset...')

    images_dict, mapping = extract(seed=seed, logger=logger)

    print('Mapping:')
    print(mapping)

    for label, images in images_dict.items():
        character = mapping[mapping[:, 0] == label][0][-1]
        print(f'Character {character}:')
        for i, image in enumerate(images):
            cv2.imshow(f'Label: {label} no. {i+1}', image)
            key = cv2.waitKey()
            cv2.destroyAllWindows()
            if key == ord('s'):
                print('Skipping to next label')
                break
            if key == ord('x'):
                print('Skipping to next dataset')
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmnist_seed', type=int, default=0)
    parser.add_argument('--emnist_seed', type=int, default=0)
    args = parser.parse_args()
    
    setup_logger('logger')
    logger = get_logger('logger')

    preview(kmnist_extract, 'kmnist', seed=args.kmnist_seed, logger=logger)
    preview(emnist_extract, 'emnist', seed=args.emnist_seed, logger=logger)


if __name__ == '__main__':
    main()
