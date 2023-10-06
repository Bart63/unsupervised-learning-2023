from data import emnist_extract, kmnist_extract
import cv2


def preview(extract, label, seed=0):
    print(f'Previewing {label} dataset...')

    images_dict, mapping = extract(seed=seed)

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
    preview(kmnist_extract, 'kmnist', seed=3)
    preview(emnist_extract, 'emnist', seed=3)


if __name__ == '__main__':
    main()
