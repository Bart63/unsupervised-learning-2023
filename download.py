from data import emnist_prepare, kmnist_prepare, winnie_prepare
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', default=False, help='Force the preparation process')
    args = parser.parse_args()

    emnist_prepare(force=args.force)
    kmnist_prepare(force=args.force)
    winnie_prepare()


if __name__ == '__main__':
    main()
