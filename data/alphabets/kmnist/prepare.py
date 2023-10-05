# Modified code from https://github.com/rois-codh/kmnist/blob/master/download_data.py

import requests
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total, unit: x  # If tqdm doesn't exist, replace it with a function that does nothing
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kuzushiji-49 (49 classes, 28x28, 270k examples)
# NumPy data format (.npz)
# Download mapping just in case or for fun
to_download = ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49_classmap.csv']

# Download a list of files
def download_list(url_list, force=False):
    for url in url_list:
        path = url.split('/')[-1]
        output_path = os.path.join(BASE_DIR, path)
        if os.path.exists(output_path) and not force:
            print(f'Skipping downloaded file {path}. Use force param to overwrite.')
            continue
        r = requests.get(url, stream=True)
        with open(output_path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)
    print('All dataset files downloaded!')

def prepare(force=False):
    download_list(to_download, force)
