import requests
import zipfile
import shutil
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total, unit: x  # If tqdm doesn't exist, replace it with a function that does nothing
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')


URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download(url, force=False):
    path = url.split('/')[-1]
    output_path = os.path.join(BASE_DIR, path)
    if os.path.exists(output_path) and not force:
        print(f'Skipping downloaded file {path}. Use force param to overwrite.')
        return
    r = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

        for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
            if chunk:
                f.write(chunk)

def prepare(force=False):
    download(URL, force)
    zip_file_path = os.path.join(BASE_DIR, 'matlab.zip')
    files_to_extract = ['matlab/emnist-bymerge.mat']

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in files_to_extract:
            if file_name in zip_ref.namelist():
                member = zip_ref.open(file_name)
                output_path = os.path.join(BASE_DIR, os.path.basename(file_name))
                with open(output_path, 'wb') as outfile:
                    shutil.copyfileobj(member, outfile)
            else:
                print(f"File '{file_name}' not found in the zip archive.")
