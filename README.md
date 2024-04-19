# unsupervised-learning-2023
Project Unsupervised Learning 2023 concentrates on solving a translation of manuscripts between image alphabets (EMNIST and KMNIST) utilizing unsupervised learning methods such as autoencoders (CAE and VAE), dimensionality reduction (PCA, UMAP) and clustering (K-Means, GMM). To simulate a real world problem, noise was added on generated manuscripts (salt and pepper, lines, rotation and scaling of letters).

## Results
Input:
![kmnist_page_0](https://github.com/Bart63/unsupervised-learning-2023/assets/30702198/63c8c272-6d5d-43e2-be82-b61b592ff7dc)

Output:
![translated](https://github.com/Bart63/unsupervised-learning-2023/assets/30702198/a2f48010-5a6f-4005-bacd-37e829350da5)


## How to run:
1. Download datasets and text `download.py`
2. (optional) Review spreadsheets of random samples to choose seeds `generate_spreadsheet.py`
3. (even more optional) Preview random (seeded) samples `preview.py`
4. Construct mapping from random (seeded) samples `generate_mapping.py`
5. Construct pages in both alphabets from mapping `generate_dataset.py`
