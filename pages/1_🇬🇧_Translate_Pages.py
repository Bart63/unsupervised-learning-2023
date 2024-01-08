import os
import glob
import joblib
from matplotlib import pyplot as plt

import numpy as np
import streamlit as st

from generate_dataset import BASE_PATH
from runtime import PageProcessor


st.set_page_config(
    page_title="Translate Pages",
    page_icon="🇬🇧",
)

st.markdown(
    """
    <style>
    /* Increase font size for all elements */
    p {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header('Translate Pages from KMNIST to EMNIST')

kmnist_pages = glob.glob(os.path.join(BASE_PATH, 'kmnist_page_*.png'))
kmnist_pages.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
emnist_pages = glob.glob(os.path.join(BASE_PATH, 'emnist_page_*.png'))
emnist_pages.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

if len(emnist_pages) == 0:
    st.warning('No EMNIST pages found. Please generate EMNIST pages first.')
else:
    st.write(f'EMNIST pages: {len(emnist_pages)}')

selected_ae = st.selectbox(
    'Select autoencoder model', 
    ['cae_bce', 'cae_mse', 'cae_ssim', 'vcae_bce', 'vcae_mse']
)

selected_cluster = st.selectbox(
    'Select clustering model',
    ['k_means', 'gaussian_mixture']
)

translate_btn = st.button('Translate')

if translate_btn:
    st.write('Translating...')
    
    base, type = selected_ae.split('_')
    pp = PageProcessor(base=base, type=type)
    pp.encode_pages()

    clustering_path = os.path.join('models', 'cluster', f'{selected_ae}_spl.joblib')
    clustering_model = joblib.load(clustering_path)

    model_keyname = 'model_kmnist' if selected_cluster == 'k_means' else 'cluster_kmnist'
    kmnist_clust_model = clustering_model[selected_cluster][model_keyname]
    kmnist_encoded = pp.encoded_kmnist_letters[:, 0, :]
    emnist_encoded = pp.encoded_emnist_letters[:, 0, :]
    clustered = kmnist_clust_model.predict(kmnist_encoded)
    
    kmnist_2_emnist = clustering_model[selected_cluster]['kmnist_to_emnist']
    translated = np.array([kmnist_2_emnist[int(l)] for l in clustered])
    
    medoids_emnist = clustering_model[selected_cluster]['medoids_emnist']
    decoded_pages_emnist = pp.decode_pages(medoids_emnist, translated)

    st.write('Translated to EMNIST pages: ', len(decoded_pages_emnist))
    for page in decoded_pages_emnist:
        st.image(page)

    st.write('Translated!')
    st.markdown('---')
    st.write('Cleaning...')

    medoids_kmnist = clustering_model[selected_cluster]['medoids_kmnist']
    decoded_pages_kmnist = pp.decode_pages(medoids_kmnist, clustered, ds='kmnist')

    st.write('Cleaned source pages:')
    for page in decoded_pages_kmnist:
        st.image(page)
    
    st.write('Cleaned!')
    st.markdown('---')
    st.write('Generating stats...')
    
    acc = pp.calc_accuracy(clustered)
    st.write(f'Accuracy: {100*acc:.2f} %')
    st.markdown('---')
    st.write('Ploting clusters...')

    reduction_classes = list(clustering_model['reduction_models'])

    num_clusters = len(set(clustered))
    cmap = plt.get_cmap('gist_ncar')
    colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    for rc in reduction_classes:
        st.write(f'Plotting {rc}...')

        st.write('KMNIST')
        reduced_kmnist = clustering_model['reduction_models'][rc]['kmnist'].transform(kmnist_encoded)
        reduced_kmnist_medoids = {
            k: clustering_model['reduction_models'][rc]['kmnist'].transform(v.reshape(1, -1))[0]
            for k, v in medoids_kmnist.items()
        }
        
        fig, ax = plt.subplots()
        for label in set(clustered):
            cluster_data = reduced_kmnist[clustered == label]
            color = colors[label % num_clusters]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color, label=label, alpha=1)
        for label in set(clustered):
            color = colors[label % num_clusters]
            ax.scatter(reduced_kmnist_medoids[label][0], reduced_kmnist_medoids[label][1], color=color, linewidths=1, edgecolor='black', alpha=0.9)
        st.pyplot(fig)

        st.write('EMNIST')
        reduced_emnist = clustering_model['reduction_models'][rc]['emnist'].transform(emnist_encoded)
        reduced_emnist_medoids = {
            k: clustering_model['reduction_models'][rc]['emnist'].transform(v.reshape(1, -1))[0]
            for k, v in medoids_emnist.items()
        }
        
        fig, ax = plt.subplots()
        for label in set(clustered):
            cluster_data = reduced_emnist[clustered == label]
            color = colors[label % num_clusters]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color, label=label, alpha=1)
        for label in set(clustered):
            color = colors[label % num_clusters]
            ax.scatter(reduced_emnist_medoids[label][0], reduced_emnist_medoids[label][1], color=color, linewidths=1, edgecolor='black', alpha=0.9)
        st.pyplot(fig)
    st.write('Finished!')
