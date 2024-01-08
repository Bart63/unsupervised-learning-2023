import os
import glob

import numpy as np
import streamlit as st

import generate_dataset as gd 


st.set_page_config(
    page_title="Generate Dataset",
    page_icon="📷",
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


def generate_dataset_btn(start_line, nb_pages, seed,
        add_roatation, add_scaling, add_salt_and_pepper, add_folding_lines, 
        num_folding_lines, salt_pepper_prob, rotation_prob, scaling_prob,
        rotation_degree, scaling_factor):
    
    generating_text = st.empty()
    generating_text.write('📷 Generating dataset...')

    gd.generate_dataset(start_line, nb_pages, seed=seed, save_pairs=False,
        add_roatation=add_roatation, add_scaling=add_scaling, add_salt_and_pepper=add_salt_and_pepper,
        add_folding_lines=add_folding_lines, num_folding_lines=num_folding_lines,
        salt_pepper_prob=salt_pepper_prob, rotation_prob=rotation_prob, scaling_prob=scaling_prob,
        rotation_degree=rotation_degree, scaling_factor=scaling_factor)

    generating_text.write('✅ Dataset generation complete!')

    kmnist_pages = glob.glob(os.path.join(gd.BASE_PATH, 'kmnist_page_*.png'))
    kmnist_pages.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    st.markdown('## KMNIST pages')
    for path in kmnist_pages:
        st.image(path)

    emnist_pages = glob.glob(os.path.join(gd.BASE_PATH, 'emnist_page_*.png'))
    emnist_pages.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    st.markdown('## EMNIST pages')
    for path in emnist_pages:
        st.image(path)
    st.balloons()

st.write("# Start by generating dataset")

start_line = st.number_input("Starting line (last line is 407)", 0, 407)
nb_pages = st.number_input("Number of pages", 1, 35)
seed = st.number_input("Seed", step=1)

st.markdown('---')

add_roatation = st.checkbox("Add roatation", value=True)
rotation_degree = st.number_input("Rotation degree", 0., 90., 15., 0.01, disabled=not add_roatation)
rotation_prob = st.number_input("Rotation probability", 0.0, 1.0, 0.3, 0.01, disabled=not add_roatation)

st.markdown('---')

add_scaling = st.checkbox("Add scaling", value=True)
scaling_factor = st.number_input("Scaling factor", 0.0, 1.0, 0.08, 0.01, disabled=not add_scaling)
scaling_prob = st.number_input("Scaling probability", 0.0, 1.0, 0.3, 0.01, disabled=not add_scaling)

st.markdown('---')

add_salt_and_pepper = st.checkbox("Add salt and pepper", value=True)
salt_pepper_prob = st.number_input("Salt and pepper probability", 0.0, 1.0, 0.01, 0.01, disabled=not add_salt_and_pepper)

st.markdown('---')

add_folding_lines = st.checkbox("Add folding lines", value=True)
num_folding_lines = st.number_input("Number of folding lines", 0, 100, 10, 1, disabled=not add_folding_lines)

st.markdown('---')

submit_btn = st.button("Generate dataset")

if submit_btn:
    generate_dataset_btn(
        start_line, nb_pages, seed, 
        add_roatation, add_scaling, 
        add_salt_and_pepper, add_folding_lines,
        num_folding_lines, salt_pepper_prob, 
        rotation_prob, scaling_prob,
        rotation_degree, scaling_factor
    )

