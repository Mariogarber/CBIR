import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
import cv2

import streamlit as st
from streamlit_cropper import st_cropper

from skimage.feature import hog
from skimage import color, exposure
from transformers import AutoFeatureExtractor, AutoModel
import logging
from preprocessor.preprocessing import image_preprocessing
from config import TRAIN_DIR, TUNED_RESNET50_MODEL_PATH, BASE_DIR, DB_PATH, DB_FILE
from feature_extraction.extract_features_hog import get_hog_features
from feature_extraction.tuned_resnet50.extract_features_tuned_resnet import get_features_from_tuned_resnet, load_tuned_resnet
from feature_extraction.extract_features_vit import load_vit, get_features_from_vit
from feature_extraction.extract_features_wavelet import extract_wavelet_features
from feature_extraction.extract_features_resnet import get_features_from_resnet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')


def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    image_list = list(df.image.values)
    return image_list

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    # Debugging print to check input type
    print(f"Type of img_query before processing: {type(img_query)}")
    # Convert PIL.Image to numpy array if needed
    if isinstance(img_query, Image.Image):
        img_query = np.array(img_query)
        print(f"Converted img_query to numpy array. Shape: {img_query.shape}")
    if feature_extractor == 'Extractor 1 (ViT)':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_vit.index'))
        feature_extractor, model = load_vit()
        img_features = get_features_from_vit(img_query, model, feature_extractor)
    elif feature_extractor == 'Extractor 2 (Wavelet)':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_wavelet.index'))
        img_features = extract_wavelet_features(img_query, wavelet='haar')  # Assign to img_features
    elif feature_extractor == 'Extractor 3 (ResNet50)':  # ResNet50
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_resnet.index'))
        img_features = get_features_from_resnet(img_query)
    elif feature_extractor == 'Extractor 4 (Tuned ResNet50)':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_tuned_resnet.index'))
        img_features = get_features_from_tuned_resnet(img_query, model=load_tuned_resnet())
    elif feature_extractor == 'Extractor 5 (Hog)':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_hog.index'))
        img_preprocessed = image_preprocessing(img_query)
        img_features = get_hog_features(img_preprocessed)
    else:
        raise ValueError(f"Extractor {feature_extractor} is not configured.")

    # Ensure img_features is a 2D NumPy array
    if isinstance(img_features, list):
        img_features = np.array(img_features)
    if img_features.ndim == 1:
        img_features = np.expand_dims(img_features, axis=0)

    # Search in the FAISS index
    _, indices = indexer.search(img_features.astype('float32'), k=n_imgs)
    return indices[0]


def main():
    st.title('CBIR IMAGE SEARCH')

    print("-----------------------------------------")
    print("Printing BASE_DIR")
    print(BASE_DIR)
    print("Printing TRAIN_DIR")
    print(TRAIN_DIR)
    print("-----------------------------------------")
    
    train_dir = TRAIN_DIR

    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        # TODO: Adapt to the type of feature extraction methods used.
        option = st.selectbox('.', ('Extractor 1 (ViT)', 'Extractor 2 (Wavelet)', 'Extractor 3 (ResNet50)' ,'Extractor 4 (Tuned ResNet50)', 'Extractor 5 (Hog)'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(train_dir, image_list[retriev[0]]))
                st.image(image, use_container_width = 'always')

            with col4:
                image = Image.open(os.path.join(train_dir, image_list[retriev[1]]))
                st.image(image, use_container_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(train_dir, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(train_dir, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(train_dir, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')

if __name__ == '__main__':
    main()