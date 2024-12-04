import os
import logging
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModel
import torch
from tqdm import tqdm
from config import SAVED_FEATURES_DIR, TRAIN_DIR
from PIL import Image
import numpy as np
from preprocessor.preprocessing import image_preprocessing


def load_vit(model_name="google/vit-base-patch16-224"):
    """
    Carga el modelo Vision Transformer (ViT) preentrenado.

    Args:
        model_name (str): Nombre del modelo a cargar.

    Returns:
        transformers.AutoFeatureExtractor: Extractor de características del modelo.
        transformers.AutoModel: Modelo Vision Transformer preentrenado.
    """

    logging.info(f"Loading model {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    logging.info("vit-base-patch16-224 model loaded")
    return feature_extractor, model

def get_features_from_vit(image_input, model, feature_extractor):
    """
    Extrae las características de una imagen utilizando el modelo Vision Transformer (ViT).

    Args:
        image_input: Path to the image file or preloaded image as numpy array.
        model (transformers.AutoModel): Modelo Vision Transformer preentrenado.
        feature_extractor (transformers.AutoFeatureExtractor): Extractor de características del modelo.

    Returns:
        torch.Tensor: Embedding de la imagen.
    """
    # Check if the input is a numpy array
    if isinstance(image_input, np.ndarray):
        # Convert numpy array to PIL.Image
        image = Image.fromarray(image_input.astype('uint8')).convert("RGB")
    else:
        # Otherwise, assume it's a path and open the image
        image = Image.open(image_input).convert("RGB")

    # Preprocess the image using the feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embedding.squeeze().tolist()

def construct_features_dict_vit(model, feature_extractor):
    """
    Construye una lista de características a partir de un conjunto de imágenes.

    Args:
        model (transformers.AutoModel): Modelo Vision Transformer preentrenado.
        feature_extractor (transformers.AutoFeatureExtractor): Extractor de características del modelo.

    Returns:   
        list: Lista de vectores de características.
    """
    features = []
    img_files = os.listdir(TRAIN_DIR)

    for img_name in tqdm(img_files, desc="Extracting ViT features", unit="image"):
        img_path = os.path.join(TRAIN_DIR, img_name)
        if os.path.isfile(img_path):
            feature = get_features_from_vit(img_path, model, feature_extractor)
            features.append(feature)
    return features


def save_features(features, filename="features_vit.csv"):
    """
    Guarda las características extraídas en un archivo CSV.

    Args:   
        features (list): Lista de vectores de características.
        filename (str): Nombre del archivo CSV de salida.

    Returns:
        None
    """
    output_path = os.path.join(SAVED_FEATURES_DIR, filename)
    pd.DataFrame(features).to_csv(output_path, index=False, header=False)
    logging.info(f"Features saved to {output_path}")


def extract_vit_features():
    logging.info("Extracting features...")
    feature_extractor, model = load_vit()
    features = construct_features_dict_vit(model, feature_extractor)
    save_features(features)
    logging.info("Features extraction completed")


if __name__ == "__main__":
    extract_vit_features()
