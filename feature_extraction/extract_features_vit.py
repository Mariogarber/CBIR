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
    logging.info(f"Loading model {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    logging.info("vit-base-patch16-224 model loaded")
    return feature_extractor, model

def get_features_from_vit(image_input, model, feature_extractor):
    """
    Extract features from an image using the ViT model.

    Args:
        image_input: Path to the image file or preloaded image as numpy array.
        model: Pretrained ViT model.
        feature_extractor: Feature extractor for the ViT model.

    Returns:
        numpy.ndarray: Extracted feature vector.
    """
    # Check if the input is a numpy array
    if isinstance(image_input, np.ndarray):
        # Convert numpy array to PIL.Image
        image = Image.fromarray(image_input.astype('uint8')).convert("RGB")
    else:
        # Otherwise, assume it's a path and open the image
        image = Image.open(image_input).convert("RGB")

    # Preprocess the image
    image = image_preprocessing(np.array(image))
    # Preprocess the image using the feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embedding.squeeze().tolist()

def construct_features_dict_vit(model, feature_extractor):
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
    Save extracted features to a CSV file without image names.

    Args:
        features (list): List of feature vectors.
        filename (str): Output file name.
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
