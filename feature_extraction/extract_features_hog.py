import os
import pandas as pd
from skimage import exposure, color
from skimage.feature import hog
from config import SAVED_FEATURES_DIR, TRAIN_DIR, LOGS_DIR
from tqdm import tqdm
from preprocessor.preprocessing import image_preprocessing
import logging


def get_hog_features(img_path):
    """
    Extrae características HOG de una imagen.

    Args:
        img_path (str): Ruta de la imagen.

    Returns:   
        numpy.ndarray: Vector de características HOG.
    """
    img = image_preprocessing(img_path)
    if img.shape[-1] > 3:
        img = img[:, :, :3]
    img = color.rgb2gray(img)
    features, hog_image = hog(
        img, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=True, channel_axis=None
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features


def construct_hog_features():
    """
    Construye un conjunto de características HOG a partir de un conjunto de imágenes.

    Returns:
        list: Lista de vectores de características HOG.
    """

    features = []
    img_files = os.listdir(TRAIN_DIR)

    for img_name in tqdm(img_files, desc="Extracting HOG features", unit="image"):
        img_path = os.path.join(TRAIN_DIR, img_name)
        if os.path.isfile(img_path):
            feature = get_hog_features(img_path)
            features.append(feature)
    return features


def save_hog_features(features, filename="features_hog.csv"):
    """
    Guarda las características HOG extraídas en un archivo CSV.

    Args:
        features (list): Lista de vectores de características HOG.
        filename (str): Nombre del archivo CSV de salida.

    Returns:   
        None
    """
    output_path = os.path.join(SAVED_FEATURES_DIR, filename)
    pd.DataFrame(features).to_csv(output_path, index=False, header=False)
    logging.info(f"HOG features saved to {output_path}")


def extract_hog_features():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),  # Guarda logs en un archivo
    ],
    ) 
    logging.info("Extracting HOG features...")
    features = construct_hog_features()
    save_hog_features(features)


if __name__ == "__main__":
    extract_hog_features()
