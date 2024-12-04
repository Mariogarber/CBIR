import os
import numpy as np
import pywt
import cv2
import pandas as pd
from tqdm import tqdm
from config import TRAIN_DIR, SAVED_FEATURES_DIR, LOGS_DIR
import logging

def wavelet_transform(image, wavelet='haar'):
    """
    Realiza la transformada wavelet discreta 2D de una imagen.
    Args:

        image (numpy.ndarray): Imagen de entrada (en escala de grises).
         wavelet (str): Tipo de wavelet a usar (por defecto: 'haar').

     Returns:

            list: Características extraídas (media y desviación estándar para cada conjunto de coeficientes).
    """
    # Perform single-level 2D Discrete Wavelet Transform
    coeffs = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs

    # Extract statistical features from coefficients
    features = []
    for coeff in [LL, LH, HL, HH]:
        features.append(np.mean(coeff))  # Mean
        features.append(np.std(coeff))  # Standard deviation

    return features


def extract_wavelet_features(image_input, wavelet='haar'):
    """
    Extrae características de una imagen utilizando la transformada wavelet discreta 2D.
    Se encarga de cargar la imagen, redimensionarla y extraer las características.

    Args:
        image_input: Path to the image file or preloaded image as numpy array.
        wavelet (str): Tipo de wavelet a usar (por defecto: 'haar').
    Returns:
        list: Características extraídas.
    """
    # Load image if a file path is provided
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
        # Convert to grayscale if the image has 3 or more channels
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Input must be a file path (str) or a numpy array.")

    # Resize the image to 224x224 (if needed)
    if image.shape[:2] != (224, 224):
        image = cv2.resize(image, (224, 224))

    # Perform Wavelet Transform and extract features
    coeffs = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs

    # Compute statistical features (mean and std) for each coefficient set
    features = []
    for coeff in [LL, LH, HL, HH]:
        features.append(np.mean(coeff))  # Mean
        features.append(np.std(coeff))  # Standard deviation

    return features


def extract_features_wavelet_main(wavelet='haar'):
    """
    Extrae las características de la transformada wavelet de las imágenes en el directorio de entrenamiento.
    Guarda las características extraídas en un archivo CSV.

    Args:
        wavelet (str): Tipo de wavelet a usar (por defecto: 'haar').

    Returns:
        None
    """
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),  # Guarda logs en un archivo
    ],
    ) 
    logging.info("Extracting Wavelet features...")
    image_files = os.listdir(TRAIN_DIR)
    all_features = []

    # Extract features for each image
    for img_name in tqdm(image_files, desc="Extracting Wavelet features", unit="image"):
        img_path = os.path.join(TRAIN_DIR, img_name)
        if os.path.isfile(img_path):
            try:
                features = extract_wavelet_features(img_path, wavelet)
                all_features.append(features)
            except Exception as e:
                logging.error(f"Error processing {img_name}: {e}")

    # Convert to DataFrame and save
    output_path = os.path.join(SAVED_FEATURES_DIR, f"features_wavelet_{wavelet}.csv")
    pd.DataFrame(all_features).to_csv(output_path, index=False, header=False)
    logging.info(f"Wavelet features saved to {output_path}")
