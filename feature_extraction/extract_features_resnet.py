import pandas as pd
import numpy as np
import cv2
import os
from keras.models import load_model
from keras.models import Model
import logging
import numpy as np
import os
import cv2
from keras.applications.resnet50 import ResNet50
from preprocessor.preprocessing import image_preprocessing
import tqdm
from config import SAVED_FEATURES_DIR, TRAIN_DIR

def load_resnet50():
    """
    Carga el modelo ResNet50 preentrenado en ImageNet.

    Returns:
        keras.Model: Modelo ResNet50 preentrenado.
    """

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=model.input, outputs=model.output)

    for layer in model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['loss'])

    logging.info('Model Resnet loaded')
    return model

def get_features_from_resnet(image_input, model):
    """
    Extrae características de una imagen utilizando el modelo ajustado ResNet50.

    Args:
        image_input: Path to the image file or preloaded image as numpy array.
        model (keras.Model): Modelo ajustado para la extracción de características.

    Returns:
        numpy.ndarray: Vector de características de la imagen.
    """
    img = image_preprocessing(image_input)  # Preprocesar la imagen
    img = img.reshape(1, 224, 224, 3)  # Ajustar al tamaño esperado por el modelo
    feature = model.predict(img)
    feature = feature[0]  # Elimina la dimensión batch
    feature = feature.mean(axis=(0, 1))  # Promedia las características de todas las regiones de la imagen
    print(f'Feature shape: {feature.shape}')
    return feature

def construct_features_dict_resnet(model):
    """
    Construye un diccionario de características a partir de un conjunto de imágenes.

    Args:
        model (keras.Model): Modelo ajustado para la extracción de características.

    Returns:
        list: Lista de vectores de características.
    """
    features = []
    img_files = os.listdir(TRAIN_DIR)

    for img_name in tqdm.tqdm(img_files, desc="Extracting ResNet50 features", unit="image"):
        img_path = os.path.join(TRAIN_DIR, img_name)
        if os.path.isfile(img_path):
            feature = get_features_from_resnet(img_path, model)
            features.append(feature)
            logging.info(f"Features extracted for {img_name}")
    return features


def save_features(features, filename="features_resnet.csv"):
    """
    Guarda las características extraídas en un archivo CSV.

    Args:
        features (list): Lista de características extraídas.
        filename (str): Nombre del archivo de salida.
    """
    output_path = os.path.join(SAVED_FEATURES_DIR, filename)
    pd.DataFrame(features).to_csv(output_path, index=False, header=False)
    logging.info(f"Features saved to {output_path}")

def extract_resnet_features():
    """
    Pipeline completo para la extracción de características utilizando el modelo ajustado ResNet50.

    Args:
        model_path (str): Ruta al archivo del modelo guardado.
        image_folder (str): Carpeta que contiene las imágenes.

    Returns:
        None
    """
    model = load_resnet50()
    features_dict = construct_features_dict_resnet(model)
    save_features(features_dict)

if __name__ == "__extract_resnet__":
    extract_resnet_features()