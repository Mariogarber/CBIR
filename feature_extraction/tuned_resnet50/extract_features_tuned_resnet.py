import os
import logging
import numpy as np
import pandas as pd
from keras.models import load_model, Model
import tqdm
from preprocessor.preprocessing import image_preprocessing
from config import SAVED_FEATURES_DIR, TRAIN_DIR, TUNED_RESNET50_MODEL_PATH


def load_tuned_resnet(model_path=TUNED_RESNET50_MODEL_PATH):
    """
    Carga el modelo ResNet50 ajustado (tuned) desde un archivo.

    Args:
        model_path (str): Ruta al archivo del modelo guardado.

    Returns:
        keras.Model: Modelo cargado listo para extracción de características.
    """
    logging.info(f"Loading tuned ResNet50 model from {model_path}...")
    model = load_model(model_path)

    model = Model(inputs=model.input, outputs=model.layers[-4].output)  # Elimina las capas densas
    feature_extraction_model = model  # Usa el modelo tal como está para la extracción
    logging.info("Tuned ResNet50 model loaded")
    return feature_extraction_model


def get_features_from_tuned_resnet(image_input, model):
    """
    Extrae características de una imagen utilizando el modelo ajustado ResNet50.

    Args:
        image_input: Path to the image file or preloaded image as numpy array.
        model (keras.Model): Modelo ajustado para la extracción de características.

    Returns:
        numpy.ndarray: Vector de características de la imagen.
    """
    img = image_preprocessing(image_input, apply_canny = True)  # Preprocesar la imagen
    img = img.reshape(1, 224, 224, 4)  # Ajustar al tamaño esperado por el modelo
    feature = model.predict(img)
    feature = feature[0]  # Elimina la dimensión batch
    feature = feature.reshape(25, 128)  # Ajusta la forma a 25x128
    print(f'Shape of the embedding: {feature.shape}')
    return feature


def construct_features_dict_tuned(model):
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
            features = get_features_from_tuned_resnet(img_path, model)
            features.append(features)
            logging.info(f"Extracted features for {img_name}")
    return features

def max_feature_metric(image_features):
    '''
    Select from each embedding component the maximun value between the features of the image

    Args:
        image_features: The features of the image (5x128)

    Returns:
        The new embedding of the image (1x128)
    '''
    max_array = np.array([0.0]*128)
    for i in range(128):
        elems = []
        for j in range(5):
            elems.append(image_features[j][i])
        max_value = float(max(elems, key=abs))
        max_array[i] = max_value
    return max_array


def get_max_features(list_features):
    '''
    Get the max features of the images of a kind of animal

    Args:
        list_features (list): List of features of the images of the animals

    Returns:
        The max features of the images of the animal
    '''
    features = []
    for feature in list_features:
        feature = max_feature_metric(feature)
        features.append(feature)
        print(f'Shape of the embedding: {feature.shape}')
        print(f'Embedding: {feature}')
    return features

def save_features(features, filename="features_tuned_resnet.csv"):
    """
    Guarda las características extraídas en un archivo CSV.

    Args:
        features (list): Lista de características extraídas.
        filename (str): Nombre del archivo de salida.
    """
    output_path = os.path.join(SAVED_FEATURES_DIR, filename)
    df = pd.DataFrame(features, columns=[f"f{i}" for i in range(128)])
    df.to_csv(output_path, index=False, header=False)
    logging.info(f"Features saved to {output_path}")


def extract_tuned_resnet_features():
    """
    Pipeline completo para la extracción de características utilizando el modelo ajustado ResNet50.

    Args:
        model_path (str): Ruta al archivo del modelo guardado.
        image_folder (str): Carpeta que contiene las imágenes.

    Returns:
        None
    """
    model = load_tuned_resnet()
    features_dict = construct_features_dict_tuned(model)
    max_features = get_max_features(features_dict)
    save_features(max_features)

if __name__ == "__extract_tuned_resnet_features__":
    extract_tuned_resnet_features()