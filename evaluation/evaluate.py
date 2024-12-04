import faiss
import os
import re
import numpy as np

import os
from preprocessor.preprocessing import image_preprocessing
from config import DB_PATH, TEST_DIR
from feature_extraction.extract_features_hog import get_hog_features
from feature_extraction.tuned_resnet50.extract_features_tuned_resnet import get_features_from_tuned_resnet, load_tuned_resnet
from feature_extraction.extract_features_vit import load_vit, get_features_from_vit
from feature_extraction.extract_features_wavelet import extract_wavelet_features
from feature_extraction.extract_features_resnet import get_features_from_resnet, load_resnet50
from ui.app import get_image_list
from utils.labels import get_label_from_filename

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def retrieve_image(img_name, feature_extractor, n_imgs=11):
    image_path = os.path.join(TEST_DIR, img_name)
    if feature_extractor == 'vit':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_vit.index'))
        feature_extractor, model = load_vit()
        img_features = get_features_from_vit(image_path, model, feature_extractor)
    elif feature_extractor == 'wavelet':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_wavelet.index'))
        img_features = extract_wavelet_features(image_path, wavelet='haar')
    elif feature_extractor == 'resnet':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_resnet.index'))
        img_features = get_features_from_resnet(image_path, model=load_resnet50())
    elif feature_extractor == 'tuned_resnet':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_tuned_resnet.index'))
        img_features = get_features_from_tuned_resnet(image_path, model=load_tuned_resnet())
    elif feature_extractor == 'hog':
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_hog.index'))
        img_preprocessed = image_preprocessing(image_path)
        img_features = get_hog_features(img_preprocessed)
    else:
        raise ValueError("Invalid extractor name, choose from 'vit', 'wavelet', 'resnet', 'tuned_resnet', 'hog'")
    
    # Convertir img_features a un array de NumPy si es una lista
    if isinstance(img_features, list):
        img_features = np.array(img_features)
    # Asegurarse de que sea bidimensional
    if img_features.ndim == 1:
        img_features = np.expand_dims(img_features, axis=0)

    _, indices = indexer.search(img_features, n_imgs)

    return indices[0]

def precision_at_k(retrieved_labels, target_label, k):
    """
    Calcula la precisión en los primeros K resultados.
    """
    retrieved_k = retrieved_labels[:k]
    relevant_at_k = sum(1 for label in retrieved_k if label == target_label)
    return relevant_at_k / k

def recall_at_k(retrieved_labels, target_label, total_relevant, k):
    """
    Calcula el recall en los primeros K resultados.
    """
    retrieved_k = retrieved_labels[:k]
    relevant_at_k = sum(1 for label in retrieved_k if label == target_label)
    return relevant_at_k / total_relevant if total_relevant > 0 else 0

def mean_average_precision(retrieved_labels, target_label):
    """
    Calcula el Mean Average Precision (MAP) para un conjunto de resultados.
    """
    relevant = 0
    precision_sum = 0
    for i, label in enumerate(retrieved_labels, start=1):
        if label == target_label:
            relevant += 1
            precision_sum += relevant / i
    return precision_sum / relevant if relevant > 0 else 0

def evaluate_indices_retrieved(img_name, feature_extractor, k=11):
    """
    Evalúa los índices recuperados para una imagen dada.
    Calcula hits, misses, Precision@K, Recall@K, y MAP.
    """
    hits = 0
    misses = 0
    indices = retrieve_image(img_name, feature_extractor)
    total_valid_images_retrieved = len(indices)

    # Obtener la etiqueta objetivo
    target_label = get_label_from_filename(os.path.basename(img_name))

    # Obtener las imágenes recuperadas
    image_list = get_image_list()
    retrieved_images = [image_list[i] for i in indices]
    retrieved_labels = [get_label_from_filename(os.path.basename(img)) for img in retrieved_images]

    # Contar hits y misses
    for label in retrieved_labels:
        if label == target_label:
            hits += 1
        else:
            misses += 1

    # Calcular métricas
    total_relevant = hits + misses  # Total de imágenes relevantes
    p_at_k = precision_at_k(retrieved_labels, target_label, k)
    r_at_k = recall_at_k(retrieved_labels, target_label, total_relevant, k)
    map_score = mean_average_precision(retrieved_labels, target_label)

    return hits, misses, total_valid_images_retrieved, p_at_k, r_at_k, map_score


from tqdm import tqdm

def evaluate_feature_extractor_on_test_set(feature_extractor, k=11):
    """
    Evalúa un extractor de características en todas las imágenes del conjunto de test.
    Calcula las métricas Precision@K, Recall@K y MAP para todas las imágenes de test.
    
    Args:
    - feature_extractor: str (nombre del extractor de características, e.g., 'vit', 'hog', etc.)
    - k: int (número de resultados en el ranking a evaluar)
    
    Returns:
    - dict: Diccionario con métricas promedio (Precision@K, Recall@K, MAP).
    """
    # Listar todas las imágenes en el directorio de test
    test_images = [img for img in os.listdir(TEST_DIR) if img.endswith(('jpg', 'jpeg', 'png', 'webp'))]
    
    total_hits = 0
    total_misses = 0
    total_valid_images = 0
    precision_sum = 0
    recall_sum = 0
    map_sum = 0

    # Iterar sobre cada imagen de test con una barra de progreso
    with tqdm(total=len(test_images), desc="Evaluating images", unit="image") as pbar:
        for img_name in test_images:
            hits, misses, total_retrieved, p_at_k, r_at_k, map_score = evaluate_indices_retrieved(img_name, feature_extractor, k)
            
            # Acumular resultados
            total_hits += hits
            total_misses += misses
            total_valid_images += total_retrieved
            precision_sum += p_at_k
            recall_sum += r_at_k
            map_sum += map_score

            # Actualizar la barra de progreso
            pbar.update(1)

    # Promediar las métricas
    num_images = len(test_images)
    avg_precision = precision_sum / num_images
    avg_recall = recall_sum / num_images
    avg_map = map_sum / num_images

    print(f"\nEvaluation completed for {feature_extractor} feature extractor:")
    print(f"Average Precision@{k}: {avg_precision}")
    print(f"Average Recall@{k}: {avg_recall}")
    print(f"Mean Average Precision (MAP): {avg_map}")

    return {
        "Average Precision@K": avg_precision,
        "Average Recall@K": avg_recall,
        "Mean Average Precision (MAP)": avg_map
    }
