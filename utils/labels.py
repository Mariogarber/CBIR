from config import TEST_DIR, TRAIN_DIR, TEST_LABEL_DIR
import os
import re
import pandas as pd


def get_label_from_filename(filename):
    """
    Obtiene la etiqueta de una imagen a partir de su nombre.
    Args:
        filename (str): Nombre de la imagen.
    Returns:
        str: Etiqueta de la imagen.
    """
    pattern = r"^(.*?) \(\d+\)\.(jpg|jpeg|png|webp)$"
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    else:
        return None

def get_labels_from_dir(dir_path):
    """
    Obtiene las etiquetas de las imágenes en un directorio.
    Args:
        dir_path (str): Ruta al directorio con las imágenes.

    Returns:
        list: Etiquetas de las imágenes en el directorio.
    """
    labels = []
    for img_name in os.listdir(dir_path):
        label = get_label_from_filename(img_name)
        if label:
            labels.append(label)
        else:
            print(f"Error: No se pudo obtener la etiqueta de la imagen {img_name}")
    return labels

def get_and_save_labels_from_test_dir():
    image_path = os.listdir(TEST_DIR)
    labels = get_labels_from_dir(TEST_DIR)
    labels_df = pd.DataFrame({'image_path': image_path, 'label': labels})
    labels_df.to_csv(os.path.join(TEST_LABEL_DIR, 'test_labels.csv'), index=False)
