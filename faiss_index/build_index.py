
import pandas as pd
import numpy as np
import faiss
import re
import os
from config import DB_PATH, SAVED_FEATURES_DIR, TRAIN_DIR

####################################################################
# 1. Cargar las características extraídas con el modelo ResNet     #
####################################################################

def load_features(features_path):
    # Cargar el archivo CSV
    features = pd.read_csv(features_path, header=None)
    return features

####################################################################
# 2. Crear un índice FAISS con las características extraídas       #
####################################################################

def create_faiss_index(features):
    # Crear índice Flat L2 (distancia euclidiana)
    d = features.shape[1]  # Dimensión de las características
    index = faiss.IndexFlatL2(d)
    # Agregar vectores al índice
    index.add(features)
    return index

####################################################################
# 3. Guardar el índice                                             #
####################################################################

def save_faiss_index(index, index_name):
    faiss.write_index(index, os.path.join(DB_PATH, index_name))

####################################################################
# 4. Crear el archivo db.csv el índice y las rutas de las imágenes #
####################################################################

def create_db_csv():
    # Crear un DataFrame que relacione las características con los nombres de las imágenes
    img_files = os.listdir(TRAIN_DIR)
    db = pd.DataFrame()
    db['index'] = np.arange(len(img_files))
    db['image'] = [os.path.join(TRAIN_DIR, img_name) for img_name in img_files]
    # Guardar el archivo db.csv
    db_file_path= os.path.join(DB_PATH, 'db.csv')
    db.to_csv(db_file_path, index=False)


####################################################################
# 5. Función principal para crear el índice con las características#
####################################################################

def build_faiss_index(feature_name, index_name):
    features = load_features(os.path.join(SAVED_FEATURES_DIR, feature_name))
    print(f"Características {feature_name} cargadas con éxito.")
    index = create_faiss_index(features)
    save_faiss_index(index, index_name)
    print(f"Índice {index_name} creado con éxito.")
