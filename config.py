import os

# Directorio raíz del proyecto
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Configuraciones de directorios
DATASET_DIR = os.path.join(BASE_DIR, "images")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessor")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
TUNED_RESNET50_MODEL_PATH = os.path.join(BASE_DIR, "feature_extraction/tuned_resnet50/tuned_resnet50.h5")
SAVED_FEATURES_DIR = os.path.join(BASE_DIR, "saved_features")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
DB_PATH = os.path.join(BASE_DIR, "database")
DB_FILE = "db.csv"
LOGS_DIR = os.path.join(BASE_DIR, "logs")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")
TEST_LABEL_DIR = os.path.join(EVALUATION_DIR, "labels")
TEST_LABELS_FILE = "train_labels.csv"
APP_DIR = os.path.join(BASE_DIR, "ui")


# Configuración de preprocesamiento
IMAGE_SIZE = (224, 224)  # Tamaño al que se redimensionarán las imágenes
NORMALIZATION_RANGE = (0, 1)  # Rango de normalización de píxeles

