import gdown
from config import TUNED_RESNET50_MODEL_PATH, LOGS_DIR
import os
import logging

def download_tuned_resnet50():
    """
    Descarga el modelo ResNet50 ajustado (tuned) desde Google Drive.
    """
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),  # Guarda logs en un archivo
    ],
    )
    url = 'https://drive.google.com/uc?id=15-kTyciL_V5GdfRQMLYSwYbPFw6kax1o'
    output = TUNED_RESNET50_MODEL_PATH

    if os.path.exists(output):
        print("Tuned ResNet50 model already downloaded")
        logging.info("Tuned ResNet50 model already downloaded")
    else:
        print(f"Downloading tuned ResNet50 model from {url}...")
        logging.info(f"Downloading tuned ResNet50 model from {url}...")
        gdown.download(url, output, quiet=False)
        print("Tuned ResNet50 model downloaded")

if __name__ == '__main__':
    download_tuned_resnet50()