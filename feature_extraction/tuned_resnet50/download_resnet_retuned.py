import gdown
from config import TUNED_RESNET50_MODEL_PATH
import os

def download_tuned_resnet50():
    """
    Descarga el modelo ResNet50 ajustado (tuned) desde Google Drive.
    """
    url = 'https://drive.google.com/uc?id=1GfeS9NdZDy99FFYJI4e3r8qCQ4IjIIEg'
    output = TUNED_RESNET50_MODEL_PATH

    if os.path.exists(output):
        print("Tuned ResNet50 model already downloaded")
    else:
        print(f"Downloading tuned ResNet50 model from {url}...")
        gdown.download(url, output, quiet=False)
        print("Tuned ResNet50 model downloaded")

if __name__ == '__main__':
    download_tuned_resnet50()