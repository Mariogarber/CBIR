import os
import shutil
from sklearn.model_selection import train_test_split

# Directorio de las carpetas de imágenes
source_dir = "images"  # Cambia esto por el directorio donde tienes las carpetas de imágenes
train_dir = "train"
test_dir = "test"

# Crear las carpetas train y test si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Proporción de la división (90% train, 10% test)
train_ratio = 0.9

# Recorrer las carpetas y dividir las imágenes
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if os.path.isdir(category_path):  # Verifica si es una carpeta
        images = [os.path.join(category_path, img) for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]
        
        # Dividir las imágenes en train y test
        train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)
        
        # Copiar imágenes al directorio train
        for img in train_images:
            shutil.copy(img, os.path.join(train_dir, os.path.basename(img)))
        
        # Copiar imágenes al directorio test
        for img in test_images:
            shutil.copy(img, os.path.join(test_dir, os.path.basename(img)))

print("Las imágenes han sido divididas y copiadas correctamente en las carpetas 'train' y 'test'.")
