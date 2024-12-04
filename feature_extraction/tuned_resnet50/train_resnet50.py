import cv2
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, Flatten, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from preprocessor.preprocessing import image_preprocessing
from config import TRAIN_DIR, TEST_DIR, TUNED_RESNET50_MODEL_PATH, LOGS_DIR

def load_dataset():
    """
    Carga las imágenes de las carpetas train y test, extrayendo las etiquetas
    desde el nombre de los archivos.

    Returns:
        imgs_train: Imágenes de entrenamiento procesadas.
        imgs_test: Imágenes de prueba procesadas.
        labels_train: Etiquetas de entrenamiento extraídas del nombre de las imágenes.
        labels_test: Etiquetas de prueba extraídas del nombre de las imágenes.
    """
    def extract_label_from_filename(filename):
        """
        Extrae la etiqueta del nombre de un archivo en el formato "Nombre animal (n).extensión".

        Args:
            filename (str): Nombre del archivo de la imagen.

        Returns:
            str: Etiqueta extraída (Nombre del animal).
        """
        # Elimina la extensión del archivo
        name_without_extension = os.path.splitext(filename)[0]
        # Extrae el nombre del animal eliminando la parte final entre paréntesis
        label = name_without_extension.rsplit("(", 1)[0].strip()
        return label

    def load_images_from_folder(folder):
        """
        Carga imágenes desde una carpeta y extrae sus etiquetas.

        Args:
            folder (str): Ruta de la carpeta que contiene las imágenes.

        Returns:
            images (list): Lista de imágenes procesadas.
            labels (list): Lista de etiquetas extraídas.
        """
        images = []
        labels = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):
                img = image_preprocessing(img_path, apply_canny=True)  # Procesar la imagen
                label = extract_label_from_filename(filename)  # Extraer la etiqueta
                images.append(img)
                labels.append(label)
        return images, labels

    # Cargar datos de entrenamiento
    imgs_train, labels_train = load_images_from_folder(TRAIN_DIR)

    # Cargar datos de prueba
    imgs_test, labels_test = load_images_from_folder(TEST_DIR)

    return np.array(imgs_train), np.array(imgs_test), labels_train, labels_test

def get_model(output_size, k=7):
    """
    Crea un modelo de red neuronal convolucional basado en ResNet50.

    Args:
        output_size (set): Conjunto de etiquetas de salida.
        k (int): Número de capas de ResNet50 hasta la ultima que congelas.

    Returns:    
        keras.Model: Modelo de red neuronal convolucional.
    """
    input_shape = (224, 224, 4)
    resnet_model = ResNet50(include_top=False, input_tensor=None,)
    new_input = Input(shape=input_shape)
    x = Conv2D(filters=3, kernel_size=(1, 1), activation="relu")(new_input)
    x = resnet_model(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dropout(0.7)(x)
    x = Dense(len(output_size), activation='softmax', kernel_initializer="he_normal")(x)
    for layer in resnet_model.layers[:-k]:
        layer.trainable = False
    model = Model(inputs=new_input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(x_train, labels, model, epochs=8, patience=3):
    '''
    Entrena el modelo con los datos de entrenamiento y las etiquetas.

    Args:
        x_train (numpy.ndarray): Datos de entrenamiento.
        labels (list): Etiquetas de las imágenes.
        model (keras.Model): Modelo de red neuronal convolucional.
    
    Returns:
        keras.Model: Modelo entrenado.
        sklearn.preprocessing.LabelEncoder: Codificador de etiquetas.
    '''
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    y_train = to_categorical(encoded_labels)
    if patience is None:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)
        return model, le
    else:
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        return model, le

def test_model(x_test, labels, model, le):
    '''
    Evalúa el modelo con los datos de prueba y las etiquetas.

    Args:
        x_test (numpy.ndarray): Datos de prueba.
        labels (list): Etiquetas de las imágenes.
        model (keras.Model): Modelo de red neuronal convolucional.
        le (sklearn.preprocessing.LabelEncoder): Codificador de etiquetas.

    Returns:
        None
    '''
    encoded_labels = le.transform(labels)
    y_test = to_categorical(encoded_labels)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    logging.info(f'Test loss: {loss}. Test accuracy: {accuracy}')

def save_model(model):
    '''
    Guarda el modelo en el directorio especificado.

    Args:
        model (keras.Model): Modelo de red neuronal convolucional.

    Returns:    
        None
    '''
    model_path = os.path.join(TUNED_RESNET50_MODEL_PATH)
    model.save(model_path)
    print(f"Model saved at {model_path}")

def train_resnet50(k=7, epochs=8, patience=3):
    '''
    Función principal para entrenar un modelo de red neuronal convolucional basado en ResNet50.
    '''
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),  # Guarda logs en un archivo
    ],
    ) 
    logging.info('Starting training process...')
    
    x_train, x_test, labels_train, labels_test = load_dataset()
    logging.info(f'Loaded {len(x_train)} training images and {len(x_test)} test images.')
    logging.info(f'Labels: {set(labels_train)}')
    logging.info(f'Freezing the last {k} layers of ResNet50.')
    model = get_model(set(labels_train), k=k)
    logging.info('Model created.')
    logging.info(f'Training model for {epochs} epochs with early stopping patience of {patience}.')
    model, le = train_model(x_train, labels_train, model, epochs=epochs, patience=patience)
    test_model(x_test, labels_test, model, le)
    save_model(model)

if __name__ == "__main__":
    train_resnet50()
