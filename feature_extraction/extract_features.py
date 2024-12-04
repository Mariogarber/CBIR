import pandas as pd
import numpy as np
import cv2
import os
from keras.models import load_model
from keras.models import Model
import logging
import numpy as np
import os
from skimage import io, color, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten
from transformers import AutoFeatureExtractor, AutoModel
from PIL import Image
import torch



IMAGE_PATH = r'C:\Users\mario\OneDrive\Documentos\GitHub\CBIR_Project\CBIR-proyect\animals'
BASE_DIR = r'C:\Users\mario\OneDrive\Documentos\GitHub\CBIR_Project\CBIR-proyect'

logging.basicConfig(
    level=logging.DEBUG,  # Nivel mínimo de logs a capturar
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=r'C:\Users\mario\OneDrive\Documentos\GitHub\CBIR_Project\CBIR-proyect\cnn_features.log',  # Guardar en archivo
    filemode='a'  # Modo de apertura: 'a' para anexar, 'w' para sobrescribir
)

def load_cnn_tuned(model_path):
    model = load_model(model_path)

    model = Model(inputs=model.input, outputs=model.layers[-4].output)

    model.summary()

    logging.info('Model loaded')
    return model


def get_features_cnn(animal_file, model):
    '''
    Get the features of the images of a kind of animal

    Args:
        animal_file: The folder name of the animal

    Returns:
        The features of the images of the animal
    '''
    features = {}
    imgs = os.listdir(os.path.join(IMAGE_PATH, animal_file))
    for img_name in imgs:
        img_path = os.path.join(IMAGE_PATH, animal_file, img_name)
        img = image_preprocessing_cnn(img_path)
        img =  img.reshape(1, 224, 224, 4)
        feature = model.predict(img)
        feature = feature[0]
        feature = feature.reshape(25, 128)
        features[img_name] = feature
    return (features)


def image_preprocessing_cnn(image_path):
    """
    Preprocess an image

     Args:
        image_path: The path to the image to preprocess

    Returns:
        The preprocessed image
    """
    # Load
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    image = cv2.resize(image, (224, 224))

    # Extract each channel
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grey_image_8bit = cv2.convertScaleAbs(grey_image, alpha=(255.0))

    grey_border = cv2.Canny(grey_image_8bit, 250, 400)

    grey_border = cv2.convertScaleAbs(grey_border, alpha=(255.0))

    image = np.stack((red_channel, green_channel, blue_channel, grey_border), axis=-1)

    return image



def construct_features_dict_cnn(file, model):
    '''
    Construct the features dict of all the animals

    Args:
        file: The file containing the animals

    Returns:
        The features dict of all the animals
    '''
    features_dict = {}
    animals = os.listdir(file)

    for animal in animals:
        features = get_features_cnn(animal, model)
        features_dict = {**features_dict, **features}
        logging.info(f'Features of {animal} extracted')
    logging.info('Features dict constructed')
    return features_dict



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

def get_max_features(animal, dict_features):
    '''
    Get the max features of the images of a kind of animal

    Args:
        animal: The folder name of the animal

    Returns:
        The max features of the images of the animal
    '''
    features = []
    imgs_name = os.listdir(os.path.join(IMAGE_PATH, animal))
    for img_name in imgs_name:
        feature = dict_features[img_name]
        feature = max_feature_metric(feature)
        features.append(feature)
    return features, imgs_name

def get_max_features_df(dict_features):
    '''
    Get the features of all the animals

    Returns:
        The features of all the animals
    '''
    df = pd.DataFrame()
    animals = os.listdir(IMAGE_PATH)
    for animal in animals:
        features, imgs_name = get_max_features(animal, dict_features)
        for i in range(len(features)):
            df[f"{imgs_name[i]}"] = features[i]
    df = df.T

    df['image'] = df.index

    return df

def get_flatten_features_df(dict_features):
    '''
    Flatten the features of all the animals

    Args:
        dict_features: The features dict of all the animals

    Returns:
        The flattened features of all the animals
    '''
    df = pd.DataFrame()
    animals = os.listdir(IMAGE_PATH)
    for animal in animals:
        features = []
        for feature in dict_features[animal]:
            feature = feature.reshape(1, 3200)
            features.append(feature[0])
        for i in range(len(features)):
            df[f"{animal}_{i}"] = features[i]
        
    df = df.T

    # df['animal'] = df.index.str.split('_').str[0]

    df.reset_index(drop=True, inplace=True)
    return df

def image_preprocessing_hog(img_path):
    img = cv2.imread(img_path)
    img = color.rgb2gray(img)
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    img = cv2.resize(img, (224, 224))
    return img

def get_hof_features(img_path):
    img = image_preprocessing_hog(img_path)
    features, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return features, hog_image_rescaled

def get_hof_features_df(file):
    df = pd.DataFrame()
    animals = os.listdir(file)
    for animal in animals:
        imgs = os.listdir(os.path.join(file, animal))
        for i, img in enumerate(imgs):
            img_path = os.path.join(file, animal, img)
            feature, _ = get_hof_features(img_path)
            df[f"{img}"] = feature
    df = df.T

    df['image'] = df.index
    
    return df

def get_resnet_features_df(features_dict):
    '''
    Get the features of all the animals

    Args:
        file: The file containing the animals
        model: The model to extract the features

    Returns:
        The features of all the animals
    '''
    df = pd.DataFrame(columns=[key for key in features_dict.keys()])

    for key in features_dict.keys():
        feature_array = features_dict[key]
        df[key] = pd.Series(feature_array)

    df = df.T

    df['image'] = df.index
    return df

def save_features(df, file):
    '''
    Save the features of all the animals to a file

    Args:
        df: The features of all the animals
        file: The file to save the features
    '''
    df.to_csv(file, index=False)

def load_transformer():
    # Cargar modelo y extractor de características
    model_name = "google/vit-base-patch16-224"
    print(f"Loading model {model_name}...")
    logging.info(f"Loading model {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("Model loaded")
    logging.info("Model loaded")

    return feature_extractor, model

def get_features_from_transformer(image_path, model, feature_extractor):
    # Cargar imagen (sustituye 'ruta_a_tu_imagen' por tu archivo)
    image = Image.open(image_path).convert("RGB")
    # Preprocesar la imagen
    inputs = feature_extractor(images=image, return_tensors="pt")
    # Generar embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # El embedding está en la última capa oculta del modelo
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
    print("Embedding shape:", embedding.shape)
    embedding_list = embedding.squeeze().tolist()
    return embedding_list

def get_features_dict_transformer(folder, model, feature_extractor):
    '''
    Get the features of all the animals

    Args:
        folder: The folder containing the animals
        model: The model to extract the features

    Returns:
        The features of all the animals
    '''
    features_dict = {}
    animals = os.listdir(folder)

    for animal in animals:
        imgs = os.listdir(os.path.join(folder, animal))
        for img in imgs:
            img_path = os.path.join(folder, animal, img)
            features = get_features_from_transformer(img_path, model, feature_extractor)
            features_dict[img] = features
    return features_dict

def get_transformer_features_df(features_dict):
    '''
    Get the features of all the animals

    Args:
        file: The file containing the animals
        model: The model to extract the features

    Returns:
        The features of all the animals
    '''
    df = pd.DataFrame(columns=[key for key in features_dict.keys()])

    for key in features_dict.keys():
        feature_array = features_dict[key]
        df[key] = pd.Series(feature_array)

    df = df.T

    df['image'] = df.index
    print(df)
    return df

def main():
    logging.info('--------------------------Start--------------------------')
    print('Start')
    cont = True
    while cont:
        print('What do you want to do?')
        print('1. Extract features from ResNet50')
        print('2. Extract features from HOG')
        print('3. Extract features from CNN')
        print('4. Extract features from Transformer')
        option = int(input())

        if option == 1:
            resnet = load_resnet50()
            dict_features = construct_features_dict_resnet(IMAGE_PATH, resnet)
            logging.info('Features extracted from the ResNet50 model')
            df = get_resnet_features_df(dict_features)
            save_features(df, os.path.join(BASE_DIR, 'cbir/features/features_resnet.csv'))
            logging.info('Features saved in "cbir/features/features_resnet.csv"')
            del resnet
            del df

        elif option == 2:
            df = get_hof_features_df(IMAGE_PATH)
            logging.info('HOG features extracted')
            save_features(df, os.path.join(BASE_DIR, 'cbir/features/features_hof.csv'))
            logging.info('HOG features saved in "cbir/features/features_hof.csv"')
            del df

        elif option == 3:
            model_tuned = load_cnn_tuned(r'C:\Users\mario\OneDrive\Documentos\GitHub\CBIR_Project\CBIR-proyect\model_weights\embedding_model.h5')
            dict_features = construct_features_dict_cnn(IMAGE_PATH, model_tuned)
            logging.info('Features extracted from the TUNED ResNet50 model')
            df = get_max_features_df(dict_features)
            save_features(df, os.path.join(BASE_DIR, 'cbir/features/features_cnn.csv'))
            del model_tuned
            del df
            logging.info('Model deleted')

        elif option == 4:
            feature_extractor, model = load_transformer()
            dict_features = get_features_dict_transformer(IMAGE_PATH, model, feature_extractor)
            logging.info('Features extracted from the Transformer model')
            df = get_transformer_features_df(dict_features)
            save_features(df, os.path.join(BASE_DIR, 'cbir/features/features_transformer.csv'))
            del model
            del feature_extractor
            del df
            logging.info('Model deleted')

        logging.info('Features saved')
        print('Do you want to continue? (y/n)')
        cont = input() == 'y'
    
    print('End')

if __name__ == '__main__':
    main()
    logging.info('--------------------------End--------------------------')