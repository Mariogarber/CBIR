import logging
import os
import subprocess

from feature_extraction.tuned_resnet50.train_resnet50 import train_resnet50
from feature_extraction.tuned_resnet50.download_resnet_retuned import download_tuned_resnet50
from feature_extraction.tuned_resnet50.extract_features_tuned_resnet import extract_tuned_resnet_features
from feature_extraction.extract_features_hog import extract_hog_features
from feature_extraction.extract_features_vit import extract_vit_features
from feature_extraction.extract_features_wavelet import extract_features_wavelet_main
from feature_extraction.extract_features_resnet import extract_resnet_features
from faiss_index.build_index import build_faiss_index, create_db_csv
from evaluation.evaluate import evaluate_indices_retrieved, evaluate_feature_extractor_on_test_set
from config import LOGS_DIR, BASE_DIR, APP_DIR

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")), # Guarda logs en un archivo
        logging.StreamHandler() # Muestra logs en consola
    ],
)   
    print("------------------------------------------------------------------------------------")
    print("Welcome to the Content Based Image Retrieval System (CBIR)")
    print("Made by Adonis and Mario")
    print("------------------------------------------------------------------------------------")
    print("This is the main menu to run the feature extraction and build the faiss index")
    input("Press Enter to continue...")
    continue_option = True
    while continue_option:
        print(' (1) Re-train the model ResNet50, type: 1')
        print(' (2) Download the tuned model based on ResNet50, please type: 2')
        print(' (3) Extract the features from the tuned model, please type: 3')
        print(' (4) Extract the features from the HOG model, please type: 4')
        print(' (5) Extract the features from the Google VIT model, please type: 5')
        print(' (6) Extract the features from the Wavelet model, please type: 6')
        print(' (7) Extract the features from the ResNet model, please type: 7')
        print(' (8) Generate the database, please type: 8')
        print(' (9) Launch the app, please type: 9')
        print('(10) Evaluate a model, please type: 10')
        print('(11) Exit, please type: exit')
        option = input("Please type the option: ")
        logging.info(f"Selected option: {option}")
        if option == '1':
            k = int(input("Please type the number of layers to do not freeze: "))
            epochs = int(input("Please type the number of epochs: "))
            patience = int(input("Please type the patience for early stopping (None to do not use EarlyStopping): "))
            logging.info(f"Training model with k={k}, epochs={epochs}, patience={patience}")
            train_resnet50(k=k, epochs=epochs, patience=patience)
            print('Training completed')
            logging.info("Training completed")

        elif option == '2':
            download_tuned_resnet50()
            print('Download completed')
            logging.info("Download completed")

        elif option == '3':
            extract_tuned_resnet_features()
            print('Extraction completed')
            build_faiss_index('features_tuned_resnet.csv', 'feat_extract_tuned_resnet.index')
            logging.info("Extraction completed using the tuned model")

        elif option == '4':
            extract_hog_features()
            print('Extraction completed')
            build_faiss_index('features_hog.csv', 'feat_extract_hog.index')
            logging.info("Extraction completed using the HOG model")

        elif option == '5':
            extract_vit_features()
            print('Extraction completed')
            build_faiss_index('features_vit.csv', 'feat_extract_vit.index')
            logging.info("Extraction completed using the VIT model")

        elif option == '6':
            extract_features_wavelet_main()
            print('Extraction completed')
            build_faiss_index('features_wavelet_haar.csv', 'feat_extract_wavelet.index')
            logging.info("Extraction completed using the Wavelet model")
        
        elif option == '7':
            extract_resnet_features()
            print('Extraction completed')
            build_faiss_index('features_resnet.csv', 'feat_extract_resnet.index')
            logging.info("Extraction completed using the ResNet model")

        elif option == '8':
            create_db_csv()
            print('Database generated')
            logging.info("Database generated")

        elif option == '9':
            create_db_csv()
            logging.info("Database generated")
            print('Running the app...')
            logging.info("Running the app")
            command = ["python", "-m", "streamlit", "run", f"{os.path.join(APP_DIR,'app.py')}"]
            print(f'Running command: {command}')
            process = subprocess.Popen(command)

        elif option == '10':
            print('Which model do you want to evaluate?')
            print(' (1) HOG')
            print(' (2) VIT')
            print(' (3) Wavelet')
            print(' (4) ResNet')
            print(' (5) Tuned ResNet')
            option = input("Please type the option: ")
            logging.info(f"Selected option: {option}")
            if option == '1':
                evaluate_feature_extractor_on_test_set("hog")
            elif option == '2':
                evaluate_feature_extractor_on_test_set("vit")
            elif option == '3':
                evaluate_feature_extractor_on_test_set("wavelet")
            elif option == '4':
                evaluate_feature_extractor_on_test_set("resnet")
            elif option == '5':
                evaluate_feature_extractor_on_test_set("tuned_resnet")
            else:
                print("Invalid option")
                logging.error("Invalid option")
                            
        continue_option = input("Do you want to continue? (yes/no): ") == 'yes'

    print("Goodbye!")
    logging.info("Session ended")

if __name__ == "__main__":
    main()