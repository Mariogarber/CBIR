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

from config import LOGS_DIR, BASE_DIR


def main():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),  # Guarda logs en un archivo
    ],
)   
    print("-----------------------------------------")
    print("Welcome to the Content Based Image Retrieval System")
    print("This is the main menu to run the feature extraction and build the faiss index")
    print(f"You are working in the directory: {BASE_DIR}")
    print(f"Logs are saved in {LOGS_DIR}")
    input("Press Enter to continue...")
    continue_option = True
    while continue_option:
        print('If you want to re-train the model, please type: train')
        print('If you want to download the tuned model, please type: download')
        print('If you want to extract the features from the tuned model, please type: tuned')
        print('If you want to extract the features from the HOG model, please type: hog')
        print('If you want to extract the features from the VIT model, please type: vit')
        print('If you want to extract the features from the Wavelet model, please type: wavelet')
        print('If you want to extract the features from the ResNet model, please type: resnet')
        print('If you want to generate the database, please type: db')
        print('If you want to access to the app, please type: app')
        option = input("Please type the option: ")
        logging.info(f"Selected option: {option}")

        if option == 'train':
            k = int(input("Please type the number of layers to do not freeze: "))
            epochs = int(input("Please type the number of epochs: "))
            patience = input("Please type the patience for early stopping (None to do not use EarlyStopping): ")
            if patience != "None":
                patience = int(patience)
            logging.info(f"Training model with k={k}, epochs={epochs}, patience={patience}")
            train_resnet50(k=k, epochs=epochs, patience=patience)
            print('Training completed')
            logging.info("Training completed")

        elif option == 'download':
            download_tuned_resnet50()
            print('Download completed')
            logging.info("Download completed")

        elif option == 'tuned':
            extract_tuned_resnet_features()
            print('Extraction completed')
            build_faiss_index('features_tuned_resnet.csv', 'feat_extract_tuned_resnet.index')
            logging.info("Extraction completed using the tuned model")

        elif option == 'hog':
            extract_hog_features()
            print('Extraction completed')
            build_faiss_index('features_hog.csv', 'feat_extract_hog.index')
            logging.info("Extraction completed using the HOG model")

        elif option == 'vit':
            extract_vit_features()
            print('Extraction completed')
            build_faiss_index('features_vit.csv', 'feat_extract_vit.index')
            logging.info("Extraction completed using the VIT model")

        elif option == 'wavelet':
            extract_features_wavelet_main()
            print('Extraction completed')
            build_faiss_index('features_wavelet_haar.csv', 'feat_extract_wavelet.index')
            logging.info("Extraction completed using the Wavelet model")
        
        elif option == 'resnet':
            extract_resnet_features()
            print('Extraction completed')
            build_faiss_index('features_resnet.csv', 'feat_extract_resnet.index')
            logging.info("Extraction completed using the ResNet model")

        elif option == 'db':
            create_db_csv()
            print('Database generated')
            logging.info("Database generated")

        elif option == 'app':
            create_db_csv()
            logging.info("Database generated")
            print('Running the app...')
            logging.info("Running the app")
            command = ["python", "-m", "streamlit", "run", f"{os.path.join(BASE_DIR, 'ui/app.py')}"]
            print(f'Running command: {command}')
            process = subprocess.Popen(command)


        continue_option = input("Do you want to continue? (yes/no): ") == 'yes'

    print("Goodbye!")
    logging.info("Session ended")

if __name__ == "__main__":
    main()