# Proyecto CBIR
Proyecto CBIR para clase

## Estructura

1. `datasets/`: Para almacenar o enlazar el conjunto de datos utilizado en el proyecto.
2. `preprocessor/`: Contiene scripts para tareas como redimensionar, normalizar y realizar aumentaciones de datos.
3. `feature_extraction/`: Scripts para implementar los distintos métodos de extracción de características.
4. `faiss_index/`: Incluye scripts para crear y manejar índices FAISS, y evaluar su rendimiento.
5. `ui/`: Carpeta dedicada a la interfaz de usuario proporcionada por los profesores.
6. `utils/`: Funciones auxiliares para manejo de archivos y visualización.
7. `docs/`: Documentación que describe el proceso, los métodos, y los resultados obtenidos.
8. `logs/`: Logs generados por la aplicación y la extracción de características.
9. `main.py`: Archivo de punto de entrada del proyecto que coordina las etapas principales.
10.  `requirements.txt`: Para listar todas las dependencias del proyecto (librerías como FAISS, OpenCV, etc.).
11. `config.py`: Archivo centralizado para configurar rutas, hiperparámetros, y otros valores globales.



RUN:
` .\cbir_env\Scripts\python.exe -m streamlit run ui/app.py`