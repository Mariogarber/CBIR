**Proyecto CBIR creado por Adonis García y Mario García**

## 1. Introducción

La búsqueda de imágenes basada en contenido (CBIR, por sus siglas en inglés) es un área activa de investigación con aplicaciones en diversos dominios, desde la gestión de bases de datos visuales hasta sistemas de reconocimiento en tiempo real. Un componente crítico de estos sistemas es la extracción de características efectivas que capturen de manera representativa las propiedades visuales de las imágenes. Estas características son transformadas en embeddings, vectores numéricos que permiten comparar imágenes en un espacio métrico utilizando índices de búsqueda eficientes como FAISS.
En el marco de esta práctica, se ha desarrollado un sistema de recuperación de imágenes basado en contenido (CBIR). Este sistema tiene como objetivo identificar y recuperar imágenes similares a una consulta proporcionada por el usuario, empleando técnicas avanzadas de extracción de características, aprendizaje profundo y búsqueda eficiente en bases de datos.
El proyecto combina múltiples tecnologías, como redes neuronales convolucionales (CNN) mediante modelos pre-entrenados, algoritmos de extracción de características como HOG (Histogram of Oriented Gradients) y transformadas Wavelet, así como métodos de búsqueda de vectores como FAISS (Facebook AI Similarity Search). Estas herramientas permiten analizar y representar imágenes de manera robusta, facilitando su comparación en espacios de alta dimensionalidad.
A través de esta práctica, no solo se han explorado técnicas para la recuperación de imágenes, sino también conceptos fundamentales en el desarrollo de aplicaciones web interactivas, integrando el sistema en una interfaz basada en Streamlit. Esto permite al usuario final interactuar con el sistema de forma intuitiva, subiendo una imagen de consulta y visualizando las imágenes más similares recuperadas desde la base de datos.
El presente trabajo describe el proceso completo llevado a cabo, desde la selección y preparación del dataset, la implementación técnica de las diferentes técnicas de extracción de características y su aplicación en el índice FAISS, hasta la evaluación exhaustiva de los resultados obtenidos. También se analizan los desafíos enfrentados durante el desarrollo del sistema, las soluciones implementadas y el impacto de cada enfoque en términos de precisión en clasificación y eficiencia computacional. Los hallazgos de este proyecto no solo contribuyen al desarrollo de sistemas CBIR más efectivos, sino que también destacan la importancia de una adecuada integración de tecnologías avanzadas en soluciones accesibles para el usuario final.	

## 2. Ejecución

### 2.0 Creación del entorno virtual (Opcional)

Se recomienda crear un nuevo entorno virtual de Python, ya sea con anaconda o con otra herramienta. De esta forma evitamos posibles conflictos de librerías y otros problemas.

### 2.1 Instalación de las librerías

A través del comando `pip install -r requeriments.txt` se instalarán todas las librerías importadas y utilizadas en este proyecto.

### 2.2 Ejecución de la aplicación

Para ejecutar la aplicación hemos desarrollado 2 formas distintas, una a través del archivo `main.py` y otra a través de `ui/app.py`. 

#### 2.2.1 Ejecución de la aplicación a través de `main.py`
Ejecutando en la consola `python main.py` se abrirá la interfaz del proyecto, pudiendo acceder a partes como el reentrenamiento de la red, la extracción de características o la reconstrucción de la base de datos. **Esto último es necesario aplicarlo al menos la 1º vez que quieres ejecutar la aplicación**. También podrás correr la aplicación desde ese menú.

#### 2.2.2 Ejecución de la aplicación a través de `ui/app.py`
Ejecutando por consola el comando `python -m streamlit run ui/app.py` se lanzará la aplicación sin necesidad de pasar por el menú del proyecto. 

## 3. Descripción del sistema de ficheros
1. `database/`: Contiene el archivo `db.csv` que relaciona los índices con las imágenes en local. También contiene los 5 índices generados.
2. `images/`: No presente en GitHub debido a su tamaño, pero contiene las imágenes usadas
3.  `preprocessor/`: Contiene el código requerido para preprocesar las imágenes en función del método que se vaya a aplicar.
4. `feature_extraction/`: Contiene el código de los 5 métodos de extracción de características implementados. También maneja la carga y descarga de modelos.
5. `faiss_index/`: Contiene el código necesario para generar los índices y almacenarlos. El código también crea el archivo `db.csv`.
6. `ui/`: Carpeta dedicada a la interfaz de usuario proporcionada por los profesores. Contiene el código referido a la aplicación que se lanza.
7. `main.py`: Archivo de punto de entrada del proyecto. Ejecuta el código del proyecto proporcionando diversas opciones al usuario.
8.  `requirements.txt`: Para listar todas las dependencias del proyecto (librerías como FAISS, OpenCV, etc.).
9. `config.py`: Continene la configuración de los path de todos los archivos del proyecto.
10. `logs/`: Contiene el log de nuestra aplicación.
11. `utils/`: Contiene distintas funciones que proporcionan utilidades al proyecto.
