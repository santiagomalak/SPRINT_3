Clustering de Usuarios a partir de Datos de Mailing


Descripción del Proyecto:
Este proyecto aplica técnicas de machine learning para segmentar usuarios según su interacción con campañas de mailing . El objetivo es identificar grupos de usuarios con comportamientos similares para optimizar las estrategias de marketing . Se emplean algoritmos de clustering como K-Means y MiniBatchKMeans , junto con análisis de calidad de clusters y generación de recomendaciones personalizadas .


Estructura del Proyecto:
SPRINT-3/
├── SPRINT-3.ipynb           # Notebook con todo el análisis y clustering 
├── README.md                # Documentación del proyecto 
│
├── Conjunto de datos/       # Datos de entrada (dataset) 
│   └── clustering_mailing.xlsx # Archivo de datos con información de clientes 
│
├── Resultados/              # Archivos generados con los resultados 
│   ├── resultados_clustering.xlsx # Clusters y recomendaciones por cliente 
│   ├── modelo_clustering.pkl     # Modelo de clustering entrenado 
│   ├── escalador.pkl            # Escalador utilizado para normalización ⚖️


️Librerías Utilizadas:
Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import joblib

1. Carga y Exploración de Datos:
Se carga el archivo clustering_mailing.xlsx del directorio Conjunto de datos/ .
Se realiza una exploración inicial para entender la estructura y características de los datos .


Variables en el dataset
Variable
Descripción
Id
Identificador único del usuario 🆔
send
Correos enviados
bounce
Correos rebotados
open
Correos abiertos
click
Clics en los correos ️
Total
Total de interacciones ➕
Comprador
1 si el usuario realizó una compra, 0 no
hour
Hora del día en que interactuó ⏰
day_of_week
Día de la semana en que interactuó
Exportar a Hojas de cálculo


✏️ 2. Limpieza y Transformación de Datos
-Se manejan valores nulos ❌.
-Se crea la variable engagement_score para medir la interacción del usuario .
-Se normalizan los datos para evitar sesgos ⚖️.


3. Segmentación con K-Means/MiniBatchKMeans
Se determina el número óptimo de clusters usando el Método del Codo y Silhouette Score .
Se aplica K-Means o MiniBatchKMeans para agrupar a los usuarios .
Se visualizan los clusters con PCA .


4. Análisis y Recomendaciones
Se analizan las características de cada cluster para entender sus diferencias .
Se generan recomendaciones de marketing personalizadas para cada grupo .
Se evalúa la calidad de los clusters con el Silhouette Score y la distribución del Silhouette Score por cluster .
Se visualizan las recomendaciones de marketing por cluster .


5. Resultados
Se identifican grupos de usuarios con diferentes niveles de interacción y comportamiento .
Se generan recomendaciones de marketing específicas para cada grupo, almacenadas en Resultados/resultados_clustering.xlsx .


6. Guardado del Modelo
El modelo entrenado se guarda como Resultados/modelo_clustering.pkl para su reutilización .
El escalador utilizado para la normalización se guarda como Resultados/escalador.pkl ⚖️.


Conclusión
Este proyecto permite segmentar usuarios de mailing para optimizar estrategias de marketing y mejorar la conversión de clientes .


Contacto
Email:santiagoaragonmalak@gmail.com 

GitHub: https://github.com/santiagomalak

Linkedin: https://www.linkedin.com/in/aragonmalak/