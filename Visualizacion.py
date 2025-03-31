#Librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Excel
archivo_excel = "C:\\pratics\\Data_Science\\Proyects\\SPRINT-3\\Resultados\\Resultados_Clustering.xlsx"

# Cargar la primera hoja del Excel
df = pd.read_excel(archivo_excel, sheet_name=0)  # O especifica el nombre de la hoja con sheet_name="nombre_hoja"

# Ver las primeras filas del dataframe
print(df.head())

#Ver comlumnas
print(df.columns)

#Visualizar la distribución de los clusters
plt.figure(figsize=(8, 5))
sns.countplot(x=df["cluster"], palette="viridis")
plt.xlabel("Cluster")
plt.ylabel("Cantidad de Clientes")
plt.title("Distribución de Clientes por Cluster")
plt.show()

#Visualizar el engagement por cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["cluster"], y=df["engagement_score"], palette="coolwarm")
plt.xlabel("Cluster")
plt.ylabel("Engagement Score")
plt.title("Engagement Score por Cluster")
plt.show()

#Visualizar los clusters con PCA

# Configuración del gráfico
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Graficar los clusters con colores distintos
scatter = sns.scatterplot(
    x=df["PCA1"],
    y=df["PCA2"],
    hue=df["Cluster"],  
    palette="tab10",   
    alpha=0.6,  # Transparencia para evitar superposiciones
    s=8         # Tamaño de puntos más pequeño
)

# Mejorar etiquetas y título
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Visualización de Clusters con PCA")

# Ajustar la leyenda
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

# Mostrar gráfico
plt.show()



