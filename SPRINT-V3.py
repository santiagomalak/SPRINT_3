# %%
#IMPORTANDO LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import joblib

# %%
#CARGA DE ARCHIVOS EXCEL
df = pd.read_excel("C:\\pratics\\Data_Science\\Dataset\\clustering_mailing.xlsx", sheet_name="Sheet1")

# %%
#EXPLORACION INICIAL DE LOS ARCHIVOS 
df.head()

# %%
df.info

# %%
df.describe()

# %%
#ELIMINAR NULOS
df.dropna()

# %%
#VISUALIZACION
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlación")
plt.show()

# %%
# Creación de una nueva variable: Engagement Score
#Objetivo : Capturar el nivel de interacción de los clientes con la campaña de correo electrónico.
df["engagement_score"] = df["open"] + (df["click"] * 2) + (df["Comprador"] * 3)

# %%
# Selección de variables para clustering
features = ["send", "bounce", "open", "click", "Total", "hour", "day_of_week", "engagement_score"]
X = df[features]

# %%
# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '8' # Reemplaza 4 con el número de núcleos que deseas usar

# %%
# Determinar número óptimo de clusters - Método del Codo
inertia = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# %%
print(np.isnan(X_scaled).sum())  # Verifica si hay valores NaN

# %%
# 8. Determinar número óptimo de clusters con una muestra para optimizar rendimiento
sample_size = min(2000, len(X_scaled))
X_sample = X_scaled[np.random.choice(len(X_scaled), sample_size, replace=False)]

inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
    kmeans.fit(X_sample)
    inertia.append(kmeans.inertia_)
    labels = kmeans.predict(X_sample)
    silhouette_scores.append(silhouette_score(X_sample, labels))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(k_range, inertia, marker='o', label='Inercia', color='blue')
ax2.plot(k_range, silhouette_scores, marker='s', label='Silhouette Score', color='red')
ax1.set_xlabel('Número de Clusters')
ax1.set_ylabel('Inercia', color='blue')
ax2.set_ylabel('Silhouette Score', color='red')
plt.title('Método del Codo y Silhouette Score')
plt.legend()
plt.show()


# %%
# 9. Evaluar la distribución del Silhouette Score por cluster
k_optimo = 6  # Ajustar según el gráfico
total_kmeans = MiniBatchKMeans(n_clusters=k_optimo, random_state=42, batch_size=256)
df["cluster"] = total_kmeans.fit_predict(X_scaled)
silhouette_vals = silhouette_samples(X_scaled, df["cluster"])
df["silhouette_score"] = silhouette_vals

plt.figure(figsize=(8, 5))
sns.violinplot(x=df["cluster"], y=df["silhouette_score"], palette="muted")
plt.xlabel("Cluster")
plt.ylabel("Silhouette Score")
plt.title("Distribución del Silhouette Score por Cluster")
plt.show()

# %%
# 10. Visualización de clusters con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["pca1"], y=df["pca2"], hue=df["cluster"], palette='tab10')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Clusters Visualizados con PCA')
plt.show()

# %%
# 11. Interpretación y Recomendaciones
cluster_summary = df.groupby("cluster").mean()
print(cluster_summary)

# %% [markdown]
# Análisis de los clusters:
# 
# Cluster 0:
# send bajo, bounce bajo, open alto, click bajo, engagement_score moderado.
# Usuarios que abren correos pero interactúan poco.
# 
# Cluster 1:
# send bajo, bounce alto, open bajo, click muy bajo, engagement_score bajo.
# Usuarios poco comprometidos con altas tasas de rebote.
# 
# Cluster 2:
# send muy alto, bounce bajo, open muy alto, click moderado, engagement_score alto.
# Usuarios altamente comprometidos con muchas aperturas e interacciones.
# 
# Cluster 3:
# send bajo, bounce muy alto, open muy bajo, click muy bajo, engagement_score muy bajo.
# Usuarios con tasas de rebote extremadamente altas y muy poco compromiso.
# 
# Cluster 4:
# send moderado, bounce bajo, open bajo, click bajo, engagement_score bajo.
# Usuarios con bajo compromiso general.
# 
# Cluster 5:
# send bajo, bounce bajo, open bajo, click bajo, engagement_score moderado.
# Usuarios con un compromiso moderado, similar al cluster 0.

# %%
# Posibles recomendaciones para marketing
for cluster_id, group in cluster_summary.iterrows():
    print(f"Cluster {cluster_id}: ")
    if group["engagement_score"] > np.mean(df["engagement_score"]):
        print(" - Clientes altamente comprometidos. Sugerencia: Ofrecer promociones exclusivas.")
    else:
        print(" - Clientes de bajo compromiso. Sugerencia: Reenviar campañas de emails con incentivos.")
    print("\n")


# %% [markdown]
# Análisis de las recomendaciones:
# 
# Clusters 0, 2, 4 y 5: Se clasifican como "Clientes altamente comprometidos" y se sugiere ofrecer promociones exclusivas. Esto indica que estos clusters tienen un buen nivel de compromiso y podrían responder bien a ofertas especiales.
# 
# Clusters 1 y 3: Se clasifican como "Clientes de bajo compromiso" y se sugiere reenviar campañas de emails con incentivos. Esto indica que estos clusters necesitan estrategias para aumentar su participación y podrían beneficiarse de incentivos para interactuar con los correos electrónicos.
# 
# Recomendaciones de marketing:
# 
# Clientes altamente comprometidos:
# Ofrecer promociones exclusivas y personalizadas.
# Implementar programas de fidelización.
# Solicitar comentarios y opiniones para mantener su compromiso.
# 
# Clientes de bajo compromiso:
# Reenviar campañas de emails con incentivos (descuentos, contenido exclusivo, etc.).
# Realizar encuestas o entrevistas para entender sus intereses.
# Segmentar aún más este grupo para personalizar el contenido.

# %%
# Posibles recomendaciones para marketing
recomendaciones = []
for cluster_id, group in cluster_summary.iterrows():
    if group["engagement_score"] > np.mean(df["engagement_score"]):
        recomendaciones.append("Alta");
    else:
        recomendaciones.append("Baja");

df_recomendaciones = pd.DataFrame({
    "Cluster": cluster_summary.index,
    "Nivel de compromiso": recomendaciones
})

plt.figure(figsize=(8, 5))
sns.barplot(x=df_recomendaciones["Cluster"], y=cluster_summary["engagement_score"], hue=df_recomendaciones["Nivel de compromiso"], palette={"Alta": "green", "Baja": "red"})
plt.xlabel("Cluster")
plt.ylabel("Engagement Score Promedio")
plt.title("Recomendaciones de Marketing por Cluster")
plt.legend(title="Nivel de compromiso")
plt.show()

# %%
# 12. Guardar el modelo para uso futuro
joblib.dump(total_kmeans, "C:\\pratics\\Data_Science\\Proyects\\SPRINT-3\\Resultados\\modelo_clustering.pkl")
joblib.dump(scaler, "C:\\pratics\\Data_Science\\Proyects\\SPRINT-3\\Resultados\\scaler.pkl")

# %%
# 13. Exportar resultados a Excel
with pd.ExcelWriter("C:\\pratics\\Data_Science\\Proyects\\SPRINT-3\\Resultados\\Resultados_Clustering.xlsx") as writer:
    df.to_excel(writer, sheet_name="Clusters", index=False)
    df_recomendaciones.to_excel(writer, sheet_name="Recomendaciones", index=False)

print("Archivo 'Resultados_Clustering.xlsx' y modelo guardados correctamente.")



