#Librerias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Título de la aplicación
st.title("Visualización de Resultados de Clustering")

# Cargar el archivo Excel
#archivo_excel = "C:\\pratics\\Data_Science\\Proyects\\SPRINT-3\\Resultados\\Resultados_Clustering.xlsx"

# Opción para que el usuario cargue el archivo Excel
uploaded_file = st.file_uploader("C:\\pratics\\Data_Science\\Proyects\\SPRINT-3\\Resultados\\Resultados_Clustering.xlsx", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Cargar el archivo Excel desde el archivo cargado
        df = pd.read_excel(uploaded_file, sheet_name=0)

        # Mostrar las primeras filas del dataframe
        st.subheader("Muestra de Datos")
        st.dataframe(df.head())

        # Mostrar las columnas del dataframe
        st.subheader("Columnas del DataFrame")
        st.write(df.columns.tolist())

        # Visualizar la distribución de los clusters (Matplotlib/Seaborn)
        st.subheader("Distribución de Clientes por Cluster")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df["cluster"], palette="viridis", ax=ax1)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Cantidad de Clientes")
        ax1.set_title("Distribución de Clientes por Cluster")
        st.pyplot(fig1)

        # Visualizar el engagement por cluster (Matplotlib/Seaborn)
        st.subheader("Engagement Score por Cluster")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df["cluster"], y=df["engagement_score"], palette="coolwarm", ax=ax2)
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Engagement Score")
        ax2.set_title("Engagement Score por Cluster")
        st.pyplot(fig2)

        # Visualizar la distribución de los clusters (Plotly)
        st.subheader("Distribución de Clientes por Cluster (Plotly)")
        fig_plotly = px.histogram(df, x="cluster", color="cluster", title="Distribución de Clientes por Cluster")
        st.plotly_chart(fig_plotly)

        # Visualizar la dispersión de Engagement Score vs. Otra Variable (Plotly)
        if "otra_variable" in df.columns:  # Verifica si la columna "otra_variable" existe
            st.subheader("Dispersión de Engagement Score vs. Otra Variable")
            fig_scatter = px.scatter(df, x="engagement_score", y="otra_variable", color="cluster", title="Dispersión por Cluster")
            st.plotly_chart(fig_scatter)

        # Métricas Resumen
        st.subheader("Métricas Resumen")
        clusters_count = df["cluster"].value_counts()
        st.write("Clientes por Cluster:", clusters_count)
        engagement_mean = df.groupby("cluster")["engagement_score"].mean()
        st.write("Promedio de Engagement por Cluster:", engagement_mean)

    except Exception as e:
        st.error(f"Ocurrió un error al cargar el archivo: {e}")