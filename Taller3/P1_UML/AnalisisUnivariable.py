import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from Taller3.P1_UML.p1_uml_util import *


def prepare_data():
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, "data")
    file_path = os.path.join(data_path, "data.csv")
    _df = read_csv_file(file_path)
    _df['V005_vent01_CO2'] = _df['V005_vent01_CO2'].fillna(
      _df['V005_vent01_CO2'].mean())
    _df['V022_vent02_CO2'] = _df['V022_vent02_CO2'].fillna(
      _df['V022_vent02_CO2'].mean())
    _df['V006_vent01_temp_out'] = _df['V006_vent01_temp_out'].fillna(
      _df['V006_vent01_temp_out'].mean())
    _df['V023_vent02_temp_out'] = _df['V023_vent02_temp_out'].fillna(
      _df['V023_vent02_temp_out'].mean())
    _df['timestamp'] = pd.to_datetime(_df['timestamp'], format='%d.%m.%Y %H:%M')
    _df.set_index('timestamp', inplace=True)
    return _df

def all_registers(_df: pd.DataFrame):
  # Agrupar por día manteniendo solo los días con 24 registros
  daily_grouped = _df.groupby(_df.index.date).filter(lambda x: len(x) == 24)

  # Volver a agrupar por día después del filtro
  daily_grouped = daily_grouped.groupby(daily_grouped.index.date)

  # Convertir los datos agrupados en matrices de 24x4 para cada día
  daily_matrices = [group.values for _, group in daily_grouped]

  # Crear un gráfico de líneas por cada variable
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))

  # Extraer cada variable y graficar
  for i, (ax, var) in enumerate(zip(axes.flat, df.columns)):
    data_by_variable = np.array([matrix[:, i] for matrix in daily_matrices if
                                 matrix.shape == (24,
                                                  4)]).T  # Asegurar que solo se incluyan días con 24x4 datos
    ax.plot(data_by_variable, alpha=0.6)
    ax.set_title(f'Variable: {var}')
    ax.set_xlabel('Hora del día (0-23)')
    ax.set_ylabel(var)

  plt.tight_layout()
  plt.show()

def k_means(_df: pd.DataFrame):
  # Agrupar por día manteniendo solo los días con 24 registros
  daily_grouped = _df.groupby(_df.index.date).filter(lambda x: len(x) == 24)

  # Volver a agrupar por día después del filtro
  daily_grouped = daily_grouped.groupby(daily_grouped.index.date)

  # Convertir los datos agrupados en matrices de 24x1 para la variable 'V022_vent02_CO2'
  daily_matrices = [group['V022_vent02_CO2'].values for _, group in
                    daily_grouped]

  # Aplicar K-Means a los datos
  n_clusters = 3  # Elige el número de clusters
  daily_data = np.array(
      [matrix for matrix in daily_matrices if len(matrix) == 24])

  kmeans = KMeans(n_clusters=n_clusters, random_state=0)
  kmeans_labels = kmeans.fit_predict(daily_data)

  # Graficar los resultados de K-Means con los clusters en un gráfico de líneas
  plt.figure(figsize=(10, 6))

  # Graficar cada día como una línea, coloreada por su cluster
  for i, (data, label) in enumerate(zip(daily_data, kmeans_labels)):
    plt.plot(range(24), data, label=f'Cluster {label}',
             color=plt.cm.viridis(label / n_clusters), alpha=0.6)

  # Ajustar el gráfico
  plt.title("K-Means Clustering de V022_vent02_CO2 con gráfico de líneas")
  plt.xlabel('Hora del día (0-23)')
  plt.ylabel('V022_vent02_CO2')
  plt.show()

def k_means_with_centroides(_df: pd.DataFrame):
  # Agrupar por día manteniendo solo los días con 24 registros
  daily_grouped = _df.groupby(_df.index.date).filter(lambda x: len(x) == 24)

  # Volver a agrupar por día después del filtro
  daily_grouped = daily_grouped.groupby(daily_grouped.index.date)

  # Convertir los datos agrupados en matrices de 24x1 para la variable 'V022_vent02_CO2'
  daily_matrices = [group['V022_vent02_CO2'].values for _, group in
                    daily_grouped]

  # Aplicar K-Means a los datos
  n_clusters = 3  # Elige el número de clusters
  daily_data = np.array(
      [matrix for matrix in daily_matrices if len(matrix) == 24])

  kmeans = KMeans(n_clusters=n_clusters, random_state=0)
  kmeans_labels = kmeans.fit_predict(daily_data)

  # Graficar los resultados de K-Means con los clusters en un gráfico de líneas
  plt.figure(figsize=(10, 6))

  # Graficar cada día como una línea, coloreada por su cluster
  for i, (data, label) in enumerate(zip(daily_data, kmeans_labels)):
    plt.plot(range(24), data, color=plt.cm.viridis(label / n_clusters),
             alpha=0.3)


  centroides = kmeans.cluster_centers_
  centroid_colors = ['red', 'blue', 'green']  # Colores para cada cluster
  for i, centroide in enumerate(centroides):
    plt.plot(range(24), centroide, color=centroid_colors[i], linewidth=3)

  # Ajustar el gráfico
  plt.title("K-Means Clustering de V022_vent02_CO2 con centroides")
  plt.xlabel('Hora del día (0-23)')
  plt.ylabel('V022_vent02_CO2')
  plt.show()

def gaussian_mixture_model(_df: pd.DataFrame):
  # Agrupar por día manteniendo solo los días con 24 registros
  daily_grouped = df.groupby(df.index.date).filter(lambda x: len(x) == 24)

  # Volver a agrupar por día después del filtro
  daily_grouped = daily_grouped.groupby(daily_grouped.index.date)

  # Convertir los datos agrupados en matrices de 24x1 para la variable 'V022_vent02_CO2'
  daily_matrices = [group['V022_vent02_CO2'].values for _, group in
                    daily_grouped]

  # Aplicar Gaussian Mixture Model (GMM) a los datos con 10 clusters
  n_components = 10  # Cambiar el número de componentes (clusters) a 10
  daily_data = np.array(
      [matrix for matrix in daily_matrices if len(matrix) == 24])

  gmm = GaussianMixture(n_components=n_components, random_state=0)
  gmm_labels = gmm.fit_predict(daily_data)

  # Graficar los resultados de GMM con los clusters en un gráfico de líneas
  plt.figure(figsize=(10, 6))

  # Graficar cada día como una línea, coloreada por su cluster
  for i, (data, label) in enumerate(zip(daily_data, gmm_labels)):
    plt.plot(range(24), data, color=plt.cm.viridis(label / n_components),
             alpha=0.3)

  # Calcular y graficar los centroides (medias gaussianas)
  centroides = gmm.means_
  centroid_colors = plt.cm.viridis(
    np.linspace(0, 1, n_components))  # Colores para cada cluster
  for i, centroide in enumerate(centroides):
    plt.plot(range(24), centroide, color=centroid_colors[i], linewidth=3)

  # Ajustar el gráfico
  plt.title("Gaussian Mixture Model de V022_vent02_CO2 con 10 clusters")
  plt.xlabel('Hora del día (0-23)')
  plt.ylabel('V022_vent02_CO2')
  plt.show()

def pca(_df: pd.DataFrame):
  # Agrupar por día manteniendo solo los días con 24 registros
  daily_grouped = df.groupby(df.index.date).filter(lambda x: len(x) == 24)

  # Volver a agrupar por día después del filtro
  daily_grouped = daily_grouped.groupby(daily_grouped.index.date)

  # Convertir los datos agrupados en matrices de 24x1 para la variable 'V022_vent02_CO2'
  daily_matrices = [group['V022_vent02_CO2'].values for _, group in
                    daily_grouped]

  # Aplicar K-Means a los datos
  n_clusters = 3  # Elige el número de clusters
  daily_data = np.array(
      [matrix for matrix in daily_matrices if len(matrix) == 24])

  kmeans = KMeans(n_clusters=n_clusters, random_state=0)
  kmeans_labels = kmeans.fit_predict(daily_data)
  distances = kmeans.transform(daily_data)  # Distancias a los centroides

  # Detectar anomalías: los puntos con mayor distancia a su cluster
  threshold = np.percentile(np.min(distances, axis=1),
                            97)  # Tomar el 5% más alejado
  anomalies = np.where(np.min(distances, axis=1) > threshold)[0]

  # Aplicar PCA para reducir la dimensionalidad
  pca = PCA(n_components=2)
  pca_data = pca.fit_transform(daily_data)

  # Graficar los resultados con anomalías
  plt.figure(figsize=(10, 6))

  # Graficar los puntos normales
  plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap='viridis',
              alpha=0.6, label='Datos normales')

  # Resaltar las anomalías
  plt.scatter(pca_data[anomalies, 0], pca_data[anomalies, 1], color='red',
              label='Anomalías', edgecolor='k', s=100)

  # Ajustar el gráfico
  plt.title("Detección de anomalías usando PCA")
  plt.xlabel('Componente Principal 1')
  plt.ylabel('Componente Principal 2')
  plt.legend()
  plt.show()

def pca_2(_df: pd.DataFrame):
  # Agrupar por día manteniendo solo los días con 24 registros
  daily_grouped = df.groupby(df.index.date).filter(lambda x: len(x) == 24)

  # Volver a agrupar por día después del filtro
  daily_grouped = daily_grouped.groupby(daily_grouped.index.date)

  # Convertir los datos agrupados en matrices de 24x1 para la variable 'V022_vent02_CO2'
  daily_matrices = [group['V022_vent02_CO2'].values for _, group in
                    daily_grouped]

  # Aplicar K-Means a los datos
  n_clusters = 3  # Elige el número de clusters
  daily_data = np.array(
      [matrix for matrix in daily_matrices if len(matrix) == 24])

  kmeans = KMeans(n_clusters=n_clusters, random_state=0)
  kmeans_labels = kmeans.fit_predict(daily_data)
  distances = kmeans.transform(daily_data)  # Distancias a los centroides

  # Detectar anomalías: los puntos con mayor distancia a su cluster
  threshold = np.percentile(np.min(distances, axis=1),
                            95)  # Tomar el 5% más alejado
  anomalies = np.where(np.min(distances, axis=1) > threshold)[0]

  plt.figure(figsize=(10, 6))

  for anomaly in anomalies:
    plt.plot(range(24), daily_data[anomaly], color='red', alpha=0.6)

  # Ajustar el gráfico
  plt.title("Anomalías detectadas en V022_vent02_CO2")
  plt.xlabel('Hora del día (0-23)')
  plt.ylabel('V022_vent02_CO2')
  plt.show()

def anomalias_result(_df: pd.DataFrame):
  # Agrupar por día manteniendo solo los días con 24 registros
  daily_grouped = df.groupby(df.index.date).filter(lambda x: len(x) == 24)

  # Volver a agrupar por día después del filtro
  daily_grouped = daily_grouped.groupby(daily_grouped.index.date)

  # Convertir los datos agrupados en matrices de 24x1 para la variable 'V022_vent02_CO2'
  daily_matrices = [group['V022_vent02_CO2'].values for _, group in
                    daily_grouped]

  # Aplicar K-Means a los datos
  n_clusters = 3  # Elige el número de clusters
  daily_data = np.array(
      [matrix for matrix in daily_matrices if len(matrix) == 24])

  kmeans = KMeans(n_clusters=n_clusters, random_state=0)
  kmeans_labels = kmeans.fit_predict(daily_data)
  distances = kmeans.transform(daily_data)  # Distancias a los centroides

  # Detectar anomalías: los puntos con mayor distancia a su cluster
  threshold = np.percentile(np.min(distances, axis=1),
                            95)  # Tomar el 5% más alejado
  anomalies = np.where(np.min(distances, axis=1) > threshold)[0]
  # Separar los días anómalos de los normales
  normal_days = np.delete(daily_data, anomalies,
                          axis=0)  # Eliminar las anomalías
  anomalous_days = daily_data[anomalies]  # Mantener las anomalías

  # Calcular estadísticas descriptivas para días normales y anómalos
  normal_stats = {
    'mean': np.mean(normal_days, axis=0),
    'min': np.min(normal_days, axis=0),
    'max': np.max(normal_days, axis=0)
  }

  anomalous_stats = {
    'mean': np.mean(anomalous_days, axis=0),
    'min': np.min(anomalous_days, axis=0),
    'max': np.max(anomalous_days, axis=0)
  }

  # Comparar visualmente las estadísticas
  plt.figure(figsize=(14, 8))

  # Gráfico de la media
  plt.subplot(2, 1, 1)
  plt.plot(range(24), normal_stats['mean'], label='Media de días normales',
           color='green')
  plt.plot(range(24), anomalous_stats['mean'], label='Media de días anómalos',
           color='red')
  plt.fill_between(range(24), anomalous_stats['min'], anomalous_stats['max'],
                   color='red', alpha=0.3, label='Rango días anómalos')
  plt.title("Comparación de días normales y anómalos (V022_vent02_CO2)")
  plt.xlabel('Hora del día (0-23)')
  plt.ylabel('V022_vent02_CO2')
  plt.legend()

  # Gráfico del rango (min, max)
  plt.subplot(2, 1, 2)
  plt.plot(range(24), normal_stats['min'], '--', color='green',
           label='Mínimo días normales')
  plt.plot(range(24), normal_stats['max'], '--', color='green',
           label='Máximo días normales')
  plt.plot(range(24), anomalous_stats['min'], '--', color='red',
           label='Mínimo días anómalos')
  plt.plot(range(24), anomalous_stats['max'], '--', color='red',
           label='Máximo días anómalos')
  plt.title("Comparación del rango de días normales y anómalos")
  plt.xlabel('Hora del día (0-23)')
  plt.ylabel('V022_vent02_CO2')
  plt.legend()

  plt.tight_layout()
  plt.show()




if __name__ == "__main__":
    df = prepare_data()
    all_registers(df)
    k_means(df)
    k_means_with_centroides(df)
    gaussian_mixture_model(df)
    pca(df)
    pca_2(df)
    anomalias_result(df)
