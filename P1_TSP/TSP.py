from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def generar_ciudades_con_distancias(n: int):
    np.random.seed(0)  # Para reproducibilidad
    ciudades = {chr(65 + i): (np.random.uniform(0, 10), np.random.uniform(0, 10)) for i in range(n)}
    return ciudades


def plotear_ruta(ciudades, ruta: List[str], mostrar_anotaciones: bool = True):
    coords = np.array([ciudades[ciudad] for ciudad in ruta])
    plt.figure(figsize=(10, 6))
    plt.plot(coords[:, 0], coords[:, 1], marker='o')
    plt.title("Ruta del Viajante de Comercio (Serdyukov)")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")

    if mostrar_anotaciones:
        for i, ciudad in enumerate(ruta):
            plt.annotate(ciudad, (coords[i, 0], coords[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.grid()
    plt.show()


def algoritmo_de_serdyukov(ciudades):
    n = len(ciudades)
    coords = np.array(list(ciudades.values()))
    distancias = distance_matrix(coords, coords)

    # Inicializar una ruta comenzando desde la primera ciudad
    ruta = [0]  # Comienza desde la primera ciudad
    visitados = set(ruta)

    while len(visitados) < n:
        ultimo = ruta[-1]
        siguiente = np.argmin([distancias[ultimo][j] if j not in visitados else np.inf for j in range(n)])
        ruta.append(siguiente)
        visitados.add(siguiente)

    # Convertir índices a letras
    return [chr(65 + ciudad) for ciudad in ruta]


class TSP:
    def __init__(self, ciudades):
        self.ciudades = ciudades

    def encontrar_la_ruta_mas_corta(self):
        return algoritmo_de_serdyukov(self.ciudades)

    def plotear_resultado(self, ruta: List[str], mostrar_anotaciones: bool = True):
        plotear_ruta(self.ciudades, ruta, mostrar_anotaciones)


def study_case_1():
    n_cities = 10
    ciudades = generar_ciudades_con_distancias(n_cities)
    tsp = TSP(ciudades)
    ruta = tsp.encontrar_la_ruta_mas_corta()
    tsp.plotear_resultado(ruta)


def study_case_2():
    n_cities = 100
    ciudades = generar_ciudades_con_distancias(n_cities)
    tsp = TSP(ciudades)
    ruta = tsp.encontrar_la_ruta_mas_corta()
    tsp.plotear_resultado(ruta, False)


if __name__ == "__main__":
    # Solve the TSP problem
    study_case_2()
    # Uncomment the following line to run study case 2
    # study_case_2()
