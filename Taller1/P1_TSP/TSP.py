import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.spatial.distance import cdist

# elegant_print - poner separadores en los prints que se usan de cabecera
def elegant_print(texto, cant_separador):
    print("="*cant_separador)
    print(texto)
    print("-"*cant_separador)

# n_points -- Funcion extrae n puntos con coordenadas enteras aleatorias
def n_points(n, rng_inf, rng_supp):
    puntos_dict = {}

    # N puntos aleatorios, key = "P" + id_nodo
    for i in range(n):
        nombre = "P" + str(i)
        x = random.randint(rng_inf, rng_supp)
        y = random.randint(rng_inf, rng_supp)
        puntos_dict[nombre] = (x, y)

        #Calculo una matriz de distancias entre puntos
        coordenadas = np.array(list(puntos_dict.values()))
        distancias = np.round(cdist(coordenadas, coordenadas), 2)

    return puntos_dict, distancias


# calcular_distancia_total -- dada una ruta, calculo la distncia total, basado  en los pesos asignados a las aristas
def calcular_distancia_total(grafo, ruta):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += grafo[ruta[i]][ruta[i + 1]]['weight']
    # Añadir la distancia de regreso al punto de inicio
    distancia_total += grafo[ruta[-1]][ruta[0]]['weight']
    return distancia_total

# mejor_ruta_tsp -- encontrar mejor ruta a partir de un nodo -- utilizamos un enfoque brute force.
def mejor_ruta_tsp(grafo, nombres, nodo_inicial):
    mejor_distancia = float('inf')
    mejor_ruta = None

    # Permutaciones para todos los nodos, partiendo del inicial
    for perm in permutations(nombres):
        if perm[0] == nodo_inicial:  # Asegurar que la ruta comience desde el nodo inicial
            distancia = calcular_distancia_total(grafo, perm)
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_ruta = perm

    return mejor_ruta, mejor_distancia


# generar_mejor ruta -- Genera la mejor ruta basado en puntos, distancias y un nodo de inicio
def generar_mejor_ruta(dic_points, dist, init_node):
    # claves del diccionario
    keys = list(dic_points.keys())

    # grafo vacio
    G = nx.Graph()

    # nodos basados en mis puntos
    for nombre, coords in dic_points.items():
        G.add_node(nombre, pos=coords)

    # tomar para cada arista la distancia correspondiente
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:  # No agregar distancia a sí mismo
                G.add_edge(keys[i], keys[j], weight=dist[i, j])

    ruta, distancia_total = mejor_ruta_tsp(G, keys, init_node)

    # sE imprime la mejor ruta
    print(
        f"La mejor ruta a partir de {init_node} es: {' -> '.join(ruta)} con una distancia total de: {distancia_total}")

    # Generar las posiciones y pesos de las aristas para graficar
    pos = nx.get_node_attributes(G, 'pos')  # posiciones
    weights = nx.get_edge_attributes(G, 'weight')  # pesos

    # El nodo de inicio se colocara con color amarillo
    color_map = ['lightblue' if node != init_node else 'yellow' for node in G.nodes()]

    # dibujar nodos
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=color_map, font_size=10)

    # En primer lugar se seleccionan los colores para aristas que se correspondan con la mejor ruta encontrada.
    for i in range(len(ruta)):

        if i == len(ruta) - 1:
            edge_color = 'red'  # Última arista de regreso al inicio
            nx.draw_networkx_edges(G, pos, edgelist=[(ruta[i], ruta[0])], edge_color=edge_color, width=2)
        else:
            edge_color = 'red'
            nx.draw_networkx_edges(G, pos, edgelist=[(ruta[i], ruta[i + 1])], edge_color=edge_color, width=2)

    # Dibujar aristas regulares
    edges_to_draw = set(G.edges()) - set(zip(ruta[:-1], ruta[1:])) - {(ruta[-1], ruta[0])}
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color='lightgray')

    # Grafo final aristas + pesos
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

    plt.title(f"Grafo de Puntos con la Mejor Ruta a partir de {init_node} en Rojo:")
    plt.show()
    # main -- flujo de ejecucion:


def main():
    # Variables:
    num_aleatorio_puntos = 6
    min_eje_coordenadas = 4
    max_eje_coordenadas = 20

    # Generacion aleatoria de puntos y distancias
    dic_points, distancias = n_points(num_aleatorio_puntos, min_eje_coordenadas, max_eje_coordenadas)
    # Nodo aleatorio del que partira el viajante
    nodo_inicial = random.choice(list(dic_points.keys()))

    # Impresión  de puntos originales:
    elegant_print("Problema del viajante: ", 150)
    print("Puntos aleatorios generados")
    print(dic_points)

    # Problema resuelto:
    elegant_print("Resolución con los puntos aleatorios generados", 150)
    generar_mejor_ruta(dic_points, distancias, nodo_inicial)

    # ejecuta el main:


if __name__ == "__main__":
    main()