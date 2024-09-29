import matplotlib.pyplot as plt
import os, sys
import numpy as np
import networkx as nx
project_path = os.path.dirname(__file__)
sys.path.append(project_path)
from P1_util import define_color, define_black_space, uniform_cost_search, reconstruct_path 


class MazeLoader:
    def __init__(self, filename):
        self.filename = filename
        self.maze = None

    def load_Maze(self):
        _maze = []
        file_path = os.path.join(project_path, self.filename)
        print("Loading Maze from", file_path)
        with open(file_path, 'r') as file:
            for line in file:
                _maze.append(list(line.strip()))
        self.maze = _maze
        return self

    def plot_maze(self):
        height = len(self.maze)
        width = len(self.maze[0])

        fig = plt.figure(figsize=(width/4, height/4))  # Ajusta el tamaño de la figura según el tamaño del Maze
        for y in range(height):
            for x in range(width):
                cell = self.maze[y][x]
                color = define_color(cell)
                plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color=color, edgecolor='black')

        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.gca().invert_yaxis()  # Invierte el eje y para que el origen esté en la esquina inferior izquierda
        plt.xticks([])
        plt.yticks([])
        fig.tight_layout()
        plt.show()
        return self
    
    #Función - get_graph_matrix - Retorna una matriz que se utilizar para verificar adyacencia (1 si hay nodo, 2 si es espacio es entrada y 3 si es salida)
    def get_graph_matrix(self):
        
        #Dimensiones
        height = len(self.maze)
        width = len(self.maze[0])

        row = []
        graph_mtrix = []
        for y in range(height):
            for x in range(width):
                cell = self.maze[y][x]
                row.append(define_black_space(cell)) 
            graph_mtrix.append(row)
            row = []

        return np.array(graph_mtrix)
    

    def get_graph(self):
        
        graph_mtrix = self.get_graph_matrix()


        # Definir grafo a utilizar
        G = nx.Graph()

        # Definir un diccionario para los colores de los nodos
        node_colors = {}
        nodes_start_end = []

        # Añadir nodos y aristas basados en las posiciones de 1,2 y 3.
        for i in range(graph_mtrix.shape[0]):
            for j in range(graph_mtrix.shape[1]):
                if graph_mtrix[i, j] in [2, 3, 1]:
                    G.add_node((i, j))  # Añadir nodo en la posición (i, j)
                    

                    if graph_mtrix[i, j] == 2:
                        node_colors[(i, j)] = 'green'  # Color verde claro
                        nodes_start_end.append((i, j)) # Define inicio fin en mi lista de inicio fin -- INICIO
                    elif graph_mtrix[i, j] == 3:
                        node_colors[(i, j)] = 'red'
                        nodes_start_end.append((i, j)) # Define inicio fin en mi lista de inicio fin -- FIN
                    else:
                        node_colors[(i, j)] = 'lightblue'   # Color azul claro
                    
                    # Verificar adyacencias (arriba, abajo, izquierda, derecha)
                    if i > 0 and graph_mtrix[i-1, j] in [2, 3, 1]:  # Adyacencia arriba
                        G.add_edge((i, j), (i-1, j))
                    if i < graph_mtrix.shape[0] - 1 and graph_mtrix[i+1, j] in [2, 3, 1]:  # Adyacencia abajo
                        G.add_edge((i, j), (i+1, j))
                    if j > 0 and graph_mtrix[i, j-1] in [2, 3, 1]:  # Adyacencia izquierda
                        G.add_edge((i, j), (i, j-1))
                    if j < graph_mtrix.shape[1] - 1 and graph_mtrix[i, j+1] in [2, 3, 1]:  # Adyacencia derecha
                        G.add_edge((i, j), (i, j+1))

        # Extraer colores para los nodos

        # Posiciones y colores
        pos = {(i, j): (j, -i) for i, j in G.nodes()}  
        node_colors_list = [node_colors[node] for node in G.nodes()]

        nx.draw(G, pos, with_labels=True, node_color=node_colors_list, node_size=500, font_size=10)
        plt.show()

        return G, pos, node_colors_list, nodes_start_end #Retorna el nodo, las posiciones para el grafico, los colores para el grafico y los nodos de incio y fin
    
    

    #Función que devuelve el grafo
    def get_solved_ucs_graph(self):
        
        G, pos, node_colors_list, nodes_start_end = self.get_graph()
        print(nodes_start_end)


