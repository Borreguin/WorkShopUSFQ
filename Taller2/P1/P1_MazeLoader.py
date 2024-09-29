import matplotlib.pyplot as plt
import os, sys
import numpy as np
import networkx as nx
import time
import math
project_path = os.path.dirname(__file__)
sys.path.append(project_path)
from P1_util import define_color, define_black_space, uniform_cost_search, reconstruct_path, elegant_print 


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
    


    def get_graph(self, solution_path=None, Graph_name = 'Laberinto', show_graph = True, small=False):
            
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
                            G.add_edge((i, j), (i-1, j),weight =1)
                        if i < graph_mtrix.shape[0] - 1 and graph_mtrix[i+1, j] in [2, 3, 1]:  # Adyacencia abajo
                            G.add_edge((i, j), (i+1, j),weight =1)
                        if j > 0 and graph_mtrix[i, j-1] in [2, 3, 1]:  # Adyacencia izquierda
                            G.add_edge((i, j), (i, j-1),weight =1)
                        if j < graph_mtrix.shape[1] - 1 and graph_mtrix[i, j+1] in [2, 3, 1]:  # Adyacencia derecha
                            G.add_edge((i, j), (i, j+1),weight =1)

            
            # Posiciones y colores
            pos = {(i, j): (j, -i) for i, j in G.nodes()}  

            if solution_path is not None:
                print(Graph_name)
                for node in solution_path:
                    if node in nodes_start_end:
                        node_colors[node] = node_colors[node]
                    else:
                        node_colors[node] = 'orange'
            
            node_colors_list = [node_colors[node] for node in G.nodes()]

            if show_graph:
                plt.figure(Graph_name)
            if small:
                nx.draw(G, pos, with_labels=False, node_color=node_colors_list, node_size=50, font_size=8)
            else:
                nx.draw(G, pos, with_labels=True, node_color=node_colors_list, node_size=500, font_size=10)
            plt.title(Graph_name)
            self.graph = G
            if show_graph:
                plt.show()

            return G, pos, node_colors_list, nodes_start_end #Retorna el nodo, las posiciones para el grafico, los colores para el grafico y los nodos de incio y fin
        


# Resolucion UCS ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #Función que devuelve el grafo
    def get_solved_ucs_graph(self,G, pos, node_colors_list, nodes_start_end, with_label_vis):
        
        inicio = time.perf_counter()  
                    

        # Example usage of the UCS function
        start_node = nodes_start_end[0]
        goal_node = nodes_start_end[1]
        result = uniform_cost_search(G, start_node, goal_node)

        total_cost, path = result
        #print(f"Least cost path from {start_node} to {goal_node}: {' -> '.join(path)} with total cost {total_cost}")

        fin = time.perf_counter()
        total_cost, path = result

        elegant_print("RESOLUCION POR MÉTODO UCS:",200)
        print("Elapsed Time: ", round((fin - inicio),6))
        print(f"Ruta de menor costo desde nodo {start_node} a nodo {goal_node}: {' -> '.join(map(str, path))} con un costo total = {total_cost} y tiempo de = {round((fin - inicio),6)} [s]")


        return path, round((fin - inicio),6)
    
# Resoulcion DLS ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def dls(self, node, goal, depth, visited=None):
            if visited is None:
                visited = set()

            # Si hemos alcanzado el nodo objetivo, retornamos el camino actual
            if node == goal:
                return [node]

            # Si hemos alcanzado la profundidad máxima permitida, devolvemos None
            if depth <= 0:
                return None

            # Marcar el nodo actual como visitado
            visited.add(node)

            # Obtener los vecinos y ordenarlos según la distancia al nodo objetivo
            neighbors = list(self.graph.neighbors(node))
            neighbors.sort(key=lambda x: math.dist(x, goal))

            # Exploración recursiva de los nodos vecinos ordenados
            for neighbor in neighbors:
                if neighbor not in visited:
                    path = self.dls(neighbor, goal, depth - 1, visited)
                    if path is not None:
                        return [node] + path

            # Si no se encuentra una solución, desmarcar el nodo para permitir otras búsquedas
            visited.remove(node)
            return None
        
    def dls_ver(self, node, goal, depth, visited=None):
            if visited is None:
                visited = set()

            # Si hemos alcanzado el nodo objetivo, retornamos el camino actual
            if node == goal:
                return [node]

            # Si hemos alcanzado la profundidad máxima permitida, devolvemos None
            if depth <= 0:
                return None

            # Marcar el nodo actual como visitado
            visited.add(node)

            # Obtener los vecinos
            neighbors = list(self.graph.neighbors(node))

            # Separar los vecinos en verticales y horizontales
            vertical_neighbors = []
            horizontal_neighbors = []

            for neighbor in neighbors:
                if neighbor[0] != node[0]:  # Cambio en la fila -> movimiento vertical
                    vertical_neighbors.append(neighbor)
                else:  # Cambio en la columna -> movimiento horizontal
                    horizontal_neighbors.append(neighbor)

            # Priorizar exploración vertical primero
            ordered_neighbors = vertical_neighbors + horizontal_neighbors

            # Exploración recursiva de los nodos vecinos ordenados
            for neighbor in ordered_neighbors:
                if neighbor not in visited:
                    path = self.dls_ver(neighbor, goal, depth - 1, visited)
                    if path is not None:
                        return [node] + path

            # Si no se encuentra una solución, desmarcar el nodo para permitir otras búsquedas
            visited.remove(node)
            return None


    def depth_limited_search(self, start, goal, limit = None, Graph_name='Depth Limited Search as', show_graph=False, vertical=True):
            if limit is None:
                limit = len(self.graph.nodes)
            start_time = time.time()
            
            # Ejecutar la búsqueda
            self.graph, _, _, nodes_start_end = self.get_graph(Graph_name=Graph_name, show_graph=show_graph)
            start_node, goal_node = nodes_start_end[0], nodes_start_end[1]
            if vertical:
                solution_path = self.dls_ver(start_node, goal_node, limit)
            else:
                solution_path = self.dls(start_node, goal_node, limit)

            end_time = time.time()
            elapsed_time = end_time - start_time

            elegant_print("RESOLUCION POR MÉTODO DLS:",200)
            print("Elapsed Time: ", round(elapsed_time,6))
            print(f"Ruta de menor costo desde nodo {start_node} a nodo {goal_node}: {' -> '.join(map(str, solution_path))} con un costo total = {len(solution_path)-1} y tiempo de = {round((elapsed_time),6)} [s]")

            return solution_path, elapsed_time
        
    def get_nodes_length(self):
            if not hasattr(self, 'graph'):
                self.graph = self.get_graph()

            # Retornar el número de nodos del grafo
            return self.graph.number_of_nodes()



