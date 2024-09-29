import matplotlib.pyplot as plt
import os, sys
project_path = os.path.dirname(__file__)
sys.path.append(project_path)
from P1_util import define_color # Se importa la función de P1_util.py para asignar colores según los caracteres del laberinto
from collections import defaultdict # Importa defaultdict para usarlo en get_graph
# defauldict es una clase similar a un diccionario que proporciona un valor predeterminado para claves que aún no existen, siendo útil para la construcción de grafos


class MazeLoader:

    # Constructor que se inicializa con el nombre del archivo del laberinto a cargar
    def __init__(self, filename):
        self.filename = filename
        self.maze = None

    # Metodo para cargar el archivo del laberinto línea por línea
    def load_Maze(self):
        _maze = []
        file_path = os.path.join(project_path, self.filename)
        print("Loading Maze from", file_path)
        with open(file_path, 'r') as file:
            for line in file:
                _maze.append(list(line.strip())) # A cada línea se le elimina el espacio en blanco adicional y se convierte en una lista de caracteres (#, E, S o espacios)
        self.maze = _maze # El laberinto se almacena en una lista de listas
        return self

    # Metodo para visualizar el laberinto
    def plot_maze(self):
        height = len(self.maze)
        width = len(self.maze[0])

        fig = plt.figure(figsize=(width/4, height/4))  # Ajusta el tamaño de la figura según el tamaño del Maze
        for y in range(height):
            for x in range(width):
                cell = self.maze[y][x]
                color = define_color(cell)

                plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], color = color, edgecolor = 'black') # Se dibuja rectángulos para cada celda del laberinto 

        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.gca().invert_yaxis()  # Invierte el eje y para que el origen esté en la esquina inferior izquierda
        plt.xticks([])
        plt.yticks([])
        fig.tight_layout()
        plt.show()

        return self

    # Metodo para construir un grafo a partir del laberinto (donde cada celda transitable (que no sea #) se trata como un nodo
    def get_graph(self):
        # Implementación de la creación del grafo a partir del laberinto
        graph = defaultdict(list)
        height = len(self.maze)
        width = len(self.maze[0])
        for y in range(height):
            for x in range(width):
                if self.maze[y][x] != '#':
                    if x > 0 and self.maze[y][x-1] != '#':
                        graph[(y, x)].append((y, x-1))
                    if x < width - 1 and self.maze[y][x+1] != '#':
                        graph[(y, x)].append((y, x+1))
                    if y > 0 and self.maze[y-1][x] != '#':
                        graph[(y, x)].append((y-1, x))
                    if y < height - 1 and self.maze[y+1][x] != '#':
                        graph[(y, x)].append((y+1, x))
        return graph