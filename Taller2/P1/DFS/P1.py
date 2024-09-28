import os, sys
project_path = os.path.dirname(__file__)
sys.path.append(project_path)
from P1_MazeLoader import MazeLoader


def study_case_1():
    print("This is study case 1")
    maze_file = 'laberinto1.txt'
    maze = MazeLoader(maze_file).load_Maze().plot_maze()
    # Aquí la implementación de la solución:
    graph = maze.get_graph()

     # Verificación básica
    print(f"Número total de nodos: {len(graph)}")
    entrada = next((node for node, info in graph.items() if info['type'] == 'entrance'), None)
    salida = next((node for node, info in graph.items() if info['type'] == 'exit'), None)
    print(f"Entrada: {entrada}")
    print(f"Salida: {salida}")
    print(f"Vecinos de la entrada: {graph[entrada]['neighbors']}")
    print(f"Vecinos de la salida: {graph[salida]['neighbors']}")



def study_case_2():
    print("This is study case 2")
    maze_file = 'laberinto2.txt'
    maze = MazeLoader(maze_file).load_Maze().plot_maze()
    # Aquí la implementación de la solución:
    graph = maze.get_graph()


def study_case_3():
    print("This is study case 2")
    maze_file = 'laberinto3.txt'
    maze = MazeLoader(maze_file).load_Maze().plot_maze()
    # Aquí la implementación de la solución:
    graph = maze.get_graph()
3

if __name__ == '__main__':
    study_case_1()
