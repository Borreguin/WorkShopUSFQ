import os, sys
project_path = os.path.dirname(__file__)
sys.path.append(project_path)
from P1_MazeLoader import MazeLoader


def study_case_1():
    print("This is study case 1")
    maze_file = 'laberinto1.txt'
    maze = MazeLoader(maze_file).load_Maze().plot_maze()

    # Aquí la implementación de la solución:
    G, pos, node_colors_list, nodes_start_end = maze.get_graph(show_graph=False)
    #UCS ----------------------------------------------------------------------------
    ucs_path, ucs_time = maze.get_solved_ucs_graph(G, pos, node_colors_list, nodes_start_end,1)
    maze.get_graph(solution_path=ucs_path, Graph_name= f"Uniform Cost Search (UCS) (Tiempo: {ucs_time:.4f} segundos - Costo: {len(ucs_path)-1})")

    #DLS ----------------------------------------------------------------------------
    dls_path, dls_time = maze.depth_limited_search(start=nodes_start_end[0], goal=nodes_start_end[1])
    maze.get_graph(solution_path=dls_path, Graph_name= f"Depth Limited Search (DLS) (Tiempo: {dls_time:.4f} segundos - Costo: {len(ucs_path)-1})")
    print("Tiempo de búsqueda DLS:", dls_time)
    


def study_case_2():
    print("This is study case 2")
    maze_file = 'laberinto2.txt'
    maze = MazeLoader(maze_file).load_Maze().plot_maze()
    # Aquí la implementación de la solución:
    G, pos, node_colors_list, nodes_start_end = maze.get_graph(show_graph=False)
    #UCS ----------------------------------------------------------------------------
    ucs_path, ucs_time = maze.get_solved_ucs_graph(G, pos, node_colors_list, nodes_start_end,1)
    maze.get_graph(solution_path=ucs_path, Graph_name= f"Uniform Cost Search (UCS) (Tiempo: {ucs_time:.4f} segundos - Costo: {len(ucs_path)-1})")

    #DLS ----------------------------------------------------------------------------
    dls_path, dls_time = maze.depth_limited_search(start=nodes_start_end[0], goal=nodes_start_end[1])
    maze.get_graph(solution_path=dls_path, Graph_name= f"Depth Limited Search (DLS) (Tiempo: {dls_time:.4f} segundos - Costo: {len(ucs_path)-1})")
    print("Tiempo de búsqueda DLS:", dls_time)


def study_case_3():
    print("This is study case 3")
    maze_file = 'laberinto3.txt'
    maze = MazeLoader(maze_file).load_Maze().plot_maze()
    # Aquí la implementación de la solución:
    G, pos, node_colors_list, nodes_start_end = maze.get_graph(show_graph=False, small=True)
    #UCS ----------------------------------------------------------------------------
    ucs_path, ucs_time = maze.get_solved_ucs_graph(G, pos, node_colors_list, nodes_start_end,1)
    maze.get_graph(solution_path=ucs_path, Graph_name= f"Uniform Cost Search (UCS) (Tiempo: {ucs_time:.4f} segundos - Costo: {len(ucs_path)-1})",small=True)

    #DLS ----------------------------------------------------------------------------
    dls_path, dls_time = maze.depth_limited_search(start=nodes_start_end[0], goal=nodes_start_end[1])
    maze.get_graph(solution_path=dls_path, Graph_name= f"Depth Limited Search (DLS) (Tiempo: {dls_time:.4f} segundos - Costo: {len(dls_path)-1})",small=True)
    print("Tiempo de búsqueda DLS:", dls_time)



if __name__ == '__main__':

    case = input("Introduce un número de 1 a 3 para resolver el caso: ")
    if int(case) == 1:
        study_case_1()
    elif int(case) == 2:
        study_case_2()
    elif int(case) == 3:
        study_case_3()
    else:
        study_case_1()