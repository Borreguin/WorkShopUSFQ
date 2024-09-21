from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
   
if __name__ == '__main__':
    #  print("Implementa tu código aquí")

    # Función para generar un nuevo estado al mover un disco
    def move_disk(state, from_tower, to_tower):
        new_state = [tower[:] for tower in state]  # Copiar estado actual
        disk = new_state[from_tower].pop()  # Sacar el disco de la torre origen
        new_state[to_tower].append(disk)  # Colocar el disco en la torre destino
        return new_state


    # Función para generar todos los estados vecinos
    def generate_neighbors(state):
        neighbors = []
        for i in range(3):  # Torre origen
            if len(state[i]) == 0:  # Si la torre está vacía, no hacer nada
                continue
            for j in range(3):  # Torre destino
                if i != j and (len(state[j]) == 0 or state[i][-1] < state[j][-1]):
                    # Mover disco de i a j solo si es permitido
                    neighbors.append(move_disk(state, i, j))
        return neighbors


    # Algoritmo BFS para encontrar la solución
    def hanoi_bfs(initial_state, final_state):
        queue = deque([(initial_state, [])])  # Cola para BFS, almacena estado y camino
        visited = set()  # Conjunto para almacenar estados visitados
        visited.add(tuple(tuple(tower) for tower in initial_state))  # Marcar como visitado

        while queue:
            current_state, path = queue.popleft()  # Sacar el primer elemento de la cola

            # Si llegamos al estado final, retornar el camino
            if current_state == final_state:
                return path

            # Generar los vecinos y explorar
            for neighbor in generate_neighbors(current_state):
                neighbor_tuple = tuple(tuple(tower) for tower in neighbor)
                if neighbor_tuple not in visited:  # Si no ha sido visitado
                    visited.add(neighbor_tuple)
                    queue.append((neighbor, path + [neighbor]))  # Añadir nuevo estado y camino


    # Estado inicial y final para 3 discos
    initial_state = [[3, 2, 1], [], []]
    final_state = [[], [], [3, 2, 1]]

    # Ejecutar el algoritmo BFS
    solution = hanoi_bfs(initial_state, final_state)

    # Imprimir la secuencia de movimientos
    for step in solution:
        print(step)

    # Función para convertir un estado en un string (para usar como etiquetas de nodos)
    def state_to_string(state):
        return str([[len(tower) for tower in state]])


    # Función para generar el grafo basado en el algoritmo BFS
    def hanoi_graph_bfs(initial_state, final_state):
        G = nx.DiGraph()  # Grafo dirigido

        queue = deque([(initial_state, [])])
        visited = set()
        visited.add(tuple(tuple(tower) for tower in initial_state))

        G.add_node(state_to_string(initial_state))  # Añadir el nodo inicial

        while queue:
            current_state, path = queue.popleft()

            if current_state == final_state:
                break

            for neighbor in generate_neighbors(current_state):
                neighbor_tuple = tuple(tuple(tower) for tower in neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    G.add_node(state_to_string(neighbor))  # Añadir nodo vecino
                    G.add_edge(state_to_string(current_state), state_to_string(neighbor))  # Añadir arista
                    queue.append((neighbor, path + [neighbor]))

        return G


    # Estado inicial y final para 3 discos
    initial_state = [[3, 2, 1], [], []]
    final_state = [[], [], [3, 2, 1]]

    # Generar el grafo utilizando BFS
    G = hanoi_graph_bfs(initial_state, final_state)

    # Dibujar el grafo
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout del grafo

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
    plt.title("Visualización del grafo para la Torre de Hanoi (3 discos)")
    plt.show()
