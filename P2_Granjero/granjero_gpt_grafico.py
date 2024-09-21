import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


# Función para verificar si un estado es válido (nadie se come a nadie)
def is_valid(state):
    farmer, wolf, goat, cabbage = state
    if wolf == goat and farmer != goat:
        return False
    if goat == cabbage and farmer != goat:
        return False
    return True


# Función para generar los estados vecinos
def generate_neighbors(state):
    farmer, wolf, goat, cabbage = state
    neighbors = []

    # El granjero se mueve solo
    new_state = (1 - farmer, wolf, goat, cabbage)
    if is_valid(new_state):
        neighbors.append(new_state)

    # El granjero lleva al lobo
    if farmer == wolf:
        new_state = (1 - farmer, 1 - wolf, goat, cabbage)
        if is_valid(new_state):
            neighbors.append(new_state)

    # El granjero lleva a la cabra
    if farmer == goat:
        new_state = (1 - farmer, wolf, 1 - goat, cabbage)
        if is_valid(new_state):
            neighbors.append(new_state)

    # El granjero lleva la col
    if farmer == cabbage:
        new_state = (1 - farmer, wolf, goat, 1 - cabbage)
        if is_valid(new_state):
            neighbors.append(new_state)

    return neighbors


# Algoritmo BFS para encontrar el camino y construir el grafo
def farmer_bfs_with_graph(initial_state, final_state):
    G = nx.DiGraph()  # Crear un grafo dirigido
    queue = deque([(initial_state, [])])
    visited = set()
    visited.add(initial_state)

    while queue:
        current_state, path = queue.popleft()

        if current_state == final_state:
            break

        for neighbor in generate_neighbors(current_state):
            if neighbor not in visited:
                visited.add(neighbor)
                G.add_edge(current_state, neighbor)  # Añadir arista al grafo
                queue.append((neighbor, path + [current_state]))

    return G


# Estado inicial y final
initial_state = (0, 0, 0, 0)  # Todos en la orilla izquierda
final_state = (1, 1, 1, 1)  # Todos en la orilla derecha

# Crear el grafo usando BFS
G = farmer_bfs_with_graph(initial_state, final_state)

# Visualización del grafo
plt.figure(figsize=(14, 10))

# Utilizar el layout de resorte (spring) con un factor de dispersión para evitar superposición
pos = nx.spring_layout(G, seed=42, k=0.5)

# Dibujar nodos y aristas
nx.draw(G, pos, with_labels=False, node_size=4000, node_color="lightblue", font_size=10, font_weight="bold",
        arrows=True)

# Mostrar etiquetas de los nodos con su estado
labels = {node: f"{node}" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')

plt.title("Visualización del grafo para el acertijo del granjero, lobo, cabra y col", fontsize=16)
plt.show()
