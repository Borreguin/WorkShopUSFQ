from collections import deque


# Función para verificar si un estado es válido (nadie se come a nadie)
def is_valid(state):
    farmer, wolf, goat, cabbage = state

    # Si el lobo está con la cabra sin el granjero, no es válido
    if wolf == goat and farmer != goat:
        return False
    # Si la cabra está con la col sin el granjero, no es válido
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


# Algoritmo BFS para encontrar el camino
def farmer_bfs(initial_state, final_state):
    queue = deque([(initial_state, [])])  # Cola para BFS
    visited = set()  # Conjunto de estados visitados
    visited.add(initial_state)

    while queue:
        current_state, path = queue.popleft()

        # Si llegamos al estado final, retornamos el camino
        if current_state == final_state:
            return path + [current_state]

        # Generar vecinos válidos y explorar
        for neighbor in generate_neighbors(current_state):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [current_state]))


# Estado inicial y final
initial_state = (0, 0, 0, 0)
final_state = (1, 1, 1, 1)

# Ejecutar el algoritmo BFS
solution = farmer_bfs(initial_state, final_state)

# Imprimir la solución paso a paso
for step in solution:
    print(step)
