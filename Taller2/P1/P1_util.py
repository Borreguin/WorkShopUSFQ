import heapq

def define_color(cell):
    if cell == '#':
        return 'black'
    elif cell == ' ':   # Espacio vacío
        return 'white'
    elif cell == 'E':   # Entrada
        return 'green'
    elif cell == 'S':   # Salida
        return 'red'
     
def define_black_space(cell):
    if cell == ' ':
        return 1 # Libre
    elif cell == 'E':   # Entrada
        return 2
    elif cell == 'S':   # Salida
        return 3
    else:
        return 0 #Pared
    

def uniform_cost_search(graph, start, goal):
    # Cola de prioridad:
    priority_queue = [(0, start)]
    # Diccionario quie contiene los nodos recorridos
    visited = {start: (0, None)}
    
    while priority_queue:
        # Buscar nodos con menor costo de la lista de prioridad
        current_cost, current_node = heapq.heappop(priority_queue)
        
        # Si acabamos, reconstruir el path y dejar el costo como esta
        if current_node == goal:
            return current_cost, reconstruct_path(visited, start, goal)
        

        for neighbor in graph.neighbors(current_node):
            total_cost = current_cost + graph[current_node][neighbor]['weight']
            # checar nodos mejores
            if neighbor not in visited or total_cost < visited[neighbor][0]:
                visited[neighbor] = (total_cost, current_node)
                heapq.heappush(priority_queue, (total_cost, neighbor))
    
    # si no se puede encontrar un camino retornar none.
    return None

def reconstruct_path(visited, start, goal):
    # Reconstruir el path verificando los nodos visitados
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = visited[current][1]  # Get the parent node
    path.reverse()
    return path

def elegant_print(message,num_sep):
    print("="*num_sep)
    print(message)
    print("-"*num_sep)
