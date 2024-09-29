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
    # Priority queue to store the frontier nodes, initialized with the start node
    priority_queue = [(0, start)]
    # Dictionary to store the cost of the shortest path to each node
    visited = {start: (0, None)}
    
    while priority_queue:
        # Pop the node with the lowest cost from the priority queue
        current_cost, current_node = heapq.heappop(priority_queue)
        
        # If we reached the goal, return the total cost and the path
        if current_node == goal:
            return current_cost, reconstruct_path(visited, start, goal)
        
        # Explore the neighbors
        for neighbor, cost in graph[current_node]:
            total_cost = current_cost + cost
            # Check if this path to the neighbor is better than any previously found
            if neighbor not in visited or total_cost < visited[neighbor][0]:
                visited[neighbor] = (total_cost, current_node)
                heapq.heappush(priority_queue, (total_cost, neighbor))
    
    # If the goal is not reachable, return None
    return None

def reconstruct_path(visited, start, goal):
    # Reconstruct the path from start to goal by following the visited nodes
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = visited[current][1]  # Get the parent node
    path.reverse()
    return path