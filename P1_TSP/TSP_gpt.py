import random
import numpy as np


# Función para calcular la distancia total de un recorrido
def calculate_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i], route[(i + 1) % len(route)]]
    return total_distance


# Función para crear una ruta aleatoria
def create_route(num_cities):
    return random.sample(range(num_cities), num_cities)


# Función para crear una población inicial
def initial_population(pop_size, num_cities):
    return [create_route(num_cities) for _ in range(pop_size)]


# Función de fitness para evaluar la aptitud de una ruta
def fitness(route, distance_matrix):
    return 1 / calculate_distance(route, distance_matrix)  # Inverso de la distancia total


# Selección basada en el fitness (ruleta)
def selection(population, fitness_scores):
    selected_indices = random.choices(range(len(population)), weights=fitness_scores, k=2)
    return population[selected_indices[0]], population[selected_indices[1]]


# Operador de cruce (crossover) basado en PMX (Partially Mapped Crossover)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]

    for i in range(start, end):
        if parent2[i] not in child:
            val = parent2[i]
            pos = i
            while start <= pos < end:
                pos = parent2.index(parent1[pos])
            child[pos] = val

    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child


# Operador de mutación (intercambio de dos ciudades)
def mutate(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route


# Función principal del algoritmo genético
def genetic_algorithm_tsp(distance_matrix, pop_size, num_generations, mutation_rate):
    num_cities = len(distance_matrix)
    population = initial_population(pop_size, num_cities)
    best_route = None
    best_distance = float('inf')

    for generation in range(num_generations):
        fitness_scores = [fitness(route, distance_matrix) for route in population]
        new_population = []

        for _ in range(pop_size // 2):
            # Selección
            parent1, parent2 = selection(population, fitness_scores)
            # Cruce
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            # Mutación
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Evaluar la nueva población
        population = new_population
        current_best_route = min(population, key=lambda route: calculate_distance(route, distance_matrix))
        current_best_distance = calculate_distance(current_best_route, distance_matrix)

        if current_best_distance < best_distance:
            best_route = current_best_route
            best_distance = current_best_distance

        print(f"Generación {generation}: Mejor distancia: {best_distance}")

    return best_route, best_distance


# Crear una matriz de distancias aleatorias para un conjunto de ciudades
def create_distance_matrix(num_cities):
    matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
    np.fill_diagonal(matrix, 0)  # La distancia entre una ciudad y sí misma es 0
    return matrix


# Parámetros del problema
num_cities = 10
distance_matrix = create_distance_matrix(num_cities)

# Parámetros del algoritmo genético
pop_size = 100
num_generations = 500
mutation_rate = 0.01

# Ejecutar el algoritmo genético
best_route, best_distance = genetic_algorithm_tsp(distance_matrix, pop_size, num_generations, mutation_rate)

print(f"Mejor ruta encontrada: {best_route}")
print(f"Mejor distancia: {best_distance}")
