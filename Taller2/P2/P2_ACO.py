import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

np.random.seed(42)


class AntColonyOptimization:
  def __init__(self, start, end, obstacles, grid_size=(10, 10), num_ants=50,
      evaporation_rate=0.2, alpha=1, beta=10):
    self.start = start
    self.end = end
    self.obstacles = obstacles
    self.grid_size = grid_size
    self.num_ants = num_ants
    self.evaporation_rate = evaporation_rate
    self.alpha = alpha
    self.beta = beta
    self.pheromones = np.ones(grid_size)
    self.best_path = None

  def _get_neighbors(self, position):
    pos_x, pos_y = position
    neighbors = []
    for i in range(-1, 2):
      for j in range(-1, 2):
        new_x, new_y = pos_x + i, pos_y + j
        if (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[
          1] and
            (new_x, new_y) != position and (
            new_x, new_y) not in self.obstacles):
          neighbors.append((new_x, new_y))
    return neighbors

  def _select_next_position(self, position, visited):
    neighbors = self._get_neighbors(position)
    probabilities = []
    total = 0
    for neighbor in neighbors:
      if neighbor not in visited:
        pheromone = self.pheromones[neighbor[1], neighbor[0]]
        heuristic = 1 / (
              np.linalg.norm(np.array(neighbor) - np.array(self.end)) + 0.1)
        probabilities.append(
            (neighbor, pheromone ** self.alpha * heuristic ** self.beta))
        total += pheromone ** self.alpha * heuristic ** self.beta
    if not probabilities:
      return None
    probabilities = [(pos, prob / total) for pos, prob in probabilities]
    selected = np.random.choice(len(probabilities),
                                p=[prob for pos, prob in probabilities])
    return probabilities[selected][0]

  def _evaporate_pheromones(self):
    self.pheromones *= (1 - self.evaporation_rate)

  def _deposit_pheromones(self, path):
    for position in path:
      self.pheromones[position[1], position[0]] += 1

  def find_best_path(self, num_iterations):
    for _ in range(num_iterations):
      all_paths = []
      for _ in range(self.num_ants):
        current_position = self.start
        path = [current_position]
        while current_position != self.end:
          next_position = self._select_next_position(current_position, path)
          if next_position is None:
            break
          path.append(next_position)
          current_position = next_position
        all_paths.append(path)

      # Considerar solo caminos que llegan al destino
      all_paths = [path for path in all_paths if path[-1] == self.end]
      all_paths.sort(key=lambda x: len(x))
      if all_paths:
        best_path = all_paths[0]

        self._evaporate_pheromones()
        self._deposit_pheromones(best_path)

        if self.best_path is None or len(best_path) <= len(self.best_path):
          self.best_path = best_path

  def plot(self):
    cmap = LinearSegmentedColormap.from_list('pheromone',
                                             ['white', 'green', 'red'])
    plt.figure(figsize=(8, 8))
    plt.imshow(self.pheromones, cmap=cmap, vmin=np.min(self.pheromones),
               vmax=np.max(self.pheromones))
    plt.colorbar(label='Pheromone intensity')
    plt.scatter(self.start[0], self.start[1], color='orange', label='Start',
                s=100)
    plt.scatter(self.end[0], self.end[1], color='magenta', label='End', s=100)
    for obstacle in self.obstacles:
      plt.scatter(obstacle[0], obstacle[1], color='gray', s=900, marker='s')
    if self.best_path:
      path_x, path_y = zip(*self.best_path)
      plt.plot(path_x, path_y, color='blue', label='Best Path', linewidth=3)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Ant Colony Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()


def calcular_distancia_total(path):
  distancia_total = 0
  for k in range(len(path) - 1):
    x1, y1 = path[k]
    x2, y2 = path[k + 1]
    distancia_total += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return distancia_total


def study_case_2():
  print("Start of Ant Colony Optimization - Second Study Case")
  start = (0, 0)
  end = (4, 7)
  obstacles = [(0, 2), (1, 2), (2, 2), (3, 2)]
  alpha_values = np.linspace(0.1, 1, 5)
  beta_values = np.arange(5, 10, 1)
  distance_matrix = np.zeros((len(alpha_values), len(beta_values)))
  for i, alpha in enumerate(alpha_values):
    for j, beta in enumerate(beta_values):
      # Aquí ejecutarías el ACO con los valores de alpha y beta
      aco = AntColonyOptimization(start, end, obstacles, num_ants=100,
                                  evaporation_rate=0.2, alpha=alpha, beta=beta)
      aco.find_best_path(500)
      best_path = aco.best_path
      distancia_total = calcular_distancia_total(best_path)

      # Guardar la distancia total en la matriz
      distance_matrix[i, j] = distancia_total

  plt.figure(figsize=(10, 8))
  sns.heatmap(distance_matrix, xticklabels=np.round(beta_values, 2),
              yticklabels=np.round(alpha_values, 2), cmap="YlGnBu", annot=True)
  plt.xlabel('Beta')
  plt.ylabel('Alpha')
  plt.title('Mapa de Calor: Alpha vs Beta vs Distancia del Mejor Camino')
  plt.show()

  min_value = np.min(distance_matrix)
  min_index = np.unravel_index(np.argmin(distance_matrix),
                               distance_matrix.shape)

  # Obtener los valores óptimos de alpha y beta
  optimal_alpha = alpha_values[min_index[0]]
  optimal_beta = beta_values[min_index[1]]

  print(f'El valor mínimo de la distancia total es: {min_value}')
  print(f'El valor óptimo de alpha es: {optimal_alpha}')
  print(f'El valor óptimo de beta es: {optimal_beta}')

  # Ejecutar el ACO con los valores óptimos de alpha y beta
  aco = AntColonyOptimization(start, end, obstacles, num_ants=100,
                              evaporation_rate=0.2, alpha=optimal_alpha,
                              beta=optimal_beta)
  aco.find_best_path(500)  # Aplicar el algoritmo una última vez
  best_path_optimo = aco.best_path  # El mejor camino final

  # Mostrar el mejor camino final y la distancia total
  distancia_total_optima = calcular_distancia_total(best_path_optimo)
  aco.plot()
  print(f'El mejor camino final es: {best_path_optimo}')
  print(
    f'La distancia total con los valores óptimos de alpha y beta es: {distancia_total_optima}')


study_case_2()