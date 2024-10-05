import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
np.random.seed(42)

class AntColonyOptimization:
    def __init__(self, start, end, obstacles, grid_size=(10, 10), num_ants=10, evaporation_rate=0.1, alpha=0.1, beta=15):
        self.start = start
        self.end = end
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromones = np.ones(grid_size)
        self.best_ant_path = None       # best path that an ant could find while the iterations occurs
        self.best_path = None           # best path based on pheromones after all iterations

    def _get_neighbors(self, position):
        pos_x, pos_y = position
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_x, new_y = pos_x + i, pos_y + j
                if (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1] and
                        (new_x, new_y) != position and (new_x, new_y) not in self.obstacles):
                    neighbors.append((new_x, new_y))
        return neighbors

    def _select_next_position(self, position, visited):
        neighbors = self._get_neighbors(position)
        if len(neighbors) == 0:
            return None
        probabilities = []
        total = 0
        for neighbor in neighbors:
            if neighbor not in visited:
                pheromone = self.pheromones[neighbor[1], neighbor[0]]
                heuristic = 1 / (np.linalg.norm(np.array(neighbor) - np.array(self.end)) + 0.1)
                probabilities.append((neighbor, pheromone ** self.alpha * heuristic ** self.beta))
                total += pheromone ** self.alpha * heuristic ** self.beta
        if not probabilities or total == 0:
            return None
        probabilities = [(pos, prob / total) for pos, prob in probabilities]
        positions, weights = zip(*probabilities)
        selected_index = np.random.choice(len(positions), p=weights)
        return probabilities[selected_index][0]

    def _evaporate_pheromones(self):
        self.pheromones *= (1 - self.evaporation_rate)

    def _deposit_pheromones(self, path):
        pheromone_deposit = 1 / len(path)  # Deposit inversely proportional to path length
        for position in path:
            self.pheromones[position[1], position[0]] += pheromone_deposit

    def find_best_path(self, num_iterations):
        self.find_best_ant_path(num_iterations)
        self.find_best_path_based_on_pheromones()
    def find_best_ant_path(self, num_iterations):
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

            # Escoger el mejor camino por su tamaño?
            # --------------------------
            all_paths.sort(key=lambda x: len(x))
            best_path = all_paths[0]

            self._evaporate_pheromones()
            self._deposit_pheromones(best_path)
            # Here we need to guarantee that the end arrived to the end
            # best_path[-1] == self.end
            if self.best_ant_path is None or len(best_path) <= len(self.best_ant_path) and best_path[-1] == self.end:
                self.best_ant_path = best_path
            # --------------------------

    def find_best_path_based_on_pheromones(self):
        self.best_path = [self.start]
        # use the pheromones to find the best path (self.pheromones)
        current_position = self.start
        while current_position != self.end:
            # use self.pheromones
            next_position = self.get_next_position_based_on_pheromones(self._get_neighbors(current_position))
            if next_position is None:
                break
            current_position = next_position
            self.best_path.append(current_position)
    def get_next_position_based_on_pheromones(self, neighbors):
        # use self.pheromones
        best_neighbor = None
        for position in neighbors:
            if position in self.best_path:
                continue
            if best_neighbor is None or self.pheromones[position[1], position[0]] > self.pheromones[best_neighbor[1], best_neighbor[0]]:
                best_neighbor = position
        return best_neighbor

    def plot(self):
        cmap = LinearSegmentedColormap.from_list('pheromone', ['white', 'green', 'red'])
        plt.figure(figsize=(8, 8))
        plt.imshow(self.pheromones, cmap=cmap, vmin=np.min(self.pheromones), vmax=np.max(self.pheromones))
        plt.colorbar(label='Pheromone intensity')
        plt.scatter(self.start[0], self.start[1], color='orange', label='Start', s=100)
        plt.scatter(self.end[0], self.end[1], color='magenta', label='End', s=100)
        for obstacle in self.obstacles:
            plt.scatter(obstacle[0], obstacle[1], color='gray', s=900, marker='s')
        if self.best_ant_path:
            path_x, path_y = zip(*self.best_ant_path)
            plt.plot(path_x, path_y, color='blue', label='Last Ant Path', linewidth=3)
        if self.best_path:
            path_x, path_y = zip(*self.best_path)
            plt.plot(path_x, path_y, color='red', label='Best Pheromone Path', linewidth=3)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title('Ant Colony Optimization')
        plt.legend()
        plt.grid(True)
        plt.show()

def study_case_1():
    print("Start of Ant Colony Optimization - First Study Case")
    start = (0, 0)
    end = (4, 7)
    obstacles = [(1, 2), (2, 2), (3, 2)]
    aco = AntColonyOptimization(start, end, obstacles)
    aco.find_best_path(100)
    aco.plot()
    print("End of Ant Colony Optimization")
    print("Best path: ", aco.best_ant_path)

def study_case_2():
    print("Start of Ant Colony Optimization - Second Study Case")
    start = (0, 0)
    end = (4, 7)
    obstacles = [(0, 2), (1, 2), (2, 2), (3, 2)]
    aco = AntColonyOptimization(start, end, obstacles)
    aco.find_best_path(100)
    aco.plot()
    print("End of Ant Colony Optimization")
    print("Best path: ", aco.best_ant_path)

if __name__ == '__main__':
    # study_case_1()
    study_case_2()



