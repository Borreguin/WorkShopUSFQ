import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Taller4.P1_GA.generalSteps import *


class GA:
    def __init__(self, population, objective, mutation_rate, n_iterations):
        self.population = population
        self.n_generation = 0
        self.n_iterations = n_iterations
        self.objective = objective
        self.mutation_rate = mutation_rate
        self.evaluation_type = AptitudeType.DEFAULT
        self.best_individual_selection_type = BestIndividualSelectionType.DEFAULT
        self.new_generation_type = NewGenerationType.DEFAULT

    def set_evaluation_type(self, evaluation_type: AptitudeType):
        self.evaluation_type = evaluation_type

    def set_best_individual_selection_type(self, _type:BestIndividualSelectionType):
        self.best_individual_selection_type = _type

    def set_new_generation_type(self, _type):
        self.new_generation_type = _type

    def run(self):
        success = False
        for _ in range(self.n_iterations):
            # las aptitudes son los valores que se obtienen al evaluar la función de aptitud
            aptitudes = [evaluate_aptitude(self.evaluation_type, individual, self.objective) for individual in self.population]
            # el mejor individuo es el que tiene la mejor aptitud
            # (esto se puede elegir como maximo o minimo, depende de como se defina la aptitud)
            best_individual, best_aptitude = select_best_individual(self.best_individual_selection_type, self.population, aptitudes)
            # si el mejor individuo es igual al objetivo, se termina el algoritmo
            if best_individual == self.objective:
                success = True
                print("Objetivo alcanzado:")
                print(f"Generación {self.n_generation}: {best_individual} - Aptitud: {best_aptitude}")
                break
            #print(f"Generación {self.n_generation}: {best_individual} - población: {len(self.population)} - Aptitud: {best_aptitude}")

            # la nueva generación se obtiene a partir de la población actual, interactuando entre los individuos
            self.population = generate_new_population(self.new_generation_type, self.population, aptitudes, self.mutation_rate)
            self.n_generation += 1

        if not success:
            print(f"Objetivo no alcanzado en las iteraciones establecidas {self.n_iterations}")


def case_study_1(_objetive):
    # Definición de la población inicial
    population = generate_population(100, len(_objetive))
    mutation_rate = 0.01
    n_iterations = 1000
    ga = GA(population, _objetive, mutation_rate, n_iterations)
    ga.run()

def case_study_2(_objetive):
    population = generate_population(100, len(_objetive))
    mutation_rate = 0.02
    n_iterations = 1000
    ga = GA(population, _objetive, mutation_rate, n_iterations)
    ga.set_evaluation_type(AptitudeType.BY_DISTANCE)
    ga.set_best_individual_selection_type(BestIndividualSelectionType.MIN_DISTANCE)
    ga.set_new_generation_type(NewGenerationType.MIN_DISTANCE)
    ga.run()


def case_study_3(_objetive):
  mutation_rate = [0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
  n_iterations = 10000
  population = generate_population(100, len(_objetive))

  n_generation = []
  for mr in mutation_rate:
    ga = GA(population, _objetive, mr, n_iterations)
    ga.run()
    n_generation.append(ga.n_generation)
  import matplotlib.pyplot as plt
  import numpy as np
  print(
    f'El mutation_rate con menor numero de generaciones es {mutation_rate[np.argmin(n_generation)]} con {np.min(n_generation)}')

  plt.plot(mutation_rate, n_generation, marker='o')
  plt.title('Pruebas de Mutation Rate')
  plt.xlabel('Mutation rate')
  plt.ylabel('Numero de generaciones')
  plt.show()


def case_study_4(_objetive):
  mutation_rate = 0.01
  n_iterations = 10000
  n_population = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

  n_generation = []
  for n in n_population:
    population = generate_population(n, len(_objetive))
    ga = GA(population, _objetive, mutation_rate, n_iterations)
    ga.run()
    n_generation.append(ga.n_generation)

  print(
    f'El mutation_rate con menor numero de generaciones es {n_population[np.argmin(n_generation)]} con {np.min(n_generation)}')
  plt.plot(n_population, n_generation, marker='o', color='red', alpha=0.7)
  plt.title('Pruebas de Tamaño Poblacion')
  plt.xlabel('Tamaño Poblacion')
  plt.ylabel('Numero de generaciones')
  plt.show()


def case_study_heatmap(_objetive):
  mutation_rate = [0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
  n_iterations = 100
  n_population = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

  n_generation = np.zeros((len(n_population), len(mutation_rate)))
  for i, n in enumerate(n_population):
    for j, mr in enumerate(mutation_rate):
      print(n, mr)
      population = generate_population(n, len(_objetive))
      ga = GA(population, _objetive, mr, n_iterations)

      ga.run()
      n_generation[i, j] = ga.n_generation
  plt.figure(figsize=(10, 8))
  sns.heatmap(n_generation, xticklabels=mutation_rate,
              yticklabels=n_population, cmap="YlGnBu", annot=True)
  plt.xlabel('Mutation Rate')
  plt.ylabel('Tamaño Poblacion')
  plt.title('Mapa de Calor: Mutation Rate y Tamaño Poblacion')
  plt.show()

  min_value = np.min(n_generation)
  min_index = np.unravel_index(np.argmin(n_generation), n_generation.shape)

  # Obtener los valores óptimos de alpha y beta
  optimal_n_population = n_population[min_index[0]]
  optimal_mutation_rate = mutation_rate[min_index[1]]

  print(f'El valor mínimo de la distancia total es: {min_value}')
  print(f'El valor óptimo de mutation rate es: {optimal_n_population}')
  print(f'El valor óptimo de tamaño poblacion es: {optimal_mutation_rate}')


def case_study_5(_objetive):
  population = generate_population(800, len(_objetive))
  mutation_rate = 0.009
  n_iterations = 1000
  ga = GA(population, _objetive, mutation_rate, n_iterations)
  ga.set_new_generation_type(NewGenerationType.TOURNAMENT)
  ga.run()


if __name__ == "__main__":
    objective = "GA Workshop! USFQ"
    #case_study_heatmap(objective)
    print("Case Study 1")
    case_study_1(objective)
    print("Case Study 2")
    case_study_2(objective)
    print("Case Study 3")
    case_study_3(objective)
    print("Case Study 4")
    case_study_4(objective)
    print("Case Study 5")
    case_study_5(objective)