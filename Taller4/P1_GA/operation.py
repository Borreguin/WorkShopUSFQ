import random
# from Taller4.P1_GA.constants import *
from constants import *
# from Taller4.P1_GA.util import *
from util import *


def parent_selection(_type: ParentSelectionType, population, aptitudes):
    if _type == ParentSelectionType.DEFAULT:
        # Selección de padres por ruleta
        cumulative = sum(aptitudes)
        selection_probability = [aptitude /
                                 cumulative for aptitude in aptitudes]
        parents = random.choices(
            population, weights=selection_probability, k=2)
        return parents
    if _type == ParentSelectionType.MIN_DISTANCE:
        # seleccionando randomicamente dos poblaciones diferentes para cada padre
        # se podria seleccionar de otra manera?
        partition_size = random.randint(1, len(population)-1)
        parent1 = choose_best_individual_by_distance(
            population[:partition_size], aptitudes[:partition_size])
        parent2 = choose_best_individual_by_distance(
            population[partition_size:], aptitudes[partition_size:])
        return parent1, parent2

    if _type == ParentSelectionType.NEW:
        # print("implement here the new parent selection")
        partition_size = random.randint(1, len(population)-1)
        # print(f"partition_size:{partition_size}")
        parent1 = tournament_selection(
            population[:partition_size], partition_size)
        parent2 = tournament_selection(
            population[partition_size:], partition_size)
        return parent1, parent2
       # return None


def crossover(_type: CrossoverType, parent1, parent2):
    if _type == CrossoverType.DEFAULT:
        # Cruce de dos padres para producir descendencia
        crossover_point = random.randint(1, len(parent1) - 1)
        # print(f'crossover_point:{len(parent1)}')
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    if _type == CrossoverType.NEW:
        # print("implement here the new crossover")

        parent1_string = parent1['individual']
        parent2_string = parent2['individual']
        crossover_point = random.randint(1, len(parent1_string) - 1)

        child1 = parent1_string[:crossover_point] + \
            parent2_string[crossover_point:]
        child2 = parent2_string[:crossover_point] + \
            parent1_string[crossover_point:]

        # Retornar los hijos
        return child1, child2
        # return None


def mutate(_type: MutationType, individual, mutation_rate):
    if _type == MutationType.DEFAULT:
        # Mutación de un individuo
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual = individual[:i] + \
                    random.choice(all_possible_gens) + individual[i + 1:]
        return individual
    if _type == MutationType.NEW:
        print("implement here the new mutation")
        return None
