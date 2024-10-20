from typing import List
import random


def word_to_array(word: str):
    return [ord(w) for w in word]

# Algo no está bien con esta función de distancia


def distance(list1: List[int], list2: List[int]):
    acc = 0
    for e1, e2 in zip(list1, list2):
        acc += (e1 - e2)
    n_size = min(len(list1), len(list2))
    if n_size == 0:
        return None
    return acc + (len(list1) - len(list2))


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def hamming_distance(s1, s2):
    """ Calcula la distancia de Hamming entre dos cadenas de la misma longitud. """
    if len(s1) != len(s2):
        raise ValueError("Las cadenas deben tener la misma longitud")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def word_distance(word1: str, word2: str):
    # return distance(word_to_array(word1), word_to_array(word2))
    # return levenshtein_distance(word_to_array(word1), word_to_array(word2))
    return hamming_distance(word_to_array(word1), word_to_array(word2))


def choose_best_individual_by_distance(population, aptitudes):
    best_individual = population[0]
    best_aptitude = aptitudes[0]
    for ind, apt in zip(population, aptitudes):
        if apt < best_aptitude:
            best_aptitude = apt
            best_individual = ind
    return best_individual


def tournament_selection(_population, partition_size, tournament_size=3):
    # Cada individuo es un diccionario con la cadena y su fitness
    # print(population[0])
    population = [{'individual': _population[i], 'fitness': 0}
                  for i in range(len(_population))]
    # print(population)

    def calculate_fitness(individual, target):
        # Calcula el fitness entre la cadena individual y la frase objetivo
        # Fitness negativo, menor es mejor
        return -hamming_distance(individual, target)

    # Calcula el fitness de cada individuo en la población
    for ind in population:
        ind['fitness'] = calculate_fitness(
            ind['individual'], "GA Workshop! USFQ")

    tournament_size = min(tournament_size, len(population))

    tournament = random.sample(population, tournament_size)
    # return max(tournament, key=lambda individual: individual.fitness)
    return max(tournament, key=lambda ind: ind['fitness'])

# print(word_distance("abc", "abc"))
# print(word_distance("abc", "abd"))
# print(word_distance("abc", "abz"))
# print(word_distance("abc", "cba"))
# print(word_distance("abc", "cbad"))
