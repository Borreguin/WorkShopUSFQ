from typing import List


def word_to_array(word: str):
    return [ord(w) for w in word]

# Algo no está bien con esta función de distancia
def distance(list1: List[int], list2: List[int]):
  # Si las listas son de diferentes longitudes, llenamos con ceros la más corta
  max_len = max(len(list1), len(list2))
  list1 += [0] * (max_len - len(list1))
  list2 += [0] * (max_len - len(list2))

  # Calculamos la distancia de Manhattan
  acc = sum(abs(e1 - e2) for e1, e2 in zip(list1, list2))
  return acc

def word_distance(word1:str, word2:str):
    return distance(word_to_array(word1), word_to_array(word2))

def choose_best_individual_by_distance(population, aptitudes):
    best_individual = population[0]
    best_aptitude = aptitudes[0]
    for ind, apt in zip(population, aptitudes):
        if apt < best_aptitude:
            best_aptitude = apt
            best_individual = ind
    return best_individual



# print(word_distance("abc", "abc"))
# print(word_distance("abc", "abd"))
# print(word_distance("abc", "abz"))
# print(word_distance("abc", "cba"))
# print(word_distance("abc", "cbad"))