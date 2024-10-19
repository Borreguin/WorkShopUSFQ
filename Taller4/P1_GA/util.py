from typing import List


def word_to_array(word: str):
    return [ord(w) for w in word]

# Algo no está bien con esta función de distancia
def distance(list1: List[int], list2: List[int]):
  m, n = len(list1), len(list2)

  # Crear una matriz de (m+1) x (n+1) para almacenar las distancias
  dp = [[0] * (n + 1) for _ in range(m + 1)]

  # Inicializar la primera fila y columna de la matriz
  for i in range(m + 1):
    dp[i][0] = i  # Costo de eliminar elementos de list1
  for j in range(n + 1):
    dp[0][j] = j  # Costo de insertar elementos en list2

  # Calcular la distancia de Levenshtein
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      if list1[i - 1] == list2[j - 1]:
        costo = 0
      else:
        costo = 1
      dp[i][j] = min(dp[i - 1][j] + 1,  # Eliminación
                     dp[i][j - 1] + 1,  # Inserción
                     dp[i - 1][j - 1] + costo)  # Sustitución

  return dp[m][n]

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