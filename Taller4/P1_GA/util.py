from typing import List


def word_to_array(word: str):
    return [ord(w) for w in word]

# Algo no está bien con esta función de distancia
# Esta forma de medir la distancia no es la más adecuada
def distance(list1:List[int], list2:List[int]):
    acc = 0
    for e1, e2 in zip(list1, list2):
        # acc += (e1 - e2)

        # Se cambia la forma de calcular la distancia
        acc += abs(e1 - e2)

    n_size = min(len(list1), len(list2))
    if n_size == 0:
        return None
    return acc + (len(list1) - len(list2))

# SE DEFINE UNA NUEVA FUNCIÓN PARA CALCULAR LA DISTANCIA ENTRE DOS PALABRAS
def levenshtein_distance(s1: str, s2: str) -> int:
    # Se crea una matriz para almacenar las distancias entre substrings
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # Se inicializa la matriz
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    # Se calcula le DISTANCIA DE LEVENSHTEIN
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,  
                               dp[i][j - 1] + 1,  
                               dp[i - 1][j - 1] + 1)  

    return dp[-1][-1]    

def word_distance(word1:str, word2:str):
    return distance(word_to_array(word1), word_to_array(word2))

# Se define la función word_distance_2 integrando la distancia de Levenshtein
def word_distance_2(word1:str, word2:str) -> int:
    return levenshtein_distance(word1, word2)

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