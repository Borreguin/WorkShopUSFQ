from typing import List


def word_to_array(word: str):
    return [ord(w) for w in word]

# Algo no está bien con esta función de distancia
def distance(list1:List[int], list2:List[int]):
    acc = 0
    for e1, e2 in zip(list1, list2):
        acc += abs(e1 - e2)
    n_size = min(len(list1), len(list2))
    if n_size == 0:
        return None
    return acc + abs((len(list1) - len(list2)))

def levenshtein_distance(str1: str, str2: str) -> int:
    n, m = len(str1), len(str2)

    # Crear una matriz (n+1) x (m+1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Inicializar la primera fila y la primera columna
    for i in range(n + 1):
        dp[i][0] = i  # Eliminar todos los caracteres de str1
    for j in range(m + 1):
        dp[0][j] = j  # Insertar todos los caracteres de str2

    # Llenar la matriz
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No hay costo si son iguales
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # Eliminación
                    dp[i][j - 1] + 1,    # Inserción
                    dp[i - 1][j - 1] + 1 # Sustitución
                )

    # La distancia de Levenshtein está en la esquina inferior derecha de la matriz
    return dp[n][m]

def word_distance(word1:str, word2:str, mthd = 1):
    if mthd == 1:
        return distance(word_to_array(word1), word_to_array(word2))
    else:
        return levenshtein_distance(word1,word2)
    

def choose_best_individual_by_distance(population, aptitudes):
    best_individual = population[0]
    best_aptitude = aptitudes[0]
    for ind, apt in zip(population, aptitudes):
        if apt < best_aptitude:
            best_aptitude = apt
            best_individual = ind
    return best_individual


def elegant_print(message,num_sep):

    print("="*num_sep)
    print(message)
    print("-"*num_sep)


'''
#print(word_distance("abc", "abc"))
print(word_distance("abc", "abd"))
print(word_distance("abc", "abz"))
print(word_distance("abc", "cba"))
print(word_distance("abc", "cbad"))'''