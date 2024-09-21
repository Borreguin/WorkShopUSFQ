from typing import List, Dict
from itertools import permutations
from P1_TSP.util import generar_ciudades_con_distancias, plotear_ruta

class TSP:
    def __init__(self, ciudades: Dict[str, tuple], distancias: Dict[str, Dict[str, float]]):
        self.ciudades = ciudades
        self.distancias = distancias

    def calcular_distancia(self, ruta: List[str]) -> float:
        distancia_total = 0.0
        for i in range(len(ruta) - 1):
            distancia_total += self.distancias[ruta[i]][ruta[i + 1]]
        distancia_total += self.distancias[ruta[-1]][ruta[0]]  # Volver al punto de inicio
        return distancia_total

    def encontrar_ruta_fuerza_bruta(self) -> List[str]:
        ciudades = list(self.ciudades.keys())
        mejor_ruta = None
        menor_distancia = float('inf')

        for permutacion in permutations(ciudades):
            distancia = self.calcular_distancia(permutacion)
            if distancia < menor_distancia:
                menor_distancia = distancia
                mejor_ruta = permutacion

        return list(mejor_ruta)

    def encontrar_ruta_vecino_mas_cercano(self) -> List[str]:
        ciudades = list(self.ciudades.keys())
        ruta = [ciudades.pop(0)]  # Empezar desde la primera ciudad
        while ciudades:
            ultima_ciudad = ruta[-1]
            ciudad_mas_cercana = min(ciudades, key=lambda ciudad: self.distancias[ultima_ciudad][ciudad])
            ruta.append(ciudad_mas_cercana)
            ciudades.remove(ciudad_mas_cercana)
        return ruta

    def plotear_resultado(self, ruta: List[str], mostrar_anotaciones: bool = True):
        plotear_ruta(self.ciudades, ruta, mostrar_anotaciones)

def study_case_comparativo():
    n_cities = 10
    ciudades, distancias = generar_ciudades_con_distancias(n_cities)
    tsp = TSP(ciudades, distancias)

    # Fuerza Bruta
    ruta_fuerza_bruta = tsp.encontrar_ruta_fuerza_bruta()
    distancia_fuerza_bruta = tsp.calcular_distancia(ruta_fuerza_bruta)
    print("Ruta más corta (Fuerza Bruta):", ruta_fuerza_bruta)
    print("Distancia total (Fuerza Bruta):", distancia_fuerza_bruta)
    tsp.plotear_resultado(ruta_fuerza_bruta)

    # Vecino Más Cercano
    ruta_vecino_mas_cercano = tsp.encontrar_ruta_vecino_mas_cercano()
    distancia_vecino_mas_cercano = tsp.calcular_distancia(ruta_vecino_mas_cercano)
    print("Ruta más corta (Vecino Más Cercano):", ruta_vecino_mas_cercano)
    print("Distancia total (Vecino Más Cercano):", distancia_vecino_mas_cercano)
    tsp.plotear_resultado(ruta_vecino_mas_cercano, True)

if __name__ == "__main__":
    # Solve the TSP problem
    study_case_comparativo()