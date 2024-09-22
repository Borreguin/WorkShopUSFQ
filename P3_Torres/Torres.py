import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Inicialización de los movimientos
movements = 0


# Definición de la función Tower of Hanoi
def TowerOfHanoi(n, source, destination, auxiliary, towers):
    global movements
    movements += 1
    if n == 1:
        move_disk(source, destination, towers)
        return
    TowerOfHanoi(n - 1, source, auxiliary, destination, towers)
    move_disk(source, destination, towers)
    TowerOfHanoi(n - 1, auxiliary, destination, source, towers)


# Función para mover el disco de una torre a otra
def move_disk(source, destination, towers):
    disk = towers[source].pop()  # Tomar el disco desde la torre de origen
    towers[destination].append(disk)  # Colocar el disco en la torre de destino
    draw_towers(towers)  # Dibujar las torres actualizadas
    time.sleep(0.5)  # Pausa para mostrar el movimiento


# Función para dibujar las torres y los discos
def draw_towers(towers):
    plt.clf()  # Limpiar el gráfico
    ax = plt.gca()

    # Dibujar las torres (3 pilares)
    for i in range(3):
        plt.plot([i, i], [0, 6], color='black', lw=5)

    # Dibujar los discos
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    for i, tower in enumerate(towers):
        height = 0  # Altura de los discos en cada torre
        for disk in tower:
            disk_width = disk  # El ancho del disco se basa en su tamaño
            ax.add_patch(patches.Rectangle((i - disk_width / 2, height), disk_width, 0.4, color=colors[disk - 1]))
            height += 0.5  # Altura para el siguiente disco

    plt.xlim(-1, 3)
    plt.ylim(0, 6)
    plt.title(f"Torre de Hanói - Movimientos: {movements}")
    plt.pause(0.01)


# Inicialización de las torres
def initialize_towers(n):
    return [[i for i in range(n, 0, -1)], [], []]  # Torre A tiene todos los discos, B y C vacías


# Función principal para ejecutar y mostrar la Torre de Hanói
def main():
    global movements
    movements = 0
    n = 3  # Número de discos
    towers = initialize_towers(n)
    plt.figure(figsize=(8, 6))
    draw_towers(towers)  # Dibujar la configuración inicial
    TowerOfHanoi(n, 0, 1, 2, towers)  # Ejecutar el algoritmo
    plt.show()


# Ejecutar la simulación
main()





