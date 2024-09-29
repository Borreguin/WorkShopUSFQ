import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Inicialización de posiciones
initial_side = {'farmer': 'left', 'wolf': 'left', 'goat': 'left', 'cabbage': 'left'}
positions = [initial_side.copy()]  # Guardaremos los estados

# Función para registrar cada movimiento
def move(item, side):
    new_pos = positions[-1].copy()
    for i in item: new_pos[i] = side
    positions.append(new_pos)



# Secuencia de movimientos
def farmer_crossing_sequence():
    move(['goat','farmer'], 'right')  # 1. Lleva la cabra
    move(['farmer'], 'left')  # 2. Vuelve solo
    move(['wolf','farmer'], 'right')  # 3. Lleva el lobo
    move(['goat','farmer'], 'left')  # 4. Vuelve con la cabra
    move(['cabbage','farmer'], 'right')  # 5. Lleva la col
    move(['farmer'], 'left')  # 6. Vuelve solo
    move(['goat','farmer'], 'right')  # 7. Lleva la cabra de nuevo

# Llamamos a la función para generar la secuencia
farmer_crossing_sequence()



positions


# Función para representar el estado en texto (para los nodos de la gráfica)
def state_to_string(state):
    left_side = [key for key, value in state.items() if value == 'left']
    right_side = [key for key, value in state.items() if value == 'right']

    left_text = ', '.join(left_side) if left_side else 'Nadie'
    right_text = ', '.join(right_side) if right_side else 'Nadie'

    return f"Izq: [{left_text}]\nDer: [{right_text}]"


# Función para mostrar el gráfico de cada paso con nodos distribuidos verticalmente y en dos columnas paralelas
def draw_graph_step(step):
    G = nx.DiGraph()

    # Añadir nodos al grafo hasta el paso actual
    for i in range(step + 1):
        G.add_node(i, label=state_to_string(positions[i]), bipartite=0 if positions[i]['farmer'] == 'left' else 1)

    # Añadir las conexiones (movimientos) hasta el paso actual
    for i in range(step):
        G.add_edge(i, i + 1)

    # Obtener los nodos de cada orilla (izquierda y derecha) para `bipartite_layout`
    left_nodes = [i for i in range(step + 1) if positions[i]['farmer'] == 'left']
    right_nodes = [i for i in range(step + 1) if positions[i]['farmer'] == 'right']

    # Usamos `bipartite_layout` para alinear los nodos en dos columnas paralelas
    pos = nx.bipartite_layout(G, left_nodes)
    pos_negadas = {clave: np.array([valor[0], -valor[1]]) for clave, valor in pos.items()}
    # Dibujar el grafo con las etiquetas
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(15, 15))  # Ajustar el tamaño para nodos verticales y paralelos
    nx.draw(G, pos_negadas, with_labels=True, labels=labels, node_size=2500, node_color="skyblue",
            font_size=10, font_weight="bold", arrows=True, arrowsize=20, verticalalignment='bottom',
            horizontalalignment="center")

    # Título con el número de paso
    plt.title(f"Cruce del Río - Paso {step}", fontsize=20)
    plt.show()



# Mostrar el gráfico por cada paso del movimiento
for step in range(len(positions)):
    draw_graph_step(step)