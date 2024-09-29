
def define_color(cell):
    if cell == '#':
        return 'black'
    elif cell == ' ':   # Espacio vacío
        return 'white'
    elif cell == 'E':   # Entrada
        return 'green'
    elif cell == 'S':   # Salida
        return 'red'
     
def define_black_space(cell):
    if cell == ' ':
        return 1 # Libre
    elif cell == 'E':   # Entrada
        return 2
    elif cell == 'S':   # Salida
        return 3
    else:
        return 0 #Pared
