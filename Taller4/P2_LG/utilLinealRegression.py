import numpy as np

# Loss function: Mean Squared Error MSE
def loss_function(y, y_hat):
    return (1 / 2 * len(y)) * sum((y - y_hat) ** 2)

# predict function using weights of the model
def predict(x, weights):
    return np.dot(x, weights)

# Gradient function
def gradient_descent(x, y, learning_rate=0.01, iterations=1000, seed=123):
    np.random.seed(seed)
    # initialize weights
    # case of study 1
    #weights = np.zeros(x.shape[1])
    # case of study 3
    weights = np.random.rand(x.shape[1])

    m = len(y)
    # just to save history of the loss function
    history = dict(cost= [], weights= [])
    for _ in range(iterations):
        # calculate the gradient
        gradients = np.dot(x.T, (np.dot(x, weights) - y)) / m
        # update weights
        weights = weights - learning_rate * gradients
        # calculate the loss function
        cost = loss_function(y, predict(x, weights))
        history['cost'].append(cost)
        history['weights'].append(weights)
    return weights, history


def stochastic_gradient_descent(x, y, learning_rate=0.01, iterations=1000, seed=123):
    np.random.seed(seed)
    weights = np.zeros(x.shape[1])
    m = len(y)
    history = dict(cost=[], weights=[])
    for _ in range(iterations):
        # Shuffle los datos para que cada iteración use un orden diferente
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        for i in range(m):
            xi = x_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            # Calcular el gradiente basado en un solo ejemplo de entrenamiento
            gradients = np.dot(xi.T, (np.dot(xi, weights) - yi))
            # Actualizar los pesos
            weights = weights - learning_rate * gradients
            
            # Guardar el costo y los pesos
            cost = loss_function(y, predict(x, weights))
            history['cost'].append(cost)
            history['weights'].append(weights)
    
    return weights, history


def mini_batch_gradient_descent(x, y, learning_rate=0.01, iterations=1000, batch_size=32, seed=123):
    np.random.seed(seed)
    weights = np.zeros(x.shape[1])
    m = len(y)
    history = dict(cost=[], weights=[])
    
    for _ in range(iterations):
        # Shuffle los datos para que cada iteración use un orden diferente
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
       # Dividir los datos en mini lotes
        for i in range(0, m, batch_size):
            xi = x_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            
        # Calcular el gradiente basado en el mini-lote    
            gradients = np.dot(xi.T, (np.dot(xi, weights) - yi)) / batch_size
            
            weights = weights - learning_rate * gradients
        
        # Guardar el costo y los pesos después de procesar todos los mini-lotes
        cost = loss_function(y, predict(x, weights))
        history['cost'].append(cost)
        history['weights'].append(weights)
    
    return weights, history




