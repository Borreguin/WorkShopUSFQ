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
    weights = np.zeros(x.shape[1])
    # case of study 3
    # weights = np.random.rand(x.shape[1])

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


def gradient_descent2(x, y, learning_rate=0.01, iterations=1000, epsilon=1e-6):
    m = len(y)
    weights = np.zeros(x.shape[1])
    history = {'cost': []}

    for i in range(iterations):
        predictions = x.dot(weights)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        history['cost'].append(cost)

        # Check for convergence
        if i > 0 and abs(history['cost'][-2] - cost) < epsilon:
            print(f'Converged after {i} iterations.')
            break

        # Update weights
        gradient = (1 / m) * x.T.dot(errors)
        weights -= learning_rate * gradient

    return weights, history
def gradient_descent3(x, y, learning_rate=0.01, iterations=1000, epsilon=1e-6):
    m = len(y)
    weights = np.zeros(x.shape[1])
    history = {'cost': []}

    for i in range(iterations):
        # Calcular las predicciones y el error
        predictions = x.dot(weights)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        history['cost'].append(cost)

        # Criterio de parada basado en el cambio de costo
        if i > 0 and abs(history['cost'][-2] - cost) < epsilon:
            print(f'Convergencia alcanzada después de {i} iteraciones.')
            break

        # Actualización de los pesos
        gradient = (1 / m) * x.T.dot(errors)
        weights -= learning_rate * gradient

    return weights, history

def gradient_descent4(x, y, learning_rate=0.01, iterations=1000, epsilon=1e-6, random_init=False):
    m = len(y)
    # Inicializar los pesos de manera aleatoria o en cero según la opción elegida
    if random_init:
        weights = np.random.randn(x.shape[1])  # Inicialización aleatoria
    else:
        weights = np.zeros(x.shape[1])  # Inicialización en cero
    history = {'cost': []}

    for i in range(iterations):
        # Calcular las predicciones y el error
        predictions = x.dot(weights)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        history['cost'].append(cost)

        # Criterio de parada basado en el cambio de costo
        if i > 0 and abs(history['cost'][-2] - cost) < epsilon:
            print(f'Convergencia alcanzada después de {i} iteraciones.')
            break

        # Actualización de los pesos
        gradient = (1 / m) * x.T.dot(errors)
        weights -= learning_rate * gradient

    return weights, history

def gradient_descent5(x, y, learning_rate=0.01, iterations=1000, epsilon=1e-6, method='batch', batch_size=32):
    m = len(y)
    weights = np.zeros(x.shape[1])  # Inicialización de los pesos en cero
    history = {'cost': []}

    for i in range(iterations):
        if method == 'batch':
            # Batch Gradient Descent: utiliza todos los datos
            predictions = x.dot(weights)
            errors = predictions - y
            gradient = (1 / m) * x.T.dot(errors)

        elif method == 'stochastic':
            # Stochastic Gradient Descent: utiliza un ejemplo aleatorio
            random_index = np.random.randint(m)
            xi = x[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            predictions = xi.dot(weights)
            errors = predictions - yi
            gradient = xi.T.dot(errors)

        elif method == 'mini-batch':
            # Mini-Batch Gradient Descent: utiliza un subconjunto de datos
            random_indices = np.random.choice(m, batch_size, replace=False)
            xi = x[random_indices]
            yi = y[random_indices]
            predictions = xi.dot(weights)
            errors = predictions - yi
            gradient = (1 / batch_size) * xi.T.dot(errors)

        # Calcular el costo
        cost = (1 / (2 * m)) * np.sum((x.dot(weights) - y) ** 2)
        history['cost'].append(cost)

        # Criterio de parada
        if i > 0 and abs(history['cost'][-2] - cost) < epsilon:
            print(f'Convergencia alcanzada después de {i} iteraciones.')
            break

        # Actualizar los pesos
        weights -= learning_rate * gradient

    return weights, history

