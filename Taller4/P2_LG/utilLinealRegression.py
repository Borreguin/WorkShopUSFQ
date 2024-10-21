import numpy as np

# Loss function: Mean Squared Error MSE
def loss_function(y, y_hat):
    return (1 / 2 * len(y)) * sum((y - y_hat) ** 2)

# predict function using weights of the model
def predict(x, weights):
    return np.dot(x, weights)

# Gradient function
def gradient_descent(x, y, learning_rate = 0.01, iterations = 1000, seed = 123):
    np.random.seed(seed)
    # initialize weights
    # case of study 1
    weights = np.zeros(x.shape[1])
    # case of study 3
    # weights = np.random.rand(x.shape[1])

    m = len(y)
    # just to save history of the loss function
    history = dict(cost = [], weights = [])
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



