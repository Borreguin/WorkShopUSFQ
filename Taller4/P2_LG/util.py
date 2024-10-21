import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

data_path = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)
normalized_file = os.path.join(os.path.dirname(__file__),  'data', 'normalized_data.pkl')

def load_iris_data():
    iris = load_iris()
    # we choose three features to predict one feature (Petal Length)
    x = iris.data[:, [0, 1, 3]]  # Sepal Length, Sepal Width, Petal Width
    # we assume that we want to predict the petal length
    y = iris.data[:, 2]  # Petal Length
    return x, y
def load_iris_data2():
    iris = load_iris()
    # Usamos las mismas características (longitud de sépalo, ancho de sépalo, ancho de pétalo), pero cambiamos la variable objetivo
    x = iris.data[:, [0, 1, 3]]  # Longitud del sépalo, ancho del sépalo, ancho del pétalo
    y = iris.data[:, 1]  # Ancho del sépalo como nueva variable objetivo
    return x, y

def load_iris_data3():
    iris = load_iris()
    # Usar solo dos características: longitud del sépalo y ancho del pétalo
    x = iris.data[:, [0, 3]]  # Longitud del sépalo, ancho del pétalo
    y = iris.data[:, 1]  # Ancho del sépalo como variable objetivo
    return x, y

def load_and_save_normalized_data_set():
    x, y = load_iris_data()

    # normalize data
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    std_x = x.std(axis=0)
    std_y = y.std()

    x = (x - x_mean) / std_x
    y = (y - y_mean) / std_y

    data = dict(x=x, y=y, x_mean=x_mean, y_mean=y_mean, std_x=std_x, std_y=std_y)
    # save data as pickle
    with open(normalized_file, 'wb') as f:
        pickle.dump(data, f)

    return data

def normalize_a_sample(x, y, x_mean, y_mean, std_x, std_y):
    x = (x - x_mean) / std_x
    y = (y - y_mean) / std_y
    return x, y

def denormalize_a_sample(x, y, x_mean, y_mean, std_x, std_y):
    x = x * std_x + x_mean
    y = y * std_y + y_mean
    return x, y

def read_normalized_data():
    # read data from pickle
    with open(normalized_file, 'rb') as f:
        data = pickle.load(f)
        return data


def split_train_and_test(x,y, test_size=0.2, seed=123):
    return train_test_split(x, y, test_size=test_size, random_state=seed)

def plot_real_vs_predicted(y_test, y_pred):
    plt.bar(np.arange(len(y_test)), y_test, alpha=1)
    plt.bar(np.arange(len(y_test)), y_pred, alpha=0.5)
    plt.legend(['Real', 'Predicted'])
    plt.show()

def plot_loss_function(history):
    plt.plot(history['cost'])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss function')
    plt.show()

def print_resuts(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error MSE: \t\t\t\t{round(mse, 3)}')
    range_y = np.max(y_test) - np.min(y_test)
    mse_percentage = (mse / range_y) * 100
    print(f'Mean Squared Error MSE percentage: \t\t{round(mse_percentage, 3)}%')

    # percentual error
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f'Mean Absolute Percentage Error MAPE: \t{round(mape, 3)}%')