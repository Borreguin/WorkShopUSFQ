from sklearn.metrics import mean_squared_error
from util import *
from utilLinealRegression import gradient_descent5, predict


def load_normalized_data_set():
    # usualmente para evitar realizar las normalizacion de los datos cada vez que se realiza entrenamiento de modelos
    # se guarda los datos normalizados, su media y desviacion estandar en un archivo
    # esto es util cuando el dataset es muy grande y la normalizacion (o preparación de datos) es costosa.
    if not os.path.exists(normalized_file):
        return load_and_save_normalized_data_set()
    return read_normalized_data()

def load_raw_data_set():
    return load_iris_data3()



def train_lineal_regression_model(study_case, learning_rate=0.01):
    if study_case == 1:
        x, y = load_raw_data_set()
    else:
        data = load_normalized_data_set()
        x = data['x']
        y = data['y']

    x_train, x_test, y_train, y_test = split_train_and_test(x, y, 0.2)

    # add column for the bias term (w0)
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]

    # train the model with the specified learning rate
    methods = ['batch','stochastic','mini-batch']
    for method in methods:
        print(f"\nTraining model with method: {method}")

        weights, history = gradient_descent5(x_train, y_train, learning_rate=learning_rate, iterations=1000, method=method)

        # predict y values
        y_pred = predict(x_test, weights)

        # print results
        print_resuts(y_test, y_pred)

        # plot prediction vs real values
        plot_real_vs_predicted(y_test, y_pred)

        # plot loss function
        plot_loss_function(history)\
    
def train_lineal_regression_model2(study_case, learning_rate=0.01):
    if study_case == 1:
        x, y = load_raw_data_set()
    else:
        data = load_normalized_data_set()
        x = data['x']
        y = data['y']

    x_train, x_test, y_train, y_test = split_train_and_test(x, y, 0.2)

    # Add column for the bias term (w0)
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]

    # Train the model
    weights, history = gradient_descent5(x_train, y_train, learning_rate=learning_rate, iterations=1000)

    # Predict y values
    y_pred = predict(x_test, weights)

    # Print results
    print_resuts(y_test, y_pred)

    # Plot prediction vs real values
    plot_real_vs_predicted(y_test, y_pred)

    # Plot loss function
    plot_loss_function(history)

def study_case_1():
    # usar los datos crudos sin normalizar
    train_lineal_regression_model(study_case=1)

def study_case_2():
    # usar los datos normalizados
    train_lineal_regression_model(study_case=2)


if __name__ == '__main__':
    # Experiment with different learning rates
    learning_rates = [0.001]
    for lr in learning_rates:
        print(f"\nEjecutando el caso 1 con taza de aprendizaje: {lr}")
        train_lineal_regression_model(study_case=2, learning_rate=lr)