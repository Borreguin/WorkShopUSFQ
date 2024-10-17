from sklearn.metrics import mean_squared_error
from Taller4.P2_LG.util import *
from Taller4.P2_LG.utilLinealRegression import gradient_descent, predict


def load_normalized_data_set():
    # usualmente para evitar realizar las normalizacion de los datos cada vez que se realiza entrenamiento de modelos
    # se guarda los datos normalizados, su media y desviacion estandar en un archivo
    # esto es util cuando el dataset es muy grande y la normalizacion (o preparación de datos) es costosa.
    if not os.path.exists(normalized_file):
        return load_and_save_normalized_data_set()
    return read_normalized_data()

def load_raw_data_set():
    return load_iris_data()



def train_lineal_regression_model(study_case):
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

    # train the model
    weights, history = gradient_descent(x_train, y_train, learning_rate=0.01, iterations=1000)

    # predict y values
    y_pred = predict(x_test, weights)

    # print results
    print_resuts(y_test, y_pred)

    # plot prediction vs real values
    plot_real_vs_predicted(y_test, y_pred)

    # plot loss function
    plot_loss_function(history)

def study_case_1():
    # usar los datos crudos sin normalizar
    train_lineal_regression_model(study_case=1)

def study_case_2():
    # usar los datos normalizados
    train_lineal_regression_model(study_case=2)


if __name__ == '__main__':
    study_case_1()
    # study_case_2()