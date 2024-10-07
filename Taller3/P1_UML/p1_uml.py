import os
from p1_uml_util import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def prepare_data():
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, "data")
    file_path = os.path.join(data_path, "data.csv")
    _df = read_csv_file(file_path)
    _df.set_index(lb_timestamp, inplace=True)
    print(_df.dtypes)
    return _df

def plot_data(_df: pd.DataFrame, lb1, lb2, legend):
    import matplotlib.pyplot as plt
    df_to_plot = _df.tail(1000)
    plt.plot(df_to_plot.index, df_to_plot[lb1], label=alias[lb_V005_vent01_CO2])
    plt.plot(df_to_plot.index, df_to_plot[lb2], label=alias[lb_V022_vent02_CO2])
    plt.xlabel(lb_timestamp)
    plt.ylabel(legend)
    plt.legend()
    plt.show()

# Apartado A - Plot de variables para observar comportamientos generales diarios.
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funcion plot_boxplot_data - Imprime un boxplot tomando como parametros de entrada un dataframe, la variable timestamp que se agrupará a solo hora, los x e y labels, el titulo y el color para el grafico y los outliers.
def plot_boxplot_data(_df, var_to_group,x_label,y_label,title,box_color):

    _df = generarte_only_timestamp(_df)
    _df = _df.dropna()

    sns.boxplot(x='just_time_stamp',color=box_color, y=var_to_group, data=df,label=var_to_group,flierprops=dict(markerfacecolor=box_color, marker='o', markersize=5, linestyle='none'))
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    #plt.close()

# Apartado B - Identificación de Patrones univariable 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Metodo 1 -Analisis por medidas de tendencia central y dispersion
def data_tendencies_method_1(_df, variable):

    _df = generarte_only_timestamp(_df)
    _df = _df.dropna()

    df_grouped = _df.groupby('just_time_stamp')[variable].agg(
        media='mean',  
        mediana='median',  
        moda=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,  
        desviacion_estandar='std',  
        iqr=lambda x: np.subtract(*np.percentile(x, [75, 25])),
        total_rows='count'
    )


    print(df_grouped)


def data_tend_m1_all_fields(_df):

    for column in df.columns:
        elegant_print("Análisis para columna " + column,150)
        data_tendencies_method_1(_df,column)

# Metodo 2 - Analisis de Frecuencias mediante histograma.
def plot_histogram_data(_df, variable):
    
    _df = generarte_only_timestamp(_df)
    _df = _df.dropna()

    _df_timestamps = _df[['just_time_stamp']].drop_duplicates().reset_index(drop=True)

    # Crear la figura y los ejes
    fig, axs = plt.subplots(4, 6, figsize=(18, 10))  # Tamaño de la figura
    axs = axs.flatten()  # Aplanar la matriz de ejes para facilitar la iteración
    count = 0

    # Crear histogramas para cada timestamp
    elegant_print('Genearando Histograma para variable - '+variable,150)

    for index, row in _df_timestamps.iterrows():
        axs[count].hist(_df[_df['just_time_stamp'] == row.iloc[0]][variable], bins=30, color='blue', alpha=0.7, edgecolor='black')  # Crear histograma
        axs[count].set_title(f'Histograma de {row.iloc[0]}')  # Título del gráfico
        axs[count].set_xlabel('Valores')  # Etiqueta del eje x
        axs[count].set_ylabel('Frecuencia')  # Etiqueta del eje y
        axs[count].grid(axis='y', alpha=0.75)  # Añadir una cuadrícula
        count = count + 1
    plt.get_current_fig_manager().window.wm_title('Análisis de Variable - ' + variable)
    plt.tight_layout()
    plt.show()  

    

def plot_histogram_all(_df):

    for column in df.columns:
        if column != 'just_time_stamp':
            plot_histogram_data(_df,column) 

# Apartado C - Identificación de anomalías univariable
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Metodo 1 - Uso de GMM


def show_outliers(_df, variable):

    _df = generarte_only_timestamp(_df)
    _df = _df.dropna()

    _df_timestamps = _df[['just_time_stamp']].drop_duplicates().reset_index(drop=True)
    _df['is_outlier'] = 0
    _df['color'] = 'lightgreen'


    for index, row in _df_timestamps.iterrows():

        Q1 = _df[(_df['just_time_stamp'] == row.iloc[0])][variable].quantile(0.25)
        Q3 = _df[(_df['just_time_stamp'] == row.iloc[0])][variable].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        _df.loc[(_df['just_time_stamp'] == row.iloc[0]) & ((_df[variable] < lower_bound) | (_df[variable] > upper_bound)),'is_outlier'] = 1
        _df.loc[(_df['just_time_stamp'] == row.iloc[0]) & ((_df[variable] < lower_bound) | (_df[variable] > upper_bound)),'color'] = 'red'

    
    # Crear un gráfico de dispersión
    plt.figure(figsize=(12, 6))  # Tamaño de la figura

    plt.scatter(
        pd.to_datetime(_df[_df['is_outlier'] == 0].index),  
        _df[_df['is_outlier'] == 0][variable],
        c='green',  # Color para puntos normales
        alpha=0.7,
        #edgecolor='black',
        s=10,
        label='Normal'  # Leyenda para puntos normales
    )

    # Puntos outliers
    plt.scatter(
        pd.to_datetime(_df[_df['is_outlier'] == 1].index),   
        _df[_df['is_outlier'] == 1][variable],
        c='red',  # Color para outliers
        alpha=0.7,
        #edgecolor='black',
        s=10,
        label='Outlier'  # Leyenda para outliers
    )

    # Título y etiquetas
    plt.title(f'Dispersión de {variable}')  # Título del gráfico
    plt.xlabel('Time Stamp')                  # Etiqueta del eje x
    plt.ylabel(variable)                      # Etiqueta del eje y
    plt.grid(True)                           # Añadir una cuadrícula
    # Agregar leyenda
    plt.legend()

    elegant_print('Outliers encontrados para la variable - ' + variable,150)
    print(_df[_df['is_outlier'] == 1])

    # Mostrar el gráfico
    plt.tight_layout()
    plt.get_current_fig_manager().window.wm_title('Análisis de Variable - ' + variable)
    plt.show()


def show_outliers_all(_df):

    for column in df.columns:
        if column != 'just_time_stamp':
            show_outliers(_df,column) 
    

# Apartado D - Búsqueda de patrones multivariable
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Método 1 - uso de clustering por GMM (Gaussian Mixture Model)

def gmm_clustering(gmm_covariance_type, gmm_max_iter, gmm_tol, gmm_n_components, gmm_init_params, df_test):


    # Definir el pipeline
    pipeline_GMM = Pipeline([
        ('scaler', StandardScaler()),    # Normaliza los datos
        ('gmm', GaussianMixture())       # Inicializa GMM sin parámetros
    ])

    # Definir el rango de parámetros para la búsqueda
    param_grid_GMM = {
        'gmm__covariance_type': gmm_covariance_type,  # Tipo de matriz de covarianza
        'gmm__max_iter': gmm_max_iter,                # Número máximo de iteraciones
        'gmm__tol': gmm_tol,                          # Tolerancia para convergencia
        'gmm__n_components': gmm_n_components,        # Número de componentes
        'gmm__init_params': gmm_init_params           # Inicialización de parámetros
    }

     # Evaluar el rendimiento usando GridSearchCV
    def silhouette_scorer(estimator, X):
        # Ajustar el modelo para obtener etiquetas de clusters
        labels = estimator.named_steps['gmm'].predict(X)
        # Solo calculamos el Silhouette Score si hay al menos dos clusters
        if len(set(labels)) > 1:
            return silhouette_score(X, labels)
        return -1

    grid_search_GMM = GridSearchCV(pipeline_GMM, param_grid_GMM, scoring=silhouette_scorer, cv=4)

    # Ajustar el GridSearchCV a los datos
    grid_search_GMM.fit(df_test)

    # Obtener los mejores parámetros
    best_params_GMM = grid_search_GMM.best_params_
    print(f"Mejores parámetros: {best_params_GMM}")

    # Obtener el mejor modelo
    best_model_GMM = grid_search_GMM.best_estimator_

    return grid_search_GMM, best_model_GMM


def visualize_gmm(grd_srch_GMM):
    
    results = pd.DataFrame(grd_srch_GMM.cv_results_)

    # Mostrar los datos de las ejecuciones
    elegant_print("Datos de las ejecuciones: ", 150)
    print(results[['param_gmm__covariance_type', 'param_gmm__init_params', 'param_gmm__max_iter', 'param_gmm__n_components','param_gmm__tol', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False))

    elegant_print("Gráfico de scores: ", 150)

    res_grouped = results.groupby('param_gmm__n_components').agg(
    max_mean_test_score=('mean_test_score', 'max')
    ).reset_index()



        # Filtrar solo algunos parámetros y el mean_test_score
    plot_data = res_grouped[['param_gmm__n_components', 'max_mean_test_score']]

    # Graficar
    plt.figure(figsize=(10,6))
    plt.plot(plot_data['param_gmm__n_components'], plot_data['max_mean_test_score'], marker='o', linestyle='-', color='b')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Mean Test Score')
    plt.title('Relación entre Número de Componentes y Score')
    plt.grid(True)
    plt.show()

def apply_best_model_dataset(df, bst_modl, mdl_name, clm_name, clm_no_use, index):

    if clm_name in df.columns:

        df[clm_name] = bst_modl.named_steps[mdl_name].fit_predict(df.iloc[:, index:])
        
    else:
        
        df[clm_name] = bst_modl.named_steps[mdl_name].fit_predict(df.iloc[:, index:])

    return df

def plot_clusters(_df,lst_variables,clst_field,is_time):
    
    _df = _df.dropna()

    if is_time == 1:
        _df = generarte_only_timestamp(_df)
        x_label_val = 'just_hour_stamp'
        _df['just_hour_stamp'] = _df['timestamp'].dt.hour
    else:
        x_label_val = 'timestamp'

    # Crear la figura y los ejes
    fig, axs = plt.subplots(2, 1, figsize=(18, 10))  # Tamaño de la figura
    axs = axs.flatten()  # Aplanar la matriz de ejes para facilitar la iteración
    count = 0

    for variable in lst_variables:

        # Crear histogramas para cada timestamp
        elegant_print('Genearando Dispersión para variable - '+variable,150)

        colors = {0: 'red', 1: 'blue', 2: 'green',3:'yellow',4:'purple',-1:'black'}


        for cluster in colors.keys():

            df_filtered = _df[_df[clst_field] == cluster]
            if is_time != 1:
                df_filtered[x_label_val] = pd.to_datetime(df_filtered[x_label_val])

            if not df_filtered.empty:
                scatter = axs[count].scatter(
                    df_filtered[x_label_val],  # Usar el timestamp como eje x
                    df_filtered[variable],  # Usar la variable como eje y
                    color=colors[cluster],  # Usar la columna 'color' para el color de los puntos
                    alpha=0.7,  # Transparencia
                    #edgecolor='black',
                    s=10,  # Tamaño de los puntos
                    label='Cluster ' + str(cluster)
                )


        # Título y etiquetas
        axs[count].set_title(f'Dispersión de Variable - ' + variable)  # Título del gráfico
        axs[count].set_xlabel('timestamp')  # Etiqueta del eje x
        axs[count].set_ylabel(variable)  # Etiqueta del eje y
        axs[count].grid(axis='y', alpha=0.75)  # Añadir una cuadrícula
        axs[count].legend()

        elegant_print('Clusterizacion de ' + variable,150)
        count=count+1

    # Mostrar el gráfico
    plt.tight_layout()
    plt.get_current_fig_manager().window.wm_title('Clusterización para Variables')
    plt.show()


# Apartado E - Búsqueda de anomalias multivariable
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Metodos - IsolationForest - LocalOutlierFactor

def show_outliers_mvar(_df,columns,lst_variables,tipo):

    _df = _df.dropna()

    if tipo == 'IsolationForest':

        # Ajustar el modelo Isolation Forest
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(_df[columns].dropna())

        # Predecir anomalías
        y_pred = model.predict(_df[columns].dropna())

        _df['Anomaly'] = y_pred

    
    else:
        
        # Ajustar el modelo Local Outlier Factor
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        y_pred = model.fit_predict(_df[columns].dropna())

        _df['Anomaly'] = y_pred


    # Crear la figura y los ejes
    fig, axs = plt.subplots(2, 1, figsize=(18, 10))  # Tamaño de la figura
    axs = axs.flatten()  # Aplanar la matriz de ejes para facilitar la iteración
    count = 0

    for variable in lst_variables:

        # Crear histogramas para cada timestamp
        elegant_print('Deteccion de anomalías para variable - '+ variable,150)

        colors = {1: 'green', -1: 'red'}


        for cluster in colors.keys():

            df_filtered = _df[_df['Anomaly'] == cluster]
        
            if not df_filtered.empty:
                scatter = axs[count].scatter(
                df_filtered['timestamp'],  # Usar el timestamp como eje x
                df_filtered[variable],  # Usar la variable como eje y
                alpha=0.7,  # Transparencia
                #edgecolor='black',
                color=colors[cluster],
                s=10,  # Tamaño de los puntos
                label='Normal' if colors[cluster] == 'green' else 'Anomalía'
                )



        # Título y etiquetas
        axs[count].set_title(f'Deteccion de anomalías usando ' + tipo +' para variable - ' + variable)  # Título del gráfico
        axs[count].set_xlabel('timestamp')  # Etiqueta del eje x
        axs[count].set_ylabel(variable)  # Etiqueta del eje y
        axs[count].grid(axis='y', alpha=0.75)  # Añadir una cuadrícula
        axs[count].legend()
        count=count+1
        elegant_print('Anomalías para variable ' + variable,150)
        print(_df[_df['Anomaly'] == -1])

    # Mostrar el gráfico
    plt.tight_layout()
    plt.get_current_fig_manager().window.wm_title('Deteccion de anomalías para Variables')
    plt.show()


if __name__ == "__main__":

    df = prepare_data()

    # Apartado A - Plot de variables
    # --------------------------------------------------------------------------------------------------
    df = prepare_data()
    plot_boxplot_data(df,'V005_vent01_CO2','timestamp','C02','C02 level vs timestamp','#1f77b4')
    plot_boxplot_data(df,'V022_vent02_CO2','timestamp','C02','C02 level vs timestamp','orange')
    plt.show()
    plot_boxplot_data(df,'V006_vent01_temp_out','timestamp','Temp','Temp vs timestamp','#1f77b4')
    plot_boxplot_data(df,'V023_vent02_temp_out','timestamp','Temp','Temp vs timestamp','orange')
    plt.show()
    

    #Apartado B - Patrones univariables
    # --------------------------------------------------------------------------------------------------
    
    # Metodo 1 - Analisis por metricas de tendencia central y dispersion:
    # --------------------------------------------------------------------
    data_tend_m1_all_fields(df)
    
    # Método 2 - Análisis por frecuencias mediante histograma
    # --------------------------------------------------------------------
    plot_histogram_all(df)
    
    

    # Apartado C - Búsqueda de Anomalías
    # --------------------------------------------------------------------------------------------------
    show_outliers_all(df)

    # Apartado D - Patrones multivariables:
    # --------------------------------------------------------------------------------------------------

    # Matriz de valores para utilizar GridSearch
    gmm__covariance_type = ['diag','tied','spherical']
    gmm__max_iter = [500,1000,1500]
    gmm__tol = [0.01,0.05,0.03,0.009,0.02,0.01]
    gmm__n_components = [2,3,4,5] #
    gmm__init_params = ['kmeans','random'] 

    # Primero variables de CO2
    # --------------------------------------------------
    #Dataset para probar y clusterizacion

    df_to_gmm =  expand_timestamp(df.reset_index(),'timestamp') 
    
    
    grd_srch_GMM_CO2, bst_mdl_GMM = gmm_clustering(gmm__covariance_type, gmm__max_iter, gmm__tol, gmm__n_components, gmm__init_params, df_to_gmm[['V005_vent01_CO2','V022_vent02_CO2','year','month','day','day_of_week','hour']].dropna())

    #Ver resultados:
    visualize_gmm(grd_srch_GMM_CO2)
    _df_clustered_CO2 = apply_best_model_dataset(df_to_gmm, bst_mdl_GMM, 'gmm', 'cluster_gmm', 'cluster',6)
    #Cluster agurpado
    print(_df_clustered_CO2)

    #Plotear los clusters en
    plot_clusters(_df_clustered_CO2,['V005_vent01_CO2','V022_vent02_CO2'],'cluster_gmm',0)
    plot_clusters(_df_clustered_CO2,['V005_vent01_CO2','V022_vent02_CO2'],'cluster_gmm',1)
    
    # Segundo variables de temperatura
    # --------------------------------------------------

    grd_srch_GMM_temp, bst_mdl_GMM_temp = gmm_clustering(gmm__covariance_type, gmm__max_iter, gmm__tol, gmm__n_components, gmm__init_params, df_to_gmm[['V006_vent01_temp_out','V023_vent02_temp_out','year','month','day','day_of_week','hour']].dropna())

    #Ver resultados:
    visualize_gmm(grd_srch_GMM_temp)
    _df_clustered_temp = apply_best_model_dataset(df_to_gmm, bst_mdl_GMM_temp, 'gmm', 'cluster_gmm', 'cluster',6)

    
    #Cluster agurpado
    print(_df_clustered_temp)

    #Plotear los clusters en
    plot_clusters(_df_clustered_temp,['V006_vent01_temp_out','V023_vent02_temp_out'],'cluster_gmm',0)
    plot_clusters(_df_clustered_temp,['V006_vent01_temp_out','V023_vent02_temp_out'],'cluster_gmm',1)

    
    # Apartado E - anomalias multivariables:
    # --------------------------------------------------------------------------------------------------

    df_anomalie_isol_for = df_to_gmm.copy()
    
    show_outliers_mvar(df_anomalie_isol_for,['V005_vent01_CO2','V022_vent02_CO2','year','month','day','day_of_week','hour'],['V005_vent01_CO2','V022_vent02_CO2'],'IsolationForest')
    show_outliers_mvar(df_anomalie_isol_for,['V006_vent01_temp_out','V023_vent02_temp_out','year','month','day','day_of_week','hour'],['V006_vent01_temp_out','V023_vent02_temp_out'],'IsolationForest')


    show_outliers_mvar(df_anomalie_isol_for,['V005_vent01_CO2','V022_vent02_CO2','year','month','day','day_of_week','hour'],['V005_vent01_CO2','V022_vent02_CO2'],'LocalOutlierFactor')
    show_outliers_mvar(df_anomalie_isol_for,['V006_vent01_temp_out','V023_vent02_temp_out','year','month','day','day_of_week','hour'],['V006_vent01_temp_out','V023_vent02_temp_out'],'LocalOutlierFactor')

    

    


    