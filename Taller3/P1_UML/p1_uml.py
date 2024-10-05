import os
from p1_uml_util import *
import seaborn as sns
import matplotlib.pyplot as plt

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


def plot_boxplot_data(_df, var_to_group):
    _df['just_time_stamp'] = _df.index
    _df['just_time_stamp'] = pd.to_datetime(_df['just_time_stamp']).dt.time

    sns.boxplot(x='just_time_stamp', y=var_to_group, data=df)
    plt.title('Boxplot de ' + var_to_group + ' por hora del día')
    plt.xticks(rotation=90)
    plt.grid(True)
    
    #plt.close()


if __name__ == "__main__":

    df = prepare_data()
    plot_boxplot_data(df,'V005_vent01_CO2')
    plot_boxplot_data(df,'V022_vent02_CO2')
    plt.show()
    plt.close()
    plot_boxplot_data(df,'V006_vent01_temp_out')
    plot_boxplot_data(df,'V023_vent02_temp_out')
    plt.show()
    plt.close()

    #plot_data(df, lb_V005_vent01_CO2, lb_V022_vent02_CO2, "CO2")
    #plot_data(df, lb_V006_vent01_temp_out, lb_V023_vent02_temp_out, "Temperature")

    