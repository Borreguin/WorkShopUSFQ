import pandas as pd

lb_timestamp = "timestamp"
lb_V005_vent01_CO2 = "V005_vent01_CO2"
lb_V022_vent02_CO2 = "V022_vent02_CO2"
lb_V006_vent01_temp_out = "V006_vent01_temp_out"
lb_V023_vent02_temp_out = "V023_vent02_temp_out"

columns = [lb_timestamp, lb_V005_vent01_CO2, lb_V022_vent02_CO2, lb_V006_vent01_temp_out, lb_V023_vent02_temp_out]
alias = {
    lb_timestamp: "timestamp",
    lb_V005_vent01_CO2: "CO2 Ventilation NE",
    lb_V022_vent02_CO2: "CO2 Ventilation SW",
    lb_V006_vent01_temp_out: "Temp. Vent. NE Out",
    lb_V023_vent02_temp_out: "Temp. Vent. SW Out"
}
def read_csv_file(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Error reading file: {file_path}")
        print(e)
        return pd.DataFrame()
    
def elegant_print(message,num_sep):

    print("="*num_sep)
    print(message)
    print("-"*num_sep)

def generarte_only_timestamp(_df):

    _df['just_time_stamp'] = _df.index
    _df['just_time_stamp'] = pd.to_datetime(_df['just_time_stamp'],dayfirst=True).dt.time

    return _df


def expand_timestamp(_df,timestamp_column):

    _df[timestamp_column] = pd.to_datetime(_df[timestamp_column],dayfirst=True,errors='coerce')
    # Extraer características
    _df['year'] = _df[timestamp_column].dt.year
    _df['month'] = _df[timestamp_column].dt.month
    _df['day'] = _df[timestamp_column].dt.day
    _df['day_of_week'] = _df[timestamp_column].dt.dayofweek  # 0 = Lunes, 6 = Domingo
    #_df['is_weekend'] = (_df[timestamp_column] >= 5).astype(int)  # 1 si es fin de semana, 0 si no
    _df['hour'] = _df[timestamp_column].dt.hour

    return _df
