"TRABAJO REALIZADO POR MARCO ANTONIO OCHOA CARDENAS Y RICARDO CARDENAS RAMOS"


# - Librerias -

import pandas as pd                                       # dataframes y utilidades
from oandapyV20 import API                                # conexion con broker OANDA
import oandapyV20.endpoints.instruments as instruments    # informacion de precios historicos
import entradas as ent





# Todos los datos del indice
df_indice = f_leer_indice(ent.file, ent.start_date_ent, ent.end_date_test)

# Datos de entrenamiento
df_indice_ent = f_leer_indice(ent.file, ent.start_date_ent, ent.end_date_ent)

# Datos de prueba
df_indice_test = f_leer_indice(ent.file, ent.start_date_test, ent.end_date_test)



def f_descarga_precios(p_instrumento, p_fecha, p_frecuencia, p_ventana):
    """
    Parameters
    ---------
    :param:
        p_instrumento: str : nombre del instruemento
        p_fecha: date : fecha de inicio de descarga de precios
        p_frecuencia: str : granularidad para descragar
        p_ventana: int : numero de precios a bajar

    Returns
    ---------
    :return:
        DataFram: precios del indice

    Debuggin
    ---------
        p_instrumento = 'EUR_USD'
        p_fecha = pd.to_datetime("2019-07-06 00:00:00")
        p_frecuencia = 'M1"
        p_ventana = 30
    """

    # Inicializar api de OANDA
    api = API(environment="practice", access_token=ent.OANDA_ACCESS_TOKEN)
    
    def validate_date(p_instrumento, p_fecha, p_frecuencia):
        p = {"count":1, "granularity":p_frecuencia, "from":p_fecha.strftime('%Y-%m-%dT%H:%M:%S')}
        r = api.request(instruments.InstrumentsCandles(instrument=p_instrumento, params=p))
        price = r.get("candles")
        return price[0]['time']
    
    fecha_diff = pd.to_datetime(
            validate_date(p_instrumento, p_fecha, 
                          p_frecuencia)).tz_convert('GMT') - p_fecha.tz_localize('GMT')
    # String fecha
    fecha = (p_fecha - fecha_diff).strftime('%Y-%m-%dT%H:%M:%S')
    # Parametros
    parameters = {"count": p_ventana, "granularity": p_frecuencia, "from": fecha}
    # Definir el instrumento del que se quiere el precio
    r = instruments.InstrumentsCandles(instrument=p_instrumento, params=parameters)
    # Descargarlo de OANDA
    response = api.request(r)
    # En fomato candles 'open, low, high, close'
    prices = response.get("candles")
    # Regresar solo los precios
    df_prices = pd.concat(
        [pd.DataFrame(
            [
                pd.to_datetime(prices[i]['time']),
                prices[i]['mid']['o'],
                prices[i]['mid']['h'],
                prices[i]['mid']['l'],
                prices[i]['mid']['c']
            ],
            index=['Date', 'Open', 'High', 'Low', 'Close'])
            for i in range(len(prices))],
        axis=1, ignore_index=True).T
                    
    num_col = ['Open', 'High', 'Low', 'Close']
    df_prices[num_col] = df_prices[num_col].apply(pd.to_numeric)
    
    return df_prices

