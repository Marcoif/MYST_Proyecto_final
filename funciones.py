"TRABAJO REALIZADO POR MARCO ANTONIO OCHOA CARDENAS Y RICARDO CARDENAS RAMOS"


# - Librerias -

import entradas as ent
import datos as dat
import pandas as pd
import proceso as pro
import numpy as np
from statsmodels.tsa.stattools import adfuller  # prueba de estacionariedad
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import visualizaciones as vs

'''
_ __ __________________________________________________________________________________ __ _ #

  Analisis de primeras visuales (ventanas de 30 min)
_ __ ____________________________________________________________________________________ __ #
  
'''


# -- ------------------------------------------------------------------------------------ -- #
# -- --------------------------------------------------------- FUNCION: Datos de visuales -- #
# -- Datos importantes para visualizaciones de la primera parte 
# -- ------------------------------------------------------------------------------------ -- #

def visuales(p_ini, p_min, p_max, tipo):
    param_data = pd.DataFrame()
    param_data['p_ini'] = p_ini
    param_data['p_min'] = p_min
    param_data['p_max'] = p_max
    param_data['posicion'] = tipo
    volumen = 463
    tp = 20
    sl = 10
    param_data['dif_min'] = ' '
    param_data['dif_max'] = ' '
    param_data['des_est'] = ' '
    param_data['stoploss'] = ' '
    param_data['take_profit'] = ' '

    for i in range(0, len(param_data)):
        if param_data['posicion'][i] == 'buy':
            param_data['dif_min'][i] = (param_data['p_ini'][i] - param_data['p_min'][i]) * 10000
            param_data['dif_max'][i] = (param_data['p_max'][i] - param_data['p_ini'][i]) * 10000
            param_data['des_est'][i] = (param_data['p_max'][i] - param_data['p_min'][i]) * 10000
            param_data['stoploss'][i] = (param_data['p_ini'][i] - (sl / 10000))
            param_data['take_profit'][i] = (param_data['p_ini'][i] + (tp / 10000))
        else:
            param_data['dif_min'][i] = (param_data['p_ini'][i] - param_data['p_min'][i]) * 10000
            param_data['dif_max'][i] = (param_data['p_max'][i] - param_data['p_ini'][i]) * 10000
            param_data['des_est'][i] = (param_data['p_max'][i] - param_data['p_min'][i]) * 10000
            param_data['stoploss'][i] = (param_data['p_ini'][i] + (sl / 10000))
            param_data['take_profit'][i] = (param_data['p_ini'][i] - (tp / 10000))

    vis_1 = pd.DataFrame(
        {
            'Concepto':
                ['Ventana', 'Precio inicial', 'Valor mínimo en Pips',
                 'Valor máximo en Pips', 'Desviación Estándar', 'Volumen',
                 'Stop loss 10 pips', 'Take profit 20 pips',
                 'Estrategia capital', 'Cuánto perder cuando se pierda',
                 'Cuanto ganar cuando se gana'],

            'Valor':
                ['30 min', param_data['p_ini'][0], param_data['dif_min'][0],
                 param_data['dif_max'][0],
                 param_data['des_est'][0], volumen, param_data['stoploss'][0],
                 param_data['take_profit'][0],
                 '5000 USD', '0.463', '0.926'],

            'Descripción':
                ['7:00 am a 7:30 am', '28/04/2020 7:00 am',
                 'Pips', 'Pips', 'Pips',
                 'unidades = 500 USD = 10% capital (5000 USD)',
                 'Precio stoploss', 'Precio take profit',
                 'Capital Inicial', 'USD', 'USD']
        }
    )

    vis_2 = pd.DataFrame(
        {
            'Concepto':
                ['Ventana', 'Precio inicial', 'Valor mínimo en Pips',
                 'Valor máximo en Pips', 'Desviación Estándar', 'Volumen',
                 'Stop loss 10 pips', 'Take profit 20 pips',
                 'Estrategia capital', 'Cuánto perder cuando se pierda',
                 'Cuanto ganar cuando se gana'],

            'Valor':
                ['30 min', param_data['p_ini'][1], param_data['dif_min'][1],
                 param_data['dif_max'][1],
                 param_data['des_est'][1], volumen, param_data['stoploss'][1],
                 param_data['take_profit'][1],
                 '5000 USD', '0.463', '0.926'],

            'Descripción':
                ['7:00 am a 7:30 am', '31/03/2020 7:00 am', 'Pips', 'Pips', 'Pips',
                 'unidades = 500 USD = 10% capital (5000 USD)',
                 'Precio stoploss', 'Precio take profit', 'Capital Inicial', 'USD', 'USD']
        }
    )

    vis_3 = pd.DataFrame(
        {
            'Concepto':
                ['Ventana', 'Precio inicial', 'Valor mínimo en Pips',
                 'Valor máximo en Pips', 'Desviación Estándar', 'Volumen',
                 'Stop loss 10 pips', 'Take profit 20 pips', 'Estrategia capital',
                 'Cuánto perder cuando se pierda', 'Cuanto ganar cuando se gana'],

            'Valor':
                ['30 min', param_data['p_ini'][2], param_data['dif_min'][2],
                 param_data['dif_max'][2],
                 param_data['des_est'][2], volumen, param_data['stoploss'][2],
                 param_data['take_profit'][2],
                 '5000 USD', '0.463', '0.926'],

            'Descripción':
                ['8:00 am a 8:30 am', '25/02/2020 8:00 am', 'Pips', 'Pips', 'Pips',
                 'unidades = 500 USD = 10% capital (5000 USD)', 'Precio stoploss',
                 'Precio take profit', 'Capital Inicial', 'USD', 'USD']
        }
    )

    vis_4 = pd.DataFrame(
        {
            'Concepto':
                ['Ventana', 'Precio inicial', 'Valor mínimo en Pips',
                 'Valor máximo en Pips', 'Desviación Estándar', 'Volumen',
                 'Stop loss 10 pips', 'Take profit 20 pips', 'Estrategia capital',
                 'Cuánto perder cuando se pierda', 'Cuanto ganar cuando se gana'],

            'Valor':
                ['30 min', param_data['p_ini'][3], param_data['dif_min'][3],
                 param_data['dif_max'][3],
                 param_data['des_est'][3], volumen,
                 param_data['stoploss'][3], param_data['take_profit'][3],
                 '5000 USD', '0.463', '0.926'],

            'Descripción':
                ['8:00 am a 8:30 am', '28/01/2020 8:00 am', 'Pips', 'Pips', 'Pips',
                 'unidades = 500 USD = 10% capital (5000 USD)', 'Precio stoploss',
                 'Precio take profit', 'Capital Inicial', 'USD', 'USD']
        }
    )

    vis_5 = pd.DataFrame(
        {
            'Concepto':
                ['Ventana', 'Precio inicial', 'Valor mínimo en Pips',
                 'Valor máximo en Pips', 'Desviación Estándar', 'Volumen',
                 'Stop loss 10 pips', 'Take profit 20 pips', 'Estrategia capital',
                 'Cuánto perder cuando se pierda', 'Cuanto ganar cuando se gana'],

            'Valor':
                ['30 min', param_data['p_ini'][4], param_data['dif_min'][4],
                 param_data['dif_max'][4],
                 param_data['des_est'][4], volumen, param_data['stoploss'][4],
                 param_data['take_profit'][4],
                 '5000 USD', '0.463', '0.926'],

            'Descripción':
                ['8:00 am a 8:30 am', '31/12/2019 8:00 am', 'Pips', 'Pips', 'Pips',
                 'unidades = 500 USD = 10% capital (5000 USD)', 'Precio stoploss',
                 'Precio take profit', 'Capital Inicial', 'USD', 'USD']
        }
    )

    return vis_1, vis_2, vis_3, vis_4, vis_5


'''
_ __ __________________________________________________________________________________ __ _ #

  Analisis de series de tiempo
_ __ ____________________________________________________________________________________ __ #
  
'''


def check_stationarity(data):
    data = data["Actual"]
    test_results = adfuller(data)
    if test_results[0] < 0 and test_results[1] <= 0.05:
        result = True
    else:
        result = False
    results = {'Resultado': result,
               'Test stadisctic': test_results[0],
               'P-value': test_results[1]
               }
    out = results
    if test_results[1] > 0.01:
        data_d1 = data.diff().dropna()
        results_d1 = adfuller(data_d1)
        if results_d1[0] < 0 and results_d1[1] <= 0.01:
            results_d1 = {'Resultado': True,
                          'Test stadistic': results_d1[0],
                          'P-value': results_d1[1]

                          }
        out = {'Datos originales': results,
               'Primera diferencia': results_d1}
    return out


def fit_arima(data):
    test_result = check_stationarity(data)
    if test_result["Resultado"] == True:
        significant_coef = lambda x: x if x > 0.5 else None
        d = 0
        p = pacf(data["Actual"])
        p = pd.DataFrame(significant_coef(p[i]) for i in range(0, 11))
        idx = 0
        for i in range(len(p)):
            if p.iloc[i] != np.nan:
                idx = i
        p = p.iloc[idx].index

        q = acf(data["Actual"])
        q = pd.DataFrame(significant_coef(q[i]) for i in range(0, 11))
        idx = 0
        for i in range(len(q)):
            if q.iloc[i] != np.nan:
                idx = i
        q = q.iloc[idx].index

        # model = ARIMA(data,order=(p,d,q))
        # model.fit(disp=0)
    return [p, q]


def norm_test(data):
    n_test = shapiro(data["Actual"])
    test_results = {'Test statistic': n_test[0],
                    'P-value': n_test[1]  # si el p-value de la prueba es menor a alpha, rechazamos H0
                    }
    return test_results



def diff_series(data, lags):
    data = data["Actual"].diff(lags).dropna()
    return data


def arch_test(data):
    data = data["Actual"]
    test = het_arch(data)
    results = {"Estadístico de prueba": test[0],
               "P-value": test[1]
               }
    results = pd.DataFrame(results, index=["Resultados"])
    return results


def get_outliers(data):
    # vs.g_AtipicalData(data)
    box = plt.boxplot(data["Actual"])
    bounds = [item.get_ydata() for item in box["whiskers"]]
    datos_atipicos = data.loc[(data["Actual"] > bounds[1][1]) | (data["Actual"] < bounds[0][1])]
    return datos_atipicos





def f_dict_ocurrencias(df_indice, reglas):
    """
    Parameters
    ---------
    :param:
        df_indice: DataFrame : Datos historicos del indice
    Returns
    ---------
    :return:
        ocurrencias: dict : Datos clasificados
    Debuggin
    ---------
        df_indice = datos.f_leer_indice('SPCase-Shiller_Home_Price_Indices(USA).csv')
    """
    return {  # Compresion de diccionario
        # Key es el nombre de los escenarios
        ent.escenarios[n]:
            [  # Compresion de lista para cada ocurrencia
                {  # Diccionario de los datos
                    'ocurrencia %d' % i:
                        df_indice.iloc[i],
                    'fecha':
                        df_indice['DateTime'][i],
                    'precios':
                        dat.f_descarga_precios(
                            ent.instrument, df_indice['DateTime'][i],
                            ent.granularity, ent.window),
                    'operacion':
                        {
                            'tipo':
                                reglas['tipos'][n],
                            'volumen':
                                reglas['volumenes'][n],
                            'takeprofit':
                                reglas['takeprofs'][n],
                            'stoploss':
                                reglas['stoploss'][n]
                        }
                }
                for i in range(len(df_indice))
                if eval(ent.condiciones[n])]
        for n in range(len(ent.escenarios))
    }


# -- ---------------------------------------------------------- FUNCION: Agregar metricas -- #
# -- Agregar las metricas calculadas con funcion f_metricas a dict
# -- ------------------------------------------------------------------------------------ -- #
def f_agregar_metricas_bt(di_ocurrencias):
    """
    Parameters
    ---------
    :param:
        di_ocurrencias: dict : Ocurrencias de f_dict_ocurrencias
    Returns
    ---------
    :return:
        ocurrencias: dict : Agregando las metricas de todas las ocurrencias
    Debuggin
    ---------
        di_ocurrencias = fn.f_dict_ocurrencias(dat.df_indice)
    """
    di_metrics = {}
    for i in ent.escenarios:
        list_metrics = []
        for j in range(len(di_ocurrencias[i])):
            # Agregar a ocurrencias una llave de metricas
            di_ocurrencias[i][j]['metricas'] = pro.f_metricas(di_ocurrencias[i][j]['precios'])

            # Agregar a ocurrencias los resultados
            parametros = [di_ocurrencias[i][j]['operacion']['volumen'],
                          di_ocurrencias[i][j]['operacion']['stoploss'],
                          di_ocurrencias[i][j]['operacion']['takeprofit']]

            resultado, pip, pips = pro.resultado_operacion(di_ocurrencias[i][j]['precios'],
                                                           di_ocurrencias[i][j]['operacion']['tipo'],
                                                           parametros)

            di_ocurrencias[i][j]['resultados'] = {'pips': pip, 'resultado': resultado}
            di_ocurrencias[i][j]['pips'] = pips

            # Hacer una lista con las metricas para cada escenario
            list_metrics.append(pro.f_metricas(di_ocurrencias[i][j]['precios']))
        # Convertir a Data Frame y tener uno con metricas de todos los casos de c/escenario
        di_metrics.update({i: pd.DataFrame.from_dict(list_metrics)})

    return di_metrics


# -- ------------------------------------------------------ FUNCION: Probabilidad metrica -- #
# -- DataFrame del Backtest
# -- ------------------------------------------------------------------------------------ -- #
def f_df_operacion(di_ocurrencias):
    df = [[], [], [], [], [], [], []]  # , [ent.cap_inicial]
    for i in ent.escenarios:
        for j in range(len(di_ocurrencias[i])):
            # Fecha de la ocurrencia
            df[0].append(di_ocurrencias[i][j]['fecha'])

            # Categoria de escenario [A, B, C, D]
            df[1].append(i)

            # Tipo de Operacion [buy. sell]
            df[2].append(di_ocurrencias[i][j]['operacion']['tipo'])

            # Volumen de la operacion
            df[3].append(di_ocurrencias[i][j]['operacion']['volumen'])

            # Resultado [ganadora, perdedora]
            df[4].append(di_ocurrencias[i][j]['resultados']['resultado'])

            # pips y capital
            pips, cap = pro.calculo_cap(di_ocurrencias[i][j]['resultados']['resultado'],
                                        di_ocurrencias[i][j]['resultados']['pips'],
                                        di_ocurrencias[i][j]['operacion']['volumen'])
            df[5].append(pips)
            df[6].append(cap)

    df_ope = pd.DataFrame(df).T
    # Nombrar las columnas
    df_ope.columns = ['timestamp', 'escenario', 'operacion', 'volumen', 'resultado', 'pips', 'capital']
    df_ope.sort_values(by='timestamp', inplace=True)
    df_ope.reset_index(drop=True, inplace=True)

    # pips y cap_acumulado
    pro.cap_acm(df_ope)

    sharp = pro.f_sharpe(df_ope)

    return df_ope, sharp


# -- ------------------------------------------------------ FUNCION: Probabilidad metrica -- #
# -- Diccionario de probabilidad de cada escenario
# -- ------------------------------------------------------------------------------------ -- #
def f_prob_metrica(di_metrics):
    """
    Parameters
    ---------
    :param:
        di_ocurrencias: dict : Ocurrencias de f_dict_ocurrencias
    Returns
    ---------
    :return:
        prob: dict : Diccionario con probabilidades de metricas
    Debuggin
    ---------
        di_ocurrencias = fn.f_dict_ocurrencias(dat.df_indice)
    """
    return {ent.escenarios[n]:
                pro.prob_metrica_mayor(di_metrics,
                                       ent.escenarios[n],
                                       'direccion',
                                       0) for n in range(len(ent.escenarios))}
