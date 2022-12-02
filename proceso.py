

# - Librerias -

import pandas as pd
import numpy as np
import entradas as ent
import datos as dat
import pyswarms as ps

def serie_normalizar(serie):
    """
    Parameters
    ---------
    :param:
        serie: pd.Serie : Precios en serie de tiempo
    Returns
    ---------
    :return:
        serie normalizada
    Debuggin
    ---------
        serie = di_ocurrencias['A'][0]['precios']['Close']
    """
    return (serie - serie.mean())/serie.std()


'''
(Dirección) Close (t_30) - Open(t_0) 
(Pips Alcistas) High(t_0 : t_30) – Open(t_0) 
(Pips Bajistas) Open(t_0) – Low(t_0 : t_30) 
(Volatilidad) High(t_-30 : t_30) ,  - mínimo low (t_-30:t_30)
'''

def f_metricas(df_precios):
    """
    Parameters
    ---------
    :param:
        df_precios: DataFrame : Datos de precios historicos (OHLC)
    Returns
    ---------
    :return:
        dict : Metricas calculadas con los precios
    Debuggin
    ---------
        df_indice = ocurrencias['A'][0]['precios']
    """
    return {
                'direccion':
                    (df_precios['Close'][len(df_precios)-1] - df_precios['Open'][0])*10000,
                    
                'alcista':
                    (df_precios['High'].max() - df_precios['Close'][0])*10000,
                    
                'bajista':
                     (df_precios['Open'][0] - df_precios['Low'].min())*10000,
                     
                'volatilidad':
                    (df_precios['High'].max() - df_precios['Low'].min())*10000
            }



def prob_metrica_mayor(di_metrics, grupo, metrica, umbral):
    """
    Parameters
    ---------
    :param:
        serie: pd.Serie : Precios en serie de tiempo
    Returns
    ---------
    :return:
        serie normalizada
    Debuggin
    ---------
        serie = di_ocurrencias['A'][0]['precios']['Close']
    """
    return len(di_metrics[grupo][metrica][
                di_metrics[grupo][metrica] > umbral]
        ) / len(di_metrics[grupo][metrica])


def resultado_operacion(df_prices, tipo, parametros):
    """
    Parameters
    ---------
    :param:
        df_prices: pd.DataFrame : Ventana de precios OHLC
        tipo: str : tipo de operacion ['buy', 'sell']
        parametrso: parametros de la operacion
    Returns
    ---------
    :return:
        resultado: str : si fue ganadora o perdedora la operacion
        pip : float : cuanto fue la diferencia con la que cerro
        pips: DataFrame : lista de pips en comparacion al primer precio
    Debuggin
    ---------
        df_prices = f_descarga_precios(ent.instrument, 
                                   pd.to_datetime("2009-01-05 13:00:00"),
                                   ent.granularity, ent.window)
        tipo = 'buy'
        parametros = [volumen, stop, take] 
    """
    # Calcular los pips que cambian a partir del primer precio
    pips = pd.DataFrame((np.array(df_prices)[:, 1:] - df_prices.iloc[0,1])*10000)
    
    if tipo == 'buy':
        # Ver cuando es mayor al takeprofit
        take = pips[pips >= parametros[2]].dropna(how = 'all')
        # Cuando es menor o igual al stoploss
        stop = pips[pips <= -parametros[1]].dropna(how = 'all')
    else:
        # Ver cuando es mayor al takeprofit
        take = pips[pips <= -parametros[2]].dropna(how = 'all')
        # Cuando es menor o igual al stoploss
        stop = pips[pips >= parametros[1]].dropna(how = 'all')
        
    # Si ambos casos sucedieron
    if len(take) > 0 and len(stop) > 0:
        # Revisar cual sucedio primero
        ind_take = take.index[0]
        ind_stop = stop.index[0]
        if ind_take > ind_stop:
            pip = parametros[2]
            resultado = 'ganadora'
        else:
            pip = parametros[1]
            resultado = 'perdedora'
    # Si nunca toco el takeprofit
    elif len(take) == 0 and len(stop) > 0:
        pip = parametros[1]
        resultado = 'perdedora'
    # Si nunca toco el stoploss
    elif len(stop) == 0 and len(take) > 0:
        pip = parametros[2]
        resultado = 'ganadora'
    # Si nunca toco ninguno
    else:
        pip = round(pips.iloc[len(pips)-1,3], 4)
        if pip > 0:
            resultado = 'ganadora'
        else:
            resultado = 'perdedora'
    
    return resultado, pip, pips

    

def calculo_cap(resultado, pips, volumen):
    """
    Parameters
    ---------
    :param:
        resultado: str : Si fue 'ganadora' o 'perdedora'
        pips: str : pips ganados en tal operacion
        volument: volumen por operacion
    Returns
    ---------
    :return:
        capital : float : cuanto se gano en dolares por operacion
        pips: DataFrame : pips positivos o negativos
    Debuggin
    ---------
        resultado = 'ganadora'
        pips = 
        volument: volumen por operacion 
    """
    if resultado == 'ganadora':
        pips = pips
        capital = (volumen * pips/10000)*ent.apalanca
    else:
        pips = -pips
        capital = (volumen * pips/10000)*ent.apalanca
    return pips, capital


def cap_acm(df_operaciones):
    """
    Parameters
    ---------
    :param:
        df_profit: DataFrame : rendimientos de las operaciones diarias
        col : str : nombre de la columna a la que se le calcula tales rendimientos
    Returns
    ---------
    :return: 
        param_profit: DataFrame
    Debuggin
    ---------
        param_profit = f_profit_diario(f_leer_archivo('archivo_tradeview_1.xlsx'))
        col = 'profit_acm'
    """
    df_operaciones['pips_acm'] = df_operaciones['pips'].cumsum()
    
    cap_acm = df_operaciones['capital'].cumsum()
    df_operaciones['capital_acm'] = [float(ent.cap_inicial + cap_acm[i]) 
                                    for i in range(len(df_operaciones['capital']))]
    


def f_sharpe(df_operaciones):
    """
    Parameters
    ---------
    :param:
        df_operaciones: DataFrame : operaciones en un periodo del tiempo
    Returns
    ---------
    :return: 
        sharpe: medida de atribucion
    Debuggin
    ---------
        df_opreaciones = funciones.66f_df_operacion(di_ocurrencias)
    """
    # -- DATOS --
    # Tasa libre de riesgo
    rf = 0
    
    # Sacar el rend de profit diario de las Operaciones
    param_profit = log_rends(df_operaciones, 'capital_acm')
    
    # Rendimientos
    rp = param_profit['rends']
    
    sharpe = (rp.mean() - rf) / rp.std() 
    return sharpe



def log_rends(df_profit, col):
    """
    Parameters
    ---------
    :param:
        df_profit: DataFrame : rendimientos de las operaciones diarias
        col : str : nombre de la columna a la que se le calcula tales rendimientos
    Returns
    ---------
    :return: 
        param_profit: DataFrame
    Debuggin
    ---------
        param_profit = f_profit_diario(f_leer_archivo('archivo_tradeview_1.xlsx'))
        col = 'profit_acm'
    """
    df_profit['rends'] = np.log(
                df_profit[col]/
                df_profit[col].shift(1)).iloc[1:]
    
    return df_profit


def no_optimized_search(df_grupo, list_pips, list_precios, init_params, n):
    """
    Parameters
    ---------
    :param:
        df_grupo: DataFrame : operaciones en un periodo del tiempo
        list_pips : list : lista de pips 
        list_precios: list : lista de precios
        init_param : list : parametros iniciales
        n : int : numero de busquedas
    Returns
    ---------
    :return: 
        best_sharpe, list_groups, init_x
    Debuggin
    ---------
        df_grupo = df_operaciones[df_operaciones['escenario']=='A'].reset_index(drop=True)
        list_pips  = [di_ocurrencias['A'][i]['pips'] 
										for i in range(len(di_ocurrencias['A']))]
        list_precios = [di_ocurrencias['A'][i]['precios'] 
										for i in range(len(di_ocurrencias['A']))]
        init_param = list : [vol, sl, tp]
        n = 25
    """
    
    # Los precios del back test
    precios = list_precios
    
    # Tomar minimos de Low y maximos de High en pips
    min_max = pd.DataFrame([[i.iloc[:,2].min(), i.iloc[:,1].max()] for i in list_pips])
    # Cuantiles que será nuestros limites para stop loss y take profit
    quants = min_max.quantile([.25, .75])
    
    # -- Limites de busqueda --
    # Valores de busqueda de volumen (100, 1000)
    vol = np.random.randint(100, 1000, (n, 1))
    # Valores de busqueda para stop
    stop = np.random.randint(abs(quants.iloc[1,0]), abs(quants.iloc[0,0]), (n, 1))
    # Valores de busqueda del take
    take = np.random.randint(abs(quants.iloc[0,1]), abs(quants.iloc[1,1]), (n, 1))
    
    # Tomar los limites de busquedas en un solo array
    init_x = np.concatenate((vol, stop, take), axis=1)
    # Para futura busqueda del df de resultados
    list_groups = []
    
    # Funcion de Sharpe
    def get_sharpe(parametros):
        grupo = df_grupo.copy()
        grupo['volumen'] = parametros[0]
        for i in range(len(grupo)):
            resultado, pip, pips = resultado_operacion(precios[i], grupo.operacion[i], 
													   parametros)
            grupo['resultado'].iloc[i] = resultado
            pips_n, capital = calculo_cap(resultado, pip, grupo['volumen'].iloc[i])
            grupo['pips'].iloc[i] = pips_n
            grupo['capital'].iloc[i] = capital
        cap_acm(grupo)
        list_groups.append(grupo)
        # Sharpe
        sharp = f_sharpe(grupo)
        return sharp
    
    # Funcion para aplucarala a mas de una lista de parametros [vol, sl, tp]
    fun = lambda x: np.apply_along_axis(get_sharpe, 1, x)
    
    # Lista de n sharpes
    n_sharpes = list(fun(init_x))
    
    # El sharpe mas grande
    best_sharpe = max(n_sharpes)
    # Localizacion de tal sharpe
    index = n_sharpes.index(best_sharpe)
    
    return best_sharpe, list_groups[index], init_x[index]
    #return n_sharpes, list_groups, init_x


# -- -------------------------------------------------------- FUNCION: Busqueda optmizada -- #
# -- Busqueda optmizada con pso para el mejor sharpe
# -- ------------------------------------------------------------------------------------ -- #
def optimized_search(df_grupo, list_pips, list_precios):
    """
    Parameters
    ---------
    :param:
        df_grupo: DataFrame : operaciones en un periodo del tiempo
        list_pips : list : lista de pips 
        list_precios: list : lista de precios
        
    Returns
    ---------
    :return: 
        best_sharpe
        grupo
        best_param
        
    Debuggin
    ---------
        df_grupo = df_operaciones[df_operaciones['escenario']=='A'].reset_index(drop=True)
        list_pips  = [di_ocurrencias['A'][i]['pips'] 
								for i in range(len(di_ocurrencias['A']))]
        list_precios = [di_ocurrencias['A'][i]['precios'] 
								for i in range(len(di_ocurrencias['A']))]
    """
    
    grupo = df_grupo.copy()
    precios = list_precios
    
    min_max = pd.DataFrame([[i.iloc[:,2].min(), i.iloc[:,1].max()] for i in list_pips])
    quants = min_max.quantile([.25, .75])

    def get_sharpe(parametros):
        grupo['volumen'] = parametros[0]
        for i in range(len(grupo)):
            resultado, pip, pips = resultado_operacion(precios[i], grupo.operacion[i], 
													   parametros)
            grupo['resultado'].iloc[i] = resultado
            pips_n, capital = calculo_cap(resultado, pip, grupo['volumen'].iloc[i])
            grupo['pips'].iloc[i] = pips_n
            grupo['capital'].iloc[i] = capital
        cap_acm(grupo)
        sharp = f_sharpe(grupo)
        return sharp
    
    def neg_sharpe(x):
        return -get_sharpe(x)
    
    fun = lambda x: np.apply_along_axis(neg_sharpe, 1, x)
    
    # Limites
    x_max = np.array([100, abs(quants.iloc[1,0]), abs(quants.iloc[0,1])])
    x_min = np.array([1000, abs(quants.iloc[0,0]), abs(quants.iloc[1,1])])
    bounds = (x_min, x_max)
    
    # Opciones de PSO
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
    optimizer = ps.single.LocalBestPSO(n_particles=4, dimensions=3, options=options, 
									   bounds=bounds)
    
    # Utilizar pso para una busqueda optimizada
    best_sharpe, best_param = optimizer.optimize(fun, 15)
    
    return -best_sharpe, grupo, best_param


# -- -------------------------------------------------------------- FUNCION: Tabla de MAD -- #
# -- Tabla MAD
# -- ------------------------------------------------------------------------------------ -- #
def f_df_mad(df_operaciones, new_df_operaciones):
	"""
    Parameters
    ---------
    :param:
        df_operaciones
		new_df_operaciones
        
    Returns
    ---------
    :return: 
        df_mad
        
    Debuggin
    ---------
        df_operaciones
		new_df_operaciones
    """
	
	# Calculamos rendimientos logaritmicos para backtesting
	ren_log_bt = np.log(df_operaciones['capital_acm'] / 
						df_operaciones['capital_acm'].shift(1))
	
	# Optimización
	
	# Calculamos rendimientos logaritmicos para optimización
	ren_log_op = np.log(new_df_operaciones['capital_acm'] / 
						new_df_operaciones['capital_acm'].shift(1))
	
	# Sacamos promedio de esos rendimientos
	rp_bt = np.mean(ren_log_bt)
	rp_op = np.mean(ren_log_op)
	
	# Sacamos la desviación estándar de los rendimientos de Optimización y backtesting
	sdp_bt = ren_log_bt.std()
	sdp_op = ren_log_op.std()
	
	# Sacamos Sharpe Ratio de optimización y backtesting
	sharpe_bt = rp_bt / sdp_bt
	sharpe_op = rp_op / sdp_op
	
	# Definimos mar diaria
	mar = ent.mar/12
	
	# Sacamos los valores de los rendimientos por debajo de mar segun corresponda
	tdd_bt = ren_log_bt[ren_log_bt <= mar]
	tdd_op = ren_log_op[ren_log_op <= mar]
	
	#  Calculamos sortino ratio para backtesting
	sortino_bt = (rp_bt - mar) / (tdd_bt.std())
	
	# Calculamos sortino ratio para Optimización
	sortino_op = (rp_op - mar) / (tdd_op.std())
	
	# Como tercera métrica de atribución al desempeño usamos Information Ratio
	sp500 = dat.f_descarga_precios('SPX500_USD', df_operaciones["timestamp"][0], "M", 
							   len(df_operaciones))
	
	# Tomamos la columna Close
	sp500 = sp500["Close"]
	
	# Creamos rendimientos logaritmicos para el benchmark
	sp500 = np.log(sp500/sp500.shift(1))[1:]
	
	rendimientos_capital = np.log(df_operaciones["capital_acm"]/
								  df_operaciones["capital_acm"].shift(1))[1:]
	rendimientos_opt = np.log(new_df_operaciones ["capital_acm"]/
							  new_df_operaciones["capital_acm"].shift(1))[1:]
	
	tracking_error = 0
	
	# Sacamos tracking error y su desviación estándar para backtesting
	tracking_error = rendimientos_capital-sp500
	tracking_error = tracking_error.std()
	
	# Calculamos Information Ratio para backtest
	IR_backtest = (rendimientos_capital.mean()-sp500.mean())/tracking_error
	
	# Calculamos tracking error y su desviación estándar para optimización
	tracking_error = rendimientos_opt-sp500
	tracking_error = tracking_error.std()
	
	# Calculamos Information Ratio para optimización
	IR_opt = (rendimientos_opt.mean()-sp500.mean())/tracking_error
	
	# Creamos el DataFrame con las 3 MAD
	mad_data = {'Metrica': ['Sharpe_bt', 'Sharpe_op', 'Sortino_bt', 'Sortino_op', 'Info_R_bt',
	                        'Info_R_op'],
	            'Valor': [sharpe_bt, sharpe_op, sortino_bt, sortino_op, IR_backtest, IR_opt],
	            'Descripción': ['Sharpe Ratio de Backtest', 'Sharpe Ratio para Optimización',
	                            'Sortino Ratio para Backtest',
	                            'Sortino Ratio para Optimización', 
								'Information Ratio para Backtest',
	                            'Information Ratio para Optimización']}
	
	df_mad = pd.DataFrame(mad_data)
	return df_mad


