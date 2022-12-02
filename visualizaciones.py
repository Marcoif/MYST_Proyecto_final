"TRABAJO REALIZADO POR MARCO ANTONIO OCHOA CARDENAS Y RICARDO CARDENAS RAMOS"

# - Librerias -

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import proceso as pro
import plotly.express as px
import scipy.stats as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


'''
_ __ __________________________________________________________________________________ __ _ #

  Analisis de series de tiempo
_ __ ____________________________________________________________________________________ __ #
  
'''

def g_SeriesDecomposition(serie_indicador):
    decomposition = seasonal_decompose(serie_indicador["Actual"], model="additive",freq=30)
    decomposition.plot()


def g_indicador(serie_indicador):
    fig = go.Figure(data=go.Scatter(x=serie_indicador["DateTime"],y = serie_indicador["Actual"],
									mode="lines+markers"))
    fig.update_layout(title="S&P Case-Shiller Home Prices Index")
                     # yaxis = dict(text="Valor reportado",rangeslider=dict(visible=False)),
                      #xaxis = dict(text="Fecha"))

    return fig


def g_velas(p0_de):

	"""
	:param p0_de: pd.DataFrame : precios OHLC(open- high-low-close) como datos de entrada
	:return: grafica final
    """
	p0_de.columns = [list(p0_de.columns)[i].lower() for i in range(0, len(p0_de.columns))]
	fig = go.Figure(data = [go.Candlestick(x = p0_de['TimeStamp'],
                                           open=p0_de['open'],
                                           high = p0_de['high'],
                                           low=p0_de['low'],
                                           close = p0_de['close'])
                            ])
	fig.update_layout(margin=go.layout.Margin(l=50,r=50,b=20,t=50,pad=0),
                      title=dict(x=0.5, y=1, text='Precios historicos OHLC'),
                      xaxis=dict(title_text='Hora del día',rangeslider=dict(visible=False)),
                      yaxis = dict(title_text = 'Precio del EurUsd'))


	fig.layout.autosize= False
	fig.layout.width = 840
	fig.layout.height=520
	fig.show()
	return fig


def g_PQ(data):
	plt.figure(1,figsize=(10,8))
	plot_acf(data["Actual"])
	plot_pacf(data["Actual"])
	plt.show()


def g_AtipicalData(data):

    fig = go.Figure()
    fig.add_trace(go.Box(y=data["Actual"]))
    fig.show()


def norm_plot(data):
	# Diagrama cuantil-cuantil
	qq_plot= st.probplot(data["Actual"],dist="norm",plot=plt) 
	
	plt.grid()
	plt.xlabel("Cuantiles teóricos distribución normal")
	plt.ylabel('Cuantiles reales')
	plt.show()
	return qq_plot

'''
_ __ __________________________________________________________________________________ __ _ #

  Analisis de clasificacion
_ __ ____________________________________________________________________________________ __ #
  
'''

# -- --------------------------------------------------- FUNCION: Histogramas de metricas -- #
# -- 4 subplots de las 4 metricas comtempladas de cada escenario
# -- ------------------------------------------------------------------------------------ -- #
def f_subplot_hist(di_metrics, grupo):
    """
    Parameters
    ---------
    :param:
        di_metrics: dict : metricas de funciones.f_agregar_metricas
        grupo: string : clasificaciones de los posibles escenarios

    Returns
    ---------
    :return:
        fig: plotly obj : Hitogramas de las 4 netricas

    Debuggin
    ---------
        di_ocurrencias = funciones.f_agregar_metricas(di_metrics)
        grupo = 'A'
    """
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Direccion", 
														"Volatilidad", 
														"Alcista", 
														"Bajista"))

    trace0 = go.Histogram(x=di_metrics[grupo]['direccion'])
    trace1 = go.Histogram(x=di_metrics[grupo]['volatilidad'])
    trace2 = go.Histogram(x=di_metrics[grupo]['alcista'])
    trace3 = go.Histogram(x=di_metrics[grupo]['bajista'])

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 2, 2)

    fig.update_layout(height=500, width=800,
                  title_text="Histogramas de Metricas para " + grupo)
    fig.show()
    return fig

# -- ----------------------------------------------------------- FUNCION: Plot de precios -- #
# -- Agregar las metricas calculadas con funcion f_metricas a dict
# -- ------------------------------------------------------------------------------------ -- #
def multi_plot_clasificacion(di_ocurrencias, grupo, precio, norm = True):
    """
    Parameters
    ---------
    :param:
        di_ocurrencias: dict : Ocurrencias de f_dict_ocurrencias
        grupo: string : clasificaciones de los posibles escenarios
        
    Returns
    ---------
    :return:
        fig: pyplot obj : Hitogramas de las 4 netricas

    Debuggin
    ---------
        di_ocurrencias = fn.f_dict_ocurrencias(dat.df_indice)
        grupo = 'A'
        precio = 'Close'

    """
    fig, ax = plt.subplots()
    for j in range(len(di_ocurrencias[grupo])):
        if norm:
            ax.plot(pro.serie_normalizar(di_ocurrencias[grupo][j]['precios'][precio]))
        else: 
            ax.plot(di_ocurrencias[grupo][j]['precios'][precio])
    
    plt.title('Precios ' + precio + ' de las ocurrencias del escenario ' + grupo)
    plt.show()
    return fig



def plot_graph(df, col):
	df[col].plot()
	plt.show()
	
def hist(df, col, title):
	fig = px.histogram(df, x=col, nbins=20)
	fig.show()
	
'''
# -- ------------------------------------------------------------------------------------ -- #           
# Checar visualmente

import matplotlib.pyplot as plt

df_operaciones['capital_acm'].plot()
plt.show()

df_operaciones['rends'].plot()
plt.show()

print('media = ', df_operaciones['rends'].mean(), ' | desv = ', df_operaciones['rends'].std())
print('sharpe = ', sharpe)


'''
