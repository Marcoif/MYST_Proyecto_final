"TRABAJO REALIZADO POR MARCO ANTONIO OCHOA CARDENAS Y RICARDO CARDENAS RAMOS"

import visualizaciones as vs
import funciones as fn
import proceso as pro
import datos as dat
import entradas as ent
from time import time
import pandas as pd




# Indicador, cargar archivo que esta en datos
indicador = dat.df_indice

# Prueba de estacionariedad
ad_fuller = fn.check_stationarity(indicador)

# Primera diferencia de la serie original
indicador_d1 = pd.DataFrame() 
indicador_d1["DateTime"]=indicador["DateTime"][1:]
indicador_d1["Actual"]=indicador["Actual"].diff().dropna()

# Resultados de la pruba para la serie original
adFuller_d1 = fn.check_stationarity(indicador_d1)

# Grafia de la serie diferenciada
plot_d1 = vs.g_SeriesDecomposition(indicador_d1)

# Componentes de autocorrelación y autocorrelacion parcial
x = vs.g_PQ(indicador_d1)

# Prueba de heterocedasticidad
arch_test = fn.arch_test(indicador_d1)

# Prueba de normalidad
norm_test=fn.norm_test(indicador_d1)

# Detección de datos atipicos
datos_atipicos = fn.get_outliers(indicador_d1)



t0 = time()

# Clasificar las ocurrencias del indice
di_ocurrencias = fn.f_dict_ocurrencias(dat.df_indice_ent, ent.reglas)


# Agregar metricas y Back test
di_metrics = fn.f_agregar_metricas_bt(di_ocurrencias)

# Probabilidad que direccion sea positivo de acuerdo al historial
probabilidades = fn.f_prob_metrica(di_metrics)

# Back test en DataFrame
df_operaciones, sharpe = fn.f_df_operacion(di_ocurrencias)

# Cuales fueron ganadoras
ganadoras = df_operaciones[df_operaciones['resultado']=='ganadora']


# Separar por escenarios
grupo_A = df_operaciones[df_operaciones['escenario']=='A'].reset_index(drop=True)
grupo_B = df_operaciones[df_operaciones['escenario']=='B'].reset_index(drop=True)
grupo_C = df_operaciones[df_operaciones['escenario']=='C'].reset_index(drop=True)
grupo_D = df_operaciones[df_operaciones['escenario']=='D'].reset_index(drop=True)

# Reset cap_acumulado
pro.cap_acm(grupo_A)
pro.cap_acm(grupo_B)
pro.cap_acm(grupo_C)
pro.cap_acm(grupo_D)

# Sharpe de cada uno separado
sharpe_A = pro.f_sharpe(grupo_A)
sharpe_B = pro.f_sharpe(grupo_B)
sharpe_C = pro.f_sharpe(grupo_C)
sharpe_D = pro.f_sharpe(grupo_D)

# Pips de cada escenarios
pips_A = [di_ocurrencias['A'][i]['pips'] for i in range(len(di_ocurrencias['A']))]
pips_B = [di_ocurrencias['B'][i]['pips'] for i in range(len(di_ocurrencias['B']))]
pips_C = [di_ocurrencias['C'][i]['pips'] for i in range(len(di_ocurrencias['C']))]
pips_D = [di_ocurrencias['D'][i]['pips'] for i in range(len(di_ocurrencias['D']))]


# 30 precios de cada ocurrencia de cada escenario
precios_A = [di_ocurrencias['A'][i]['precios'] for i in range(len(di_ocurrencias['A']))]
precios_B = [di_ocurrencias['B'][i]['precios'] for i in range(len(di_ocurrencias['B']))]
precios_C = [di_ocurrencias['C'][i]['precios'] for i in range(len(di_ocurrencias['C']))]
precios_D = [di_ocurrencias['D'][i]['precios'] for i in range(len(di_ocurrencias['D']))]


# Tiempo
print(time() - t0)

 

a_best_sharpe_bnp, a_grupo_bnp, a_best_param_bnp = pro.no_optimized_search(grupo_A, pips_A, 
                                               precios_A, 
                                               [ent.volumen_a, ent.stoplos_a, ent.takeprof_a], 10)

b_best_sharpe_bnp, b_grupo_bnp, b_best_param_bnp = pro.no_optimized_search(grupo_B, pips_B, 
                                               precios_B, 
                                               [ent.volumen_b, ent.stoplos_b, ent.takeprof_b], 10)

c_best_sharpe_bnp, c_grupo_bnp, c_best_param_bnp = pro.no_optimized_search(grupo_C, pips_C, 
                                               precios_C, 
                                               [ent.volumen_c, ent.stoplos_c, ent.takeprof_c], 10)

d_best_sharpe_bnp, d_grupo_bnp, d_best_param_bnp = pro.no_optimized_search(grupo_D, pips_D, 
                                               precios_D, 
                                               [ent.volumen_d, ent.stoplos_d, ent.takeprof_d], 10)



# PSO
a_best_sharpe_pso, a_grupo_pso, a_best_param_pso = pro.optimized_search(grupo_A, pips_A, 
                                               precios_A)

b_best_sharpe_pso, b_grupo_pso, b_best_param_pso = pro.optimized_search(grupo_B, pips_B, 
                                               precios_B)

c_best_sharpe_pso, c_grupo_pso, c_best_param_pso = pro.optimized_search(grupo_C, pips_C, 
                                               precios_C)

d_best_sharpe_pso, d_grupo_pso, d_best_param_pso = pro.optimized_search(grupo_D, pips_D, 
                                               precios_D)
 

new_reglas = {
            'tipos':
                [ent.tipo_a, ent.tipo_b, ent.tipo_c, ent.tipo_d],
            'volumenes':
                [a_best_param_pso[0], b_best_param_pso[0], 
                 c_best_param_pso[0], d_best_param_pso[0]],
            'takeprofs':
                [a_best_param_pso[2], b_best_param_pso[2], 
                 c_best_param_pso[2], d_best_param_pso[2]],
            'stoploss':
                [a_best_param_pso[1], b_best_param_pso[1], 
                 c_best_param_pso[1], d_best_param_pso[1]]
            }

new_di_ocurrencias = fn.f_dict_ocurrencias(dat.df_indice, new_reglas)

# Agregar metricas y Back test
new_di_metrics = fn.f_agregar_metricas_bt(new_di_ocurrencias)

# Probabilidad que direccion sea positivo de acuerdo al historial
new_probabilidades = fn.f_prob_metrica(new_di_metrics)

# Back test en DataFrame
new_df_operaciones, new_sharpe = fn.f_df_operacion(new_di_ocurrencias)

print(time() - t0)




df_mad = pro.f_df_mad(df_operaciones, new_df_operaciones)

