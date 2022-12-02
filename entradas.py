
# Clasificaciones y condiciones

escenarios = ['A', 'B', 'C', 'D']

# Condiciones de cada escenario

# Actual >= Consensus >= Previous
cond_1 = "df_indice['Actual'][i] >= df_indice['Consensus'][i] >= df_indice['Previous'][i]"
# Actual >= Consensus < Previous
cond_2 = "df_indice['Actual'][i] >= df_indice['Consensus'][i] < df_indice['Previous'][i]"
# Actual < Consensus >= Previous
cond_3 = "df_indice['Actual'][i] < df_indice['Consensus'][i] >= df_indice['Previous'][i]"
# Actual < Consensus < Previous
cond_4 = "df_indice['Actual'][i] < df_indice['Consensus'][i] < df_indice['Previous'][i]"

condiciones = [cond_1, cond_2, cond_3, cond_4]
valores = [1, 2, 3, 4]


# -- ------------------------------------------------------------------------------------ -- #

# Parametros Iniciales de la cuenta

cap_inicial = 100000
riesgo_max = 1000
apalanca = 100

# Tasa libre de riesgo anual
rf = 0.08

# Propuesta de Parametros para cada escenario

# -- 'A' --
# Sentido [-1, 1]
tipo_a = 'buy'
# Dinero puesto en la operacion
volumen_a = 500
# Precio para tomar ganancia 
takeprof_a = 13
# Preio para tomar perdida
stoplos_a = 10

# -- 'B' --
# Sentido [-1, 1]
tipo_b = 'buy'
# Dinero puesto en la operacion
volumen_b = 700
# Precio para tomar ganancia
takeprof_b = 17
# Preio para tomar perdida
stoplos_b = 15

# -- 'C' --
# Sentido [-1, 1]
tipo_c = 'sell'
# Dinero puesto en la operacion
volumen_c = 600
# Precio para tomar ganancia
takeprof_c = 15
# Preio para tomar perdida
stoplos_c = 11

# -- 'D' --
# Sentido [-1, 1]
tipo_d = 'sell'
# Dinero puesto en la operacion
volumen_d = 300
# Precio para tomar ganancia
takeprof_d = 13
# Preio para tomar perdida
stoplos_d = 10

# Vectores de todos
reglas = {
            'tipos':
                [tipo_a, tipo_b, tipo_c, tipo_d],
            'volumenes':
                [volumen_a, volumen_b, volumen_c, volumen_d],
            'takeprofs':
                [takeprof_a, takeprof_b, takeprof_c, takeprof_d],
            'stoploss':
                [stoplos_a, stoplos_b, stoplos_c, stoplos_d]
            }
	
# -- ------------------------------------------------------------------------------------ -- #
	
# Datos para MAD

mar = 0.3






