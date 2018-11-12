source("Rfunctions/EstrategiaTradingSimulacion.R")
model_MA = source("R1- base_model_MA.R")
resultados = model_MA$value$regresion$MA2$results

# Vemos las variaciones según diferentes porcentajes de entrada y salida
for(i in seq(0.1,1,0.1) ){
    estrategiaTradingSimulacion(resultados,modo = "regresion", cap_inicial = 10000, porc_entrada = i,porc_salida = i)
}

# Probamos lo mismo con la MA10 y vemos como coherentemente salen peores resultados
resultados = model_MA$value$regresion$MA10$results
for(i in seq(0.1,1,0.1) ){
    estrategiaTradingSimulacion(resultados,modo = "regresion", cap_inicial = 10000, porc_entrada = i,porc_salida = i)
}