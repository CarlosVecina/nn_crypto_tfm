#setwd("C:/Users/Carlos/Desktop/Data Science/00 -TFM/9- Parsing & Modelling")
library(magrittr)
library(purrr)
library(dplyr)
library(tidyr)
library(lubridate)
library(tidyquant)  
library(plotly)
library(xgboost)
source("Rfunctions/LoadFormatData.R")   # Recibe un archivo y lo formatea a Ts,Yreg,Yclas, X explandiendolo en fechas y MA si se le indica.
source("Rfunctions/DataSplitXGB.R")     # Recibe la matriz con X e Y y la convierte en Train, Test, Validation XGB
source("Rfunctions/ErrorRegression.R")  # Calcula el error en un aproax de regresión, MAE y MSE
source("Rfunctions/ErrorClassification.R")  # Calcula el error con un aproax clasif, con matriz de confusion.

# Params
set.seed(12345)
forbbiden_hours = 5
outputMA = list()

dataBTC_USD <- LoadFormatData(file = "dataBTC_USD_0007.csv",forbbiden_hours = forbbiden_hours, expand = T,mavg = T)
sets <- DataSplitXGB(dataBTC_USD,0.8,0.1)[1:3] # De la lista os quedaremos con los DF, no con los objetos xgb


###################################################################################
# 2. CALCULO DE ERRORES REGRESIÓN
# Creamos el data frame con la Y real en el momento t, y el valor predicho por la media movil

outputMA$regresion$MA2=ErrorRegresion(testSetDf = sets$testSet, testPredsVec = sets$testSet$Mean_2)
outputMA$regresion$MA5=ErrorRegresion(testSetDf = sets$testSet, testPredsVec = sets$testSet$Mean_5)
outputMA$regresion$MA10=ErrorRegresion(testSetDf = sets$testSet, testPredsVec = sets$testSet$Mean_10)
outputMA$regresion$MA2
outputMA$regresion$MA5
outputMA$regresion$MA10

###################################################################################
# 3. CALCULO DE ERRORES CLASIFICACION
ErrorClassification(testSetDf = sets$testSet, testPredsVec = sets$testSet$Mean_2)
ErrorClassification(testSetDf = sets$testSet, testPredsVec = sets$testSet$Mean_5)
ErrorClassification(testSetDf = sets$testSet, testPredsVec = sets$testSet$Mean_10)


outputMA


