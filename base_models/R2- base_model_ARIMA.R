setwd("C:/Users/Carlos/Desktop/Data Science/00 -TFM/9- Parsing & Modelling")
library(magrittr)
library(purrr)
library(dplyr)
library(tidyr)
library(lubridate)
library(tidyquant)  # Loads tidyverse, tidyquant, financial pkgs, xts/zoo
library(plotly)
library(forecast)
library(tseries)

forbbiden_hours = 5
# 1. PREPARACIÓN DE LOS DATOS Y DIFERENTES MAs

# Cargamos los datos
features <- c("Timestamp", "Size", "Price_mean", "Norders", "Price_min", "Price_max", "Price_close")
dataBTCUSD <- read.csv("dataBTC_USD_0007.csv") %>% as_tibble() %>% set_colnames(features) %>% mutate(Timestamp = as_datetime(ymd_hms(Timestamp))) %>% as_tibble() 

# Vector con todas las fechas entre la última fecha y la primera
expandedTimestamps <- seq(first(dataBTCUSD$Timestamp), last(dataBTCUSD$Timestamp), by="sec") %>% as_tibble() %>% set_colnames("Timestamp")

# Sacamos los NAs implicitos
dataBTCUSDexpand <- left_join(expandedTimestamps  , dataBTCUSD, by="Timestamp")

# Rellenamos los NAs según corresponda asignarle el último valor o un 0
naReplacement <- dataBTCUSDexpand %>% select(Size, Norders) %>% zoo() %>% imputeTS::na.replace(.,0) %>% as_tibble() 
naLocf <- dataBTCUSDexpand %>% select(Price_mean, Price_min, Price_max, Price_close) %>% zoo() %>% imputeTS::na.locf(.) %>% as_tibble() 
dataBTCUSD_Parsed <- bind_cols(dataBTCUSDexpand %>% select(Timestamp), naReplacement,naLocf)

ggplot(dataBTCUSD_Parsed, aes(Timestamp, Price_close)) + geom_line()

# CREAMOS EL ZOO
trainLength = round(nrow(dataBTCUSD_Parsed)*0.8)
trainSet = dataBTCUSD_Parsed[0:trainLength,]
testSet = dataBTCUSD_Parsed[(trainLength+1):nrow(dataBTCUSD_Parsed),]

count_ma = ts(na.omit(trainSet$Price_close),frequency = 300)
decomp = stl(count_ma, s.window="periodic")
deseasonal_cnt <- seasadj(decomp)
plot(decomp)
 
#estacionario. Necesitamos estacionaridad, con media y desviación constante en el tiempo
adf.test(count_ma, alternative = "stationary")
Acf(count_ma, main='')

Pacf(count_ma, main='')

count_d1 = diff(deseasonal_cnt, differences = 1)
plot(count_d1)
adf.test(count_d1, alternative = "stationary")

Acf(count_d1, main='ACF for Differenced Series') # en la hora hay un pico de autocorrelación
Pacf(count_d1, main='PACF for Differenced Series')


fit<-auto.arima(deseasonal_cnt, seasonal=FALSE)
tsdisplay(residuals(fit), lag.max=360, main='(1,1,1) Model Residuals')

fit2 = arima(deseasonal_cnt, order=c(1,0,0))
tsdisplay(residuals(fit2), lag.max=15, main='Seasonal Model Residuals')

fcast <- forecast(fit2, h=6)
plot(fcast)

# ¡CUANTO MENOR ES TRAINSTEP MAS TARDA!
predVector = NULL
trainStep = 5
for (d in seq(0,nrow(testSet),trainStep)){
    print(d/nrow(testSet))
    count_ma = ts(na.omit(c(trainSet$Price_close,testSet$Price_close[0:d])),frequency = 300)
    decomp = stl(count_ma, s.window="periodic")
    deseasonal_cnt <- seasadj(decomp)
    fit2 = auto.arima(deseasonal_cnt, seasonal=FALSE)
    fcast = forecast(fit2, h=forbbiden_hours+trainStep)$mean[(forbbiden_hours+1):(forbbiden_hours+trainStep)]
    # ultima iteracion
    if(nrow(testSet)-d < trainStep){
        dist=nrow(testSet)-d
        fcast = fcast[0:dist]
    }
    predVector = c(predVector, fcast)
    print(predVector %>% length())
    }

predVector %>% length()



###################################################################################
# 2. CALCULO DE ERRORES REGRESIÓN

results <- testSet %>% select(Timestamp,Price_close) %>% bind_cols(predVector %>% as_tibble() %>% set_names("predsArima"))
graph <- results %>% gather(key="key", value="value", -Timestamp)

ggplotly(ggplot(graph, aes(x= Timestamp, y= value, col=key)) + 
             geom_line()
)

resultsErrors <- results %>% mutate(Error_Arima = abs(Price_close-predsArima))

AE <- resultsErrors %>% summarise_at( vars(contains("Error_"))  ,funs(sum(.)))
MAE <- AE/nrow(results) # Error_Arima 0.2567148

SE <- resultsErrors %>% summarise_at( vars(contains("Error_"))  ,funs(sum(.^2)))
MSE <- SE/nrow(results) # Error_Arima 0.1855787

###################################################################################
# 3. CALCULO DE ERRORES CLASIFICACION
testSetClasif <- c(0,sign(diff(testSet$Price_close))) %>% as_tibble() %>% set_names("Price_close")
resultsErrorsClasif <- c(0,sign(diff(resultsErrors$predsArima))) %>% as_tibble() %>% set_names("ErrorBin")
resultsClasifDef <- bind_cols(testSetClasif,resultsErrorsClasif)

library(caret)
confusionMatrix(resultsClasifDef$Price_close, resultsClasifDef$ErrorBin) # Acc 0.0849, no predice ningun 0 y hay muchos
#

