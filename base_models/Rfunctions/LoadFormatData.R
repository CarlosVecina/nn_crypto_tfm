setwd("C:/Users/Carlos/Desktop/Data Science/00 -TFM/9- Parsing & Modelling")
library(magrittr)
library(purrr)
library(dplyr)
library(tidyr)
library(lubridate)
library(tidyquant)  # Loads tidyverse, tidyquant, financial pkgs, xts/zoo
library(plotly)
library(xgboost)

# Hay que tener en cuenta que el Ylabel lo desplaza automáticamente 1 ts, por lo que las forbbiden se añadiran a ese 1.
LoadFormatData <- function(file="dataBTC_USD_0007.csv",forbbiden_hours=5, expand = T, mavg = T){
    
    features <- c("Timestamp", "Size", "Price_mean", "Norders", "Price_min", "Price_max", "Price_close")
    dataCrypto <- read.csv(file) %>% as_tibble() %>% set_colnames(features) %>% mutate(Timestamp = as_datetime(ymd_hms(Timestamp))) %>% as_tibble() 

    Ylabel <- c(dataCrypto$Price_close[-(1:(1+forbbiden_hours))],rep(NA,(1+forbbiden_hours))) %>% as_tibble() %>% set_names("Ylabel")
    dataCrypto_Parsed <- bind_cols(dataCrypto,Ylabel) %>% mutate(YlabelClass = sign(Ylabel - Price_close)+1) %>%  # se sobreescribe si hay expand o MA
        select(Timestamp, Ylabel, YlabelClass, everything())
    
    if(expand){
        # Vector con todas las fechas entre la última fecha y la primera
        expandedTimesteps <- seq(first(dataCrypto$Timestamp), last(dataCrypto$Timestamp), by="sec") %>% as_tibble() %>% set_colnames("Timestamp")
        
        # Sacamos los NAs implicitos
        dataCryptoexpand <- left_join(expandedTimesteps  , dataCrypto, by="Timestamp")
        
        # Rellenamos los NAs según corresponda asignarle el último valor o un 0
        naReplacement <- dataCryptoexpand %>% select(Size, Norders) %>% zoo() %>% imputeTS::na.replace(.,0) %>% as_tibble() 
        naLocf <- dataCryptoexpand %>% select(Price_mean, Price_min, Price_max, Price_close) %>% zoo() %>% imputeTS::na.locf(.) %>% as_tibble() 
        
        
        Ylabel <- c(naLocf$Price_close[-(1:(1+forbbiden_hours))],rep(NA,(1+forbbiden_hours))) %>% as_tibble() %>% set_names("Ylabel")
        dataCrypto_Parsed <- bind_cols(dataCryptoexpand %>% select(Timestamp), naReplacement,naLocf,Ylabel) %>% mutate(YlabelClass = sign(Ylabel - Price_close)+1) %>%  # se sobreescribe si hay expand o MA
            select(Timestamp, Ylabel, YlabelClass, everything())
    }
    
    if(mavg){
        # Creamos las MA2, MA5 y M10
        
        tidyverse_rollmean <- dataCrypto_Parsed %>%
            tq_mutate(
                # tq_mutate args
                select     = Price_close,
                mutate_fun = rollapply, 
                # rollapply args
                width      = 2,
                align      = "right",
                FUN        = mean,
                # mean args
                na.rm      = TRUE,
                # tq_mutate args
                col_rename = "Mean_2"
            ) %>%
            tq_mutate(
                select     = Price_close,
                mutate_fun = rollapply,
                width      = 5,
                align      = "right",
                FUN        = mean,
                na.rm      = TRUE,
                col_rename = "Mean_5"
            ) %>%
            tq_mutate(
                select     = Price_close,
                mutate_fun = rollapply,
                width      = 10,
                align      = "right",
                FUN        = mean,
                na.rm      = TRUE,
                col_rename = "Mean_10"
            )
        
        # Una vez tenemos la MA que corresponde a cada observación, dejaremos un GAP de 5 segundos donde no dispondremos 
        # de estos datos
        dataCrypto_Parsed <- tidyverse_rollmean %>% mutate_at(vars(contains("Mean_")), funs(lag(.,forbbiden_hours))) %>% 
            select(Timestamp, Ylabel, YlabelClass, everything())
        
        
    }
    
    
    return(dataCrypto_Parsed)
    
}


