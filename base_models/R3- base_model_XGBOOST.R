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
plotXGBImportance = F
forbbiden_hours = 5

dataBTC_USD <- LoadFormatData(file = "dataBTC_USD_0007.csv",forbbiden_hours = forbbiden_hours, expand = T,mavg = T)



###################### XGBOOST REGRESION

sets <- DataSplitXGB(dataBTC_USD, trainPerc = 0.85,validPerc = 0.1)


params = list(eta = .05, max_depth = 4, gamma = 0, alpha = 0, lambda = 1,
              colsample_bytree = 1, min_child_weight = 5, subsample = .8)

watchlist <- list(train = sets$trainXGB,
                  validation = sets$validationXGB)

xgbModel <- xgb.train(data = sets$trainXGB ,
                      params = params,
                      verbose = 1,
                      seed = 1234,
                      print_every_n = 50,
                      nrounds = 10000,
                      watchlist = watchlist,
                      early_stopping_rounds = 15)

testPreds <- predict(xgbModel, sets$testXGB) 

if(plotXGBImportance){
imp_matrix <- xgb.importance(feature_names = colnames(testSet_xgb), model = xgbModel)
print(xgb.plot.importance(importance_matrix = imp_matrix))
}


ErrorRegresion(sets$testSet,testPreds,plotly=T)
ErrorClassification(sets$testSet,testPreds)


###################### XGBOOST CLASIFICACION
sets <- DataSplitXGB(dataBTC_USD, trainPerc = 0.85,validPerc = 0.1, YlabelVar = "YlabelClass")



params = list(eta = .05, max_depth = 4, gamma = 0, alpha = 0, lambda = 1,
              colsample_bytree = 1, min_child_weight = 5, subsample = .8, objective = "multi:softprob", num_class = 3)
watchlist <- list(train = sets$trainXGB,
                  validation = sets$validationXGB)
numberOfClasses <- length(unique(sets$trainSet$YlabelClass))
xgb_params <- list("objective" = "multi:softprob",
                   #"eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
#
## CrossValidation: http://appliedpredictivemodeling.com/blog/2014/11/27/vpuig01pqbklmi72b8lcl3ij5hj2qm
#
xgbModelClass <- xgb.train(params = xgb_params,
                      data = sets$trainXGB,
                      verbose = 1,
                      seed = 1234,
                      print_every_n = 50,
                      nrounds = 10000,
                      watchlist = watchlist,
                      early_stopping_rounds = 25)

test_pred <- predict(xgbModelClass, newdata = sets$testXGB)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
    t() %>%
    data.frame() %>%
    mutate(XGB_pred_class = max.col(., "last"),
           Price_closeClassLabel = sets$testSet$YlabelClass)


confusionMatrix(test_prediction$Price_closeClassLabel %>% na.omit(), test_prediction$XGB_pred_class[0:length(test_prediction$Price_closeClassLabel %>% na.omit())])
