ErrorClassification <- function(testSetDf, testPredsVec, model="regresion"){
    if(model=="regresion"){
    
    testSetClasif <- testSetDf %>% select(Timestamp, YlabelClass)
    resultsClasifDef <- cbind(Price_closeY = testSetClasif$YlabelClass[-1], PredsClass = sign(diff(testPredsVec))+1) %>% 
        as_tibble()#%>% mutate_all(funs(as.factor(.))) %>% as_tibble()
    
    library(caret)
    print(confusionMatrix(resultsClasifDef$Price_closeY, resultsClasifDef$PredsClass)) # Accuracy : 0.7271 
    }
    
    
}

