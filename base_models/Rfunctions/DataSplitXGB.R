DataSplitXGB <- function(data, trainPerc = 0.7, validPerc = 0.1, YlabelVar = "Ylabel"){
    
    trainLength = round(nrow(data)*trainPerc)
    validationLength = round(nrow(data)*validPerc) + trainLength
    trainSet = data[0:trainLength,]
    validationSet = data[trainLength:validationLength,]
    testSet = data[(validationLength+1):nrow(data),]
    
    trainSet_xgb <- xgb.DMatrix(data.matrix(trainSet %>% select(-Timestamp, -contains("Ylabel"))), label = data.matrix(trainSet %>% select(one_of(YlabelVar))))
    validationSet_xgb <- xgb.DMatrix(data.matrix(validationSet %>% select(-Timestamp,-contains("Ylabel"))), label = data.matrix(validationSet %>% select(one_of(YlabelVar))))
    testSet_xgb <- xgb.DMatrix(data.matrix(testSet %>% select(-Timestamp,-contains("Ylabel"))), label = data.matrix(testSet %>% select(one_of(YlabelVar))))
    
    output <- list(trainSet = trainSet,
                   validationSet = validationSet,
                   testSet = testSet,
                   trainXGB = trainSet_xgb,
                   validationXGB = validationSet_xgb,
                   testXGB = testSet_xgb)
    return(output)
    
}
