ErrorRegresion <- function(testSetDf, testPredsVec, plotly = T){
    
    results <- testSetDf %>% select(Timestamp, Ylabel) %>% bind_cols(testPredsVec %>% as_tibble() %>% set_names("Preds"))
    graph <- results %>% gather(key="key", value="value", -Timestamp)
    
    if(plotly){
    plotly <- (ggplotly(ggplot(graph, aes(x= Timestamp, y= value, col=key)) + 
                 geom_line()
    ))
    }
    
    resultsErrors <- results %>% mutate(Error = abs(Ylabel-Preds)) %>% na.omit() #la ultima prediccion no existe en el testset
    
    AE <- resultsErrors %>% summarise_at( vars(contains("Error"))  ,funs(sum(.)))
    MAE <- AE/nrow(results) #  4.391711
    
    SE <- resultsErrors %>% summarise_at( vars(contains("Error"))  ,funs(sum(.^2)))
    MSE <- SE/nrow(results) # 63.42483
    
    output <- list(MAE = MAE,
                   MSE = MSE,
                   Plot = plotly
                   )
    return(output)
    
}
