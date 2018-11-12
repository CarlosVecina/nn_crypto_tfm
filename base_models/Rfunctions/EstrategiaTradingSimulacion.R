estrategiaTradingSimulacion <- function(resultados,modo= "regresion" ,cap_inicial=10000,
                                        porc_entrada=0.1,porc_salida=0.1, intervalo_tolerancia=0,
                                        fee_entrada=0, fee_salida=0, liquidar=T){
    # Espera Timestamp Price_Close Ylabel Preds
    if (modo == "regresion"){
        #Preparamos el Dataset
        resultadosDf <-  resultados %>% select(Timestamp,Price_close,Ylabel,Preds)
        resultadosDf <-  resultadosDf %>% mutate(YlabelClas = if_else((Ylabel-Price_close)>0,1,if_else((Ylabel-Price_close)==0,0,-1))) %>% 
            mutate(PredsClas=if_else(Preds-Price_close>0,1,if_else(Preds-Price_close==0,0,-1)))
        # Inicializamos variables
        no_invertido = cap_inicial
        coins_en_cartera = compra_instante = venta_instante = contador_compras = contador_ventas = 0
        
        for(i in (1:nrow(resultadosDf))){
            # Si hay senal de compra del Moodelo
            senal_suficiente = if_else(abs(resultadosDf$Preds[i]-resultadosDf$Price_close[i]) > intervalo_tolerancia,T,F)
            if(resultadosDf$PredsClas[i]>0 && no_invertido>0 && senal_suficiente){
                compra_instante <-  (porc_entrada*no_invertido)/resultadosDf$Price_close[i]
                coins_en_cartera <-  coins_en_cartera + compra_instante - compra_instante*fee_entrada
                no_invertido <-  no_invertido-porc_entrada*no_invertido
                contador_compras = contador_compras +1

            }
            # Si hay senal de venta del Modelo
            senal_suficiente = if_else(abs(resultadosDf$Preds[i]-resultadosDf$Price_close[i]) > intervalo_tolerancia,T,F)
            if(resultadosDf$PredsClas[i]<0 && coins_en_cartera>0 && senal_suficiente){
                venta_instante <-  coins_en_cartera*porc_salida*resultadosDf$Price_close[i]
                no_invertido <-  no_invertido + venta_instante - venta_instante*fee_salida
                coins_en_cartera <-  coins_en_cartera - coins_en_cartera*porc_salida
                contador_ventas <-  contador_ventas +1

            }
            #Liquido la posición al final del periodo
            if(i == nrow(resultadosDf) && liquidar){
                venta_instante <-  coins_en_cartera*1*resultadosDf$Price_close[i]
                no_invertido <-  no_invertido + venta_instante - venta_instante*fee_salida
                #print(venta_instante*fee_salida)
                contador_ventas <-  if_else(coins_en_cartera>0,contador_ventas +1,contador_ventas) 
                coins_en_cartera <-  coins_en_cartera - coins_en_cartera*1
                print(contador_ventas)
                print(no_invertido)
            }
        }
    }
    output <- list(capital_final=no_invertido,coins=coins_en_cartera,
                   ncompras = contador_compras, nventas = contador_ventas)
    return(output)
}