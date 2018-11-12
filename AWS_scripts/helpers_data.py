# encoding: utf8
from datetime import timedelta
from itertools import cycle

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical

import medias as m


def cachedCycle(gen):
    # Repetimos los datos del generador hasta el infinito, guardando en memoria el resultado
    results = []
    for r in gen:
        results.append(r)
        yield r
    # Cuando se acaban los datos, repetimos en loop
    yield from cycle(results)


def procesar_datos_para_conteo(datos, args):
    # Dividimos el fragmento en los cortes de los datos mayores que 5 minutos
    fragmentos = dividir_fragmentos(datos)
    # Dividimos por batches
    batches = generar_batches(fragmentos, stride=args.stride, batch_size=args.batch_size, length=args.length)
    return batches


def procesar_datos(datos, args):
    datos = (normalizar_ohlc_total(x,  # Normalizamos el dataset
                                   media_precio=m.medias[args.dataset]["precio"],
                                   media_ordenes=m.medias[args.dataset]["num_orders"],
                                   std_precio=m.stds[args.dataset]["precio"],
                                   std_ordenes=m.stds[args.dataset]["num_orders"]
                                   ) for x in datos)

    # Dividimos el fragmento en los cortes de los datos mayores que 5 minutos
    fragmentos = dividir_fragmentos(datos)
    # Rellenamos los huecos
    fragmentos = (rellenar_huecos(x, args.frequency) for x in fragmentos)
    # Desplazamos el horizonte temporal a predecir
    fragmentos = (aniadir_horizonte_temporal(x, args.forbidden_timesteps) for x in fragmentos)
    # Cambiamos la prediccion a los distintos modos: regresión, clasificación o delta
    fragmentos = (procesar_prediccion(x, mode=args.mode, min_delta=args.min_delta) for x in fragmentos)
    # Dividimos por batches
    batches = generar_batches(fragmentos, stride=args.stride, batch_size=args.batch_size, length=args.length)
    # Repetimos en loop los batches
    return cachedCycle(batches)


def dividir_fragmentos(df_generator, salto_maximo=300):
    for df in df_generator:
        # Buscamos las posiciones donde hay una separación de más de salto_máximo segundos
        posicion_divisiones = (df.loc[df.loc[:, "time"].diff() > timedelta(seconds=salto_maximo), "time"].index) + 1

        if len(posicion_divisiones) == 0:
            yield df
        else:
            # Dividimos en fragmentos consecutivos
            inicio = 0
            for posicion_final in posicion_divisiones:
                yield df.loc[inicio:posicion_final, :]
                inicio = posicion_final


def rellenar_huecos(df, frecuencia=15, variables_padding=["max", "min", "open", "close", "mean_price"]):
    time_min = df.loc[:, "time"].min()
    time_max = df.loc[:, "time"].max()
    df = df.set_index("time")
    df = df.reindex(pd.date_range(start=str(time_min), freq=f'{frecuencia}s', end=str(time_max)))
    for column in df.columns:
        if column in variables_padding:
            df.loc[:, column].fillna(method="pad", inplace=True)
        else:
            df.loc[:, column].fillna(value=0, inplace=True)
    return df


def normalizar_columna(c):
    return (c - np.mean(c)) / np.std(c)


def normalizar_ohlc_por_batches(df):  # En desuso
    media_precio = np.mean(df.loc[:, ["mean_price"]])[0]
    std_precio = np.std(df.loc[:, ["mean_price"]])[0]
    df.loc[:, ["max", "min", "open", "close", "mean_price"]] = \
        (df.loc[:, ["max", "min", "open", "close", "mean_price"]] - media_precio) / std_precio

    df.loc[:, "num_orders"] = normalizar_columna(df.loc[:, "num_orders"])

    return df


def normalizar_ohlc_total(df, media_precio, std_precio, media_ordenes, std_ordenes):
    df.loc[:, ["max", "min", "open", "close", "mean_price"]] = \
        (df.loc[:, ["max", "min", "open", "close", "mean_price"]] - media_precio) / std_precio

    df.loc[:, "num_orders"] = (df.loc[:, "num_orders"] - media_ordenes) / std_ordenes

    return df


def dividir_training_test(x_element):
    # Quitamos la predicción en la X y el dia y el tiempo
    X = x_element.loc[:, ~x_element.columns.str.startswith('pred')]
    X = X.loc[:, ~X.columns.isin(["day", "time"])]
    # Cogemos las predicciones para Y
    Y = x_element.loc[:, x_element.columns.str.startswith('pred')]

    # Lo convertimos en arrays
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def generar_batches(gen, stride=5, length=50, batch_size=300):
    for X, Y in (dividir_training_test(x) for x in gen):
        gen = TimeseriesGenerator(data=X, targets=Y, length=length, stride=stride,
                                  start_index=0, end_index=None,
                                  shuffle=False, reverse=False, batch_size=batch_size)
        for i in range(len(gen)):
            yield gen[i]


def aniadir_horizonte_temporal(x, forbidden, variable="mean_price"):
    # Nota: el +1 es porque keras ya adelanta una posición, asi que hay que mover uno menos de lo que queremos
    desplazado = x.loc[:, variable].shift(-forbidden + 1)
    x.loc[:, "predict"] = desplazado
    # Borramos los registros del final sin predicción.
    x = x.iloc[:-(forbidden + 1), :]
    return x


def procesar_prediccion(x, mode, min_delta):
    # modos: (regression, categorization, delta)
    if mode == "regression":
        pass  # No hay nada que hacer, ya se ha desplazado el precio
    elif mode == "classification":
        delta = x.loc[:, "predict"] - x.loc[:, "mean_price"]
        abs_delta = np.abs(delta)
        predict = np.where(abs_delta < min_delta, 0, np.sign(delta))
        # Guardamos en tres columnas cada categoria
        x.loc[:, "predict_side"] = (predict == 0) * 1
        x.loc[:, "predict_down"] = (predict == -1) * 1
        x.loc[:, "predict_up"] = (predict == 1) * 1
        # Borramos la prediccion original
        x = x.drop(["predict"], 1)
    elif mode == "delta":
        x.loc[:, "predict"] = x.loc[:, "predict"] - x.loc[:, "mean_price"]
    elif mode == "logreturn":
        x.loc[:, "predict"] = np.log(x.loc[:, "predict"] / x.loc[:, "close"])
    return (x)
