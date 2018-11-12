# Inicialice app
import pyspark.sql.functions as F
from pyspark.sql import *

# os.chdir("/home/ubuntu")

spark = SparkSession.builder.appName("prueba").master("local[*]").getOrCreate()
sc = spark.sparkContext

# Creo la funcion para parsear las fechas
import datetime
import time


def formatoFecha(elemento):
    if elemento.endswith('Z') and elemento[10] == "T" and elemento[19] == ".":
        output = elemento[0:10] + " " + elemento[11:19] + elemento[20:]
        output = output[:-1]

    elementoFormateado = time.strptime(output, "%Y-%m-%d %H:%M:%S%f")
    # return output
    return datetime.datetime(elementoFormateado[0], elementoFormateado[1], elementoFormateado[2], elementoFormateado[3],
                             elementoFormateado[4], elementoFormateado[5],
                             elementoFormateado[6])  # .replace(tzinfo=pytz.utc)


# Probamos que funciona bien
import pytz

a = "2018-06-25T10:42:56.366000Z"
b = formatoFecha(a).replace(tzinfo=pytz.utc)
b

# Cargamos los datos desde el fichero
#import os

#os.chdir("/data")
data_file = "data/gdax_data_0007.gz"
data = spark.read.json(data_file)  # .sample(False,0.2,1234)

# Cambiamos los tipos de los datos
data = data.withColumn("price", data.price.cast("float"))
data = data.withColumn("size", data.size.cast("float"))
data = data.withColumn("remaining_size", data.remaining_size.cast("float"))
# NO VA data.withColumn("depth_x", formatoFecha(data.time))


# Dividimos los diferentes pares

# dataBTC_USD=data.filter(data.product_id == "BTC-USD")
# dataBTC_EUR=data.filter(data.product_id == "BTC-EUR")
# dataBTC_GPB=data.filter(data.product_id == "BTC-GPB")

# dataETH_USD=data.filter(data.product_id == "ETH-USD")
# dataETH_EUR=data.filter(data.product_id == "ETH-EUR")

# dataLTC_USD=data.filter(data.product_id == "LTC-USD)
# dataLTC_EUR=data.filter(data.product_id == "LTC-EUR")

# dataLTC_BTC=data.filter(data.product_id == "LTC-BTC")
# dataETH_BTC=data.filter(data.product_id == "ETH-BTC")


dataParsedBTC_USD = data.filter(data.product_id == "BTC-USD")
dataParsedBTC_USD.persist()

########## 1. Añadimos Precio Máximo, Mínimo, Medio y Last ##########
dataBTC_USD_match = dataParsedBTC_USD.filter("type == 'match' ").select("time", "size", "price").groupBy("time") #AND size != 'None'
dataBTC_USD_matchReducedDEF = dataBTC_USD_match.agg(F.sum("size").alias("size"),
                                                    F.mean("price").alias("mean"),
                                                    F.count(F.lit(1)).alias("num_orders"),
                                                    F.max("price").alias("max"),
                                                    F.min("price").alias("min"),
                                                    F.last("price").alias("last")
                                                    )




DFdataBTC_USD_matchReducedDEF = dataBTC_USD_matchReducedDEF.toDF("time", "size", "price_mean", "num_orders", "price_max", "price_min", "price_last")

DFdataBTC_USD_matchReducedDEFSorted = DFdataBTC_USD_matchReducedDEF.sort(F.col("time").asc())
#a = DFdataBTC_USD_matchReducedDEFSorted.rdd.map(lambda x: (str(x[0]), x[1], x[2], x[3], x[4], x[5], x[6]))
#b = a.toDF()
#c = b.toPandas()
#c.to_csv("dataBTC_USD_0007.csv", index=False)

pdData = DFdataBTC_USD_matchReducedDEFSorted.toPandas()


############################ 2. Order book ############################
