import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True
)

parser.add_argument(
    "--output",
    required=True
)

args = parser.parse_args()

spark = (
    SparkSession.builder
    .appName("05_red_vial_geo_bronze_to_silver")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("orden", StringType(), True),
    StructField("codruta", StringType(), True),
    StructField("trayectori", StringType(), True),
    StructField("iddpto", StringType(), True),
    StructField("inicio", StringType(), True),
    StructField("fin", StringType(), True),
    StructField("longitud", StringType(), True),
    StructField("carriles", StringType(), True),
    StructField("superfic", StringType(), True),
    StructField("superficie", StringType(), True),
    StructField("estado", StringType(), True),
    StructField("GEOMETRIA_WKT", StringType(), True),
    StructField("LAT_INICIO", StringType(), True),
    StructField("LON_INICIO", StringType(), True),
    StructField("LAT_FIN", StringType(), True),
    StructField("LON_FIN", StringType(), True),
    StructField("NRO_PUNTOS", StringType(), True)
])

bronze = (
    spark.read
    .option("header", "true")
    .option("sep", ";")
    .option("quote", '"')
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .option("maxCharsPerColumn", "-1")
    .schema(schema)
    .csv(args.input)
)

silver = (
    bronze

    .withColumn(
        "ORDEN",
        F.col("orden").cast("long")
    )

    .withColumn(
        "CODIGO_RUTA",
        F.upper(
            F.trim(F.col("codruta"))
        )
    )

    .withColumn(
        "TRAYECTORIA",
        F.trim(F.col("trayectori"))
    )

    .withColumn(
        "ID_DEPARTAMENTO",
        F.lpad(
            F.trim(F.col("iddpto")),
            2,
            "0"
        )
    )

    .withColumn(
        "INICIO_KM",
        F.col("inicio").cast("double")
    )

    .withColumn(
        "FIN_KM",
        F.col("fin").cast("double")
    )

    .withColumn(
        "LONGITUD_KM",
        F.col("longitud").cast("double")
    )

    .withColumn(
        "NRO_CARRILES",
        F.when(
            F.trim(F.col("carriles")) == "",
            F.lit(None)
        ).otherwise(
            F.col("carriles").cast("int")
        )
    )

    .withColumn(
        "CODIGO_SUPERFICIE",
        F.col("superfic").cast("int")
    )

    .withColumn(
        "SUPERFICIE_ORIGINAL",
        F.trim(F.col("superficie"))
    )

    .withColumn(
        "SUPERFICIE_NORMALIZADA",
        F.upper(
            F.trim(F.col("superficie"))
        )
    )

    .withColumn(
        "ESTADO_ORIGINAL",
        F.trim(F.col("estado"))
    )

    .withColumn(
        "ESTADO_NORMALIZADO",
        F.upper(
            F.trim(F.col("estado"))
        )
    )

    .withColumn(
        "GEOMETRIA_WKT",
        F.col("GEOMETRIA_WKT")
    )

    .withColumn(
        "LAT_INICIO",
        F.col("LAT_INICIO").cast("double")
    )

    .withColumn(
        "LON_INICIO",
        F.col("LON_INICIO").cast("double")
    )

    .withColumn(
        "LAT_FIN",
        F.col("LAT_FIN").cast("double")
    )

    .withColumn(
        "LON_FIN",
        F.col("LON_FIN").cast("double")
    )

    .withColumn(
        "NRO_PUNTOS",
        F.col("NRO_PUNTOS").cast("long")
    )

    # La fuente original almacenaba:
    # X = latitud
    # Y = longitud
    #
    # El staging ya convirtió la geometría:
    # X = longitud
    # Y = latitud

    .withColumn(
        "FLAG_EJES_CORREGIDOS",
        F.lit(1)
    )

    .withColumn(
        "FLAG_CARRILES_NO_DISPONIBLE",
        F.when(
            F.col("NRO_CARRILES").isNull(),
            1
        ).otherwise(0)
    )

    .withColumn(
        "FLAG_GEOMETRIA_INVALIDA",
        F.when(
            F.col("GEOMETRIA_WKT").isNull()
            | (
                ~F.col("GEOMETRIA_WKT")
                .startswith("LINESTRING(")
            )
            | (
                F.col("NRO_PUNTOS") < 2
            ),
            1
        ).otherwise(0)
    )

    .withColumn(
        "FLAG_COORDENADAS_EXTREMOS_INVALIDAS",
        F.when(
            ~(
                F.col("LAT_INICIO").between(-20, 1)
                & F.col("LAT_FIN").between(-20, 1)
                & F.col("LON_INICIO").between(-82, -68)
                & F.col("LON_FIN").between(-82, -68)
            ),
            1
        ).otherwise(0)
    )

    .withColumn(
        "FECHA_CORTE",
        F.to_date(
            F.lit("2024-12-31")
        )
    )

    .withColumn(
        "REGISTRO_VALIDO",
        F.when(
            F.col("ORDEN").isNotNull()
            & F.col("CODIGO_RUTA").isNotNull()
            & (
                F.length(
                    F.col("CODIGO_RUTA")
                ) > 0
            )
            & F.col("LONGITUD_KM").isNotNull()
            & (
                F.col("LONGITUD_KM") > 0
            )
            & (
                F.col(
                    "FLAG_GEOMETRIA_INVALIDA"
                ) == 0
            )
            & (
                F.col(
                    "FLAG_COORDENADAS_EXTREMOS_INVALIDAS"
                ) == 0
            ),
            1
        ).otherwise(0)
    )

    .withColumn(
        "ORIGEN_ARCHIVO",
        F.lit(args.input)
    )

    .select(
        "ORDEN",
        "CODIGO_RUTA",
        "TRAYECTORIA",
        "ID_DEPARTAMENTO",
        "INICIO_KM",
        "FIN_KM",
        "LONGITUD_KM",
        "NRO_CARRILES",
        "CODIGO_SUPERFICIE",
        "SUPERFICIE_ORIGINAL",
        "SUPERFICIE_NORMALIZADA",
        "ESTADO_ORIGINAL",
        "ESTADO_NORMALIZADO",
        "GEOMETRIA_WKT",
        "LAT_INICIO",
        "LON_INICIO",
        "LAT_FIN",
        "LON_FIN",
        "NRO_PUNTOS",
        "FLAG_EJES_CORREGIDOS",
        "FLAG_CARRILES_NO_DISPONIBLE",
        "FLAG_GEOMETRIA_INVALIDA",
        "FLAG_COORDENADAS_EXTREMOS_INVALIDAS",
        "FECHA_CORTE",
        "REGISTRO_VALIDO",
        "ORIGEN_ARCHIVO"
    )
)

silver.cache()

total_bronze = bronze.count()
total_silver = silver.count()

carriles_no_disponibles = (
    silver
    .filter(
        F.col(
            "FLAG_CARRILES_NO_DISPONIBLE"
        ) == 1
    )
    .count()
)

geometrias_invalidas = (
    silver
    .filter(
        F.col(
            "FLAG_GEOMETRIA_INVALIDA"
        ) == 1
    )
    .count()
)

coordenadas_invalidas = (
    silver
    .filter(
        F.col(
            "FLAG_COORDENADAS_EXTREMOS_INVALIDAS"
        ) == 1
    )
    .count()
)

registros_invalidos = (
    silver
    .filter(
        F.col("REGISTRO_VALIDO") == 0
    )
    .count()
)

print("=== METRICAS SILVER RED VIAL GEO ===")

print(
    "BRONZE:",
    total_bronze
)

print(
    "SILVER:",
    total_silver
)

print(
    "CARRILES NO DISPONIBLES:",
    carriles_no_disponibles
)

print(
    "GEOMETRIAS INVALIDAS:",
    geometrias_invalidas
)

print(
    "COORDENADAS EXTREMOS INVALIDAS:",
    coordenadas_invalidas
)

print(
    "REGISTROS INVALIDOS:",
    registros_invalidos
)

print("\n=== ESQUEMA SILVER ===")

silver.printSchema()

print("\n=== MUESTRA ===")

(
    silver
    .select(
        "ORDEN",
        "CODIGO_RUTA",
        "LONGITUD_KM",
        "SUPERFICIE_NORMALIZADA",
        "ESTADO_NORMALIZADO",
        "NRO_PUNTOS",
        "LAT_INICIO",
        "LON_INICIO",
        "REGISTRO_VALIDO"
    )
    .show(
        10,
        truncate=False
    )
)

(
    silver.write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(args.output)
)

print("\nSILVER ESCRITO EN:")
print(args.output)

silver.unpersist()

spark.stop()