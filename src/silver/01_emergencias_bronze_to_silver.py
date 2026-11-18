from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import (
    col,
    trim,
    upper,
    regexp_replace,
    when,
    to_date,
    current_timestamp,
    lit
)

spark = (
    SparkSession.builder
    .appName("Ransa-Emergencias-Bronze-to-Silver")
    .getOrCreate()
)

INPUT = "gs://ransa-big-data-2026-lake/staging/emergencias/emergencias_utf8.csv"
OUTPUT = "gs://ransa-big-data-2026-lake/silver/emergencias/"

schema = StructType([
    StructField("ID_EMERGENCIA", StringType(), True),
    StructField("CODIGO_REPORTE", StringType(), True),
    StructField("FECHA_OCURRENCIA", StringType(), True),
    StructField("CLASE_FENOMENO", StringType(), True),
    StructField("FENOMENO", StringType(), True),
    StructField("EVENTO", StringType(), True),
    StructField("DEPARTEMENTO", StringType(), True),
    StructField("PROVINCIA", StringType(), True),
    StructField("DISTRITO", StringType(), True),
    StructField("AFECTACION", StringType(), True),
    StructField("RED_AFECTADA", StringType(), True),
    StructField("TRAMO_AFECADO", StringType(), True),
    StructField("SECTOR_AFECTADO", StringType(), True),
    StructField("HECHOS", StringType(), True),
    StructField("NOVEDADES", StringType(), True),
    StructField("FUENTE", StringType(), True),
    StructField("ESTADO_INICIAL", StringType(), True),
    StructField("FECHA_CIERRE", StringType(), True),
    StructField("ESTADO_ACTUAL", StringType(), True),
    StructField("LATITUD", StringType(), True),
    StructField("LONGITUD", StringType(), True),
    StructField("FECHA_CORTE", StringType(), True)
])

df = (
    spark.read
    .option("header", True)
    .option("sep", ";")
    .option("encoding", "UTF-8")
    .option("multiLine", True)
    .option("quote", '"')
    .option("mode", "PERMISSIVE")
    .schema(schema)
    .csv(INPUT)
)

total_bronze = df.count()

df_silver = (
    df

    # Corrección de nombres de columnas del archivo fuente
    .withColumnRenamed("DEPARTEMENTO", "DEPARTAMENTO")
    .withColumnRenamed("TRAMO_AFECADO", "TRAMO_AFECTADO")

    # Identificador
    .withColumn(
        "ID_EMERGENCIA",
        trim(col("ID_EMERGENCIA")).cast("long")
    )

    # Fechas
    .withColumn(
        "FECHA_OCURRENCIA",
        to_date(trim(col("FECHA_OCURRENCIA")), "dd/MM/yyyy")
    )
    .withColumn(
        "FECHA_CIERRE",
        when(
            trim(col("FECHA_CIERRE")) == "",
            None
        ).otherwise(
            to_date(trim(col("FECHA_CIERRE")), "dd/MM/yyyy")
        )
    )
    .withColumn(
        "FECHA_CORTE",
        to_date(trim(col("FECHA_CORTE")), "yyyyMMdd")
    )

    # Coordenadas
    .withColumn(
        "LATITUD",
        trim(col("LATITUD")).cast("double")
    )
    .withColumn(
        "LONGITUD",
        trim(col("LONGITUD")).cast("double")
    )

    # Variables categóricas
    .withColumn("DEPARTAMENTO", upper(trim(col("DEPARTAMENTO"))))
    .withColumn("PROVINCIA", upper(trim(col("PROVINCIA"))))
    .withColumn("DISTRITO", upper(trim(col("DISTRITO"))))
    .withColumn("ESTADO_INICIAL", upper(trim(col("ESTADO_INICIAL"))))
    .withColumn("ESTADO_ACTUAL", upper(trim(col("ESTADO_ACTUAL"))))

    # Normalización básica de la ruta
    .withColumn(
        "RED_AFECTADA_NORMALIZADA",
        regexp_replace(
            upper(trim(col("RED_AFECTADA"))),
            r"\s+",
            " "
        )
    )

    # Flags de calidad
    .withColumn(
        "FLAG_FECHA_CIERRE_FALTANTE",
        col("FECHA_CIERRE").isNull()
    )
    .withColumn(
        "FLAG_COORDENADA_INVALIDA",
        (
            col("LATITUD").isNull()
            | col("LONGITUD").isNull()
            | (col("LATITUD") < -90)
            | (col("LATITUD") > 90)
            | (col("LONGITUD") < -180)
            | (col("LONGITUD") > 180)
        )
    )
    .withColumn(
        "REGISTRO_VALIDO",
        (
            col("ID_EMERGENCIA").isNotNull()
            & col("FECHA_OCURRENCIA").isNotNull()
            & col("FECHA_CORTE").isNotNull()
            & (~col("FLAG_COORDENADA_INVALIDA"))
        )
    )

    # Trazabilidad
    .withColumn(
        "ORIGEN_DATOS",
        lit("MTC - Emergencias Red Vial Nacional 2026")
    )
    .withColumn(
        "FECHA_PROCESAMIENTO",
        current_timestamp()
    )
)

total_silver = df_silver.count()

fechas_cierre_faltantes = (
    df_silver
    .filter(col("FLAG_FECHA_CIERRE_FALTANTE"))
    .count()
)

coordenadas_invalidas = (
    df_silver
    .filter(col("FLAG_COORDENADA_INVALIDA"))
    .count()
)

registros_invalidos = (
    df_silver
    .filter(~col("REGISTRO_VALIDO"))
    .count()
)

print("==============================================")
print("EMERGENCIAS - BRONZE A SILVER")
print("==============================================")
print("REGISTROS BRONZE:", total_bronze)
print("REGISTROS SILVER:", total_silver)
print("FECHA_CIERRE FALTANTE:", fechas_cierre_faltantes)
print("COORDENADAS INVALIDAS:", coordenadas_invalidas)
print("REGISTROS INVALIDOS:", registros_invalidos)

print("\nESQUEMA SILVER:")
df_silver.printSchema()

print("\nMUESTRA SILVER:")
df_silver.select(
    "ID_EMERGENCIA",
    "FECHA_OCURRENCIA",
    "DEPARTAMENTO",
    "RED_AFECTADA",
    "RED_AFECTADA_NORMALIZADA",
    "ESTADO_ACTUAL",
    "LATITUD",
    "LONGITUD",
    "REGISTRO_VALIDO"
).show(10, truncate=False)

(
    df_silver.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUTPUT)
)

print("\nSALIDA SILVER:", OUTPUT)
print("PROCESO BRONZE -> SILVER FINALIZADO CORRECTAMENTE")

spark.stop()