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
    .appName("Ransa-Red-Vial-Bronze-to-Silver")
    .getOrCreate()
)

INPUT = "gs://ransa-big-data-2026-lake/staging/red_vial/red_vial_utf8.csv"
OUTPUT = "gs://ransa-big-data-2026-lake/silver/red_vial/"

schema = StructType([
    StructField("ID_RVN", StringType(), True),
    StructField("ID_DEPARTAMENTO", StringType(), True),
    StructField("DEPARTAMENTO", StringType(), True),
    StructField("CODIGO_RUTA", StringType(), True),
    StructField("TRAYECTORIA", StringType(), True),
    StructField("INICIO", StringType(), True),
    StructField("FINAL", StringType(), True),
    StructField("CLASIFICACION_EJE", StringType(), True),
    StructField("JERARQUIA", StringType(), True),
    StructField("LONGITUD", StringType(), True),
    StructField("NRO_CARRILES", StringType(), True),
    StructField("SUPERFICIE", StringType(), True),
    StructField("SUPERFICIE_L", StringType(), True),
    StructField("ESTADO", StringType(), True),
    StructField("ESTADO_L", StringType(), True),
    StructField("CODIGO_CONCESION", StringType(), True),
    StructField("NOMBRE_CONCESION", StringType(), True),
    StructField("CODIGO_LOGISTICO", StringType(), True),
    StructField("CORREDOR_LOGISTICO", StringType(), True),
    StructField("FECHA_CORTE", StringType(), True),
    StructField("ES_CONCES", StringType(), True)
])

df = (
    spark.read
    .option("header", True)
    .option("sep", ";")
    .option("encoding", "UTF-8")
    .option("quote", '"')
    .option("mode", "PERMISSIVE")
    .schema(schema)
    .csv(INPUT)
)

total_bronze = df.count()

df_silver = (
    df

    # Identificadores
    .withColumn(
        "ID_RVN",
        trim(col("ID_RVN")).cast("long")
    )
    .withColumn(
        "ID_DEPARTAMENTO",
        trim(col("ID_DEPARTAMENTO")).cast("int")
    )

    # Ubicación
    .withColumn(
        "DEPARTAMENTO",
        upper(trim(col("DEPARTAMENTO")))
    )

    # Normalización del código de ruta
    .withColumn(
        "CODIGO_RUTA",
        regexp_replace(
            upper(trim(col("CODIGO_RUTA"))),
            r"\s+",
            ""
        )
    )

    # Datos kilométricos
    .withColumn(
        "INICIO",
        trim(col("INICIO")).cast("double")
    )
    .withColumn(
        "FINAL",
        trim(col("FINAL")).cast("double")
    )
    .withColumn(
        "LONGITUD",
        trim(col("LONGITUD")).cast("double")
    )

    # Conservación del número de carriles original
    .withColumn(
        "NRO_CARRILES_ORIGINAL",
        trim(col("NRO_CARRILES"))
    )

    # Vacío y -1 representan información no disponible
    .withColumn(
        "NRO_CARRILES",
        when(
            (trim(col("NRO_CARRILES")) == "")
            | (trim(col("NRO_CARRILES")) == "-1"),
            None
        ).otherwise(
            trim(col("NRO_CARRILES")).cast("int")
        )
    )

    # Normalización de superficie
    .withColumn(
        "SUPERFICIE_NORMALIZADA",
        when(
            upper(trim(col("SUPERFICIE_L"))).isin(
                "PAVIMENTADO",
                "PAVIMENTADA"
            ),
            "PAVIMENTADO"
        )
        .when(
            upper(trim(col("SUPERFICIE_L"))).isin(
                "AFIRMADO",
                "AFIRMADA"
            ),
            "AFIRMADO"
        )
        .when(
            upper(trim(col("SUPERFICIE_L"))).isin(
                "PROYECTADO",
                "PROYECTADA"
            ),
            "PROYECTADO"
        )
        .when(
            upper(trim(col("SUPERFICIE_L")))
            == "ASFALTADO ECONÓMICO",
            "ASFALTADO ECONOMICO"
        )
        .when(
            upper(trim(col("SUPERFICIE_L")))
            == "SIN AFIRMAR",
            "SIN AFIRMAR"
        )
        .when(
            upper(trim(col("SUPERFICIE_L")))
            == "TROCHA",
            "TROCHA"
        )
        .otherwise(
            upper(trim(col("SUPERFICIE_L")))
        )
    )

    # Estado normalizado
    .withColumn(
        "ESTADO_NORMALIZADO",
        upper(trim(col("ESTADO_L")))
    )

    # Fecha de corte
    .withColumn(
        "FECHA_CORTE",
        to_date(
            trim(col("FECHA_CORTE")),
            "yyyyMMdd"
        )
    )

    # Indicador de concesión
    .withColumn(
        "ES_CONCES",
        trim(col("ES_CONCES")).cast("int")
    )

    # Flags de calidad
    .withColumn(
        "FLAG_CARRILES_NO_DISPONIBLE",
        col("NRO_CARRILES").isNull()
    )
    .withColumn(
        "FLAG_ESTADO_NO_DISPONIBLE",
        upper(trim(col("ESTADO_L")))
        == "INFORMACIÓN NO DISPONIBLE"
    )
    .withColumn(
        "FLAG_LONGITUD_INVALIDA",
        col("LONGITUD").isNull()
        | (col("LONGITUD") <= 0)
    )

    # Registro válido
    .withColumn(
        "REGISTRO_VALIDO",
        (
            col("ID_RVN").isNotNull()
            & col("CODIGO_RUTA").isNotNull()
            & (trim(col("CODIGO_RUTA")) != "")
            & col("FECHA_CORTE").isNotNull()
            & (~col("FLAG_LONGITUD_INVALIDA"))
        )
    )

    # Trazabilidad
    .withColumn(
        "ORIGEN_DATOS",
        lit("MTC - Red Vial Nacional 2022-2025")
    )
    .withColumn(
        "FECHA_PROCESAMIENTO",
        current_timestamp()
    )
)

total_silver = df_silver.count()

carriles_no_disponibles = (
    df_silver
    .filter(col("FLAG_CARRILES_NO_DISPONIBLE"))
    .count()
)

estado_no_disponible = (
    df_silver
    .filter(col("FLAG_ESTADO_NO_DISPONIBLE"))
    .count()
)

longitudes_invalidas = (
    df_silver
    .filter(col("FLAG_LONGITUD_INVALIDA"))
    .count()
)

registros_invalidos = (
    df_silver
    .filter(~col("REGISTRO_VALIDO"))
    .count()
)

print("==============================================")
print("RED VIAL - BRONZE A SILVER")
print("==============================================")
print("REGISTROS BRONZE:", total_bronze)
print("REGISTROS SILVER:", total_silver)
print("CARRILES NO DISPONIBLES:", carriles_no_disponibles)
print("ESTADO NO DISPONIBLE:", estado_no_disponible)
print("LONGITUDES INVALIDAS:", longitudes_invalidas)
print("REGISTROS INVALIDOS:", registros_invalidos)

print("\nESQUEMA SILVER:")
df_silver.printSchema()

print("\nMUESTRA SILVER:")
df_silver.select(
    "ID_RVN",
    "DEPARTAMENTO",
    "CODIGO_RUTA",
    "LONGITUD",
    "NRO_CARRILES",
    "NRO_CARRILES_ORIGINAL",
    "SUPERFICIE_L",
    "SUPERFICIE_NORMALIZADA",
    "ESTADO_L",
    "FECHA_CORTE",
    "REGISTRO_VALIDO"
).show(10, truncate=False)

(
    df_silver.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUTPUT)
)

print("\nSALIDA SILVER:", OUTPUT)
print(
    "PROCESO RED VIAL BRONZE -> SILVER "
    "FINALIZADO CORRECTAMENTE"
)

spark.stop()