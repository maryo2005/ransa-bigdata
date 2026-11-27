from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType
)
from pyspark.sql.functions import (
    col,
    trim,
    upper,
    regexp_replace,
    when,
    to_date,
    concat,
    lit,
    current_timestamp
)


spark = (
    SparkSession.builder
    .appName("Ransa-Flujo-Peajes-Bronze-to-Silver")
    .getOrCreate()
)


INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "staging/flujo_peajes/flujo_peajes_utf8.csv"
)

OUTPUT = (
    "gs://ransa-big-data-2026-lake/"
    "silver/flujo_peajes/"
)


columnas = [
    "ADMINIST",
    "CODIGO_PEAJE",
    "DEPARTAMENTO",
    "NOMBRE_PEAJE",
    "VEH_LIGEROS_TAR_DIF",
    "VEH_LIGEROS_AUTOMOVILES",
    "VEH_PESADOS_TAR_DIF",
    "VEH_PESADOS_2E",
    "VEH_PESADOS_3E",
    "VEH_PESADOS_4E",
    "VEH_PESADOS_5E",
    "VEH_PESADOS_6E",
    "VEH_PESADOS_7E",
    "VEH_LIGEROS_TOTAL",
    "VEH_LIGEROS_IMD",
    "VEH_PESADOS_TOTAL",
    "VEH_PESADOS_IMD",
    "VEH_TOTAL",
    "VEH_IMD",
    "ANIO",
    "MES",
    "PERIODO",
    "EXTRA_23",
    "EXTRA_24",
    "VALOR_EXTRA_SIN_CABECERA"
]


schema = StructType([
    StructField(c, StringType(), True)
    for c in columnas
])


df = (
    spark.read
    .option("header", True)
    .option("sep", ";")
    .option("encoding", "UTF-8")
    .schema(schema)
    .csv(INPUT)
)


total_bronze = df.count()


campos_numericos = [
    "VEH_LIGEROS_TAR_DIF",
    "VEH_LIGEROS_AUTOMOVILES",
    "VEH_PESADOS_TAR_DIF",
    "VEH_PESADOS_2E",
    "VEH_PESADOS_3E",
    "VEH_PESADOS_4E",
    "VEH_PESADOS_5E",
    "VEH_PESADOS_6E",
    "VEH_PESADOS_7E",
    "VEH_LIGEROS_TOTAL",
    "VEH_LIGEROS_IMD",
    "VEH_PESADOS_TOTAL",
    "VEH_PESADOS_IMD",
    "VEH_TOTAL",
    "VEH_IMD"
]


df_silver = df


for campo in campos_numericos:
    df_silver = df_silver.withColumn(
        campo,
        regexp_replace(
            trim(col(campo)),
            r"\s+",
            ""
        ).cast("long")
    )


df_silver = (
    df_silver

    .withColumn(
        "ANIO",
        trim(col("ANIO")).cast("int")
    )

    .withColumn(
        "MES",
        trim(col("MES")).cast("int")
    )

    .withColumn(
        "PERIODO_ORIGINAL",
        trim(col("PERIODO"))
    )

    .withColumn(
        "PERIODO_FECHA",
        to_date(
            concat(
                trim(col("PERIODO")),
                lit("01")
            ),
            "yyyyMMdd"
        )
    )

    .withColumn(
        "ADMINIST_NORMALIZADO",
        upper(trim(col("ADMINIST")))
    )

    .withColumn(
        "DEPARTAMENTO_NORMALIZADO",
        upper(trim(col("DEPARTAMENTO")))
    )

    .withColumn(
        "NOMBRE_PEAJE_NORMALIZADO",
        upper(trim(col("NOMBRE_PEAJE")))
    )

    .withColumn(
        "FLAG_VALOR_EXTRA_SIN_CABECERA",
        col("VALOR_EXTRA_SIN_CABECERA").isNotNull()
        & (
            trim(col("VALOR_EXTRA_SIN_CABECERA"))
            != ""
        )
    )

    .withColumn(
        "FLAG_TOTAL_LIGEROS_INCONSISTENTE",
        col("VEH_LIGEROS_TOTAL")
        != (
            col("VEH_LIGEROS_TAR_DIF")
            + col("VEH_LIGEROS_AUTOMOVILES")
        )
    )

    .withColumn(
        "FLAG_TOTAL_PESADOS_INCONSISTENTE",
        col("VEH_PESADOS_TOTAL")
        != (
            col("VEH_PESADOS_TAR_DIF")
            + col("VEH_PESADOS_2E")
            + col("VEH_PESADOS_3E")
            + col("VEH_PESADOS_4E")
            + col("VEH_PESADOS_5E")
            + col("VEH_PESADOS_6E")
            + col("VEH_PESADOS_7E")
        )
    )

    .withColumn(
        "FLAG_TOTAL_VEHICULAR_INCONSISTENTE",
        col("VEH_TOTAL")
        != (
            col("VEH_LIGEROS_TOTAL")
            + col("VEH_PESADOS_TOTAL")
        )
    )

    .withColumn(
        "FLAG_PERIODO_INCONSISTENTE",
        trim(col("PERIODO"))
        != concat(
            col("ANIO").cast("string"),
            lpad(
                col("MES").cast("string"),
                2,
                "0"
            )
        )
    )

    .withColumn(
        "REGISTRO_VALIDO",
        (
            col("CODIGO_PEAJE").isNotNull()
            & (trim(col("CODIGO_PEAJE")) != "")
            & col("PERIODO_FECHA").isNotNull()
            & col("VEH_TOTAL").isNotNull()
            & (col("VEH_TOTAL") >= 0)
        )
    )

    .withColumn(
        "ORIGEN_DATOS",
        lit(
            "MTC - Flujo vehicular registrado "
            "en peajes 2014-2026"
        )
    )

    .withColumn(
        "FECHA_PROCESAMIENTO",
        current_timestamp()
    )
)


total_silver = df_silver.count()


extras = (
    df_silver
    .filter(
        col("FLAG_VALOR_EXTRA_SIN_CABECERA")
    )
    .count()
)


ligeros_inconsistentes = (
    df_silver
    .filter(
        col("FLAG_TOTAL_LIGEROS_INCONSISTENTE")
    )
    .count()
)


pesados_inconsistentes = (
    df_silver
    .filter(
        col("FLAG_TOTAL_PESADOS_INCONSISTENTE")
    )
    .count()
)


totales_inconsistentes = (
    df_silver
    .filter(
        col("FLAG_TOTAL_VEHICULAR_INCONSISTENTE")
    )
    .count()
)


periodos_inconsistentes = (
    df_silver
    .filter(
        col("FLAG_PERIODO_INCONSISTENTE")
    )
    .count()
)


registros_invalidos = (
    df_silver
    .filter(
        ~col("REGISTRO_VALIDO")
    )
    .count()
)


print("==============================================")
print("FLUJO PEAJES - BRONZE A SILVER")
print("==============================================")

print("REGISTROS BRONZE:", total_bronze)
print("REGISTROS SILVER:", total_silver)
print("VALORES EXTRA SIN CABECERA:", extras)

print(
    "TOTALES LIGEROS INCONSISTENTES:",
    ligeros_inconsistentes
)

print(
    "TOTALES PESADOS INCONSISTENTES:",
    pesados_inconsistentes
)

print(
    "TOTALES VEHICULARES INCONSISTENTES:",
    totales_inconsistentes
)

print(
    "PERIODOS INCONSISTENTES:",
    periodos_inconsistentes
)

print(
    "REGISTROS INVALIDOS:",
    registros_invalidos
)


print("\nESQUEMA SILVER:")
df_silver.printSchema()


print("\nMUESTRA SILVER:")

df_silver.select(
    "CODIGO_PEAJE",
    "NOMBRE_PEAJE",
    "DEPARTAMENTO",
    "ANIO",
    "MES",
    "PERIODO_FECHA",
    "VEH_LIGEROS_TOTAL",
    "VEH_PESADOS_TOTAL",
    "VEH_TOTAL",
    "VEH_IMD",
    "FLAG_TOTAL_LIGEROS_INCONSISTENTE",
    "FLAG_VALOR_EXTRA_SIN_CABECERA",
    "REGISTRO_VALIDO"
).show(
    10,
    truncate=False
)


(
    df_silver.write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUTPUT)
)


print(
    "\nSALIDA SILVER:",
    OUTPUT
)

print(
    "PROCESO FLUJO PEAJES BRONZE -> SILVER "
    "FINALIZADO CORRECTAMENTE"
)


spark.stop()