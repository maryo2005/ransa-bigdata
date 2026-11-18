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
    current_timestamp,
    lit,
    concat,
    lpad,
    to_date,
    sha2,
    concat_ws,
    coalesce,
    sum as spark_sum
)


spark = (
    SparkSession.builder
    .appName(
        "Ransa-Accidentes-Bronze-to-Silver"
    )
    .getOrCreate()
)


INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "staging/accidentes/accidentes_utf8.csv"
)

OUTPUT = (
    "gs://ransa-big-data-2026-lake/"
    "silver/accidentes/"
)


schema = StructType([
    StructField(
        "ANIO",
        StringType(),
        True
    ),
    StructField(
        "MES",
        StringType(),
        True
    ),
    StructField(
        "ENTIDAD_PRESTADORA",
        StringType(),
        True
    ),
    StructField(
        "CONCESION",
        StringType(),
        True
    ),
    StructField(
        "SIGLAS",
        StringType(),
        True
    ),
    StructField(
        "TIPO_ACCIDENTE",
        StringType(),
        True
    ),
    StructField(
        "CAUSA_ACCIDENTE",
        StringType(),
        True
    ),
    StructField(
        "TIPO_TRANSPORTE",
        StringType(),
        True
    ),
    StructField(
        "TIPO_VEHICULO",
        StringType(),
        True
    ),
    StructField(
        "CLIMA",
        StringType(),
        True
    ),
    StructField(
        "TIPO_VIA",
        StringType(),
        True
    ),
    StructField(
        "RUTA",
        StringType(),
        True
    ),
    StructField(
        "TRAMO",
        StringType(),
        True
    ),
    StructField(
        "CANTIDAD VEHINVOLUCRADOS",
        StringType(),
        True
    ),
    StructField(
        "CANTIDAD AFECTADOS",
        StringType(),
        True
    ),
    StructField(
        "CANTIDAD ACCIDENTES",
        StringType(),
        True
    )
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


# Hash de la fila original.
# Permite identificar duplicados sin eliminarlos.
columnas_originales = df.columns

df = df.withColumn(
    "FILA_HASH",
    sha2(
        concat_ws(
            "||",
            *[
                coalesce(
                    trim(col(c)),
                    lit("")
                )
                for c in columnas_originales
            ]
        ),
        256
    )
)


df_silver = (
    df

    # Columnas con espacios
    .withColumnRenamed(
        "CANTIDAD VEHINVOLUCRADOS",
        "CANTIDAD_VEHINVOLUCRADOS"
    )
    .withColumnRenamed(
        "CANTIDAD AFECTADOS",
        "CANTIDAD_AFECTADOS"
    )
    .withColumnRenamed(
        "CANTIDAD ACCIDENTES",
        "CANTIDAD_ACCIDENTES"
    )

    # Año y mes
    .withColumn(
        "ANIO",
        trim(col("ANIO")).cast("int")
    )
    .withColumn(
        "MES",
        trim(col("MES")).cast("int")
    )

    # Periodo mensual
    .withColumn(
        "PERIODO",
        to_date(
            concat(
                col("ANIO").cast("string"),
                lit("-"),
                lpad(
                    col("MES").cast("string"),
                    2,
                    "0"
                ),
                lit("-01")
            ),
            "yyyy-MM-dd"
        )
    )

    # Campos numéricos
    .withColumn(
        "CANTIDAD_VEHINVOLUCRADOS",
        trim(
            col("CANTIDAD_VEHINVOLUCRADOS")
        ).cast("int")
    )
    .withColumn(
        "CANTIDAD_AFECTADOS",
        trim(
            col("CANTIDAD_AFECTADOS")
        ).cast("int")
    )
    .withColumn(
        "CANTIDAD_ACCIDENTES",
        trim(
            col("CANTIDAD_ACCIDENTES")
        ).cast("int")
    )

    # Conservación de la ruta original
    .withColumn(
        "RUTA_ORIGINAL",
        trim(col("RUTA"))
    )

    # Normalización conservadora de ruta
    .withColumn(
        "RUTA_NORMALIZADA",
        when(
            upper(
                trim(col("RUTA"))
            ) == "SIN INFO",
            None
        ).otherwise(
            regexp_replace(
                upper(
                    trim(col("RUTA"))
                ),
                r"\s+",
                ""
            )
        )
    )

    # Normalización de tramo
    .withColumn(
        "TRAMO_NORMALIZADO",
        regexp_replace(
            upper(
                trim(col("TRAMO"))
            ),
            r"\s+",
            " "
        )
    )

    # Normalización del tipo de accidente
    .withColumn(
        "TIPO_ACCIDENTE_NORMALIZADO",
        regexp_replace(
            upper(
                trim(col("TIPO_ACCIDENTE"))
            ),
            r"\s+",
            " "
        )
    )

    # Normalización de causa
    .withColumn(
        "CAUSA_ACCIDENTE_NORMALIZADA",
        regexp_replace(
            upper(
                trim(col("CAUSA_ACCIDENTE"))
            ),
            r"\s+",
            " "
        )
    )

    # Normalización del tipo de vía
    .withColumn(
        "TIPO_VIA_NORMALIZADO",
        regexp_replace(
            upper(
                trim(col("TIPO_VIA"))
            ),
            r"\s+",
            " "
        )
    )

    # Normalización del clima
    .withColumn(
        "CLIMA_NORMALIZADO",
        regexp_replace(
            upper(
                trim(col("CLIMA"))
            ),
            r"\s+",
            " "
        )
    )

    # Ruta no disponible
    .withColumn(
        "FLAG_RUTA_NO_DISPONIBLE",
        col(
            "RUTA_NORMALIZADA"
        ).isNull()
    )

    # Transporte faltante
    .withColumn(
        "FLAG_TIPO_TRANSPORTE_FALTANTE",
        col(
            "TIPO_TRANSPORTE"
        ).isNull()
        | (
            trim(
                col("TIPO_TRANSPORTE")
            ) == ""
        )
    )

    # Vehículo faltante
    .withColumn(
        "FLAG_TIPO_VEHICULO_FALTANTE",
        col(
            "TIPO_VEHICULO"
        ).isNull()
        | (
            trim(
                col("TIPO_VEHICULO")
            ) == ""
        )
    )

    # Textos con posibles problemas
    # de codificación o calidad
    .withColumn(
        "FLAG_TEXTO_SOSPECHOSO",
        col(
            "CAUSA_ACCIDENTE_NORMALIZADA"
        ).rlike("¿|Ã|Â")
        | col(
            "CLIMA_NORMALIZADO"
        ).rlike("¿|Ã|Â")
        | upper(
            trim(
                col("TIPO_VEHICULO")
            )
        ).rlike("¿|Ã|Â")
    )

    # Validez estructural
    .withColumn(
        "REGISTRO_VALIDO",
        (
            col("ANIO").between(
                2019,
                2026
            )
            & col("MES").between(
                1,
                12
            )
            & col("TRAMO").isNotNull()
            & (
                trim(
                    col("TRAMO")
                ) != ""
            )
            & col(
                "CANTIDAD_VEHINVOLUCRADOS"
            ).isNotNull()
            & (
                col(
                    "CANTIDAD_VEHINVOLUCRADOS"
                ) >= 0
            )
            & col(
                "CANTIDAD_AFECTADOS"
            ).isNotNull()
            & (
                col(
                    "CANTIDAD_AFECTADOS"
                ) >= 0
            )
            & col(
                "CANTIDAD_ACCIDENTES"
            ).isNotNull()
            & (
                col(
                    "CANTIDAD_ACCIDENTES"
                ) > 0
            )
        )
    )

    # Trazabilidad
    .withColumn(
        "ORIGEN_DATOS",
        lit(
            "OSITRAN - Accidentes "
            "de Transito en Carreteras"
        )
    )
    .withColumn(
        "FECHA_PROCESAMIENTO",
        current_timestamp()
    )
)


total_silver = df_silver.count()


rutas_no_disponibles = (
    df_silver
    .filter(
        col("FLAG_RUTA_NO_DISPONIBLE")
    )
    .count()
)


transportes_faltantes = (
    df_silver
    .filter(
        col(
            "FLAG_TIPO_TRANSPORTE_FALTANTE"
        )
    )
    .count()
)


vehiculos_faltantes = (
    df_silver
    .filter(
        col(
            "FLAG_TIPO_VEHICULO_FALTANTE"
        )
    )
    .count()
)


textos_sospechosos = (
    df_silver
    .filter(
        col("FLAG_TEXTO_SOSPECHOSO")
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


# Los duplicados se contabilizan,
# pero NO se eliminan.
duplicados_exactos = (
    df_silver
    .groupBy("FILA_HASH")
    .count()
    .select(
        spark_sum(
            when(
                col("count") > 1,
                col("count") - 1
            ).otherwise(0)
        ).alias("DUPLICADOS")
    )
    .collect()[0]["DUPLICADOS"]
)


print(
    "=============================================="
)
print(
    "ACCIDENTES - BRONZE A SILVER"
)
print(
    "=============================================="
)

print(
    "REGISTROS BRONZE:",
    total_bronze
)

print(
    "REGISTROS SILVER:",
    total_silver
)

print(
    "DUPLICADOS EXACTOS CONSERVADOS:",
    duplicados_exactos
)

print(
    "RUTAS NO DISPONIBLES:",
    rutas_no_disponibles
)

print(
    "TIPO_TRANSPORTE FALTANTE:",
    transportes_faltantes
)

print(
    "TIPO_VEHICULO FALTANTE:",
    vehiculos_faltantes
)

print(
    "TEXTOS SOSPECHOSOS:",
    textos_sospechosos
)

print(
    "REGISTROS INVALIDOS:",
    registros_invalidos
)


print("\nESQUEMA SILVER:")

df_silver.printSchema()


print("\nMUESTRA SILVER:")

df_silver.select(
    "ANIO",
    "MES",
    "PERIODO",
    "TIPO_ACCIDENTE",
    "RUTA_ORIGINAL",
    "RUTA_NORMALIZADA",
    "TRAMO",
    "CANTIDAD_ACCIDENTES",
    "FLAG_RUTA_NO_DISPONIBLE",
    "REGISTRO_VALIDO"
).show(
    10,
    truncate=False
)


(
    df_silver.write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(OUTPUT)
)


print(
    "\nSALIDA SILVER:",
    OUTPUT
)

print(
    "PROCESO ACCIDENTES BRONZE -> SILVER "
    "FINALIZADO CORRECTAMENTE"
)


spark.stop()