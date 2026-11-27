import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


parser = argparse.ArgumentParser()

parser.add_argument("--input-22-24", required=True)
parser.add_argument("--input-25", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()


spark = (
    SparkSession.builder
    .appName("06_transporte_carga_bronze_to_silver")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


schema = StructType([
    StructField("ID", StringType(), True),
    StructField("RUC", StringType(), True),
    StructField("RAZON_SOCIAL", StringType(), True),
    StructField("FECHA_RESOLUCION", StringType(), True),
    StructField("VIGENCIA_HASTA", StringType(), True),
    StructField("PERMISO_OPER", StringType(), True),
    StructField("PLACA", StringType(), True),
    StructField("ANIO_FAB", StringType(), True),
    StructField("N_CHASIS", StringType(), True),
    StructField("N_MOTOR", StringType(), True),
    StructField("MARCA", StringType(), True),
    StructField("SERVICIO", StringType(), True),
    StructField("CLASE", StringType(), True),
    StructField("COMBUSTIBLE", StringType(), True),
    StructField("N_ASIENTOS", StringType(), True),
    StructField("N_LLANTAS", StringType(), True),
    StructField("N_EJES", StringType(), True),
    StructField("CARGA_UTIL", StringType(), True),
    StructField("P_SECO", StringType(), True),
    StructField("P_BRUTO", StringType(), True),
    StructField("LARGO", StringType(), True),
    StructField("ANCHO", StringType(), True),
    StructField("ALTO", StringType(), True),
    StructField("DEPARTAMENTO", StringType(), True),
    StructField("FECHA_CORTE", StringType(), True),
    StructField("FUENTE_PERIODO", StringType(), True)
])


bronze = (
    spark.read
    .option("header", "true")
    .option("sep", ";")
    .option("quote", '"')
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .schema(schema)
    .csv([
        args.input_22_24,
        args.input_25
    ])
)


def limpiar_numero(nombre):
    return F.regexp_replace(
        F.trim(F.col(nombre)),
        ",",
        "."
    )


def parse_double(nombre):
    limpio = limpiar_numero(nombre)

    return F.when(
        limpio.rlike(r"^-?[0-9]+([.][0-9]+)?$"),
        limpio.cast("double")
    )


def parse_int(nombre):
    limpio = F.trim(F.col(nombre))

    return F.when(
        limpio.rlike(r"^-?[0-9]+$"),
        limpio.cast("int")
    )


def parse_fecha(nombre):
    columna = F.trim(F.col(nombre))

    return F.coalesce(
        F.to_date(columna, "yyyyMMdd"),
        F.to_date(columna, "dd/MM/yyyy"),
        F.to_date(columna, "yyyy-MM-dd")
    )


def flag_numero_invalido(nombre):
    original = F.upper(F.trim(F.col(nombre)))
    convertido = parse_double(nombre)

    return F.when(
        (~original.isin("", "ND", "N/D"))
        & convertido.isNull(),
        1
    ).otherwise(0)


silver = (
    bronze

    .withColumn(
        "ID_ORIGEN",
        F.col("ID").cast("long")
    )

    .withColumn(
        "TRANSPORTISTA_HASH",
        F.when(
            F.length(F.trim(F.col("RUC"))) > 0,
            F.sha2(
                F.trim(F.col("RUC")),
                256
            )
        )
    )

    .withColumn(
        "VEHICULO_HASH",
        F.when(
            F.length(F.trim(F.col("PLACA"))) > 0,
            F.sha2(
                F.upper(F.trim(F.col("PLACA"))),
                256
            )
        )
    )

    .withColumn(
        "FECHA_RESOLUCION_ORIGINAL",
        F.trim(F.col("FECHA_RESOLUCION"))
    )

    .withColumn(
        "FECHA_RESOLUCION_DATE",
        parse_fecha("FECHA_RESOLUCION")
    )

    .withColumn(
        "VIGENCIA_HASTA_ORIGINAL",
        F.trim(F.col("VIGENCIA_HASTA"))
    )

    .withColumn(
        "VIGENCIA_HASTA_DATE",
        parse_fecha("VIGENCIA_HASTA")
    )

    .withColumn(
        "PERMISO_OPER",
        F.upper(F.trim(F.col("PERMISO_OPER")))
    )

    .withColumn(
        "ANIO_FAB",
        parse_int("ANIO_FAB")
    )

    .withColumn(
        "MARCA",
        F.upper(F.trim(F.col("MARCA")))
    )

    .withColumn(
        "SERVICIO",
        F.upper(F.trim(F.col("SERVICIO")))
    )

    .withColumn(
        "CLASE",
        F.upper(F.trim(F.col("CLASE")))
    )

    .withColumn(
        "COMBUSTIBLE",
        F.upper(F.trim(F.col("COMBUSTIBLE")))
    )

    .withColumn(
        "N_ASIENTOS",
        parse_int("N_ASIENTOS")
    )

    .withColumn(
        "N_LLANTAS",
        parse_int("N_LLANTAS")
    )

    .withColumn(
        "N_EJES",
        parse_int("N_EJES")
    )

    .withColumn(
        "CARGA_UTIL_ORIGINAL",
        F.trim(F.col("CARGA_UTIL"))
    )

    .withColumn(
        "CARGA_UTIL",
        parse_double("CARGA_UTIL")
    )

    .withColumn(
        "P_SECO",
        parse_double("P_SECO")
    )

    .withColumn(
        "P_BRUTO_ORIGINAL",
        F.trim(F.col("P_BRUTO"))
    )

    .withColumn(
        "P_BRUTO",
        parse_double("P_BRUTO")
    )

    .withColumn(
        "LARGO",
        parse_double("LARGO")
    )

    .withColumn(
        "ANCHO",
        parse_double("ANCHO")
    )

    .withColumn(
        "ALTO",
        parse_double("ALTO")
    )

    .withColumn(
        "DEPARTAMENTO",
        F.upper(F.trim(F.col("DEPARTAMENTO")))
    )

    .withColumn(
        "FECHA_CORTE_DATE",
        parse_fecha("FECHA_CORTE")
    )

    .withColumn(
        "ANIO_CORTE",
        F.year(F.col("FECHA_CORTE_DATE"))
    )

    .withColumn(
        "FLAG_N_ASIENTOS_NO_DISPONIBLE",
        F.when(
            F.col("N_ASIENTOS").isNull(),
            1
        ).otherwise(0)
    )

    .withColumn(
        "FLAG_N_LLANTAS_NO_DISPONIBLE",
        F.when(
            F.col("N_LLANTAS").isNull(),
            1
        ).otherwise(0)
    )

    .withColumn(
        "FLAG_VIGENCIA_NO_FECHA",
        F.when(
            F.col("VIGENCIA_HASTA_DATE").isNull(),
            1
        ).otherwise(0)
    )

    .withColumn(
        "FLAG_CARGA_UTIL_INVALIDA_ORIGEN",
        flag_numero_invalido(
            "CARGA_UTIL_ORIGINAL"
        )
    )

    .withColumn(
        "FLAG_P_BRUTO_INVALIDO_ORIGEN",
        flag_numero_invalido(
            "P_BRUTO_ORIGINAL"
        )
    )

    .withColumn(
        "FLAG_IDENTIFICADORES_PSEUDONIMIZADOS",
        F.lit(1)
    )

    .withColumn(
        "REGISTRO_VALIDO",
        F.when(
            F.col("ID_ORIGEN").isNotNull()
            & F.col("VEHICULO_HASH").isNotNull()
            & F.col("TRANSPORTISTA_HASH").isNotNull()
            & F.col("FECHA_RESOLUCION_DATE").isNotNull()
            & F.col("FECHA_CORTE_DATE").isNotNull()
            & F.col("ANIO_FAB").isNotNull()
            & F.col("N_EJES").isNotNull()
            & F.col("CARGA_UTIL").isNotNull()
            & F.col("P_BRUTO").isNotNull(),
            1
        ).otherwise(0)
    )

    .select(
        "ID_ORIGEN",
        "FUENTE_PERIODO",

        "TRANSPORTISTA_HASH",
        "VEHICULO_HASH",

        "FECHA_RESOLUCION_ORIGINAL",
        "FECHA_RESOLUCION_DATE",

        "VIGENCIA_HASTA_ORIGINAL",
        "VIGENCIA_HASTA_DATE",

        "PERMISO_OPER",
        "ANIO_FAB",
        "MARCA",
        "SERVICIO",
        "CLASE",
        "COMBUSTIBLE",

        "N_ASIENTOS",
        "N_LLANTAS",
        "N_EJES",

        "CARGA_UTIL",
        "P_SECO",
        "P_BRUTO",

        "LARGO",
        "ANCHO",
        "ALTO",

        "DEPARTAMENTO",

        "FECHA_CORTE_DATE",
        "ANIO_CORTE",

        "FLAG_N_ASIENTOS_NO_DISPONIBLE",
        "FLAG_N_LLANTAS_NO_DISPONIBLE",
        "FLAG_VIGENCIA_NO_FECHA",

        "FLAG_CARGA_UTIL_INVALIDA_ORIGEN",
        "FLAG_P_BRUTO_INVALIDO_ORIGEN",

        "FLAG_IDENTIFICADORES_PSEUDONIMIZADOS",

        "REGISTRO_VALIDO"
    )
)


silver.cache()

total_bronze = bronze.count()
total_silver = silver.count()

asientos_nd = (
    silver
    .filter(
        F.col(
            "FLAG_N_ASIENTOS_NO_DISPONIBLE"
        ) == 1
    )
    .count()
)

llantas_nd = (
    silver
    .filter(
        F.col(
            "FLAG_N_LLANTAS_NO_DISPONIBLE"
        ) == 1
    )
    .count()
)

vigencia_no_fecha = (
    silver
    .filter(
        F.col(
            "FLAG_VIGENCIA_NO_FECHA"
        ) == 1
    )
    .count()
)

carga_invalida = (
    silver
    .filter(
        F.col(
            "FLAG_CARGA_UTIL_INVALIDA_ORIGEN"
        ) == 1
    )
    .count()
)

pbruto_invalido = (
    silver
    .filter(
        F.col(
            "FLAG_P_BRUTO_INVALIDO_ORIGEN"
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


print("=== METRICAS SILVER TRANSPORTE CARGA ===")

print("BRONZE:", total_bronze)
print("SILVER:", total_silver)

print(
    "N_ASIENTOS NO DISPONIBLE:",
    asientos_nd
)

print(
    "N_LLANTAS NO DISPONIBLE:",
    llantas_nd
)

print(
    "VIGENCIA NO FECHA:",
    vigencia_no_fecha
)

print(
    "CARGA_UTIL INVALIDA ORIGEN:",
    carga_invalida
)

print(
    "P_BRUTO INVALIDO ORIGEN:",
    pbruto_invalido
)

print(
    "REGISTROS INVALIDOS:",
    registros_invalidos
)


print("\n=== DISTRIBUCION POR CORTE ===")

(
    silver
    .groupBy(
        "FECHA_CORTE_DATE",
        "FUENTE_PERIODO"
    )
    .count()
    .orderBy("FECHA_CORTE_DATE")
    .show(20, truncate=False)
)


print("\n=== ESQUEMA SILVER ===")
silver.printSchema()


print("\n=== MUESTRA NO SENSIBLE ===")

(
    silver
    .select(
        "FUENTE_PERIODO",
        "ANIO_FAB",
        "MARCA",
        "SERVICIO",
        "CLASE",
        "COMBUSTIBLE",
        "N_EJES",
        "CARGA_UTIL",
        "P_BRUTO",
        "DEPARTAMENTO",
        "FECHA_CORTE_DATE",
        "REGISTRO_VALIDO"
    )
    .show(10, truncate=False)
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