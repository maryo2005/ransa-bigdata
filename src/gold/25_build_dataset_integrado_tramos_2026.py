from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("build_dataset_integrado_tramos_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

BASE = "gs://ransa-big-data-2026-lake/gold/"

MASTER = BASE + "master_tramos_2024/"
TARGET = BASE + "afectacion_tramo_2026/"
ACC = BASE + "accidentes_features_ruta_hist_2025/"
FLUJO = BASE + "flujo_features_ruta_hist_2025/"
CARGA = BASE + "carga_features_departamento_2025/"

OUT = BASE + "dataset_integrado_tramos_2026/"

m = spark.read.parquet(MASTER)
t = spark.read.parquet(TARGET)
a = spark.read.parquet(ACC)
f = spark.read.parquet(FLUJO)
c = spark.read.parquet(CARGA)

# =========================================================
# 1. Funciones de normalización
# =========================================================

def canonical_route(columna):

    limpio = F.regexp_replace(
        F.upper(F.trim(columna)),
        r"[^A-Z0-9]",
        ""
    )

    valido = limpio.rlike(
        r"^PE[0-9]+[A-Z]*$"
    )

    numero = F.regexp_extract(
        limpio,
        r"^PE([0-9]+)([A-Z]*)$",
        1
    )

    sufijo = F.regexp_extract(
        limpio,
        r"^PE([0-9]+)([A-Z]*)$",
        2
    )

    return F.when(
        valido,
        F.concat(
            F.lit("PE"),
            numero.cast("int").cast("string"),
            sufijo
        )
    )


def norm_dep(columna):

    return F.translate(
        F.upper(F.trim(columna)),
        "ÁÉÍÓÚÜÑ",
        "AEIOUUN"
    )


# =========================================================
# 2. Preparar Master
# =========================================================

master = (
    m
    .withColumn(
        "RUTA_JOIN",
        canonical_route(
            F.col("CODIGO_RUTA")
        )
    )
    .withColumn(
        "DEP_JOIN",
        norm_dep(
            F.col("DEPARTAMENTO")
        )
    )
)

# =========================================================
# 3. Target - SOLO LABEL
# Evitamos leakage de información de Emergencias 2026.
# =========================================================

target = (
    t
    .select(
        "ID_TRAMO",
        "AFECTACION_TRAMO"
    )
)

# =========================================================
# 4. Features Accidentes
# =========================================================

acc = (
    a
    .select(
        F.col("RUTA_FINAL")
        .alias("RUTA_JOIN_ACC"),

        "ACCIDENTES_HIST_2019_2025",
        "AFECTADOS_HIST_2019_2025",
        "VEHICULOS_INV_HIST_2019_2025",
        "ACCIDENTES_2025",
        "ACCIDENTES_2024_2025",
        "ANIOS_CON_ACCIDENTES",
        "MESES_CON_ACCIDENTES",
        "PROM_AFECTADOS_POR_ACCIDENTE",
        "PROM_VEHICULOS_POR_ACCIDENTE",
        "ACCIDENTES_POR_100KM_HIST",
        "ACCIDENTES_2025_POR_100KM",
        "ACCIDENTES_24M_POR_100KM"
    )
)

# =========================================================
# 5. Features Flujo vehicular
# =========================================================

flujo = (
    f
    .select(
        F.col("RUTA_FINAL")
        .alias("RUTA_JOIN_FLUJO"),

        "N_PEAJES_RUTA",
        "N_OBSERVACIONES_FLUJO_HIST",
        "MESES_OBSERVADOS_HIST",
        "VEH_TOTAL_OBSERVADO_2014_2025",
        "PROM_IMD_2014_2025",
        "PROM_IMD_PESADOS_2014_2025",
        "PROM_IMD_2024",
        "VEH_TOTAL_OBSERVADO_2025",
        "VEH_LIGEROS_OBSERVADO_2025",
        "VEH_PESADOS_OBSERVADO_2025",
        "PROM_VEH_MENSUAL_PEAJE_2025",
        "PROM_IMD_2025",
        "MAX_IMD_2025",
        "PROM_IMD_PESADOS_2025",
        "MESES_OBSERVADOS_2025",
        "N_FLAGS_LIGEROS_INCONSISTENTES",
        "PCT_PESADOS_2025",
        "VAR_IMD_2025_VS_2024_PCT"
    )
)

# =========================================================
# 6. Features Carga departamental
# =========================================================

carga = (
    c
    .select(
        F.col("DEP_N")
        .alias("DEP_JOIN_CARGA"),

        "FLOTA_CARGA_2025",
        "TRANSPORTISTAS_CARGA_2025",
        "MEDIANA_EJES_2025",
        "P95_EJES_2025",
        "MEDIANA_CARGA_UTIL_2025",
        "P95_CARGA_UTIL_2025",
        "MEDIANA_P_BRUTO_2025",
        "P95_P_BRUTO_2025",
        "VEHICULOS_POR_TRANSPORTISTA_2025",
        "LOG_FLOTA_CARGA_2025",
        "LOG_TRANSPORTISTAS_CARGA_2025",
        "PCT_CAMION_2025",
        "PCT_SEMIREMOLQUE_2025",
        "PCT_REMOLCADOR_2025",
        "PCT_PICKUP_2025",
        "PCT_DIESEL_2025"
    )
)

# =========================================================
# 7. Integración
# =========================================================

df = (
    master
    .join(
        target,
        "ID_TRAMO",
        "inner"
    )
    .join(
        acc,
        master.RUTA_JOIN
        ==
        acc.RUTA_JOIN_ACC,
        "left"
    )
    .join(
        flujo,
        master.RUTA_JOIN
        ==
        flujo.RUTA_JOIN_FLUJO,
        "left"
    )
    .join(
        carga,
        master.DEP_JOIN
        ==
        carga.DEP_JOIN_CARGA,
        "left"
    )
)

# =========================================================
# 8. Indicadores de disponibilidad
# =========================================================

df = (
    df
    .withColumn(
        "HAS_ACCIDENTES",
        F.when(
            F.col("RUTA_JOIN_ACC").isNotNull(),
            1
        ).otherwise(0)
    )
    .withColumn(
        "HAS_FLUJO",
        F.when(
            F.col("RUTA_JOIN_FLUJO").isNotNull(),
            1
        ).otherwise(0)
    )
    .withColumn(
        "HAS_CARGA",
        F.when(
            F.col("DEP_JOIN_CARGA").isNotNull(),
            1
        ).otherwise(0)
    )
    .drop(
        "RUTA_JOIN_ACC",
        "RUTA_JOIN_FLUJO",
        "DEP_JOIN_CARGA"
    )
    .withColumn(
        "FECHA_CORTE_PREDICTORES",
        F.to_date(
            F.lit("2025-12-31")
        )
    )
    .withColumn(
        "FECHA_INICIO_TARGET",
        F.to_date(
            F.lit("2026-01-01")
        )
    )
    .withColumn(
        "FECHA_FIN_TARGET",
        F.to_date(
            F.lit("2026-08-20")
        )
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_DATASET",
        F.current_timestamp()
    )
)

# =========================================================
# 9. Validaciones críticas
# =========================================================

total = df.count()
unicos = df.select("ID_TRAMO").distinct().count()

print("\n======================================")
print("DATASET INTEGRADO")
print("======================================")

print("FILAS:", total)
print("ID_TRAMO UNICOS:", unicos)
print("COLUMNAS:", len(df.columns))

if total != 7058:
    raise Exception(
        f"ERROR: se esperaban 7058 filas "
        f"y se obtuvieron {total}"
    )

if unicos != 7058:
    raise Exception(
        f"ERROR: se esperaban 7058 "
        f"ID_TRAMO únicos y hay {unicos}"
    )

# =========================================================
# 10. Target
# =========================================================

print("\n======================================")
print("DISTRIBUCION TARGET")
print("======================================")

(
    df
    .groupBy("AFECTACION_TRAMO")
    .count()
    .orderBy("AFECTACION_TRAMO")
    .show()
)

pos = (
    df
    .filter(
        F.col("AFECTACION_TRAMO") == 1
    )
    .count()
)

neg = (
    df
    .filter(
        F.col("AFECTACION_TRAMO") == 0
    )
    .count()
)

if pos != 726 or neg != 6332:
    raise Exception(
        f"ERROR target: positivos={pos}, "
        f"negativos={neg}"
    )

# =========================================================
# 11. Cobertura
# =========================================================

print("\n======================================")
print("COBERTURA DE FUENTES")
print("======================================")

cobertura = (
    df
    .agg(
        F.sum("HAS_ACCIDENTES")
        .alias("CON_ACCIDENTES"),

        F.sum("HAS_FLUJO")
        .alias("CON_FLUJO"),

        F.sum("HAS_CARGA")
        .alias("CON_CARGA")
    )
    .first()
)

print(
    "CON_ACCIDENTES:",
    cobertura["CON_ACCIDENTES"]
)

print(
    "CON_FLUJO:",
    cobertura["CON_FLUJO"]
)

print(
    "CON_CARGA:",
    cobertura["CON_CARGA"]
)

if cobertura["CON_ACCIDENTES"] != 2892:
    raise Exception("Cobertura Accidentes inesperada")

if cobertura["CON_FLUJO"] != 2990:
    raise Exception("Cobertura Flujo inesperada")

if cobertura["CON_CARGA"] != 6986:
    raise Exception("Cobertura Carga inesperada")

# =========================================================
# 12. Combinaciones
# =========================================================

print("\n======================================")
print("COMBINACIONES DE COBERTURA")
print("======================================")

(
    df
    .groupBy(
        "HAS_ACCIDENTES",
        "HAS_FLUJO",
        "HAS_CARGA"
    )
    .count()
    .orderBy(
        "HAS_ACCIDENTES",
        "HAS_FLUJO",
        "HAS_CARGA"
    )
    .show(20, truncate=False)
)

# =========================================================
# 13. Nulos principales
# =========================================================

print("\n======================================")
print("NULOS CLAVE")
print("======================================")

cols_check = [
    "LONGITUD_KM",
    "SUPERFICIE",
    "ESTADO",
    "CLASIFICACION_EJE",
    "JERARQUIA",
    "NRO_CARRILES",
    "ES_CONCES",
    "ACCIDENTES_2025",
    "PROM_IMD_2025",
    "FLOTA_CARGA_2025"
]

exprs = [
    F.sum(
        F.when(
            F.col(x).isNull(),
            1
        ).otherwise(0)
    ).alias(x)
    for x in cols_check
]

df.agg(*exprs).show(
    truncate=False
)

# =========================================================
# 14. Escritura
# =========================================================

(
    df
    .repartition(2)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(OUT)
)

print("\n======================================")
print("GOLD INTEGRADO GENERADO")
print("======================================")

print("OUTPUT:", OUT)

spark.stop()