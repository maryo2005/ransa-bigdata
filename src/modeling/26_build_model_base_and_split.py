from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("build_model_base_and_split")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_integrado_tramos_2026/"
)

OUT_DATASET = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026/"
)

OUT_SPLIT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/split_rutas_modelado_2026/"
)

SEED = 2026

df = spark.read.parquet(INPUT)

# =========================================================
# 1. Variables seleccionadas
#
# No incluimos:
# - geometría
# - coordenadas
# - fechas del target
# - información de emergencias 2026
# - identificadores de concesiones
# - flags de calidad del proceso de integración
#
# RUTA_JOIN se conserva SOLO para controlar el split.
# =========================================================

columnas = [

    # Identificación
    "ID_TRAMO",
    "RUTA_JOIN",

    # Contexto geográfico
    "DEPARTAMENTO",

    # Estructurales del tramo
    "LONGITUD_KM",
    "SUPERFICIE",
    "ESTADO",
    "CLASIFICACION_EJE",
    "JERARQUIA",
    "NRO_CARRILES",
    "ES_CONCES",

    # Disponibilidad
    "HAS_ACCIDENTES",
    "HAS_FLUJO",
    "HAS_CARGA",

    # Accidentes históricos <= 2025
    "ACCIDENTES_POR_100KM_HIST",
    "ACCIDENTES_2025_POR_100KM",
    "ACCIDENTES_24M_POR_100KM",
    "PROM_AFECTADOS_POR_ACCIDENTE",
    "PROM_VEHICULOS_POR_ACCIDENTE",
    "ANIOS_CON_ACCIDENTES",
    "MESES_CON_ACCIDENTES",

    # Flujo vehicular <= 2025
    "N_PEAJES_RUTA",
    "MESES_OBSERVADOS_2025",
    "PROM_IMD_2025",
    "PROM_IMD_PESADOS_2025",
    "PCT_PESADOS_2025",
    "VAR_IMD_2025_VS_2024_PCT",

    # Transporte de carga 2025
    "LOG_FLOTA_CARGA_2025",
    "LOG_TRANSPORTISTAS_CARGA_2025",
    "VEHICULOS_POR_TRANSPORTISTA_2025",
    "MEDIANA_EJES_2025",
    "P95_EJES_2025",
    "MEDIANA_CARGA_UTIL_2025",
    "P95_CARGA_UTIL_2025",
    "MEDIANA_P_BRUTO_2025",
    "P95_P_BRUTO_2025",
    "PCT_CAMION_2025",
    "PCT_SEMIREMOLQUE_2025",
    "PCT_REMOLCADOR_2025",
    "PCT_PICKUP_2025",
    "PCT_DIESEL_2025",

    # Target
    "AFECTACION_TRAMO"
]

base = df.select(*columnas)

# =========================================================
# 2. Flag específico para NRO_CARRILES faltante
# La imputación real se hará posteriormente SOLO con TRAIN.
# =========================================================

base = (
    base
    .withColumn(
        "FLAG_NRO_CARRILES_FALTANTE",
        F.when(
            F.col("NRO_CARRILES").isNull(),
            1
        ).otherwise(0)
    )
)

# =========================================================
# 3. Validaciones iniciales
# =========================================================

print("\n======================================")
print("DATASET BASE")
print("======================================")

print("FILAS:", base.count())
print(
    "ID_TRAMO UNICOS:",
    base.select("ID_TRAMO").distinct().count()
)
print(
    "RUTAS:",
    base.select("RUTA_JOIN").distinct().count()
)

rutas_nulas = (
    base
    .filter(F.col("RUTA_JOIN").isNull())
    .count()
)

print("RUTA_JOIN NULOS:", rutas_nulas)

if base.count() != 7058:
    raise Exception("ERROR: dataset base != 7058")

if base.select("ID_TRAMO").distinct().count() != 7058:
    raise Exception("ERROR: ID_TRAMO duplicado")

if rutas_nulas != 0:
    raise Exception("ERROR: existen rutas nulas")

# =========================================================
# 4. Estadísticas del target por ruta
# =========================================================

route_stats = (
    base
    .groupBy("RUTA_JOIN")
    .agg(
        F.count("*").alias("N_TRAMOS"),
        F.sum("AFECTACION_TRAMO").alias("N_POSITIVOS"),
        F.avg("AFECTACION_TRAMO").alias("TASA_AFECTACION")
    )
)

n_rutas = route_stats.count()

print("\n======================================")
print("RUTAS PARA SPLIT")
print("======================================")

print("TOTAL RUTAS:", n_rutas)

# =========================================================
# 5. Estratos según tasa de afectación
#
# Se forman 5 grupos de rutas según su tasa histórica
# observada de afectación en el target.
# =========================================================

w_estrato = Window.orderBy(
    F.col("TASA_AFECTACION"),
    F.col("RUTA_JOIN")
)

route_stats = (
    route_stats
    .withColumn(
        "ESTRATO_SPLIT",
        F.ntile(5).over(w_estrato)
    )
)

# =========================================================
# 6. Split determinista 80/20 dentro de cada estrato
#
# Ninguna ruta puede aparecer simultáneamente
# en TRAIN y TEST.
# =========================================================

w_hash = (
    Window
    .partitionBy("ESTRATO_SPLIT")
    .orderBy(
        F.xxhash64(
            F.col("RUTA_JOIN"),
            F.lit(SEED)
        ),
        F.col("RUTA_JOIN")
    )
)

w_count = (
    Window
    .partitionBy("ESTRATO_SPLIT")
)

route_split = (
    route_stats
    .withColumn(
        "RN",
        F.row_number().over(w_hash)
    )
    .withColumn(
        "N_RUTAS_ESTRATO",
        F.count("*").over(w_count)
    )
    .withColumn(
        "LIMITE_TRAIN",
        F.ceil(
            F.col("N_RUTAS_ESTRATO") * F.lit(0.80)
        ).cast("int")
    )
    .withColumn(
        "SPLIT",
        F.when(
            F.col("RN") <= F.col("LIMITE_TRAIN"),
            F.lit("TRAIN")
        ).otherwise(
            F.lit("TEST")
        )
    )
)

# =========================================================
# 7. Incorporar split al dataset
# =========================================================

dataset = (
    base
    .join(
        route_split.select(
            "RUTA_JOIN",
            "ESTRATO_SPLIT",
            "SPLIT"
        ),
        "RUTA_JOIN",
        "inner"
    )
)

# =========================================================
# 8. Validar que ninguna ruta esté en ambos conjuntos
# =========================================================

rutas_cruzadas = (
    dataset
    .groupBy("RUTA_JOIN")
    .agg(
        F.countDistinct("SPLIT").alias("N_SPLITS")
    )
    .filter(
        F.col("N_SPLITS") > 1
    )
    .count()
)

print("\n======================================")
print("VALIDACION SPLIT")
print("======================================")

print(
    "RUTAS EN TRAIN Y TEST SIMULTANEAMENTE:",
    rutas_cruzadas
)

if rutas_cruzadas != 0:
    raise Exception(
        "ERROR: una ruta aparece en TRAIN y TEST"
    )

# =========================================================
# 9. Distribución del split
# =========================================================

print("\n======================================")
print("FILAS POR SPLIT")
print("======================================")

(
    dataset
    .groupBy("SPLIT")
    .agg(
        F.count("*").alias("TRAMOS"),
        F.countDistinct(
            "RUTA_JOIN"
        ).alias("RUTAS"),
        F.sum(
            "AFECTACION_TRAMO"
        ).alias("POSITIVOS"),
        F.avg(
            "AFECTACION_TRAMO"
        ).alias("TASA_POSITIVOS")
    )
    .orderBy("SPLIT")
    .show(truncate=False)
)

print("\n======================================")
print("TARGET POR SPLIT")
print("======================================")

(
    dataset
    .groupBy(
        "SPLIT",
        "AFECTACION_TRAMO"
    )
    .count()
    .orderBy(
        "SPLIT",
        "AFECTACION_TRAMO"
    )
    .show()
)

# =========================================================
# 10. Cobertura por split
# =========================================================

print("\n======================================")
print("COBERTURA POR SPLIT")
print("======================================")

(
    dataset
    .groupBy("SPLIT")
    .agg(
        F.sum("HAS_ACCIDENTES")
        .alias("CON_ACCIDENTES"),

        F.sum("HAS_FLUJO")
        .alias("CON_FLUJO"),

        F.sum("HAS_CARGA")
        .alias("CON_CARGA")
    )
    .orderBy("SPLIT")
    .show(truncate=False)
)

# =========================================================
# 11. Nulos importantes
# =========================================================

print("\n======================================")
print("NULOS POR SPLIT")
print("======================================")

(
    dataset
    .groupBy("SPLIT")
    .agg(
        F.sum(
            F.when(
                F.col("NRO_CARRILES").isNull(),
                1
            ).otherwise(0)
        ).alias("NULL_NRO_CARRILES"),

        F.sum(
            F.when(
                F.col(
                    "ACCIDENTES_2025_POR_100KM"
                ).isNull(),
                1
            ).otherwise(0)
        ).alias("NULL_ACCIDENTES"),

        F.sum(
            F.when(
                F.col("PROM_IMD_2025").isNull(),
                1
            ).otherwise(0)
        ).alias("NULL_FLUJO"),

        F.sum(
            F.when(
                F.col(
                    "LOG_FLOTA_CARGA_2025"
                ).isNull(),
                1
            ).otherwise(0)
        ).alias("NULL_CARGA")
    )
    .orderBy("SPLIT")
    .show(truncate=False)
)

# =========================================================
# 12. Escritura
# =========================================================

(
    route_split
    .select(
        "RUTA_JOIN",
        "N_TRAMOS",
        "N_POSITIVOS",
        "TASA_AFECTACION",
        "ESTRATO_SPLIT",
        "SPLIT"
    )
    .repartition(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUT_SPLIT)
)

(
    dataset
    .repartition(2)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUT_DATASET)
)

print("\n======================================")
print("DATASET DE MODELADO BASE GENERADO")
print("======================================")

print("DATASET:", OUT_DATASET)
print("SPLIT:", OUT_SPLIT)
print("SEED:", SEED)

spark.stop()