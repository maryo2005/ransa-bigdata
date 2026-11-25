from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("build_carga_features_departamento_2025")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

CARGA = (
    "gs://ransa-big-data-2026-lake/"
    "silver/transporte_carga/"
)

MASTER = (
    "gs://ransa-big-data-2026-lake/"
    "gold/master_tramos_2024/"
)

OUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/carga_features_departamento_2025/"
)

df = spark.read.parquet(CARGA)
m = spark.read.parquet(MASTER)

# =========================================================
# 1. Normalización
# =========================================================

def norm_text(c):
    return F.translate(
        F.upper(F.trim(c)),
        "ÁÉÍÓÚÜÑ",
        "AEIOUUN"
    )

# =========================================================
# 2. Corte 2025 válido
# =========================================================

d25 = (
    df
    .filter(F.col("ANIO_CORTE") == 2025)
    .filter(F.col("REGISTRO_VALIDO") == 1)
    .withColumn(
        "DEP_N",
        norm_text(F.col("DEPARTAMENTO"))
    )
)

print("\n======================================")
print("BASE 2025 VALIDA")
print("======================================")

print("FILAS:", d25.count())

print(
    "VEHICULOS:",
    d25.select("VEHICULO_HASH")
    .distinct()
    .count()
)

# =========================================================
# 3. Eliminar vehículos con departamento ambiguo
# =========================================================

veh_dep = (
    d25
    .groupBy("VEHICULO_HASH")
    .agg(
        F.countDistinct(
            "DEP_N"
        ).alias("N_DEPARTAMENTOS")
    )
)

veh_estables = (
    veh_dep
    .filter(F.col("N_DEPARTAMENTOS") == 1)
    .select("VEHICULO_HASH")
)

base = (
    d25
    .join(
        veh_estables,
        "VEHICULO_HASH",
        "inner"
    )
)

print("\n======================================")
print("BASE ESTABLE")
print("======================================")

print("FILAS:", base.count())

print(
    "VEHICULOS ESTABLES:",
    base.select("VEHICULO_HASH")
    .distinct()
    .count()
)

print(
    "VEHICULOS EXCLUIDOS POR DEP AMBIGUO:",
    veh_dep
    .filter(F.col("N_DEPARTAMENTOS") > 1)
    .count()
)

# =========================================================
# 4. Variables auxiliares
# =========================================================

base = (
    base
    .withColumn(
        "FLAG_CAMION",
        F.when(
            F.col("CLASE") == "CAMION",
            1
        ).otherwise(0)
    )
    .withColumn(
        "FLAG_SEMIREMOLQUE",
        F.when(
            F.col("CLASE") == "SEMIREMOLQUE",
            1
        ).otherwise(0)
    )
    .withColumn(
        "FLAG_REMOLCADOR",
        F.when(
            F.col("CLASE") == "REMOLCADOR",
            1
        ).otherwise(0)
    )
    .withColumn(
        "FLAG_PICKUP",
        F.when(
            F.col("CLASE") == "CMTA PICKUP",
            1
        ).otherwise(0)
    )
    .withColumn(
        "COMBUSTIBLE_N",
        norm_text(F.col("COMBUSTIBLE"))
    )
    .withColumn(
        "FLAG_DIESEL",
        F.when(
            F.col("COMBUSTIBLE_N") == "DIESEL",
            1
        ).otherwise(0)
    )
)

# =========================================================
# 5. Features robustas por departamento
# =========================================================

features = (
    base
    .groupBy(
        "DEP_N"
    )
    .agg(
        F.countDistinct(
            "VEHICULO_HASH"
        ).alias(
            "FLOTA_CARGA_2025"
        ),

        F.countDistinct(
            "TRANSPORTISTA_HASH"
        ).alias(
            "TRANSPORTISTAS_CARGA_2025"
        ),

        F.avg(
            "FLAG_CAMION"
        ).alias(
            "PROP_CAMION_2025"
        ),

        F.avg(
            "FLAG_SEMIREMOLQUE"
        ).alias(
            "PROP_SEMIREMOLQUE_2025"
        ),

        F.avg(
            "FLAG_REMOLCADOR"
        ).alias(
            "PROP_REMOLCADOR_2025"
        ),

        F.avg(
            "FLAG_PICKUP"
        ).alias(
            "PROP_PICKUP_2025"
        ),

        F.avg(
            "FLAG_DIESEL"
        ).alias(
            "PROP_DIESEL_2025"
        ),

        F.expr(
            "percentile_approx(N_EJES, 0.5)"
        ).alias(
            "MEDIANA_EJES_2025"
        ),

        F.expr(
            "percentile_approx(N_EJES, 0.95)"
        ).alias(
            "P95_EJES_2025"
        ),

        F.expr(
            """
            percentile_approx(
                CARGA_UTIL,
                0.5
            )
            """
        ).alias(
            "MEDIANA_CARGA_UTIL_2025"
        ),

        F.expr(
            """
            percentile_approx(
                CARGA_UTIL,
                0.95
            )
            """
        ).alias(
            "P95_CARGA_UTIL_2025"
        ),

        F.expr(
            """
            percentile_approx(
                P_BRUTO,
                0.5
            )
            """
        ).alias(
            "MEDIANA_P_BRUTO_2025"
        ),

        F.expr(
            """
            percentile_approx(
                P_BRUTO,
                0.95
            )
            """
        ).alias(
            "P95_P_BRUTO_2025"
        )
    )
)

# =========================================================
# 6. Variables derivadas
# =========================================================

features = (
    features
    .withColumn(
        "VEHICULOS_POR_TRANSPORTISTA_2025",
        F.round(
            F.col("FLOTA_CARGA_2025")
            /
            F.col("TRANSPORTISTAS_CARGA_2025"),
            4
        )
    )
    .withColumn(
        "LOG_FLOTA_CARGA_2025",
        F.log1p(
            F.col("FLOTA_CARGA_2025")
        )
    )
    .withColumn(
        "LOG_TRANSPORTISTAS_CARGA_2025",
        F.log1p(
            F.col("TRANSPORTISTAS_CARGA_2025")
        )
    )
)

# Porcentajes más legibles

for c in [
    "PROP_CAMION_2025",
    "PROP_SEMIREMOLQUE_2025",
    "PROP_REMOLCADOR_2025",
    "PROP_PICKUP_2025",
    "PROP_DIESEL_2025"
]:

    features = (
        features
        .withColumn(
            c.replace("PROP_", "PCT_"),
            F.round(
                F.col(c) * 100.0,
                4
            )
        )
        .drop(c)
    )

# =========================================================
# 7. Relación con Master
# =========================================================

deps_master = (
    m
    .select(
        norm_text(
            F.col("DEPARTAMENTO")
        ).alias("DEP_N")
    )
    .distinct()
)

features = (
    features
    .join(
        deps_master,
        "DEP_N",
        "inner"
    )
    .withColumn(
        "NIVEL_AGREGACION",
        F.lit("DEPARTAMENTO")
    )
    .withColumn(
        "FECHA_CORTE_PREDICTORES",
        F.to_date(
            F.lit("2025-12-31")
        )
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

# =========================================================
# 8. Validaciones
# =========================================================

print("\n======================================")
print("FEATURES DEPARTAMENTALES")
print("======================================")

print(
    "DEPARTAMENTOS:",
    features.count()
)

if features.count() != 23:
    raise Exception(
        "ERROR: se esperaban 23 "
        "departamentos con Carga."
    )

(
    features
    .orderBy(
        F.desc("FLOTA_CARGA_2025")
    )
    .show(
        30,
        truncate=False
    )
)

# =========================================================
# 9. Escritura Gold
# =========================================================

(
    features
    .repartition(1)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(OUT)
)

print("\n======================================")
print("GOLD CARGA GENERADO")
print("======================================")

print("OUTPUT:", OUT)

spark.stop()