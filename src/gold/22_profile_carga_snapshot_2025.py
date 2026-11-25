from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("profile_carga_snapshot_2025")
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

df = spark.read.parquet(CARGA)
m = spark.read.parquet(MASTER)

# =========================================================
# 1. Corte 2025
# =========================================================

d25 = (
    df
    .filter(F.col("ANIO_CORTE") == 2025)
)

print("\n======================================")
print("SNAPSHOT CARGA 2025")
print("======================================")

print("FILAS:", d25.count())

print(
    "VEHICULOS UNICOS:",
    d25.select("VEHICULO_HASH")
    .distinct()
    .count()
)

print(
    "TRANSPORTISTAS UNICOS:",
    d25.select("TRANSPORTISTA_HASH")
    .distinct()
    .count()
)

print(
    "REGISTROS VALIDOS:",
    d25.filter(
        F.col("REGISTRO_VALIDO") == 1
    ).count()
)

# =========================================================
# 2. Repetición de vehículos dentro del mismo corte
# =========================================================

veh_counts = (
    d25
    .groupBy("VEHICULO_HASH")
    .agg(
        F.count("*").alias("N_REGISTROS")
    )
)

print("\n======================================")
print("REPETICION VEHICULOS EN 2025")
print("======================================")

print(
    "VEHICULOS CON MAS DE 1 REGISTRO:",
    veh_counts.filter(
        F.col("N_REGISTROS") > 1
    ).count()
)

print(
    "MAX REGISTROS MISMO VEHICULO:",
    veh_counts.agg(
        F.max("N_REGISTROS")
    ).first()[0]
)

(
    veh_counts
    .groupBy("N_REGISTROS")
    .count()
    .orderBy("N_REGISTROS")
    .show(30, truncate=False)
)

# =========================================================
# 3. Vehículos asociados a varios departamentos
# =========================================================

veh_dep = (
    d25
    .groupBy("VEHICULO_HASH")
    .agg(
        F.countDistinct(
            "DEPARTAMENTO"
        ).alias("N_DEPARTAMENTOS")
    )
)

print("\n======================================")
print("ESTABILIDAD DEPARTAMENTO")
print("======================================")

print(
    "VEHICULOS EN >1 DEPARTAMENTO:",
    veh_dep.filter(
        F.col("N_DEPARTAMENTOS") > 1
    ).count()
)

print(
    "MAX DEPARTAMENTOS POR VEHICULO:",
    veh_dep.agg(
        F.max("N_DEPARTAMENTOS")
    ).first()[0]
)

# =========================================================
# 4. Distribución departamental
# =========================================================

print("\n======================================")
print("FLOTA POR DEPARTAMENTO 2025")
print("======================================")

(
    d25
    .groupBy("DEPARTAMENTO")
    .agg(
        F.count("*").alias("FILAS"),
        F.countDistinct(
            "VEHICULO_HASH"
        ).alias("VEHICULOS_UNICOS"),
        F.countDistinct(
            "TRANSPORTISTA_HASH"
        ).alias("TRANSPORTISTAS_UNICOS")
    )
    .orderBy(
        F.desc("VEHICULOS_UNICOS")
    )
    .show(50, truncate=False)
)

# =========================================================
# 5. Comparar departamentos con Master
# =========================================================

def norm_dep(c):

    return F.translate(
        F.upper(F.trim(c)),
        "ÁÉÍÓÚÜÑ",
        "AEIOUUN"
    )


deps_carga = (
    d25
    .select(
        norm_dep(
            F.col("DEPARTAMENTO")
        ).alias("DEP")
    )
    .distinct()
)

deps_master = (
    m
    .select(
        norm_dep(
            F.col("DEPARTAMENTO")
        ).alias("DEP")
    )
    .distinct()
)

print("\n======================================")
print("DEPARTAMENTOS VS MASTER")
print("======================================")

print(
    "DEPARTAMENTOS CARGA:",
    deps_carga.count()
)

print(
    "DEPARTAMENTOS MASTER:",
    deps_master.count()
)

print(
    "COINCIDENTES:",
    deps_carga
    .join(
        deps_master,
        "DEP",
        "inner"
    )
    .count()
)

print("\nCARGA SIN DEPARTAMENTO EN MASTER:")

(
    deps_carga
    .join(
        deps_master,
        "DEP",
        "left_anti"
    )
    .show(50, truncate=False)
)

# =========================================================
# 6. Cuantiles robustos
# =========================================================

cols_num = [
    "ANIO_FAB",
    "N_EJES",
    "N_LLANTAS",
    "CARGA_UTIL",
    "P_SECO",
    "P_BRUTO",
    "LARGO",
    "ANCHO",
    "ALTO"
]

print("\n======================================")
print("CUANTILES VARIABLES NUMERICAS")
print("======================================")

for c in cols_num:

    valores = (
        d25
        .filter(F.col(c).isNotNull())
        .approxQuantile(
            c,
            [
                0.00,
                0.01,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
                0.999,
                1.00
            ],
            0.001
        )
    )

    print(c, valores)

# =========================================================
# 7. Extremos superiores
# =========================================================

print("\n======================================")
print("TOP CARGA UTIL")
print("======================================")

(
    d25
    .select(
        "VEHICULO_HASH",
        "DEPARTAMENTO",
        "CLASE",
        "N_EJES",
        "CARGA_UTIL",
        "P_BRUTO",
        "LARGO",
        "ANCHO",
        "ALTO"
    )
    .orderBy(
        F.desc("CARGA_UTIL")
    )
    .show(20, truncate=False)
)

print("\n======================================")
print("TOP P_BRUTO")
print("======================================")

(
    d25
    .select(
        "VEHICULO_HASH",
        "DEPARTAMENTO",
        "CLASE",
        "N_EJES",
        "CARGA_UTIL",
        "P_BRUTO"
    )
    .orderBy(
        F.desc("P_BRUTO")
    )
    .show(20, truncate=False)
)

# =========================================================
# 8. Clases principales
# =========================================================

print("\n======================================")
print("CLASES VEHICULARES 2025")
print("======================================")

(
    d25
    .groupBy("CLASE")
    .count()
    .orderBy(
        F.desc("count")
    )
    .show(50, truncate=False)
)

# =========================================================
# 9. Combustibles
# =========================================================

print("\n======================================")
print("COMBUSTIBLE 2025")
print("======================================")

(
    d25
    .groupBy("COMBUSTIBLE")
    .count()
    .orderBy(
        F.desc("count")
    )
    .show(30, truncate=False)
)

spark.stop()