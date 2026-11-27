from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("audit_final_gold_inputs")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

BASE = "gs://ransa-big-data-2026-lake/gold/"

paths = {
    "MASTER": BASE + "master_tramos_2024/",
    "TARGET": BASE + "afectacion_tramo_2026/",
    "ACCIDENTES": BASE + "accidentes_features_ruta_hist_2025/",
    "FLUJO": BASE + "flujo_features_ruta_hist_2025/",
    "CARGA": BASE + "carga_features_departamento_2025/"
}

dfs = {
    nombre: spark.read.parquet(path)
    for nombre, path in paths.items()
}

# =========================================================
# 1. Esquemas y tamaños
# =========================================================

for nombre, df in dfs.items():

    print("\n======================================")
    print(nombre)
    print("======================================")

    print("FILAS:", df.count())
    print("COLUMNAS:", len(df.columns))
    print("COLUMNAS:")
    print(df.columns)

    df.printSchema()

# =========================================================
# 2. Unicidad de claves
# =========================================================

master = dfs["MASTER"]
target = dfs["TARGET"]
acc = dfs["ACCIDENTES"]
flujo = dfs["FLUJO"]
carga = dfs["CARGA"]

print("\n======================================")
print("UNICIDAD DE CLAVES")
print("======================================")

print(
    "MASTER ID_TRAMO:",
    master.count(),
    master.select("ID_TRAMO").distinct().count()
)

print(
    "TARGET ID_TRAMO:",
    target.count(),
    target.select("ID_TRAMO").distinct().count()
)

print(
    "ACC RUTA_FINAL:",
    acc.count(),
    acc.select("RUTA_FINAL").distinct().count()
)

print(
    "FLUJO RUTA_FINAL:",
    flujo.count(),
    flujo.select("RUTA_FINAL").distinct().count()
)

print(
    "CARGA DEP_N:",
    carga.count(),
    carga.select("DEP_N").distinct().count()
)

# =========================================================
# 3. Verificar cobertura Target sobre Master
# =========================================================

master_ids = master.select("ID_TRAMO").distinct()
target_ids = target.select("ID_TRAMO").distinct()

print("\n======================================")
print("MASTER VS TARGET")
print("======================================")

print(
    "MASTER SIN TARGET:",
    master_ids.join(
        target_ids,
        "ID_TRAMO",
        "left_anti"
    ).count()
)

print(
    "TARGET SIN MASTER:",
    target_ids.join(
        master_ids,
        "ID_TRAMO",
        "left_anti"
    ).count()
)

# =========================================================
# 4. Distribución Target
# =========================================================

print("\n======================================")
print("TARGET")
print("======================================")

if "AFECTACION_TRAMO" in target.columns:

    target.groupBy(
        "AFECTACION_TRAMO"
    ).count().orderBy(
        "AFECTACION_TRAMO"
    ).show()

# =========================================================
# 5. Normalización para medir cobertura
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


master_norm = (
    master
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
# 6. Cobertura sobre los 7058 tramos
# =========================================================

acc_keys = (
    acc
    .select(
        F.col("RUTA_FINAL").alias("RUTA_JOIN")
    )
    .distinct()
    .withColumn(
        "HAS_ACCIDENTES",
        F.lit(1)
    )
)

flujo_keys = (
    flujo
    .select(
        F.col("RUTA_FINAL").alias("RUTA_JOIN")
    )
    .distinct()
    .withColumn(
        "HAS_FLUJO",
        F.lit(1)
    )
)

carga_keys = (
    carga
    .select(
        F.col("DEP_N").alias("DEP_JOIN")
    )
    .distinct()
    .withColumn(
        "HAS_CARGA",
        F.lit(1)
    )
)

audit = (
    master_norm
    .join(
        acc_keys,
        "RUTA_JOIN",
        "left"
    )
    .join(
        flujo_keys,
        "RUTA_JOIN",
        "left"
    )
    .join(
        carga_keys,
        "DEP_JOIN",
        "left"
    )
    .fillna(
        {
            "HAS_ACCIDENTES": 0,
            "HAS_FLUJO": 0,
            "HAS_CARGA": 0
        }
    )
)

print("\n======================================")
print("COBERTURA SOBRE TRAMOS MASTER")
print("======================================")

audit.agg(
    F.count("*").alias("TOTAL_TRAMOS"),

    F.sum("HAS_ACCIDENTES")
    .alias("TRAMOS_CON_ACCIDENTES"),

    F.sum("HAS_FLUJO")
    .alias("TRAMOS_CON_FLUJO"),

    F.sum("HAS_CARGA")
    .alias("TRAMOS_CON_CARGA")
).show(truncate=False)

print("\n======================================")
print("COMBINACIONES DE COBERTURA")
print("======================================")

(
    audit
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

spark.stop()