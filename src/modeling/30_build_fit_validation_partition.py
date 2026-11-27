from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import random

spark = (
    SparkSession.builder
    .appName("build_fit_validation_partition_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# CONFIGURACION
# ============================================================

BASE = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026_v2/"
)

ML_FINAL = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_final_2026/"
)

OUT_DATASET = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_ml_particionado_2026/"
)

OUT_RUTAS = (
    "gs://ransa-big-data-2026-lake/"
    "gold/particion_rutas_ml_2026/"
)

SEED = 2026
VALID_FRAC = 0.20
ITERACIONES = 50000

# ============================================================
# 1. CARGAR DATASETS
# ============================================================

base = spark.read.parquet(BASE)
ml = spark.read.parquet(ML_FINAL)

print("\n======================================")
print("ENTRADAS")
print("======================================")
print("BASE:", base.count())
print("ML FINAL:", ml.count())

# ============================================================
# 2. VERIFICAR SPLIT EXTERNO
# ============================================================

split_rutas = (
    base
    .select("RUTA_JOIN", "SPLIT")
    .distinct()
)

rutas_duplicadas_split = (
    split_rutas
    .groupBy("RUTA_JOIN")
    .agg(F.countDistinct("SPLIT").alias("N_SPLITS"))
    .filter(F.col("N_SPLITS") > 1)
    .count()
)

if rutas_duplicadas_split != 0:
    raise Exception(
        f"ERROR: existen {rutas_duplicadas_split} rutas en mas de un SPLIT"
    )

train_base = base.filter(F.col("SPLIT") == "TRAIN")
test_base = base.filter(F.col("SPLIT") == "TEST")

print("\n======================================")
print("SPLIT EXTERNO CONGELADO")
print("======================================")
print("TRAIN TRAMOS:", train_base.count())
print(
    "TRAIN RUTAS:",
    train_base.select("RUTA_JOIN").distinct().count()
)
print("TEST TRAMOS:", test_base.count())
print(
    "TEST RUTAS:",
    test_base.select("RUTA_JOIN").distinct().count()
)

# ============================================================
# 3. RESUMEN POR RUTA SOLO EN TRAIN
# ============================================================

rutas_train = (
    train_base
    .groupBy("RUTA_JOIN")
    .agg(
        F.count("*").alias("N_TRAMOS"),

        F.sum(
            F.col("AFECTACION_TRAMO").cast("long")
        ).alias("N_POSITIVOS"),

        F.sum(
            F.coalesce(
                F.col("HAS_ACCIDENTES").cast("long"),
                F.lit(0)
            )
        ).alias("N_ACCIDENTES_COVERAGE"),

        F.sum(
            F.coalesce(
                F.col("HAS_FLUJO").cast("long"),
                F.lit(0)
            )
        ).alias("N_FLUJO_COVERAGE"),

        F.sum(
            F.coalesce(
                F.col("HAS_CARGA").cast("long"),
                F.lit(0)
            )
        ).alias("N_CARGA_COVERAGE")
    )
    .orderBy("RUTA_JOIN")
)

rows = rutas_train.collect()

if len(rows) != 128:
    raise Exception(
        f"ERROR: se esperaban 128 rutas TRAIN y se obtuvieron {len(rows)}"
    )

n_valid_routes = round(len(rows) * VALID_FRAC)

# ============================================================
# 4. TOTALES TRAIN
# ============================================================

totales = {
    "tramos": sum(int(r["N_TRAMOS"] or 0) for r in rows),
    "positivos": sum(int(r["N_POSITIVOS"] or 0) for r in rows),
    "acc": sum(int(r["N_ACCIDENTES_COVERAGE"] or 0) for r in rows),
    "flujo": sum(int(r["N_FLUJO_COVERAGE"] or 0) for r in rows),
    "carga": sum(int(r["N_CARGA_COVERAGE"] or 0) for r in rows),
}

print("\n======================================")
print("OBJETIVO VALIDATION")
print("======================================")
print("RUTAS TRAIN:", len(rows))
print("RUTAS VALIDATION:", n_valid_routes)
print("FRACCION OBJETIVO:", VALID_FRAC)
print("TOTALES TRAIN:", totales)

# ============================================================
# 5. BUSQUEDA REPRODUCIBLE DE RUTAS VALIDATION
# ============================================================

rng = random.Random(SEED)

indices = list(range(len(rows)))

best_score = None
best_indices = None
best_stats = None

for _ in range(ITERACIONES):

    seleccion = rng.sample(indices, n_valid_routes)

    stats = {
        "tramos": 0,
        "positivos": 0,
        "acc": 0,
        "flujo": 0,
        "carga": 0,
    }

    for i in seleccion:
        r = rows[i]

        stats["tramos"] += int(r["N_TRAMOS"] or 0)
        stats["positivos"] += int(r["N_POSITIVOS"] or 0)
        stats["acc"] += int(r["N_ACCIDENTES_COVERAGE"] or 0)
        stats["flujo"] += int(r["N_FLUJO_COVERAGE"] or 0)
        stats["carga"] += int(r["N_CARGA_COVERAGE"] or 0)

    def desviacion(nombre):
        total = totales[nombre]

        if total == 0:
            return 0.0

        proporcion = stats[nombre] / total

        return abs(proporcion - VALID_FRAC)

    score = (
        6.0 * desviacion("tramos")
        + 6.0 * desviacion("positivos")
        + 2.0 * desviacion("acc")
        + 2.0 * desviacion("flujo")
        + 1.0 * desviacion("carga")
    )

    if best_score is None or score < best_score:
        best_score = score
        best_indices = set(seleccion)
        best_stats = stats.copy()

print("\n======================================")
print("MEJOR PARTICION ENCONTRADA")
print("======================================")
print("SCORE:", best_score)
print("VALIDATION STATS:", best_stats)

for k in ["tramos", "positivos", "acc", "flujo", "carga"]:
    print(
        k,
        "=",
        best_stats[k],
        "/",
        totales[k],
        "=",
        round(best_stats[k] / totales[k], 6)
        if totales[k] > 0
        else None
    )

# ============================================================
# 6. CREAR MAPA DE PARTICIONES
# ============================================================

assignments = []

for i, r in enumerate(rows):

    partition = (
        "VALIDATION"
        if i in best_indices
        else "FIT"
    )

    assignments.append(
        (
            r["RUTA_JOIN"],
            partition
        )
    )

test_routes = (
    test_base
    .select("RUTA_JOIN")
    .distinct()
    .collect()
)

for r in test_routes:
    assignments.append(
        (
            r["RUTA_JOIN"],
            "TEST"
        )
    )

route_map = spark.createDataFrame(
    assignments,
    ["RUTA_JOIN", "PARTITION"]
)

# ============================================================
# 7. VALIDAR MAPA DE RUTAS
# ============================================================

total_rutas = route_map.count()
rutas_unicas = (
    route_map
    .select("RUTA_JOIN")
    .distinct()
    .count()
)

duplicadas = (
    route_map
    .groupBy("RUTA_JOIN")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print("\n======================================")
print("MAPA DE PARTICIONES")
print("======================================")

(
    route_map
    .groupBy("PARTITION")
    .count()
    .orderBy("PARTITION")
    .show(truncate=False)
)

print("TOTAL RUTAS:", total_rutas)
print("RUTAS UNICAS:", rutas_unicas)
print("RUTAS DUPLICADAS:", duplicadas)

if total_rutas != 160:
    raise Exception(
        f"ERROR: se esperaban 160 rutas y hay {total_rutas}"
    )

if rutas_unicas != 160:
    raise Exception(
        f"ERROR: se esperaban 160 rutas unicas y hay {rutas_unicas}"
    )

if duplicadas != 0:
    raise Exception(
        f"ERROR: existen {duplicadas} rutas duplicadas"
    )

# ============================================================
# 8. APLICAR PARTICION AL DATASET ML YA PREPROCESADO
# ============================================================

dataset = (
    ml
    .join(
        route_map,
        "RUTA_JOIN",
        "left"
    )
)

total = dataset.count()

ids_unicos = (
    dataset
    .select("ID_TRAMO")
    .distinct()
    .count()
)

sin_partition = (
    dataset
    .filter(F.col("PARTITION").isNull())
    .count()
)

rutas_multi = (
    dataset
    .groupBy("RUTA_JOIN")
    .agg(
        F.countDistinct("PARTITION")
        .alias("N_PARTITIONS")
    )
    .filter(F.col("N_PARTITIONS") > 1)
    .count()
)

print("\n======================================")
print("VALIDACION DATASET FINAL")
print("======================================")
print("TOTAL:", total)
print("ID_TRAMO UNICOS:", ids_unicos)
print("SIN PARTITION:", sin_partition)
print("RUTAS EN MAS DE UNA PARTICION:", rutas_multi)

if total != 7058:
    raise Exception(
        f"ERROR: se esperaban 7058 filas y hay {total}"
    )

if ids_unicos != 7058:
    raise Exception(
        f"ERROR: ID_TRAMO no es unico: {ids_unicos}"
    )

if sin_partition != 0:
    raise Exception(
        f"ERROR: existen {sin_partition} filas sin PARTITION"
    )

if rutas_multi != 0:
    raise Exception(
        f"ERROR: existen {rutas_multi} rutas compartidas"
    )

# ============================================================
# 9. RESUMEN FINAL FIT / VALIDATION / TEST
# ============================================================

print("\n======================================")
print("PARTITION FINAL")
print("======================================")

(
    dataset
    .groupBy("PARTITION")
    .agg(
        F.count("*").alias("N_TRAMOS"),
        F.countDistinct("RUTA_JOIN").alias("N_RUTAS"),
        F.sum(
            F.col("AFECTACION_TRAMO").cast("long")
        ).alias("N_POSITIVOS"),
        F.avg(
            F.col("AFECTACION_TRAMO").cast("double")
        ).alias("TASA_POSITIVOS")
    )
    .orderBy("PARTITION")
    .show(truncate=False)
)

print("\n======================================")
print("TARGET POR PARTICION")
print("======================================")

(
    dataset
    .groupBy(
        "PARTITION",
        "AFECTACION_TRAMO"
    )
    .count()
    .orderBy(
        "PARTITION",
        "AFECTACION_TRAMO"
    )
    .show(truncate=False)
)

# ============================================================
# 10. COBERTURA POR PARTICION USANDO BASE RAW
# ============================================================

base_part = (
    base
    .join(
        route_map,
        "RUTA_JOIN",
        "left"
    )
)

print("\n======================================")
print("COBERTURA POR PARTICION")
print("======================================")

(
    base_part
    .groupBy("PARTITION")
    .agg(
        F.count("*").alias("N_TRAMOS"),

        F.sum(
            F.coalesce(
                F.col("HAS_ACCIDENTES").cast("long"),
                F.lit(0)
            )
        ).alias("CON_ACCIDENTES"),

        F.sum(
            F.coalesce(
                F.col("HAS_FLUJO").cast("long"),
                F.lit(0)
            )
        ).alias("CON_FLUJO"),

        F.sum(
            F.coalesce(
                F.col("HAS_CARGA").cast("long"),
                F.lit(0)
            )
        ).alias("CON_CARGA")
    )
    .orderBy("PARTITION")
    .show(truncate=False)
)

# ============================================================
# 11. ESCRIBIR RESULTADOS
# ============================================================

(
    route_map
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUT_RUTAS)
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
print("PARTICION ML GENERADA")
print("======================================")
print("DATASET:", OUT_DATASET)
print("MAPA RUTAS:", OUT_RUTAS)
print("SEED:", SEED)
print("ITERACIONES:", ITERACIONES)
print("TEST EXTERNO: CONGELADO")
print("======================================")

spark.stop()