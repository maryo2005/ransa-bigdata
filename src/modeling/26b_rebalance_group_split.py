from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import random

spark = (
    SparkSession.builder
    .appName("rebalance_group_split")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026/"
)

OUT_DATASET = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026_v2/"
)

OUT_SPLIT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/split_rutas_modelado_2026_v2/"
)

SEED = 2026
TEST_FRAC = 0.20
ITERACIONES = 50000

df = spark.read.parquet(INPUT)

# Quitamos el split anterior.
base = df.drop(
    "SPLIT",
    "ESTRATO_SPLIT"
)

# =========================================================
# 1. Estadísticas por ruta
# =========================================================

route_df = (
    base
    .groupBy("RUTA_JOIN")
    .agg(
        F.count("*").alias("N_TRAMOS"),
        F.sum("AFECTACION_TRAMO").alias("N_POSITIVOS"),
        F.sum("HAS_ACCIDENTES").alias("N_ACCIDENTES"),
        F.sum("HAS_FLUJO").alias("N_FLUJO"),
        F.sum("HAS_CARGA").alias("N_CARGA")
    )
)

rows = route_df.collect()

stats = {
    r["RUTA_JOIN"]: {
        "tramos": r["N_TRAMOS"],
        "positivos": r["N_POSITIVOS"],
        "accidentes": r["N_ACCIDENTES"],
        "flujo": r["N_FLUJO"],
        "carga": r["N_CARGA"]
    }
    for r in rows
}

rutas = sorted(stats.keys())

# =========================================================
# 2. Totales y objetivos TEST
# =========================================================

totales = {
    k: sum(x[k] for x in stats.values())
    for k in [
        "tramos",
        "positivos",
        "accidentes",
        "flujo",
        "carga"
    ]
}

objetivos = {
    k: totales[k] * TEST_FRAC
    for k in totales
}

n_test_rutas = round(
    len(rutas) * TEST_FRAC
)

print("\n======================================")
print("OBJETIVOS SPLIT V2")
print("======================================")

print("RUTAS TOTAL:", len(rutas))
print("RUTAS TEST:", n_test_rutas)

for k in objetivos:
    print(
        k.upper(),
        "TOTAL=",
        totales[k],
        "TARGET_TEST=",
        round(objetivos[k], 2)
    )

# =========================================================
# 3. Función de evaluación
# =========================================================

def sumar(seleccion):

    out = {
        "tramos": 0,
        "positivos": 0,
        "accidentes": 0,
        "flujo": 0,
        "carga": 0
    }

    for ruta in seleccion:
        for k in out:
            out[k] += stats[ruta][k]

    return out


def score(seleccion):

    s = sumar(seleccion)

    # Mayor peso a tamaño del TEST y clase positiva.
    pesos = {
        "tramos": 6.0,
        "positivos": 6.0,
        "accidentes": 2.0,
        "flujo": 2.0,
        "carga": 1.0
    }

    resultado = 0.0

    for k, peso in pesos.items():

        target = objetivos[k]

        if target > 0:
            resultado += (
                peso
                *
                abs(s[k] - target)
                /
                target
            )

    return resultado


# =========================================================
# 4. Búsqueda reproducible
# =========================================================

rng = random.Random(SEED)

mejor = None
mejor_score = float("inf")

for _ in range(ITERACIONES):

    candidato = tuple(
        sorted(
            rng.sample(
                rutas,
                n_test_rutas
            )
        )
    )

    sc = score(candidato)

    if sc < mejor_score:
        mejor = candidato
        mejor_score = sc

test_rutas = set(mejor)

print("\n======================================")
print("MEJOR SOLUCION")
print("======================================")

print("SCORE:", mejor_score)
print("RUTAS TEST:", len(test_rutas))
print("METRICAS TEST:", sumar(test_rutas))

# =========================================================
# 5. Crear mapa de split
# =========================================================

split_rows = [
    (
        ruta,
        "TEST" if ruta in test_rutas
        else "TRAIN"
    )
    for ruta in rutas
]

split_df = spark.createDataFrame(
    split_rows,
    [
        "RUTA_JOIN",
        "SPLIT"
    ]
)

dataset = (
    base
    .join(
        split_df,
        "RUTA_JOIN",
        "inner"
    )
)

# =========================================================
# 6. Validar aislamiento de rutas
# =========================================================

cruzadas = (
    dataset
    .groupBy("RUTA_JOIN")
    .agg(
        F.countDistinct(
            "SPLIT"
        ).alias("N_SPLITS")
    )
    .filter(
        F.col("N_SPLITS") > 1
    )
    .count()
)

print("\n======================================")
print("VALIDACION")
print("======================================")

print(
    "RUTAS EN AMBOS SPLITS:",
    cruzadas
)

if cruzadas != 0:
    raise Exception(
        "ERROR: leakage entre rutas"
    )

if dataset.count() != 7058:
    raise Exception(
        "ERROR: no se conservaron 7058 tramos"
    )

# =========================================================
# 7. Métricas finales
# =========================================================

print("\n======================================")
print("RESULTADO TRAIN / TEST V2")
print("======================================")

(
    dataset
    .groupBy("SPLIT")
    .agg(
        F.count("*")
        .alias("TRAMOS"),

        F.countDistinct(
            "RUTA_JOIN"
        ).alias("RUTAS"),

        F.sum(
            "AFECTACION_TRAMO"
        ).alias("POSITIVOS"),

        F.avg(
            "AFECTACION_TRAMO"
        ).alias("TASA_POSITIVOS"),

        F.sum(
            "HAS_ACCIDENTES"
        ).alias("CON_ACCIDENTES"),

        F.avg(
            "HAS_ACCIDENTES"
        ).alias("PCT_COB_ACC"),

        F.sum(
            "HAS_FLUJO"
        ).alias("CON_FLUJO"),

        F.avg(
            "HAS_FLUJO"
        ).alias("PCT_COB_FLUJO"),

        F.sum(
            "HAS_CARGA"
        ).alias("CON_CARGA"),

        F.avg(
            "HAS_CARGA"
        ).alias("PCT_COB_CARGA")
    )
    .orderBy("SPLIT")
    .show(
        truncate=False
    )
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
# 8. Nulos
# =========================================================

print("\n======================================")
print("NULOS V2")
print("======================================")

(
    dataset
    .groupBy("SPLIT")
    .agg(
        F.sum(
            F.col(
                "NRO_CARRILES"
            ).isNull().cast("int")
        ).alias(
            "NULL_NRO_CARRILES"
        ),

        F.sum(
            F.col(
                "ACCIDENTES_2025_POR_100KM"
            ).isNull().cast("int")
        ).alias(
            "NULL_ACCIDENTES"
        ),

        F.sum(
            F.col(
                "PROM_IMD_2025"
            ).isNull().cast("int")
        ).alias(
            "NULL_FLUJO"
        ),

        F.sum(
            F.col(
                "LOG_FLOTA_CARGA_2025"
            ).isNull().cast("int")
        ).alias(
            "NULL_CARGA"
        )
    )
    .orderBy("SPLIT")
    .show(
        truncate=False
    )
)

# =========================================================
# 9. Escritura
# =========================================================

(
    split_df
    .repartition(1)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(
        OUT_SPLIT
    )
)

(
    dataset
    .repartition(2)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(
        OUT_DATASET
    )
)

print("\n======================================")
print("SPLIT V2 GENERADO")
print("======================================")

print("DATASET:", OUT_DATASET)
print("SPLIT:", OUT_SPLIT)
print("SEED:", SEED)
print("ITERACIONES:", ITERACIONES)

spark.stop()