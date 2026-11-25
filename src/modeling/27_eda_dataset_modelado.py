from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import math

spark = (
    SparkSession.builder
    .appName("eda_dataset_modelado_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026_v2/"
)

OUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/eda_modelado_2026/"
)

df = spark.read.parquet(INPUT)

# =========================================================
# 1. Variables candidatas
# =========================================================

numeric_cols = [
    "LONGITUD_KM",
    "NRO_CARRILES",
    "ES_CONCES",
    "HAS_ACCIDENTES",
    "HAS_FLUJO",
    "HAS_CARGA",
    "FLAG_NRO_CARRILES_FALTANTE",

    "ACCIDENTES_POR_100KM_HIST",
    "ACCIDENTES_2025_POR_100KM",
    "ACCIDENTES_24M_POR_100KM",
    "PROM_AFECTADOS_POR_ACCIDENTE",
    "PROM_VEHICULOS_POR_ACCIDENTE",
    "ANIOS_CON_ACCIDENTES",
    "MESES_CON_ACCIDENTES",

    "N_PEAJES_RUTA",
    "MESES_OBSERVADOS_2025",
    "PROM_IMD_2025",
    "PROM_IMD_PESADOS_2025",
    "PCT_PESADOS_2025",
    "VAR_IMD_2025_VS_2024_PCT",

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
    "PCT_DIESEL_2025"
]

categorical_cols = [
    "DEPARTAMENTO",
    "SUPERFICIE",
    "ESTADO",
    "CLASIFICACION_EJE",
    "JERARQUIA"
]

# =========================================================
# 2. Validación inicial
# =========================================================

print("\n======================================")
print("BASE EDA")
print("======================================")

print("FILAS:", df.count())
print(
    "ID_TRAMO UNICOS:",
    df.select("ID_TRAMO").distinct().count()
)
print(
    "RUTAS:",
    df.select("RUTA_JOIN").distinct().count()
)

(
    df
    .groupBy("SPLIT")
    .agg(
        F.count("*").alias("TRAMOS"),
        F.countDistinct("RUTA_JOIN").alias("RUTAS"),
        F.sum("AFECTACION_TRAMO").alias("POSITIVOS"),
        F.avg("AFECTACION_TRAMO").alias("TASA_POSITIVOS")
    )
    .orderBy("SPLIT")
    .show(truncate=False)
)

# =========================================================
# 3. Resumen numérico general
# =========================================================

numeric_rows = []

total = df.count()

for c in numeric_cols:

    stats = (
        df
        .agg(
            F.count(F.col(c)).alias("N_VALIDOS"),
            F.sum(
                F.col(c).isNull().cast("int")
            ).alias("N_NULOS"),
            F.mean(c).alias("MEDIA"),
            F.stddev(c).alias("STDDEV")
        )
        .first()
    )

    q = df.approxQuantile(
        c,
        [0.00, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00],
        0.01
    )

    if len(q) == 7:
        minimo, p25, p50, p75, p95, p99, maximo = q
    else:
        minimo = p25 = p50 = p75 = None
        p95 = p99 = maximo = None

    n_nulos = stats["N_NULOS"] or 0

    numeric_rows.append(
        (
            c,
            int(stats["N_VALIDOS"] or 0),
            int(n_nulos),
            float(n_nulos / total * 100.0),
            float(stats["MEDIA"]) if stats["MEDIA"] is not None else None,
            float(stats["STDDEV"]) if stats["STDDEV"] is not None else None,
            minimo,
            p25,
            p50,
            p75,
            p95,
            p99,
            maximo
        )
    )

numeric_summary = spark.createDataFrame(
    numeric_rows,
    [
        "VARIABLE",
        "N_VALIDOS",
        "N_NULOS",
        "PCT_NULOS",
        "MEDIA",
        "STDDEV",
        "MIN",
        "P25",
        "P50",
        "P75",
        "P95",
        "P99",
        "MAX"
    ]
)

print("\n======================================")
print("RESUMEN NUMERICO")
print("======================================")

numeric_summary.show(
    100,
    truncate=False
)

# =========================================================
# 4. Drift TRAIN vs TEST
# Standardized Mean Difference (SMD)
# =========================================================

train = df.filter(
    F.col("SPLIT") == "TRAIN"
)

test = df.filter(
    F.col("SPLIT") == "TEST"
)

n_train = train.count()
n_test = test.count()

drift_rows = []

for c in numeric_cols:

    tr = (
        train
        .agg(
            F.mean(c).alias("MEAN"),
            F.stddev(c).alias("STD"),
            F.sum(
                F.col(c).isNull().cast("int")
            ).alias("NULLS")
        )
        .first()
    )

    te = (
        test
        .agg(
            F.mean(c).alias("MEAN"),
            F.stddev(c).alias("STD"),
            F.sum(
                F.col(c).isNull().cast("int")
            ).alias("NULLS")
        )
        .first()
    )

    mean_tr = tr["MEAN"]
    mean_te = te["MEAN"]

    std_tr = tr["STD"]
    std_te = te["STD"]

    smd = None

    if (
        mean_tr is not None
        and mean_te is not None
        and std_tr is not None
        and std_te is not None
    ):

        pooled = math.sqrt(
            (
                float(std_tr) ** 2
                +
                float(std_te) ** 2
            ) / 2.0
        )

        if pooled > 0:
            smd = abs(
                float(mean_tr)
                -
                float(mean_te)
            ) / pooled
        else:
            smd = 0.0

    pct_null_train = (
        (tr["NULLS"] or 0)
        / n_train
        * 100.0
    )

    pct_null_test = (
        (te["NULLS"] or 0)
        / n_test
        * 100.0
    )

    drift_rows.append(
        (
            c,
            float(mean_tr) if mean_tr is not None else None,
            float(mean_te) if mean_te is not None else None,
            float(smd) if smd is not None else None,
            float(pct_null_train),
            float(pct_null_test),
            float(
                abs(
                    pct_null_train
                    -
                    pct_null_test
                )
            )
        )
    )

drift = spark.createDataFrame(
    drift_rows,
    [
        "VARIABLE",
        "MEDIA_TRAIN",
        "MEDIA_TEST",
        "SMD_ABS",
        "PCT_NULL_TRAIN",
        "PCT_NULL_TEST",
        "DIF_NULL_PCT"
    ]
)

print("\n======================================")
print("MAYOR DRIFT TRAIN VS TEST")
print("======================================")

(
    drift
    .orderBy(
        F.desc_nulls_last("SMD_ABS")
    )
    .show(
        100,
        truncate=False
    )
)

# =========================================================
# 5. Categorías y tasa de afectación
# =========================================================

categorical_summary = None

for c in categorical_cols:

    temp = (
        df
        .groupBy(c)
        .agg(
            F.count("*").alias("N"),
            F.sum(
                "AFECTACION_TRAMO"
            ).alias("POSITIVOS"),
            F.avg(
                "AFECTACION_TRAMO"
            ).alias("TASA_AFECTACION")
        )
        .withColumn(
            "VARIABLE",
            F.lit(c)
        )
        .withColumnRenamed(
            c,
            "CATEGORIA"
        )
        .select(
            "VARIABLE",
            F.col("CATEGORIA").cast("string"),
            "N",
            "POSITIVOS",
            "TASA_AFECTACION"
        )
    )

    if categorical_summary is None:
        categorical_summary = temp
    else:
        categorical_summary = (
            categorical_summary
            .unionByName(temp)
        )

print("\n======================================")
print("CATEGORICAS VS TARGET")
print("======================================")

(
    categorical_summary
    .orderBy(
        "VARIABLE",
        F.desc("N")
    )
    .show(
        200,
        truncate=False
    )
)

# =========================================================
# 6. Cobertura de fuentes vs target
# =========================================================

coverage_target = (
    df
    .groupBy(
        "HAS_ACCIDENTES",
        "HAS_FLUJO",
        "HAS_CARGA"
    )
    .agg(
        F.count("*").alias("TRAMOS"),
        F.sum(
            "AFECTACION_TRAMO"
        ).alias("POSITIVOS"),
        F.avg(
            "AFECTACION_TRAMO"
        ).alias("TASA_AFECTACION")
    )
    .orderBy(
        "HAS_ACCIDENTES",
        "HAS_FLUJO",
        "HAS_CARGA"
    )
)

print("\n======================================")
print("COBERTURA VS TARGET")
print("======================================")

coverage_target.show(
    20,
    truncate=False
)

# =========================================================
# 7. Escritura de evidencia EDA
# =========================================================

(
    numeric_summary
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "resumen_numerico/"
    )
)

(
    drift
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "drift_train_test/"
    )
)

(
    categorical_summary
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "categoricas_target/"
    )
)

(
    coverage_target
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "cobertura_target/"
    )
)

print("\n======================================")
print("EDA GOLD GENERADO")
print("======================================")

print("OUTPUT:", OUT)

spark.stop()