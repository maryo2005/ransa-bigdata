from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.stat import Correlation
import math

spark = (
    SparkSession.builder
    .appName("audit_features_train_only")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026_v2/"
)

OUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/audit_features_train_2026/"
)

df = spark.read.parquet(INPUT)

train = df.filter(
    F.col("SPLIT") == "TRAIN"
)

test = df.filter(
    F.col("SPLIT") == "TEST"
)

# =========================================================
# 1. Variables numéricas candidatas
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
    "PROM_IMD_2025",
    "PROM_IMD_PESADOS_2025",
    "PCT_PESADOS_2025",

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
    "CLASIFICACION_EJE"
]

print("\n======================================")
print("BASE TRAIN")
print("======================================")

print("TRAIN:", train.count())
print("TEST:", test.count())

print(
    "RUTAS TRAIN:",
    train.select("RUTA_JOIN").distinct().count()
)

print(
    "RUTAS TEST:",
    test.select("RUTA_JOIN").distinct().count()
)

# =========================================================
# 2. Varianza / cardinalidad TRAIN
# =========================================================

print("\n======================================")
print("VARIANZA NUMERICA TRAIN")
print("======================================")

variance_rows = []

for c in numeric_cols:

    r = (
        train
        .agg(
            F.count(F.col(c)).alias("VALIDOS"),
            F.stddev(c).alias("STDDEV"),
            F.countDistinct(c).alias("DISTINTOS")
        )
        .first()
    )

    variance_rows.append(
        (
            c,
            int(r["VALIDOS"] or 0),
            int(r["DISTINTOS"] or 0),
            float(r["STDDEV"])
            if r["STDDEV"] is not None
            else None
        )
    )

variance_df = spark.createDataFrame(
    variance_rows,
    [
        "VARIABLE",
        "N_VALIDOS",
        "N_DISTINTOS",
        "STDDEV"
    ]
)

variance_df.orderBy(
    "N_DISTINTOS",
    "VARIABLE"
).show(100, truncate=False)

# =========================================================
# 3. Imputación SOLO ajustada con TRAIN
# Se usa únicamente para poder calcular correlaciones.
# No estamos materializando aún el pipeline final.
# =========================================================

imputed_cols = [
    c + "__IMP"
    for c in numeric_cols
]

imputer = Imputer(
    strategy="median",
    inputCols=numeric_cols,
    outputCols=imputed_cols
)

imputer_model = imputer.fit(train)

train_imp = imputer_model.transform(train)

# =========================================================
# 4. Correlación entre predictores
# =========================================================

assembler = VectorAssembler(
    inputCols=imputed_cols,
    outputCol="features_audit"
)

vector_df = assembler.transform(
    train_imp
).select("features_audit")

matrix = (
    Correlation
    .corr(
        vector_df,
        "features_audit",
        "pearson"
    )
    .head()[0]
    .toArray()
)

high_corr = []

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):

        value = matrix[i][j]

        if (
            not math.isnan(value)
            and abs(value) >= 0.85
        ):
            high_corr.append(
                (
                    numeric_cols[i],
                    numeric_cols[j],
                    float(value),
                    float(abs(value))
                )
            )

high_corr.sort(
    key=lambda x: x[3],
    reverse=True
)

print("\n======================================")
print("CORRELACIONES ALTAS |R| >= 0.85")
print("======================================")

if high_corr:

    high_corr_df = spark.createDataFrame(
        high_corr,
        [
            "VARIABLE_1",
            "VARIABLE_2",
            "CORRELACION",
            "ABS_CORRELACION"
        ]
    )

    high_corr_df.show(
        200,
        truncate=False
    )

else:

    high_corr_df = spark.createDataFrame(
        [],
        """
        VARIABLE_1 string,
        VARIABLE_2 string,
        CORRELACION double,
        ABS_CORRELACION double
        """
    )

    print("No se encontraron pares >= 0.85")

# =========================================================
# 5. Correlación numérica con TARGET
# SOLO TRAIN
# =========================================================

target_corr_rows = []

for c, imp in zip(
    numeric_cols,
    imputed_cols
):

    value = (
        train_imp
        .select(
            F.corr(
                F.col(imp),
                F.col("AFECTACION_TRAMO")
            ).alias("CORR")
        )
        .first()["CORR"]
    )

    target_corr_rows.append(
        (
            c,
            float(value)
            if value is not None
            and not math.isnan(value)
            else None,
            abs(float(value))
            if value is not None
            and not math.isnan(value)
            else None
        )
    )

target_corr_df = spark.createDataFrame(
    target_corr_rows,
    [
        "VARIABLE",
        "CORR_TARGET_TRAIN",
        "ABS_CORR_TARGET_TRAIN"
    ]
)

print("\n======================================")
print("CORRELACION CON TARGET - SOLO TRAIN")
print("======================================")

(
    target_corr_df
    .orderBy(
        F.desc_nulls_last(
            "ABS_CORR_TARGET_TRAIN"
        )
    )
    .show(
        100,
        truncate=False
    )
)

# =========================================================
# 6. Categóricas SOLO TRAIN
# =========================================================

print("\n======================================")
print("CARDINALIDAD CATEGORICAS TRAIN")
print("======================================")

for c in categorical_cols:

    print("\nVARIABLE:", c)

    (
        train
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
        .orderBy(
            F.desc("N")
        )
        .show(
            100,
            truncate=False
        )
    )

# =========================================================
# 7. Niveles de TEST no vistos en TRAIN
# No se usa target de TEST.
# =========================================================

print("\n======================================")
print("CATEGORIAS TEST NO VISTAS EN TRAIN")
print("======================================")

unseen_rows = []

for c in categorical_cols:

    train_vals = (
        train
        .select(c)
        .distinct()
    )

    test_vals = (
        test
        .select(c)
        .distinct()
    )

    unseen = (
        test_vals
        .join(
            train_vals,
            c,
            "left_anti"
        )
    )

    n = unseen.count()

    print(c, ":", n)

    if n > 0:
        unseen.show(
            100,
            truncate=False
        )

    unseen_rows.append(
        (
            c,
            n
        )
    )

unseen_df = spark.createDataFrame(
    unseen_rows,
    [
        "VARIABLE",
        "N_CATEGORIAS_TEST_NO_VISTAS"
    ]
)

# =========================================================
# 8. Guardar evidencia
# =========================================================

(
    variance_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "varianza/"
    )
)

(
    high_corr_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "correlaciones_altas/"
    )
)

(
    target_corr_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "correlacion_target_train/"
    )
)

(
    unseen_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(
        OUT + "categorias_test_no_vistas/"
    )
)

print("\n======================================")
print("AUDITORIA TRAIN GENERADA")
print("======================================")

print("OUTPUT:", OUT)

spark.stop()