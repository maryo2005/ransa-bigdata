from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

spark = (
    SparkSession.builder
    .appName("build_common_ml_features_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026_v2/"
)

OUTPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_final_2026/"
)

MODEL_OUT = (
    "gs://ransa-big-data-2026-lake/"
    "models/preprocesamiento_comun_2026/"
)

# =========================================================
# 1. Variables definitivas
# =========================================================

numeric_cols = [
    "LONGITUD_KM",
    "NRO_CARRILES",
    "FLAG_NRO_CARRILES_FALTANTE",
    "ES_CONCES",

    "HAS_ACCIDENTES",
    "ACCIDENTES_24M_POR_100KM",
    "PROM_AFECTADOS_POR_ACCIDENTE",
    "PROM_VEHICULOS_POR_ACCIDENTE",
    "MESES_CON_ACCIDENTES",

    "HAS_FLUJO",
    "N_PEAJES_RUTA",
    "PROM_IMD_2025",
    "PCT_PESADOS_2025",

    "HAS_CARGA",
    "LOG_FLOTA_CARGA_2025",
    "P95_CARGA_UTIL_2025",
    "MEDIANA_P_BRUTO_2025",
    "PCT_CAMION_2025",
    "PCT_PICKUP_2025"
]

categorical_cols = [
    "SUPERFICIE",
    "ESTADO",
    "CLASIFICACION_EJE"
]

df = spark.read.parquet(INPUT)

train = df.filter(
    F.col("SPLIT") == "TRAIN"
)

test = df.filter(
    F.col("SPLIT") == "TEST"
)

print("\n======================================")
print("ENTRADAS")
print("======================================")

print("TRAIN:", train.count())
print("TEST:", test.count())

# =========================================================
# 2. Imputación
#
# El modelo del Imputer se ajusta SOLO con TRAIN.
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

# =========================================================
# 3. Categóricas
#
# StringIndexer también se ajusta SOLO con TRAIN.
# handleInvalid=keep protege ante categorías futuras.
# =========================================================

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "__IDX",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

indexed_cols = [
    c + "__IDX"
    for c in categorical_cols
]

ohe_cols = [
    c + "__OHE"
    for c in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=indexed_cols,
    outputCols=ohe_cols,
    handleInvalid="keep",
    dropLast=True
)

# =========================================================
# 4. Vector común
# =========================================================

assembler = VectorAssembler(
    inputCols=imputed_cols + ohe_cols,
    outputCol="features",
    handleInvalid="error"
)

pipeline = Pipeline(
    stages=[
        imputer,
        *indexers,
        encoder,
        assembler
    ]
)

# =========================================================
# 5. FIT SOLO CON TRAIN
# =========================================================

pipeline_model = pipeline.fit(train)

train_p = pipeline_model.transform(train)
test_p = pipeline_model.transform(test)

print("\n======================================")
print("IMPUTACION APRENDIDA SOLO EN TRAIN")
print("======================================")

imputer_model = pipeline_model.stages[0]
imputer_model.surrogateDF.show(
    truncate=False
)

# =========================================================
# 6. Pesos de clase
#
# Calculados SOLO con distribución TRAIN.
# =========================================================

n_train = train.count()

n0 = train.filter(
    F.col("AFECTACION_TRAMO") == 0
).count()

n1 = train.filter(
    F.col("AFECTACION_TRAMO") == 1
).count()

w0 = n_train / (2.0 * n0)
w1 = n_train / (2.0 * n1)

print("\n======================================")
print("PESOS DE CLASE")
print("======================================")

print("TRAIN TOTAL:", n_train)
print("CLASE 0:", n0)
print("CLASE 1:", n1)
print("PESO CLASE 0:", w0)
print("PESO CLASE 1:", w1)

def preparar(df_p):

    return (
        df_p
        .withColumn(
            "label",
            F.col(
                "AFECTACION_TRAMO"
            ).cast("double")
        )
        .withColumn(
            "CLASS_WEIGHT",
            F.when(
                F.col("AFECTACION_TRAMO") == 1,
                F.lit(w1)
            ).otherwise(
                F.lit(w0)
            )
        )
        .select(
            "ID_TRAMO",
            "RUTA_JOIN",
            "SPLIT",
            "AFECTACION_TRAMO",
            "label",
            "CLASS_WEIGHT",
            "features"
        )
    )

train_final = preparar(train_p)
test_final = preparar(test_p)

final_df = (
    train_final
    .unionByName(test_final)
)

# =========================================================
# 7. Validaciones
# =========================================================

print("\n======================================")
print("VALIDACION FINAL")
print("======================================")

print("FILAS:", final_df.count())

print(
    "ID_TRAMO UNICOS:",
    final_df.select(
        "ID_TRAMO"
    ).distinct().count()
)

print(
    "FEATURES NULOS:",
    final_df.filter(
        F.col("features").isNull()
    ).count()
)

dimension = (
    final_df
    .select("features")
    .first()["features"]
    .size
)

print(
    "DIMENSION VECTOR FEATURES:",
    dimension
)

(
    final_df
    .groupBy("SPLIT")
    .agg(
        F.count("*").alias("TRAMOS"),
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

if final_df.count() != 7058:
    raise Exception(
        "ERROR: total distinto de 7058"
    )

if (
    final_df
    .select("ID_TRAMO")
    .distinct()
    .count()
    != 7058
):
    raise Exception(
        "ERROR: ID_TRAMO no único"
    )

if (
    final_df
    .filter(
        F.col("features").isNull()
    )
    .count()
    != 0
):
    raise Exception(
        "ERROR: features nulos"
    )

# =========================================================
# 8. Persistencia
# =========================================================

(
    final_df
    .repartition(2)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(OUTPUT)
)

pipeline_model.write().overwrite().save(
    MODEL_OUT
)

print("\n======================================")
print("DATASET ML FINAL GENERADO")
print("======================================")

print("DATASET:", OUTPUT)
print("PIPELINE:", MODEL_OUT)
print("NUMERICAS:", len(numeric_cols))
print("CATEGORICAS:", len(categorical_cols))
print("SEED SPLIT: 2026")

spark.stop()