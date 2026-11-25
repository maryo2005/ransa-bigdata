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
    .appName("build_tuning_features_fit_only_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# RUTAS
# ============================================================

BASE = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_base_tramos_2026_v2/"
)

PARTITIONS = (
    "gs://ransa-big-data-2026-lake/"
    "gold/particion_rutas_ml_2026/"
)

OUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_tuning_fit_validation_2026/"
)

MODEL_OUT = (
    "gs://ransa-big-data-2026-lake/"
    "models/preprocesamiento_tuning_fit_2026/"
)

# ============================================================
# VARIABLES CONGELADAS
# ============================================================

NUMERIC_COLS = [
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

CATEGORICAL_COLS = [
    "SUPERFICIE",
    "ESTADO",
    "CLASIFICACION_EJE"
]

# ============================================================
# 1. CARGAR DATOS
# ============================================================

base = spark.read.parquet(BASE)
partitions = spark.read.parquet(PARTITIONS)

df = (
    base
    .drop("SPLIT")
    .join(
        partitions,
        "RUTA_JOIN",
        "inner"
    )
)

fit_df = df.filter(F.col("PARTITION") == "FIT")
val_df = df.filter(F.col("PARTITION") == "VALIDATION")

print("\n======================================")
print("ENTRADAS TUNING")
print("======================================")
print("FIT:", fit_df.count())
print("VALIDATION:", val_df.count())
print(
    "FIT RUTAS:",
    fit_df.select("RUTA_JOIN").distinct().count()
)
print(
    "VALIDATION RUTAS:",
    val_df.select("RUTA_JOIN").distinct().count()
)

# ============================================================
# 2. PESOS DE CLASE APRENDIDOS SOLO EN FIT
# ============================================================

fit_total = fit_df.count()

fit_pos = (
    fit_df
    .filter(F.col("AFECTACION_TRAMO") == 1)
    .count()
)

fit_neg = fit_total - fit_pos

w0 = fit_total / (2.0 * fit_neg)
w1 = fit_total / (2.0 * fit_pos)

print("\n======================================")
print("PESOS DE CLASE FIT-ONLY")
print("======================================")
print("FIT TOTAL:", fit_total)
print("FIT NEGATIVOS:", fit_neg)
print("FIT POSITIVOS:", fit_pos)
print("W0:", w0)
print("W1:", w1)

# ============================================================
# 3. IMPUTACION NUMERICA
# ============================================================

IMPUTED_COLS = [
    f"{c}__IMP"
    for c in NUMERIC_COLS
]

imputer = Imputer(
    strategy="median",
    inputCols=NUMERIC_COLS,
    outputCols=IMPUTED_COLS
)

# ============================================================
# 4. INDEXACION CATEGORICA
# ============================================================

indexers = []

for c in CATEGORICAL_COLS:
    indexers.append(
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}__IDX",
            handleInvalid="keep"
        )
    )

INDEXED_COLS = [
    f"{c}__IDX"
    for c in CATEGORICAL_COLS
]

# ============================================================
# 5. ONE HOT ENCODING
# ============================================================

OHE_COLS = [
    f"{c}__OHE"
    for c in CATEGORICAL_COLS
]

encoder = OneHotEncoder(
    inputCols=INDEXED_COLS,
    outputCols=OHE_COLS,
    dropLast=True,
    handleInvalid="keep"
)

# ============================================================
# 6. VECTOR FINAL
# ============================================================

assembler = VectorAssembler(
    inputCols=IMPUTED_COLS + OHE_COLS,
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

# ============================================================
# 7. FIT SOLO CON FIT
# ============================================================

print("\n======================================")
print("AJUSTANDO PREPROCESAMIENTO SOLO EN FIT")
print("======================================")

pipeline_model = pipeline.fit(fit_df)

fit_t = pipeline_model.transform(fit_df)
val_t = pipeline_model.transform(val_df)

# ============================================================
# 8. AGREGAR LABEL Y PESOS
# ============================================================

def preparar(df_in):
    return (
        df_in
        .withColumn(
            "label",
            F.col("AFECTACION_TRAMO").cast("double")
        )
        .withColumn(
            "CLASS_WEIGHT_TUNING",
            F.when(
                F.col("AFECTACION_TRAMO") == 1,
                F.lit(w1)
            ).otherwise(F.lit(w0))
        )
        .select(
            "ID_TRAMO",
            "RUTA_JOIN",
            "PARTITION",
            "AFECTACION_TRAMO",
            "label",
            "CLASS_WEIGHT_TUNING",
            "features"
        )
    )

fit_final = preparar(fit_t)
val_final = preparar(val_t)

final = fit_final.unionByName(val_final)

# ============================================================
# 9. VALIDACIONES
# ============================================================

total = final.count()

ids_unicos = (
    final
    .select("ID_TRAMO")
    .distinct()
    .count()
)

features_null = (
    final
    .filter(F.col("features").isNull())
    .count()
)

print("\n======================================")
print("VALIDACION DATASET TUNING")
print("======================================")

(
    final
    .groupBy("PARTITION")
    .agg(
        F.count("*").alias("N_TRAMOS"),
        F.countDistinct("RUTA_JOIN").alias("N_RUTAS"),
        F.sum("AFECTACION_TRAMO").alias("N_POSITIVOS")
    )
    .orderBy("PARTITION")
    .show(truncate=False)
)

print("TOTAL:", total)
print("ID_TRAMO UNICOS:", ids_unicos)
print("FEATURES NULL:", features_null)

if total != 5627:
    raise Exception(
        f"ERROR: se esperaban 5627 filas FIT+VALIDATION y hay {total}"
    )

if ids_unicos != 5627:
    raise Exception(
        f"ERROR: se esperaban 5627 ID_TRAMO unicos y hay {ids_unicos}"
    )

if features_null != 0:
    raise Exception(
        f"ERROR: existen {features_null} features nulas"
    )

# ============================================================
# 10. DIMENSION VECTOR
# ============================================================

dimension = (
    final
    .select("features")
    .first()[0]
    .size
)

print("\n======================================")
print("VECTOR FEATURES")
print("======================================")
print("DIMENSION:", dimension)

# ============================================================
# 11. GUARDAR
# ============================================================

(
    final
    .repartition(2)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUT)
)

pipeline_model.write().overwrite().save(MODEL_OUT)

print("\n======================================")
print("TUNING DATASET GENERADO")
print("======================================")
print("DATASET:", OUT)
print("PREPROCESAMIENTO:", MODEL_OUT)
print("PREPROCESS FIT: FIT-ONLY")
print("CLASS WEIGHTS: FIT-ONLY")
print("VALIDATION NO USADO PARA AJUSTAR")
print("TEST NO TOCADO")
print("======================================")

spark.stop()