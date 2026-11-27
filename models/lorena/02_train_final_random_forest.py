from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ============================================================
# CONFIGURACION
# ============================================================

spark = (
    SparkSession.builder
    .appName("lorena_random_forest_final")
    .getOrCreate()
)

DATA_PATH = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_final_2026/"
)

OUTPUT_PATH = (
    "gs://ransa-big-data-2026-lake/"
    "models/lorena/random_forest/"
)

SEED = 2026

# Configuracion ganadora obtenida SOLO con FIT -> VALIDATION
NUM_TREES = 100
MAX_DEPTH = 8
MIN_INSTANCES = 1


print("=" * 80)
print("RANDOM FOREST - MODELO FINAL")
print("Integrante: Lorena Vasquez")
print("=" * 80)

print("\nConfiguracion congelada:")
print("numTrees =", NUM_TREES)
print("maxDepth =", MAX_DEPTH)
print("minInstancesPerNode =", MIN_INSTANCES)


# ============================================================
# LEER DATASET FINAL CONGELADO
# ============================================================

df = spark.read.parquet(DATA_PATH)

print("\nColumnas disponibles:")
print(df.columns)

required_cols = {
    "SPLIT",
    "features",
    "label",
    "CLASS_WEIGHT",
    "ID_TRAMO",
    "RUTA_JOIN",
}

missing = required_cols - set(df.columns)

assert not missing, (
    f"ERROR: faltan columnas requeridas: {missing}"
)


# ============================================================
# PARTICIONES TRAIN / TEST
# ============================================================

print("\nDistribucion SPLIT:")
df.groupBy("SPLIT").count().show()

train_df = (
    df.filter(F.col("SPLIT") == "TRAIN")
    .cache()
)

test_df = (
    df.filter(F.col("SPLIT") == "TEST")
    .cache()
)

n_train = train_df.count()
n_test = test_df.count()

train_pos = train_df.filter(F.col("label") == 1).count()
test_pos = test_df.filter(F.col("label") == 1).count()

print("\nTAMANOS CONGELADOS")
print("TRAIN:", n_train)
print("TRAIN positivos:", train_pos)
print("TEST:", n_test)
print("TEST positivos:", test_pos)

assert n_train == 5627, (
    f"TRAIN esperado=5627, obtenido={n_train}"
)

assert n_test == 1431, (
    f"TEST esperado=1431, obtenido={n_test}"
)

assert test_pos == 145, (
    f"TEST positivos esperado=145, obtenido={test_pos}"
)

print("\nContrato final: OK")
print("No se realiza tuning adicional.")
print("TEST se evaluara una sola vez.")


# ============================================================
# ENTRENAR RANDOM FOREST FINAL
# ============================================================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="CLASS_WEIGHT",
    numTrees=NUM_TREES,
    maxDepth=MAX_DEPTH,
    minInstancesPerNode=MIN_INSTANCES,
    seed=SEED,
)

print("\nEntrenando modelo final con TRAIN...")

model = rf.fit(train_df)

print("Entrenamiento completado.")


# ============================================================
# EVALUAR TEST UNA SOLA VEZ
# ============================================================

print("\nEvaluando TEST...")

pred = model.transform(test_df).cache()

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR",
)

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",
)

pr_auc = pr_evaluator.evaluate(pred)
roc_auc = roc_evaluator.evaluate(pred)


# ============================================================
# MATRIZ DE CONFUSION
# ============================================================

cm_rows = (
    pred.groupBy("label", "prediction")
    .count()
    .collect()
)

cm = {
    (float(r["label"]), float(r["prediction"])): int(r["count"])
    for r in cm_rows
}

tn = cm.get((0.0, 0.0), 0)
fp = cm.get((0.0, 1.0), 0)
fn = cm.get((1.0, 0.0), 0)
tp = cm.get((1.0, 1.0), 0)


# ============================================================
# METRICAS
# ============================================================

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)

accuracy = (
    (tp + tn) / (tp + tn + fp + fn)
    if (tp + tn + fp + fn) > 0
    else 0.0
)


print("\n" + "=" * 80)
print("RESULTADOS FINALES - TEST")
print("=" * 80)

print(f"PR-AUC:    {pr_auc:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1:        {f1:.6f}")
print(f"Accuracy:  {accuracy:.6f}")

print("\nMATRIZ DE CONFUSION")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")


# ============================================================
# GUARDAR MODELO
# ============================================================

print("\nGuardando modelo final...")

model.write().overwrite().save(
    OUTPUT_PATH + "final_model/"
)


# ============================================================
# GUARDAR METRICAS
# ============================================================

metrics_df = spark.createDataFrame(
    [
        (
            "RandomForest",
            NUM_TREES,
            MAX_DEPTH,
            MIN_INSTANCES,
            float(pr_auc),
            float(roc_auc),
            float(precision),
            float(recall),
            float(f1),
            float(accuracy),
            int(tn),
            int(fp),
            int(fn),
            int(tp),
        )
    ],
    [
        "model",
        "numTrees",
        "maxDepth",
        "minInstancesPerNode",
        "PR_AUC",
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
)

(
    metrics_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(OUTPUT_PATH + "test_metrics/")
)


# ============================================================
# GUARDAR PREDICCIONES TEST
# ============================================================

(
    pred.select(
        "ID_TRAMO",
        "RUTA_JOIN",
        "label",
        "probability",
        "prediction",
    )
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH + "test_predictions/")
)


print("\n" + "=" * 80)
print("MODELO FINAL COMPLETADO")
print("=" * 80)

print(
    "Modelo: "
    + OUTPUT_PATH
    + "final_model/"
)

print(
    "Metricas: "
    + OUTPUT_PATH
    + "test_metrics/"
)

print(
    "Predicciones: "
    + OUTPUT_PATH
    + "test_predictions/"
)

pred.unpersist()
train_df.unpersist()
test_df.unpersist()

spark.stop()
