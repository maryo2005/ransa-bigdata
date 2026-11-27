from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

spark = (
    SparkSession.builder
    .appName("fit_final_logistic_and_test_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# CONFIGURACION CONGELADA DESDE VALIDATION
# ============================================================

REG_PARAM = 0.01
ELASTIC_NET_PARAM = 0.0
THRESHOLD = 0.65

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_modelado_final_2026/"
)

MODEL_OUT = (
    "gs://ransa-big-data-2026-lake/"
    "models/final/logistic_regression_2026/"
)

METRICS_OUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/model_metrics/logistic_regression_test_2026/"
)

PREDICTIONS_OUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/predictions/logistic_regression_test_2026/"
)

# ============================================================
# 1. CARGAR DATASET FINAL
# ============================================================

df = spark.read.parquet(INPUT)

train = (
    df
    .filter(F.col("SPLIT") == "TRAIN")
    .cache()
)

test = (
    df
    .filter(F.col("SPLIT") == "TEST")
    .cache()
)

n_train = train.count()
n_test = test.count()

train_pos = (
    train
    .filter(F.col("label") == 1.0)
    .count()
)

test_pos = (
    test
    .filter(F.col("label") == 1.0)
    .count()
)

print("\n======================================")
print("DATASET FINAL")
print("======================================")
print("TRAIN:", n_train)
print("TRAIN POSITIVOS:", train_pos)
print("TEST:", n_test)
print("TEST POSITIVOS:", test_pos)

if n_train != 5627:
    raise Exception(
        f"ERROR TRAIN esperado=5627, obtenido={n_train}"
    )

if n_test != 1431:
    raise Exception(
        f"ERROR TEST esperado=1431, obtenido={n_test}"
    )

if test_pos != 145:
    raise Exception(
        f"ERROR TEST positivos esperado=145, obtenido={test_pos}"
    )

# ============================================================
# 2. ENTRENAR MODELO FINAL
# ============================================================

print("\n======================================")
print("ENTRENANDO LOGISTIC REGRESSION FINAL")
print("======================================")
print("REG_PARAM:", REG_PARAM)
print("ELASTIC_NET_PARAM:", ELASTIC_NET_PARAM)
print("THRESHOLD:", THRESHOLD)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="CLASS_WEIGHT",
    regParam=REG_PARAM,
    elasticNetParam=ELASTIC_NET_PARAM,
    maxIter=100,
    tol=1e-6,
    standardization=True,
    threshold=THRESHOLD
)

model = lr.fit(train)

# ============================================================
# 3. EVALUACION FINAL EN TEST
# ============================================================

pred = (
    model
    .transform(test)
    .withColumn(
        "SCORE_RIESGO",
        vector_to_array("probability")[1]
    )
    .cache()
)

# ============================================================
# 4. PR-AUC Y ROC-AUC
# ============================================================

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_auc = float(
    pr_evaluator.evaluate(pred)
)

roc_auc = float(
    roc_evaluator.evaluate(pred)
)

# ============================================================
# 5. MATRIZ DE CONFUSION
# ============================================================

r = (
    pred
    .agg(
        F.sum(
            F.when(
                (F.col("label") == 1.0)
                & (F.col("prediction") == 1.0),
                1
            ).otherwise(0)
        ).alias("TP"),

        F.sum(
            F.when(
                (F.col("label") == 0.0)
                & (F.col("prediction") == 1.0),
                1
            ).otherwise(0)
        ).alias("FP"),

        F.sum(
            F.when(
                (F.col("label") == 0.0)
                & (F.col("prediction") == 0.0),
                1
            ).otherwise(0)
        ).alias("TN"),

        F.sum(
            F.when(
                (F.col("label") == 1.0)
                & (F.col("prediction") == 0.0),
                1
            ).otherwise(0)
        ).alias("FN")
    )
    .first()
)

tp = int(r["TP"] or 0)
fp = int(r["FP"] or 0)
tn = int(r["TN"] or 0)
fn = int(r["FN"] or 0)

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
    (tp + tn) / (tp + fp + tn + fn)
)

# ============================================================
# 6. BASELINE MAYORITARIO
#    Predice siempre clase 0
# ============================================================

baseline_tn = n_test - test_pos
baseline_fn = test_pos
baseline_tp = 0
baseline_fp = 0

baseline_accuracy = baseline_tn / n_test
baseline_precision = 0.0
baseline_recall = 0.0
baseline_f1 = 0.0

# ============================================================
# 7. MOSTRAR RESULTADOS
# ============================================================

print("\n======================================")
print("LOGISTIC REGRESSION - TEST FINAL")
print("======================================")
print("THRESHOLD:", THRESHOLD)
print("PR-AUC:", pr_auc)
print("ROC-AUC:", roc_auc)
print("ACCURACY:", accuracy)
print("PRECISION:", precision)
print("RECALL:", recall)
print("F1:", f1)

print("\nMATRIZ DE CONFUSION")
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)

print("\n======================================")
print("BASELINE MAYORITARIO - TEST")
print("======================================")
print("Prediccion: todos los tramos = 0")
print("ACCURACY:", baseline_accuracy)
print("PRECISION:", baseline_precision)
print("RECALL:", baseline_recall)
print("F1:", baseline_f1)
print("TP:", baseline_tp)
print("FP:", baseline_fp)
print("TN:", baseline_tn)
print("FN:", baseline_fn)

# ============================================================
# 8. GUARDAR METRICAS
# ============================================================

metrics_rows = [
    (
        "LOGISTIC_REGRESSION",
        float(REG_PARAM),
        float(ELASTIC_NET_PARAM),
        float(THRESHOLD),
        float(pr_auc),
        float(roc_auc),
        float(accuracy),
        float(precision),
        float(recall),
        float(f1),
        int(tp),
        int(fp),
        int(tn),
        int(fn)
    ),
    (
        "BASELINE_MAYORITARIO",
        None,
        None,
        None,
        None,
        None,
        float(baseline_accuracy),
        float(baseline_precision),
        float(baseline_recall),
        float(baseline_f1),
        int(baseline_tp),
        int(baseline_fp),
        int(baseline_tn),
        int(baseline_fn)
    )
]

schema = """
MODELO string,
REG_PARAM double,
ELASTIC_NET_PARAM double,
THRESHOLD double,
PR_AUC double,
ROC_AUC double,
ACCURACY double,
PRECISION double,
RECALL double,
F1 double,
TP long,
FP long,
TN long,
FN long
"""

metrics_df = spark.createDataFrame(
    metrics_rows,
    schema=schema
)

(
    metrics_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(METRICS_OUT)
)

# ============================================================
# 9. GUARDAR PREDICCIONES TEST
# ============================================================

pred_output = (
    pred
    .select(
        "ID_TRAMO",
        "RUTA_JOIN",
        "AFECTACION_TRAMO",
        "label",
        "SCORE_RIESGO",
        F.col("prediction")
        .cast("int")
        .alias("PREDICCION_FINAL")
    )
)

(
    pred_output
    .repartition(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(PREDICTIONS_OUT)
)

# ============================================================
# 10. GUARDAR MODELO
# ============================================================

model.write().overwrite().save(MODEL_OUT)

print("\n======================================")
print("MODELO FINAL GUARDADO")
print("======================================")
print("MODELO:", MODEL_OUT)
print("METRICAS:", METRICS_OUT)
print("PREDICCIONES:", PREDICTIONS_OUT)
print("TRAIN USADO:", 5627)
print("TEST USADO:", 1431)
print("TEST EVALUADO: SI - EVALUACION FINAL")
print("======================================")

pred.unpersist()
train.unpersist()
test.unpersist()

spark.stop()