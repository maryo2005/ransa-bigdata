from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

spark = (
    SparkSession.builder
    .appName("tune_logistic_regression_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# RUTAS
# ============================================================

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_tuning_fit_validation_2026/"
)

OUT_TUNING = (
    "gs://ransa-big-data-2026-lake/"
    "gold/model_metrics/logistic_regression_tuning_2026/"
)

OUT_THRESHOLDS = (
    "gs://ransa-big-data-2026-lake/"
    "gold/model_metrics/logistic_regression_thresholds_2026/"
)

MODEL_OUT = (
    "gs://ransa-big-data-2026-lake/"
    "models/tuning/logistic_regression_fit_2026/"
)

# ============================================================
# 1. CARGAR FIT Y VALIDATION
# ============================================================

df = spark.read.parquet(INPUT)

fit_df = (
    df
    .filter(F.col("PARTITION") == "FIT")
    .cache()
)

val_df = (
    df
    .filter(F.col("PARTITION") == "VALIDATION")
    .cache()
)

n_fit = fit_df.count()
n_val = val_df.count()

print("\n======================================")
print("DATOS LOGISTIC REGRESSION")
print("======================================")
print("FIT:", n_fit)
print("VALIDATION:", n_val)

if n_fit != 4501:
    raise Exception(
        f"ERROR: FIT esperado=4501, obtenido={n_fit}"
    )

if n_val != 1126:
    raise Exception(
        f"ERROR: VALIDATION esperado=1126, obtenido={n_val}"
    )

# ============================================================
# 2. EVALUADORES
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

# ============================================================
# 3. FUNCION MATRIZ DE CONFUSION
# ============================================================

def metricas_clasificacion(df_pred, prediction_col="prediction"):

    r = (
        df_pred
        .agg(
            F.sum(
                F.when(
                    (F.col("label") == 1.0)
                    & (F.col(prediction_col) == 1.0),
                    1
                ).otherwise(0)
            ).alias("TP"),

            F.sum(
                F.when(
                    (F.col("label") == 0.0)
                    & (F.col(prediction_col) == 1.0),
                    1
                ).otherwise(0)
            ).alias("FP"),

            F.sum(
                F.when(
                    (F.col("label") == 0.0)
                    & (F.col(prediction_col) == 0.0),
                    1
                ).otherwise(0)
            ).alias("TN"),

            F.sum(
                F.when(
                    (F.col("label") == 1.0)
                    & (F.col(prediction_col) == 0.0),
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

    total = tp + fp + tn + fn

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
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "PRECISION": precision,
        "RECALL": recall,
        "F1": f1,
        "ACCURACY": accuracy,
    }

# ============================================================
# 4. GRID DE HIPERPARAMETROS
# ============================================================

REG_PARAMS = [
    0.0,
    0.01,
    0.05,
    0.10
]

ELASTIC_PARAMS = [
    0.0,
    0.5,
    1.0
]

resultados = []
modelos = {}

config_id = 0

print("\n======================================")
print("INICIO TUNING")
print("======================================")

for reg in REG_PARAMS:
    for elastic in ELASTIC_PARAMS:

        config_id += 1

        print(
            f"\nCONFIG {config_id}: "
            f"regParam={reg}, "
            f"elasticNetParam={elastic}"
        )

        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="CLASS_WEIGHT_TUNING",
            regParam=reg,
            elasticNetParam=elastic,
            maxIter=100,
            tol=1e-6,
            standardization=True
        )

        model = lr.fit(fit_df)

        pred = model.transform(val_df).cache()

        pr_auc = float(
            pr_evaluator.evaluate(pred)
        )

        roc_auc = float(
            roc_evaluator.evaluate(pred)
        )

        m = metricas_clasificacion(pred)

        resultado = {
            "CONFIG_ID": int(config_id),
            "REG_PARAM": float(reg),
            "ELASTIC_NET_PARAM": float(elastic),
            "PR_AUC": pr_auc,
            "ROC_AUC": roc_auc,
            "ACCURACY_05": float(m["ACCURACY"]),
            "PRECISION_05": float(m["PRECISION"]),
            "RECALL_05": float(m["RECALL"]),
            "F1_05": float(m["F1"]),
            "TP_05": int(m["TP"]),
            "FP_05": int(m["FP"]),
            "TN_05": int(m["TN"]),
            "FN_05": int(m["FN"])
        }

        resultados.append(resultado)
        modelos[config_id] = model

        print(
            "PR-AUC=",
            round(pr_auc, 6),
            "ROC-AUC=",
            round(roc_auc, 6),
            "F1@0.5=",
            round(m["F1"], 6),
            "Recall@0.5=",
            round(m["RECALL"], 6)
        )

        pred.unpersist()

# ============================================================
# 5. SELECCIONAR MEJOR HIPERPARAMETRO
#    Principal: PR-AUC
#    Desempate: F1 y Recall
# ============================================================

mejor = max(
    resultados,
    key=lambda x: (
        x["PR_AUC"],
        x["F1_05"],
        x["RECALL_05"]
    )
)

best_id = mejor["CONFIG_ID"]
best_model = modelos[best_id]

print("\n======================================")
print("MEJOR CONFIGURACION POR VALIDATION")
print("======================================")

for k, v in mejor.items():
    print(k, "=", v)

# ============================================================
# 6. AJUSTAR THRESHOLD SOLO EN VALIDATION
# ============================================================

best_pred = (
    best_model
    .transform(val_df)
    .withColumn(
        "SCORE_POSITIVO",
        vector_to_array("probability")[1]
    )
    .cache()
)

thresholds = [
    round(x / 100.0, 2)
    for x in range(10, 91, 5)
]

threshold_results = []

print("\n======================================")
print("TUNING DE THRESHOLD")
print("======================================")

for threshold in thresholds:

    pred_t = (
        best_pred
        .withColumn(
            "PRED_THRESHOLD",
            F.when(
                F.col("SCORE_POSITIVO")
                >= F.lit(threshold),
                F.lit(1.0)
            ).otherwise(F.lit(0.0))
        )
    )

    m = metricas_clasificacion(
        pred_t,
        "PRED_THRESHOLD"
    )

    row = {
        "CONFIG_ID": int(best_id),
        "THRESHOLD": float(threshold),
        "ACCURACY": float(m["ACCURACY"]),
        "PRECISION": float(m["PRECISION"]),
        "RECALL": float(m["RECALL"]),
        "F1": float(m["F1"]),
        "TP": int(m["TP"]),
        "FP": int(m["FP"]),
        "TN": int(m["TN"]),
        "FN": int(m["FN"])
    }

    threshold_results.append(row)

    print(
        "THRESHOLD=",
        threshold,
        "F1=",
        round(m["F1"], 6),
        "PRECISION=",
        round(m["PRECISION"], 6),
        "RECALL=",
        round(m["RECALL"], 6)
    )

# ============================================================
# 7. MEJOR THRESHOLD
#    Principal: F1
#    Desempate: Recall y Precision
# ============================================================

best_threshold = max(
    threshold_results,
    key=lambda x: (
        x["F1"],
        x["RECALL"],
        x["PRECISION"]
    )
)

print("\n======================================")
print("MEJOR THRESHOLD VALIDATION")
print("======================================")

for k, v in best_threshold.items():
    print(k, "=", v)

# ============================================================
# 8. GUARDAR RESULTADOS
# ============================================================

tuning_df = spark.createDataFrame(resultados)

threshold_df = spark.createDataFrame(
    threshold_results
)

(
    tuning_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUT_TUNING)
)

(
    threshold_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(OUT_THRESHOLDS)
)

best_model.write().overwrite().save(MODEL_OUT)

print("\n======================================")
print("LOGISTIC REGRESSION TUNING COMPLETO")
print("======================================")
print("MEJOR CONFIG ID:", best_id)
print("REG PARAM:", mejor["REG_PARAM"])
print(
    "ELASTIC NET:",
    mejor["ELASTIC_NET_PARAM"]
)
print("VALIDATION PR-AUC:", mejor["PR_AUC"])
print("VALIDATION ROC-AUC:", mejor["ROC_AUC"])
print(
    "MEJOR THRESHOLD:",
    best_threshold["THRESHOLD"]
)
print(
    "F1 THRESHOLD:",
    best_threshold["F1"]
)
print(
    "RECALL THRESHOLD:",
    best_threshold["RECALL"]
)
print(
    "PRECISION THRESHOLD:",
    best_threshold["PRECISION"]
)
print("TEST UTILIZADO: NO")
print("======================================")

best_pred.unpersist()
fit_df.unpersist()
val_df.unpersist()

spark.stop()