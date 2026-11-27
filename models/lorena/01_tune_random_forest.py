from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ============================================================
# CONFIGURACION
# ============================================================

spark = (
    SparkSession.builder
    .appName("lorena_random_forest_tuning")
    .getOrCreate()
)

INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/dataset_tuning_fit_validation_2026/"
)

OUTPUT = (
    "gs://ransa-big-data-2026-lake/"
    "models/lorena/random_forest/"
)

SEED_RF = 2026

print("=" * 80)
print("RANDOM FOREST - TUNING")
print("Integrante: Lorena Vasquez")
print("=" * 80)


# ============================================================
# LEER DATASET CONGELADO
# ============================================================

df = spark.read.parquet(INPUT)

print("\nColumnas:")
print(df.columns)

print("\nDistribucion PARTITION:")
df.groupBy("PARTITION").count().show()


# ============================================================
# VALIDAR CONTRATO
# ============================================================

required_cols = {
    "PARTITION",
    "features",
    "label",
    "CLASS_WEIGHT_TUNING",
}

missing = required_cols - set(df.columns)

assert not missing, (
    f"ERROR: faltan columnas requeridas: {missing}"
)


# ============================================================
# FIT Y VALIDATION
# ============================================================

fit_df = (
    df.filter(F.col("PARTITION") == "FIT")
    .cache()
)

validation_df = (
    df.filter(F.col("PARTITION") == "VALIDATION")
    .cache()
)

n_fit = fit_df.count()
n_validation = validation_df.count()

fit_pos = fit_df.filter(F.col("label") == 1).count()
val_pos = validation_df.filter(F.col("label") == 1).count()

print("\nTAMANOS CONGELADOS")
print("FIT:", n_fit)
print("FIT POSITIVOS:", fit_pos)
print("VALIDATION:", n_validation)
print("VALIDATION POSITIVOS:", val_pos)

assert n_fit == 4501, (
    f"FIT esperado=4501, obtenido={n_fit}"
)

assert n_validation == 1126, (
    f"VALIDATION esperado=1126, obtenido={n_validation}"
)

print("\nContrato de tuning: OK")
print("TEST NO ES UTILIZADO EN ESTE SCRIPT.")


# ============================================================
# EVALUADORES
# ============================================================

pr_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR",
)

roc_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",
)


def calcular_metricas(pred):
    pr_auc = pr_eval.evaluate(pred)
    roc_auc = roc_eval.evaluate(pred)

    cm = {
        (float(r["label"]), float(r["prediction"])): r["count"]
        for r in pred.groupBy("label", "prediction").count().collect()
    }

    tn = cm.get((0.0, 0.0), 0)
    fp = cm.get((0.0, 1.0), 0)
    fn = cm.get((1.0, 0.0), 0)
    tp = cm.get((1.0, 1.0), 0)

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

    return {
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


# ============================================================
# GRID DE HIPERPARAMETROS
# ============================================================

grid = []

for num_trees in [100, 200]:
    for max_depth in [5, 8]:
        for min_instances in [1, 5]:
            grid.append(
                (num_trees, max_depth, min_instances)
            )

print("\nConfiguraciones a probar:", len(grid))


# ============================================================
# TUNING
# ============================================================

resultados = []

best_pr = -1.0
best_config = None

for num_trees, max_depth, min_instances in grid:

    print("\n" + "-" * 80)

    print(
        f"numTrees={num_trees} | "
        f"maxDepth={max_depth} | "
        f"minInstancesPerNode={min_instances}"
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="CLASS_WEIGHT_TUNING",
        numTrees=num_trees,
        maxDepth=max_depth,
        minInstancesPerNode=min_instances,
        seed=SEED_RF,
    )

    model = rf.fit(fit_df)

    predictions = model.transform(validation_df).cache()

    m = calcular_metricas(predictions)

    print(f"PR-AUC:    {m['PR_AUC']:.6f}")
    print(f"ROC-AUC:   {m['ROC_AUC']:.6f}")
    print(f"Precision: {m['Precision']:.6f}")
    print(f"Recall:    {m['Recall']:.6f}")
    print(f"F1:        {m['F1']:.6f}")
    print(f"Accuracy:  {m['Accuracy']:.6f}")

    print(
        f"CM: TN={m['TN']} FP={m['FP']} "
        f"FN={m['FN']} TP={m['TP']}"
    )

    resultados.append(
        (
            num_trees,
            max_depth,
            min_instances,
            m["PR_AUC"],
            m["ROC_AUC"],
            m["Precision"],
            m["Recall"],
            m["F1"],
            m["Accuracy"],
            m["TP"],
            m["FP"],
            m["TN"],
            m["FN"],
        )
    )

    if m["PR_AUC"] > best_pr:

        best_pr = m["PR_AUC"]

        best_config = {
            "numTrees": num_trees,
            "maxDepth": max_depth,
            "minInstancesPerNode": min_instances,
        }

    predictions.unpersist()


# ============================================================
# TABLA FINAL
# ============================================================

cols = [
    "numTrees",
    "maxDepth",
    "minInstancesPerNode",
    "PR_AUC",
    "ROC_AUC",
    "Precision",
    "Recall",
    "F1",
    "Accuracy",
    "TP",
    "FP",
    "TN",
    "FN",
]

results_df = spark.createDataFrame(
    resultados,
    cols,
)

print("\n" + "=" * 80)
print("RESULTADOS VALIDATION")
print("=" * 80)

results_df.orderBy(
    F.desc("PR_AUC")
).show(
    truncate=False
)


# ============================================================
# GANADOR
# ============================================================

print("\n" + "=" * 80)
print("CONFIGURACION GANADORA")
print("=" * 80)

print(best_config)
print(f"PR-AUC VALIDATION = {best_pr:.6f}")


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

(
    results_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(OUTPUT + "validation_results/")
)

best_df = spark.createDataFrame(
    [
        (
            best_config["numTrees"],
            best_config["maxDepth"],
            best_config["minInstancesPerNode"],
            float(best_pr),
        )
    ],
    [
        "numTrees",
        "maxDepth",
        "minInstancesPerNode",
        "PR_AUC_VALIDATION",
    ]
)

(
    best_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(OUTPUT + "best_config/")
)

print("\nTUNING COMPLETADO")
print("TEST NO UTILIZADO.")

fit_df.unpersist()
validation_df.unpersist()

spark.stop()