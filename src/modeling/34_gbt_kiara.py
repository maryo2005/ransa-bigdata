import json
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

TUNING_PATH = "gs://ransa-big-data-2026-lake/gold/dataset_tuning_fit_validation_2026/"
FINAL_PATH = "gs://ransa-big-data-2026-lake/gold/dataset_modelado_final_2026/"
OUT_GCS = "gs://ransa-big-data-2026-lake/models/kiara"


def evaluate_predictions(pred_df, label_col="label", raw_col="rawPrediction", pred_col="prediction"):
    pr_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderROC")

    pr_auc = float(pr_eval.evaluate(pred_df))
    roc_auc = float(roc_eval.evaluate(pred_df))

    tp = pred_df.filter((col(label_col) == 1.0) & (col(pred_col) == 1.0)).count()
    tn = pred_df.filter((col(label_col) == 0.0) & (col(pred_col) == 0.0)).count()
    fp = pred_df.filter((col(label_col) == 0.0) & (col(pred_col) == 1.0)).count()
    fn = pred_df.filter((col(label_col) == 1.0) & (col(pred_col) == 0.0)).count()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main():
    spark = SparkSession.builder.appName("gbt-kiara-ransa").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # 1. TUNING: FIT para entrenar, VALIDATION para elegir hiperparámetros (TEST prohibido)
    df_tuning = spark.read.parquet(TUNING_PATH).select(
        col("PARTITION"),
        col("features"),
        col("label").cast("double").alias("label"),
        col("CLASS_WEIGHT_TUNING").cast("double").alias("weight"),
    ).cache()

    fit_df = df_tuning.filter(col("PARTITION") == "FIT").drop("PARTITION")
    val_df = df_tuning.filter(col("PARTITION") == "VALIDATION").drop("PARTITION")

    candidates = [
        {"maxIter": 30, "maxDepth": 4, "stepSize": 0.1, "subsamplingRate": 0.8, "seed": 42},
        {"maxIter": 50, "maxDepth": 5, "stepSize": 0.1, "subsamplingRate": 0.8, "seed": 42},
        {"maxIter": 50, "maxDepth": 6, "stepSize": 0.05, "subsamplingRate": 0.8, "seed": 42},
        {"maxIter": 80, "maxDepth": 5, "stepSize": 0.05, "subsamplingRate": 0.8, "seed": 42},
    ]

    tuning_results = []
    best_cfg = None
    best_val_pr_auc = -1.0

    for cfg in candidates:
        t0 = time.perf_counter()
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="weight",
            maxIter=cfg["maxIter"],
            maxDepth=cfg["maxDepth"],
            stepSize=cfg["stepSize"],
            subsamplingRate=cfg["subsamplingRate"],
            seed=cfg["seed"],
        )
        model = gbt.fit(fit_df)
        val_preds = model.transform(val_df)
        metrics = evaluate_predictions(val_preds)
        elapsed = time.perf_counter() - t0

        record = {**cfg, **metrics, "train_val_seconds": round(elapsed, 2)}
        tuning_results.append(record)
        print("VALIDATION CANDIDATE:", json.dumps(record))

        if metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = metrics["pr_auc"]
            best_cfg = cfg

    df_tuning.unpersist()

    # 2. ENTRENAMIENTO FINAL EN TRAIN Y EVALUACIÓN EN TEST (UNA SOLA VEZ)
    df_final = spark.read.parquet(FINAL_PATH).select(
        col("ID_TRAMO"),
        col("RUTA_JOIN"),
        col("SPLIT"),
        col("features"),
        col("label").cast("double").alias("label"),
        col("CLASS_WEIGHT").cast("double").alias("weight"),
    ).cache()

    train_df = df_final.filter(col("SPLIT") == "TRAIN").drop("SPLIT")
    test_df = df_final.filter(col("SPLIT") == "TEST").drop("SPLIT")

    t_final0 = time.perf_counter()
    final_gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="weight",
        maxIter=best_cfg["maxIter"],
        maxDepth=best_cfg["maxDepth"],
        stepSize=best_cfg["stepSize"],
        subsamplingRate=best_cfg["subsamplingRate"],
        seed=best_cfg["seed"],
    )
    final_model = final_gbt.fit(train_df)
    final_train_seconds = round(time.perf_counter() - t_final0, 2)

    test_preds = final_model.transform(test_df).cache()
    test_metrics = evaluate_predictions(test_preds)

    # Baseline común: predecir siempre clase 0 en TEST
    baseline_preds = test_preds.withColumn("prediction", when(col("label") >= 0, 0.0).otherwise(0.0))
    baseline_metrics = evaluate_predictions(baseline_preds)

    final_model.write().overwrite().save(f"{OUT_GCS}/gbt_model_final")
    test_preds.select("ID_TRAMO", "RUTA_JOIN", "label", "prediction", "probability").write.mode("overwrite").parquet(
        f"{OUT_GCS}/test_predictions"
    )

    summary = {
        "model_family": "GBTClassifier",
        "owner": "kiara",
        "best_config": best_cfg,
        "final_train_seconds": final_train_seconds,
        "tuning_validation_results": tuning_results,
        "test_metrics": test_metrics,
        "baseline_always_0_test_metrics": baseline_metrics,
    }

    report_df = spark.createDataFrame([(json.dumps(summary, indent=2),)], ["report_json"])
    report_df.coalesce(1).write.mode("overwrite").text(f"{OUT_GCS}/metrics_report")

    print("=== FINAL REPORT ===")
    print(json.dumps(summary, indent=2))

    test_preds.unpersist()
    df_final.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
