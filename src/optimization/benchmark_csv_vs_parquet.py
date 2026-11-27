import json
import statistics
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BUCKET = "ransa-big-data-2026-lake"
PARQUET_PATH = f"gs://{BUCKET}/silver/transporte_carga/"
CSV_PATH = f"gs://{BUCKET}/temp/bench_transporte_carga_csv/"
RESULTS_PATH = f"gs://{BUCKET}/temp/bench_results.json"
RUNS = 3


def ensure_csv_exists(spark: SparkSession) -> None:
    df_parquet = spark.read.parquet(PARQUET_PATH)
    jvm = spark.sparkContext._jvm
    conf = spark._jsc.hadoopConfiguration()
    hadoop_path = jvm.org.apache.hadoop.fs.Path(CSV_PATH)
    fs = hadoop_path.getFileSystem(conf)
    if not fs.exists(hadoop_path):
        df_parquet.write.mode("overwrite").option("header", True).csv(CSV_PATH)


def run_single_trial(df) -> dict:
    start = time.perf_counter()
    cols = df.columns
    selected_cols = cols[: min(4, len(cols))]
    group_col = selected_cols[0]
    agg_df = (
        df.select(*selected_cols)
        .filter(F.col(group_col).isNotNull())
        .groupBy(group_col)
        .agg(F.count("*").alias("total_filas"))
    )
    grouped_rows = agg_df.count()
    elapsed_sec = round(time.perf_counter() - start, 4)
    return {
        "elapsed_sec": elapsed_sec,
        "grouped_rows": grouped_rows,
    }


def benchmark_format(spark: SparkSession, fmt: str, path: str) -> dict:
    trials = []
    for i in range(1, RUNS + 1):
        if fmt == "parquet":
            df = spark.read.parquet(path)
        else:
            df = spark.read.option("header", True).option("inferSchema", False).csv(path)
        metrics = run_single_trial(df)
        metrics["run"] = i
        trials.append(metrics)
        print(f"[{fmt.upper()}] Corrida {i}/{RUNS}: {metrics['elapsed_sec']} s (grupos={metrics['grouped_rows']})")

    durations = [t["elapsed_sec"] for t in trials]
    median_sec = round(statistics.median(durations), 4)
    return {
        "format": fmt,
        "path": path,
        "runs": trials,
        "median_sec": median_sec,
    }


def main() -> None:
    spark = SparkSession.builder.appName("RansaBenchmarkCSVvsParquet").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    ensure_csv_exists(spark)

    csv_summary = benchmark_format(spark, "csv", CSV_PATH)
    parquet_summary = benchmark_format(spark, "parquet", PARQUET_PATH)

    report = {
        "dataset": "silver/transporte_carga",
        "runs_per_format": RUNS,
        "csv": csv_summary,
        "parquet_snappy": parquet_summary,
        "median_diff_sec": round(csv_summary["median_sec"] - parquet_summary["median_sec"], 4),
    }

    print("\n=== RESUMEN MEDIANAS (CSV vs PARQUET+SNAPPY) ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    summary_rdd = spark.sparkContext.parallelize([json.dumps(report, ensure_ascii=False)])
    summary_rdd.coalesce(1).saveAsTextFile(RESULTS_PATH.replace(".json", "_out"))

    spark.stop()


if __name__ == "__main__":
    main()
