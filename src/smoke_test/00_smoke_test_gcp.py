from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("ransa_smoke_test_gcp")
    .getOrCreate()
)

df = spark.range(0, 1000000)

print("Cantidad de registros:", df.count())
print("Cantidad de particiones:", df.rdd.getNumPartitions())

output_path = "gs://ransa-big-data-2026-lake/temp/smoke_test"

(
    df.repartition(2)
    .write
    .mode("overwrite")
    .parquet(output_path)
)

print("Salida:", output_path)
print("Resultado: SUCCESS")

spark.stop()