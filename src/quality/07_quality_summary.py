from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("ransa_quality_summary")
    .getOrCreate()
)

BUCKET = "gs://ransa-big-data-2026-lake"

DATASETS = {
    "emergencias": f"{BUCKET}/silver/emergencias/",
    "red_vial": f"{BUCKET}/silver/red_vial/",
    "accidentes": f"{BUCKET}/silver/accidentes/",
    "flujo_peajes": f"{BUCKET}/silver/flujo_peajes/",
    "red_vial_geo": f"{BUCKET}/silver/red_vial_geo/",
    "transporte_carga": f"{BUCKET}/silver/transporte_carga/",
}

resultados = []

for nombre, ruta in DATASETS.items():
    print("=" * 80)
    print(f"DATASET: {nombre}")
    print(f"RUTA: {ruta}")

    df = spark.read.parquet(ruta)

    total = df.count()
    columnas = len(df.columns)

    duplicados = (
        df.groupBy(df.columns)
        .count()
        .filter(F.col("count") > 1)
        .agg(F.sum(F.col("count") - 1).alias("duplicados"))
        .collect()[0]["duplicados"]
    )

    duplicados = duplicados or 0

    null_counts = (
        df.select([
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in df.columns
        ])
        .collect()[0]
        .asDict()
    )

    columnas_con_nulos = sum(
        1 for _, valor in null_counts.items()
        if valor and valor > 0
    )

    total_nulos = sum(
        valor or 0
        for valor in null_counts.values()
    )

    flags = [c for c in df.columns if c.upper().startswith("FLAG_")]

    flags_activos = 0

    for flag in flags:
        try:
            cantidad = (
                df.filter(F.col(flag) == True).count()
            )
            flags_activos += cantidad
        except Exception:
            pass

    resultados.append(
        (
            nombre,
            total,
            columnas,
            duplicados,
            total_nulos,
            columnas_con_nulos,
            len(flags),
            flags_activos,
        )
    )

    print(f"Registros: {total}")
    print(f"Columnas: {columnas}")
    print(f"Duplicados exactos detectados: {duplicados}")
    print(f"Valores nulos totales: {total_nulos}")
    print(f"Columnas con nulos: {columnas_con_nulos}")
    print(f"Columnas FLAG_: {len(flags)}")
    print(f"Flags activos: {flags_activos}")

schema = [
    "dataset",
    "registros_silver",
    "columnas",
    "duplicados_exactos",
    "valores_nulos",
    "columnas_con_nulos",
    "cantidad_flags",
    "flags_activos",
]

summary_df = spark.createDataFrame(resultados, schema=schema)

summary_df.show(truncate=False)

OUTPUT = f"{BUCKET}/quality/quality_summary"

(
    summary_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(OUTPUT)
)

PARQUET_OUTPUT = f"{BUCKET}/quality/quality_summary_parquet"

(
    summary_df
    .write
    .mode("overwrite")
    .parquet(PARQUET_OUTPUT)
)

print("==============================================")
print("QUALITY SUMMARY COMPLETADO")
print("CSV:", OUTPUT)
print("PARQUET:", PARQUET_OUTPUT)
print("==============================================")

spark.stop()