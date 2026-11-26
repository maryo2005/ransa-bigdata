import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


DEFAULT_SILVER_INPUT = (
    "gs://ransa-big-data-2026-lake/"
    "silver/transporte_carga/"
)

DEFAULT_SCALABILITY_BASE = (
    "gs://ransa-big-data-2026-lake/"
    "test/scalability/"
)


def validar_ruta_escritura_segura(ruta: str) -> None:
    ruta_norm = ruta.strip().lower().rstrip("/")
    rutas_prohibidas = ["/bronze", "/silver"]
    for prohibida in rutas_prohibidas:
        if prohibida in ruta_norm:
            raise ValueError(
                f"Escritura bloqueada por seguridad en ruta protegida: {ruta}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera datasets de escalabilidad (1x, 10x, 100x) a partir "
            "del Parquet Silver de Transporte de Carga."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_SILVER_INPUT,
        help="Ruta GCS del Parquet Silver de Transporte de Carga (solo lectura)."
    )
    parser.add_argument(
        "--output-base",
        default=DEFAULT_SCALABILITY_BASE,
        help="Ruta base GCS para escribir 1x/, 10x/ y 100x/."
    )
    args = parser.parse_args()

    output_base = args.output_base.rstrip("/")
    validar_ruta_escritura_segura(output_base)

    spark = (
        SparkSession.builder
        .appName("01_generate_scalability_data")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("\n======================================")
    print("GENERACION DE DATOS DE ESCALABILIDAD")
    print("======================================")
    print("INPUT SILVER (SOLO LECTURA):", args.input)
    print("OUTPUT BASE:", f"{output_base}/")

    df_silver = spark.read.parquet(args.input)
    df_silver.cache()

    base_count = df_silver.count()
    print(f"FILAS BASE EN SILVER (1x): {base_count}")

    escalas = [
        ("1x", 1, False),
        ("10x", 10, True),
        ("100x", 100, True),
    ]

    for nombre_escala, factor, es_sintetico in escalas:
        destino = f"{output_base}/{nombre_escala}/"
        validar_ruta_escritura_segura(destino)

        print(f"\n--- Generando escala {nombre_escala} (factor={factor}) ---")

        if factor == 1:
            df_out = (
                df_silver
                .withColumn("replica_id", F.lit(0).cast("long"))
                .withColumn("ES_DATO_SINTETICO", F.lit(es_sintetico))
            )
        else:
            df_out = (
                df_silver
                .crossJoin(
                    spark.range(factor).withColumnRenamed("id", "replica_id")
                )
                .withColumn("ES_DATO_SINTETICO", F.lit(es_sintetico))
            )

        (
            df_out.write
            .mode("overwrite")
            .option("compression", "snappy")
            .parquet(destino)
        )

        filas_esperadas = base_count * factor
        print(f"Escala {nombre_escala} escrita en: {destino}")
        print(f"Filas esperadas ({nombre_escala}): {filas_esperadas}")
        print(f"ES_DATO_SINTETICO: {es_sintetico}")

    df_silver.unpersist()

    print("\n======================================")
    print("DATOS DE ESCALABILIDAD GENERADOS OK")
    print("======================================")

    spark.stop()


if __name__ == "__main__":
    main()
