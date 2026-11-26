import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    IntegerType,
    DoubleType,
)


DEFAULT_SCALABILITY_BASE = (
    "gs://ransa-big-data-2026-lake/"
    "test/scalability/"
)

DEFAULT_OUTPUT_BASE = (
    "gs://ransa-big-data-2026-lake/"
    "test/scalability/output/"
)

DEFAULT_METRICS_OUTPUT = (
    "gs://ransa-big-data-2026-lake/"
    "test/scalability/metrics/"
)


def validar_ruta_escritura_segura(ruta: str) -> None:
    ruta_norm = ruta.strip().lower().rstrip("/")
    rutas_prohibidas = ["/bronze", "/silver"]
    for prohibida in rutas_prohibidas:
        if prohibida in ruta_norm:
            raise ValueError(
                f"Escritura bloqueada por seguridad en ruta protegida: {ruta}"
            )


def obtener_num_executors(spark_session: SparkSession) -> int:
    sc = spark_session.sparkContext

    try:
        num_execs = int(sc._jsc.sc().getExecutorIds().size())
        if num_execs > 0:
            return num_execs
    except Exception:
        pass

    try:
        total_nodos = int(sc._jsc.sc().getExecutorMemoryStatus().size())
        return max(total_nodos - 1, 1)
    except Exception:
        pass

    conf_execs = sc.getConf().get("spark.executor.instances", None)
    if conf_execs and conf_execs.isdigit():
        return int(conf_execs)

    return 1


def ejecutar_carga_benchmark(
    spark: SparkSession,
    input_path: str,
    output_path: str,
) -> tuple:
    t_inicio = time.perf_counter()

    df = spark.read.parquet(input_path)

    df_validos = df.filter(F.col("REGISTRO_VALIDO") == 1)

    resultado = (
        df_validos
        .groupBy(
            "DEPARTAMENTO",
            "CLASE",
            "COMBUSTIBLE",
            "ANIO_CORTE",
        )
        .agg(
            F.count("*").alias("TOTAL_REGISTROS_VALIDOS"),
            F.sum("CARGA_UTIL").alias("SUM_CARGA_UTIL"),
            F.avg("CARGA_UTIL").alias("AVG_CARGA_UTIL"),
            F.sum("P_BRUTO").alias("SUM_P_BRUTO"),
            F.avg("P_BRUTO").alias("AVG_P_BRUTO"),
            F.sum("P_SECO").alias("SUM_P_SECO"),
            F.avg("P_SECO").alias("AVG_P_SECO"),
            F.sum("N_EJES").alias("SUM_N_EJES"),
            F.avg("N_EJES").alias("AVG_N_EJES"),
            F.avg("ANIO_FAB").alias("AVG_ANIO_FAB"),
        )
    )

    (
        resultado.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(output_path)
    )

    tiempo_segundos = round(time.perf_counter() - t_inicio, 4)

    num_executors = obtener_num_executors(spark)
    particiones = df.rdd.getNumPartitions()
    filas = df.count()

    return filas, particiones, num_executors, tiempo_segundos


def imprimir_tabla_resumen(metricas: list) -> None:
    print("\n======================================================================================")
    print("RESUMEN DE BENCHMARK DE ESCALABILIDAD - TRANSPORTE DE CARGA")
    print("======================================================================================")
    encabezado = (
        f"{'ESCALA':<8} | "
        f"{'FILAS':>14} | "
        f"{'PARTICIONES':>12} | "
        f"{'EXECUTORS/WORKERS':>18} | "
        f"{'TIEMPO (s)':>12} | "
        f"{'ESTADO':<10}"
    )
    print(encabezado)
    print("-" * len(encabezado))

    for m in metricas:
        print(
            f"{m['escala']:<8} | "
            f"{m['filas']:>14,d} | "
            f"{m['particiones']:>12d} | "
            f"{m['num_executors']:>18d} | "
            f"{m['tiempo_segundos']:>12.4f} | "
            f"{m['estado']:<10}"
        )
    print("======================================================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta benchmark de escalabilidad secuencial sobre "
            "1x, 10x y 100x y guarda métricas en CSV."
        )
    )
    parser.add_argument(
        "--input-base",
        default=DEFAULT_SCALABILITY_BASE,
        help="Ruta base GCS donde residen 1x/, 10x/ y 100x/."
    )
    parser.add_argument(
        "--output-base",
        default=DEFAULT_OUTPUT_BASE,
        help="Ruta base GCS para resultados agregados por escala."
    )
    parser.add_argument(
        "--metrics-output",
        default=DEFAULT_METRICS_OUTPUT,
        help="Ruta GCS donde se guardará el CSV de métricas."
    )
    args = parser.parse_args()

    input_base = args.input_base.rstrip("/")
    output_base = args.output_base.rstrip("/")
    metrics_output = args.metrics_output.rstrip("/") + "/"

    validar_ruta_escritura_segura(output_base)
    validar_ruta_escritura_segura(metrics_output)

    spark = (
        SparkSession.builder
        .appName("02_benchmark_scalability")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    escalas = ["1x", "10x", "100x"]
    metricas = []

    print("\n======================================")
    print("INICIO BENCHMARK DE ESCALABILIDAD")
    print("======================================")

    for escala in escalas:
        input_escala = f"{input_base}/{escala}/"
        output_escala = f"{output_base}/{escala}/"
        validar_ruta_escritura_segura(output_escala)

        print(f"\n>>> Ejecutando benchmark para escala: {escala}")
        print(f"    Lectura : {input_escala}")
        print(f"    Salida  : {output_escala}")

        spark.catalog.clearCache()
        t_fallback = time.perf_counter()

        try:
            filas, particiones, num_executors, tiempo_segundos = (
                ejecutar_carga_benchmark(
                    spark=spark,
                    input_path=input_escala,
                    output_path=output_escala,
                )
            )
            estado = "OK"
            print(
                f"    Completado -> filas={filas:,}, particiones={particiones}, "
                f"executors={num_executors}, tiempo={tiempo_segundos:.4f}s"
            )
        except Exception as exc:
            tiempo_segundos = round(time.perf_counter() - t_fallback, 4)
            filas = 0
            particiones = 0
            num_executors = obtener_num_executors(spark)
            mensaje_corto = str(exc).splitlines()[0][:120]
            estado = f"ERROR: {mensaje_corto}"
            print(f"    Fallo en escala {escala}: {estado}")

        metricas.append({
            "escala": escala,
            "filas": int(filas),
            "particiones": int(particiones),
            "num_executors": int(num_executors),
            "tiempo_segundos": float(tiempo_segundos),
            "estado": estado,
        })

    imprimir_tabla_resumen(metricas)

    esquema_metricas = StructType([
        StructField("escala", StringType(), False),
        StructField("filas", LongType(), False),
        StructField("particiones", IntegerType(), False),
        StructField("num_executors", IntegerType(), False),
        StructField("tiempo_segundos", DoubleType(), False),
        StructField("estado", StringType(), False),
    ])

    filas_df = [
        (
            m["escala"],
            m["filas"],
            m["particiones"],
            m["num_executors"],
            m["tiempo_segundos"],
            m["estado"],
        )
        for m in metricas
    ]

    df_metricas = spark.createDataFrame(filas_df, schema=esquema_metricas)

    (
        df_metricas
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(metrics_output)
    )

    print("METRICAS GUARDADAS EN CSV:")
    print(metrics_output)

    spark.stop()


if __name__ == "__main__":
    main()
