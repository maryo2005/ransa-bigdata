from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("build_flujo_features_route_2025")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

FLUJO = (
    "gs://ransa-big-data-2026-lake/"
    "silver/flujo_peajes/"
)

MASTER = (
    "gs://ransa-big-data-2026-lake/"
    "gold/master_tramos_2024/"
)

OUT_CATALOGO = (
    "gs://ransa-big-data-2026-lake/"
    "gold/flujo_peajes_catalogo_ruta/"
)

OUT_MAPEO = (
    "gs://ransa-big-data-2026-lake/"
    "gold/flujo_peajes_mapeo_hist_2025/"
)

OUT_FEATURES = (
    "gs://ransa-big-data-2026-lake/"
    "gold/flujo_features_ruta_hist_2025/"
)

f = spark.read.parquet(FLUJO)
m = spark.read.parquet(MASTER)

# =========================================================
# 1. Funciones de normalización
# =========================================================

def canonical_route(columna):

    limpio = F.regexp_replace(
        F.upper(F.trim(columna)),
        r"[^A-Z0-9]",
        ""
    )

    valido = limpio.rlike(
        r"^PE[0-9]+[A-Z]*$"
    )

    numero = F.regexp_extract(
        limpio,
        r"^PE([0-9]+)([A-Z]*)$",
        1
    )

    sufijo = F.regexp_extract(
        limpio,
        r"^PE([0-9]+)([A-Z]*)$",
        2
    )

    return F.when(
        valido,
        F.concat(
            F.lit("PE"),
            numero.cast("int").cast("string"),
            sufijo
        )
    )


def norm_text(columna):

    return F.translate(
        F.upper(F.trim(columna)),
        "ÁÉÍÓÚÜÑ",
        "AEIOUUN"
    )


# =========================================================
# 2. Extraer ruta desde CODIGO_PEAJE
# =========================================================

f = (
    f
    .withColumn(
        "CODIGO_CORE",
        F.regexp_replace(
            F.col("CODIGO_PEAJE"),
            r"^[0-9]{2}",
            ""
        )
    )
    .withColumn(
        "RUTA_CODIGO_RAW",
        F.regexp_replace(
            F.col("CODIGO_CORE"),
            r"[A-Z]{3}[0-9]?$",
            ""
        )
    )
    .withColumn(
        "RUTA_FINAL",
        canonical_route(
            F.col("RUTA_CODIGO_RAW")
        )
    )
    .withColumn(
        "DEP_N",
        norm_text(
            F.col("DEPARTAMENTO")
        )
    )
)

m = (
    m
    .withColumn(
        "RUTA_MASTER",
        canonical_route(
            F.col("CODIGO_RUTA")
        )
    )
    .withColumn(
        "DEP_MASTER_N",
        norm_text(
            F.col("DEPARTAMENTO")
        )
    )
)

ruta_dep_master = (
    m
    .select(
        "RUTA_MASTER",
        "DEP_MASTER_N"
    )
    .filter(
        F.col("RUTA_MASTER").isNotNull()
    )
    .distinct()
)

# =========================================================
# 3. Catálogo validado de peajes
# =========================================================

catalogo_base = (
    f
    .select(
        "CODIGO_PEAJE",
        "NOMBRE_PEAJE_NORMALIZADO",
        "DEPARTAMENTO_NORMALIZADO",
        "DEP_N",
        "RUTA_FINAL"
    )
    .distinct()
)

catalogo = (
    catalogo_base.alias("p")
    .join(
        ruta_dep_master.alias("m"),
        (
            F.col("p.RUTA_FINAL")
            ==
            F.col("m.RUTA_MASTER")
        )
        &
        (
            F.col("p.DEP_N")
            ==
            F.col("m.DEP_MASTER_N")
        ),
        "inner"
    )
    .select(
        "p.CODIGO_PEAJE",
        "p.NOMBRE_PEAJE_NORMALIZADO",
        "p.DEPARTAMENTO_NORMALIZADO",
        "p.RUTA_FINAL"
    )
    .distinct()
    .withColumn(
        "METODO_MAPEO",
        F.lit("CODIGO_PEAJE_RUTA_DEP")
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

print("\n======================================")
print("CATALOGO PEAJES")
print("======================================")

print(
    "PEAJES:",
    catalogo.count()
)

if catalogo.count() != 82:
    raise Exception(
        f"ERROR: se esperaban 82 peajes "
        f"y se obtuvieron {catalogo.count()}"
    )

# =========================================================
# 4. Histórico <= 2025
# =========================================================

hist = (
    f
    .filter(
        F.col("ANIO") <= 2025
    )
)

mapping = (
    hist.alias("f")
    .join(
        catalogo
        .select(
            "CODIGO_PEAJE",
            "RUTA_FINAL"
        )
        .alias("c"),
        F.col("f.CODIGO_PEAJE")
        ==
        F.col("c.CODIGO_PEAJE"),
        "inner"
    )
    .select(
        "f.*",
        F.col(
            "c.RUTA_FINAL"
        ).alias(
            "RUTA_FINAL_VALIDADA"
        )
    )
    .drop("RUTA_FINAL")
    .withColumnRenamed(
        "RUTA_FINAL_VALIDADA",
        "RUTA_FINAL"
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

total_hist = hist.count()
total_mapping = mapping.count()

print("\n======================================")
print("MAPPING HISTORICO")
print("======================================")

print(
    "TOTAL <=2025:",
    total_hist
)

print(
    "MAPEADOS:",
    total_mapping
)

print(
    "COBERTURA_PCT:",
    round(
        total_mapping /
        total_hist * 100,
        2
    )
)

if total_hist != 11296:
    raise Exception(
        f"ERROR: se esperaban 11296 "
        f"registros históricos y se "
        f"obtuvieron {total_hist}"
    )

if total_mapping != 11296:
    raise Exception(
        f"ERROR: no se obtuvo cobertura "
        f"completa: {total_mapping}"
    )

# =========================================================
# 5. Features por ruta
# =========================================================

features = (
    mapping
    .groupBy(
        "RUTA_FINAL"
    )
    .agg(

        F.countDistinct(
            "CODIGO_PEAJE"
        ).alias(
            "N_PEAJES_RUTA"
        ),

        F.count(
            "*"
        ).alias(
            "N_OBSERVACIONES_FLUJO_HIST"
        ),

        F.countDistinct(
            "PERIODO_FECHA"
        ).alias(
            "MESES_OBSERVADOS_HIST"
        ),

        # ---------------------------------
        # Histórico completo
        # ---------------------------------

        F.sum(
            "VEH_TOTAL"
        ).alias(
            "VEH_TOTAL_OBSERVADO_2014_2025"
        ),

        F.avg(
            "VEH_IMD"
        ).alias(
            "PROM_IMD_2014_2025"
        ),

        F.avg(
            "VEH_PESADOS_IMD"
        ).alias(
            "PROM_IMD_PESADOS_2014_2025"
        ),

        # ---------------------------------
        # 2024
        # ---------------------------------

        F.avg(
            F.when(
                F.col("ANIO") == 2024,
                F.col("VEH_IMD")
            )
        ).alias(
            "PROM_IMD_2024"
        ),

        # ---------------------------------
        # 2025
        # ---------------------------------

        F.sum(
            F.when(
                F.col("ANIO") == 2025,
                F.col("VEH_TOTAL")
            ).otherwise(0)
        ).alias(
            "VEH_TOTAL_OBSERVADO_2025"
        ),

        F.sum(
            F.when(
                F.col("ANIO") == 2025,
                F.col("VEH_LIGEROS_TOTAL")
            ).otherwise(0)
        ).alias(
            "VEH_LIGEROS_OBSERVADO_2025"
        ),

        F.sum(
            F.when(
                F.col("ANIO") == 2025,
                F.col("VEH_PESADOS_TOTAL")
            ).otherwise(0)
        ).alias(
            "VEH_PESADOS_OBSERVADO_2025"
        ),

        F.avg(
            F.when(
                F.col("ANIO") == 2025,
                F.col("VEH_TOTAL")
            )
        ).alias(
            "PROM_VEH_MENSUAL_PEAJE_2025"
        ),

        F.avg(
            F.when(
                F.col("ANIO") == 2025,
                F.col("VEH_IMD")
            )
        ).alias(
            "PROM_IMD_2025"
        ),

        F.max(
            F.when(
                F.col("ANIO") == 2025,
                F.col("VEH_IMD")
            )
        ).alias(
            "MAX_IMD_2025"
        ),

        F.avg(
            F.when(
                F.col("ANIO") == 2025,
                F.col("VEH_PESADOS_IMD")
            )
        ).alias(
            "PROM_IMD_PESADOS_2025"
        ),

        F.countDistinct(
            F.when(
                F.col("ANIO") == 2025,
                F.col("PERIODO_FECHA")
            )
        ).alias(
            "MESES_OBSERVADOS_2025"
        ),

        F.sum(
            F.when(
                F.col(
                    "FLAG_TOTAL_LIGEROS_INCONSISTENTE"
                ),
                1
            ).otherwise(0)
        ).alias(
            "N_FLAGS_LIGEROS_INCONSISTENTES"
        )
    )
)

# =========================================================
# 6. Variables derivadas
# =========================================================

features = (
    features
    .withColumn(
        "PCT_PESADOS_2025",
        F.when(
            F.col(
                "VEH_TOTAL_OBSERVADO_2025"
            ) > 0,
            F.round(
                (
                    F.col(
                        "VEH_PESADOS_OBSERVADO_2025"
                    )
                    /
                    F.col(
                        "VEH_TOTAL_OBSERVADO_2025"
                    )
                ) * 100.0,
                4
            )
        )
    )
    .withColumn(
        "VAR_IMD_2025_VS_2024_PCT",
        F.when(
            F.col(
                "PROM_IMD_2024"
            ) > 0,
            F.round(
                (
                    (
                        F.col(
                            "PROM_IMD_2025"
                        )
                        -
                        F.col(
                            "PROM_IMD_2024"
                        )
                    )
                    /
                    F.col(
                        "PROM_IMD_2024"
                    )
                ) * 100.0,
                4
            )
        )
    )
    .withColumn(
        "FECHA_CORTE_PREDICTORES",
        F.to_date(
            F.lit("2025-12-31")
        )
    )
    .withColumn(
        "COBERTURA_FLUJO_MAPEADO_PCT",
        F.lit(100.0)
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

# =========================================================
# 7. Validación
# =========================================================

print("\n======================================")
print("FEATURES DE FLUJO POR RUTA")
print("======================================")

print(
    "RUTAS CON FLUJO:",
    features.count()
)

(
    features
    .select(
        "RUTA_FINAL",
        "N_PEAJES_RUTA",
        "MESES_OBSERVADOS_HIST",
        "PROM_IMD_2014_2025",
        "PROM_IMD_2024",
        "PROM_IMD_2025",
        "MAX_IMD_2025",
        "PROM_IMD_PESADOS_2025",
        "PCT_PESADOS_2025",
        "VAR_IMD_2025_VS_2024_PCT"
    )
    .orderBy(
        F.desc(
            "PROM_IMD_2025"
        )
    )
    .show(
        100,
        truncate=False
    )
)

# =========================================================
# 8. Escribir Gold
# =========================================================

(
    catalogo
    .repartition(1)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(
        OUT_CATALOGO
    )
)

(
    mapping
    .repartition(1)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(
        OUT_MAPEO
    )
)

(
    features
    .repartition(1)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(
        OUT_FEATURES
    )
)

print("\n======================================")
print("GOLD FLUJO GENERADO")
print("======================================")

print(
    "CATALOGO:",
    OUT_CATALOGO
)

print(
    "MAPEO:",
    OUT_MAPEO
)

print(
    "FEATURES:",
    OUT_FEATURES
)

spark.stop()