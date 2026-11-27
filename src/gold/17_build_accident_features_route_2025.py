from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("build_accident_features_route_2025")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

ACCIDENTES = (
    "gs://ransa-big-data-2026-lake/"
    "silver/accidentes/"
)

MASTER = (
    "gs://ransa-big-data-2026-lake/"
    "gold/master_tramos_2024/"
)

OUT_MAPEO = (
    "gs://ransa-big-data-2026-lake/"
    "gold/accidentes_mapeo_ruta_hist_2025/"
)

OUT_SIN_MAPEO = (
    "gs://ransa-big-data-2026-lake/"
    "gold/accidentes_sin_mapeo_hist_2025/"
)

OUT_FEATURES = (
    "gs://ransa-big-data-2026-lake/"
    "gold/accidentes_features_ruta_hist_2025/"
)

a = spark.read.parquet(ACCIDENTES)
m = spark.read.parquet(MASTER)

# =========================================================
# 1. Histórico anterior al target 2026
# =========================================================

hist = (
    a
    .filter(
        F.col("PERIODO")
        <= F.lit("2025-12-31")
    )
    .dropDuplicates(
        ["FILA_HASH"]
    )
)

print("\n======================================")
print("BASE HISTORICA ACCIDENTES")
print("======================================")

print(
    "TOTAL UNICOS:",
    hist.count()
)

# =========================================================
# 2. Canonicalización de rutas
# =========================================================

def canonical_route(columna):

    limpio = F.regexp_replace(
        F.upper(
            F.trim(columna)
        ),
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

    return (
        F.when(
            valido,
            F.concat(
                F.lit("PE"),
                numero
                .cast("int")
                .cast("string"),
                sufijo
            )
        )
    )


hist = (
    hist
    .withColumn(
        "RUTA_CANONICA",
        canonical_route(
            F.col("RUTA_NORMALIZADA")
        )
    )
)

m = (
    m
    .withColumn(
        "RUTA_CANONICA_MASTER",
        canonical_route(
            F.col("CODIGO_RUTA")
        )
    )
)

rutas_master = (
    m
    .select(
        "RUTA_CANONICA_MASTER"
    )
    .filter(
        F.col(
            "RUTA_CANONICA_MASTER"
        ).isNotNull()
    )
    .distinct()
)

# =========================================================
# 3. Mapeo directo
# =========================================================

directos = (
    hist.alias("a")
    .join(
        rutas_master.alias("m"),
        F.col("a.RUTA_CANONICA")
        ==
        F.col("m.RUTA_CANONICA_MASTER"),
        "inner"
    )
    .select("a.*")
    .withColumn(
        "RUTA_FINAL",
        F.col("RUTA_CANONICA")
    )
    .withColumn(
        "METODO_MAPEO_RUTA",
        F.lit("RUTA_CANONICA_DIRECTA")
    )
)

print("\n======================================")
print("MAPEO DIRECTO")
print("======================================")

print(
    "DIRECTOS:",
    directos.count()
)

# =========================================================
# 4. Diccionario TRAMO -> RUTA
# =========================================================

base_dict = (
    directos
    .filter(
        F.col(
            "TRAMO_NORMALIZADO"
        ).isNotNull()
    )
    .select(
        "TRAMO_NORMALIZADO",
        "RUTA_FINAL"
    )
)

stats_tramo = (
    base_dict
    .groupBy(
        "TRAMO_NORMALIZADO"
    )
    .agg(
        F.countDistinct(
            "RUTA_FINAL"
        ).alias("N_RUTAS"),

        F.count("*")
        .alias("N_EVIDENCIAS")
    )
)

map_tramo = (
    base_dict
    .select(
        "TRAMO_NORMALIZADO",
        "RUTA_FINAL"
    )
    .distinct()
    .join(
        stats_tramo,
        "TRAMO_NORMALIZADO",
        "inner"
    )
)

genericos = [
    "VARIANTE",
    "SIN INFO",
    "NO PRECISA",
    "N.P.",
    "-",
    "SERPENTÍN",
    "SERPENTIN"
]

diccionario_seguro = (
    map_tramo
    .filter(
        F.col("N_RUTAS") == 1
    )
    .filter(
        F.col("N_EVIDENCIAS") >= 2
    )
    .filter(
        ~F.col(
            "TRAMO_NORMALIZADO"
        ).isin(genericos)
    )
    .select(
        "TRAMO_NORMALIZADO",
        F.col(
            "RUTA_FINAL"
        ).alias(
            "RUTA_RECUPERADA"
        ),
        "N_EVIDENCIAS"
    )
)

# =========================================================
# 5. Pendientes
# =========================================================

ids_directos = (
    directos
    .select("FILA_HASH")
    .distinct()
)

pendientes = (
    hist
    .join(
        ids_directos,
        "FILA_HASH",
        "left_anti"
    )
)

# =========================================================
# 6. Recuperación conservadora
# =========================================================

recuperados = (
    pendientes.alias("p")
    .join(
        diccionario_seguro.alias("d"),
        F.col("p.TRAMO_NORMALIZADO")
        ==
        F.col("d.TRAMO_NORMALIZADO"),
        "inner"
    )
    .select(
        "p.*",
        F.col(
            "d.RUTA_RECUPERADA"
        )
    )
    .withColumn(
        "RUTA_FINAL",
        F.col(
            "RUTA_RECUPERADA"
        )
    )
    .withColumn(
        "METODO_MAPEO_RUTA",
        F.lit(
            "TRAMO_UNIVOCO_RECUPERADO"
        )
    )
    .drop(
        "RUTA_RECUPERADA"
    )
)

print("\n======================================")
print("RECUPERACION")
print("======================================")

print(
    "RECUPERADOS:",
    recuperados.count()
)

# =========================================================
# 7. Consolidar mapping
# =========================================================

base_cols = hist.columns

mapping = (
    directos
    .select(
        *base_cols,
        "RUTA_FINAL",
        "METODO_MAPEO_RUTA"
    )
    .unionByName(
        recuperados.select(
            *base_cols,
            "RUTA_FINAL",
            "METODO_MAPEO_RUTA"
        )
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

total_mapping = mapping.count()

unicos_mapping = (
    mapping
    .select("FILA_HASH")
    .distinct()
    .count()
)

print("\n======================================")
print("MAPPING FINAL")
print("======================================")

print(
    "TOTAL:",
    total_mapping
)

print(
    "FILA_HASH UNICOS:",
    unicos_mapping
)

(
    mapping
    .groupBy(
        "METODO_MAPEO_RUTA"
    )
    .count()
    .show(
        20,
        truncate=False
    )
)

# =========================================================
# 8. No mapeados
# =========================================================

ids_mapping = (
    mapping
    .select("FILA_HASH")
    .distinct()
)

sin_mapeo_base = (
    hist
    .join(
        ids_mapping,
        "FILA_HASH",
        "left_anti"
    )
)

rutas_master_flag = (
    rutas_master
    .withColumn(
        "RUTA_EXISTE_MASTER",
        F.lit(True)
    )
)

sin_mapeo = (
    sin_mapeo_base
    .join(
        rutas_master_flag,
        sin_mapeo_base.RUTA_CANONICA
        ==
        rutas_master_flag.RUTA_CANONICA_MASTER,
        "left"
    )
    .withColumn(
        "MOTIVO_SIN_MAPEO",
        F.when(
            F.col("RUTA_CANONICA")
            .isNull(),
            "SIN_RUTA_CANONICA_Y_TRAMO_NO_RECUPERABLE"
        )
        .when(
            F.col(
                "RUTA_EXISTE_MASTER"
            ).isNull(),
            "RUTA_NO_EXISTE_MASTER_2024"
        )
        .otherwise(
            "TRAMO_NO_RECUPERABLE"
        )
    )
    .drop(
        "RUTA_CANONICA_MASTER",
        "RUTA_EXISTE_MASTER"
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

total_sin = sin_mapeo.count()

print("\n======================================")
print("SIN MAPEO")
print("======================================")

print(
    "TOTAL:",
    total_sin
)

(
    sin_mapeo
    .groupBy(
        "MOTIVO_SIN_MAPEO"
    )
    .count()
    .show(
        20,
        truncate=False
    )
)

# =========================================================
# 9. Validaciones de cobertura
# =========================================================

if total_mapping != unicos_mapping:
    raise Exception(
        "ERROR: FILA_HASH duplicado "
        "en mapping."
    )

if total_mapping != 31384:
    raise Exception(
        f"ERROR: se esperaban 31384 "
        f"mapeados y se obtuvieron "
        f"{total_mapping}"
    )

if total_sin != 5243:
    raise Exception(
        f"ERROR: se esperaban 5243 "
        f"sin mapeo y se obtuvieron "
        f"{total_sin}"
    )

if (
    total_mapping
    + total_sin
    != hist.count()
):
    raise Exception(
        "ERROR: cobertura no suma "
        "el total histórico."
    )

# =========================================================
# 10. Features históricas por ruta
# =========================================================

features = (
    mapping
    .groupBy(
        "RUTA_FINAL"
    )
    .agg(
        F.count(
            "*"
        ).alias(
            "ACCIDENTES_HIST_2019_2025"
        ),

        F.sum(
            F.coalesce(
                F.col(
                    "CANTIDAD_AFECTADOS"
                ),
                F.lit(0)
            )
        ).alias(
            "AFECTADOS_HIST_2019_2025"
        ),

        F.sum(
            F.coalesce(
                F.col(
                    "CANTIDAD_VEHINVOLUCRADOS"
                ),
                F.lit(0)
            )
        ).alias(
            "VEHICULOS_INV_HIST_2019_2025"
        ),

        F.sum(
            F.when(
                F.col("ANIO") == 2025,
                1
            ).otherwise(0)
        ).alias(
            "ACCIDENTES_2025"
        ),

        F.sum(
            F.when(
                F.col("ANIO")
                .between(
                    2024,
                    2025
                ),
                1
            ).otherwise(0)
        ).alias(
            "ACCIDENTES_2024_2025"
        ),

        F.countDistinct(
            "ANIO"
        ).alias(
            "ANIOS_CON_ACCIDENTES"
        ),

        F.countDistinct(
            F.date_format(
                "PERIODO",
                "yyyy-MM"
            )
        ).alias(
            "MESES_CON_ACCIDENTES"
        )
    )
)

# =========================================================
# 11. Longitud de la ruta
# =========================================================

longitud_ruta = (
    m
    .filter(
        F.col(
            "RUTA_CANONICA_MASTER"
        ).isNotNull()
    )
    .groupBy(
        "RUTA_CANONICA_MASTER"
    )
    .agg(
        F.sum(
            "LONGITUD_KM"
        ).alias(
            "LONGITUD_RUTA_KM"
        ),

        F.countDistinct(
            "ID_TRAMO"
        ).alias(
            "N_TRAMOS_RUTA"
        )
    )
)

features = (
    features
    .join(
        longitud_ruta,
        features.RUTA_FINAL
        ==
        longitud_ruta.RUTA_CANONICA_MASTER,
        "left"
    )
    .drop(
        "RUTA_CANONICA_MASTER"
    )
)

# =========================================================
# 12. Variables derivadas
# =========================================================

features = (
    features
    .withColumn(
        "PROM_AFECTADOS_POR_ACCIDENTE",
        F.round(
            F.col(
                "AFECTADOS_HIST_2019_2025"
            )
            /
            F.col(
                "ACCIDENTES_HIST_2019_2025"
            ),
            4
        )
    )
    .withColumn(
        "PROM_VEHICULOS_POR_ACCIDENTE",
        F.round(
            F.col(
                "VEHICULOS_INV_HIST_2019_2025"
            )
            /
            F.col(
                "ACCIDENTES_HIST_2019_2025"
            ),
            4
        )
    )
    .withColumn(
        "ACCIDENTES_POR_100KM_HIST",
        F.when(
            F.col(
                "LONGITUD_RUTA_KM"
            ) > 0,
            F.round(
                F.col(
                    "ACCIDENTES_HIST_2019_2025"
                )
                /
                F.col(
                    "LONGITUD_RUTA_KM"
                )
                * 100.0,
                4
            )
        )
    )
    .withColumn(
        "ACCIDENTES_2025_POR_100KM",
        F.when(
            F.col(
                "LONGITUD_RUTA_KM"
            ) > 0,
            F.round(
                F.col(
                    "ACCIDENTES_2025"
                )
                /
                F.col(
                    "LONGITUD_RUTA_KM"
                )
                * 100.0,
                4
            )
        )
    )
    .withColumn(
        "ACCIDENTES_24M_POR_100KM",
        F.when(
            F.col(
                "LONGITUD_RUTA_KM"
            ) > 0,
            F.round(
                F.col(
                    "ACCIDENTES_2024_2025"
                )
                /
                F.col(
                    "LONGITUD_RUTA_KM"
                )
                * 100.0,
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
        "COBERTURA_ACCIDENTES_MAPEADOS_PCT",
        F.lit(85.69)
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

# =========================================================
# 13. Resumen features
# =========================================================

print("\n======================================")
print("FEATURES POR RUTA")
print("======================================")

print(
    "RUTAS CON FEATURES:",
    features.count()
)

print(
    "RUTAS MASTER:",
    rutas_master.count()
)

(
    features
    .select(
        "RUTA_FINAL",
        "ACCIDENTES_HIST_2019_2025",
        "ACCIDENTES_2025",
        "ACCIDENTES_2024_2025",
        "LONGITUD_RUTA_KM",
        "ACCIDENTES_POR_100KM_HIST",
        "ACCIDENTES_2025_POR_100KM"
    )
    .orderBy(
        F.desc(
            "ACCIDENTES_HIST_2019_2025"
        )
    )
    .show(
        30,
        truncate=False
    )
)

# =========================================================
# 14. Escritura Gold
# =========================================================

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
    sin_mapeo
    .repartition(1)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(
        OUT_SIN_MAPEO
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
print("GOLD ACCIDENTES GENERADO")
print("======================================")

print(
    "MAPEO:",
    OUT_MAPEO
)

print(
    "SIN MAPEO:",
    OUT_SIN_MAPEO
)

print(
    "FEATURES:",
    OUT_FEATURES
)

spark.stop()