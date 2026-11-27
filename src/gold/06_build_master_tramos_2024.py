from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("build_master_tramos_2024")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

TABULAR = "gs://ransa-big-data-2026-lake/silver/red_vial/"
GEO = "gs://ransa-big-data-2026-lake/silver/red_vial_geo/"

OUTPUT = (
    "gs://ransa-big-data-2026-lake/"
    "gold/master_tramos_2024/"
)

# =========================================================
# 1. Cargar Silver
# =========================================================

rv = spark.read.parquet(TABULAR)
geo = spark.read.parquet(GEO)

# =========================================================
# 2. Preparar Red Vial tabular - corte diciembre 2024
# =========================================================

rv24 = (
    rv
    .filter(
        F.col("FECHA_CORTE")
        == F.to_date(F.lit("2024-12-31"))
    )
    .select(
        "ID_RVN",
        "ID_DEPARTAMENTO",
        "DEPARTAMENTO",
        "CODIGO_RUTA",
        "INICIO",
        "FINAL",
        "CLASIFICACION_EJE",
        "JERARQUIA",
        "NRO_CARRILES",
        "ES_CONCES",
        "CODIGO_CONCESION",
        "NOMBRE_CONCESION",
        "CODIGO_LOGISTICO",
        "CORREDOR_LOGISTICO"
    )
    .withColumn(
        "DEP",
        F.lpad(
            F.col("ID_DEPARTAMENTO")
            .cast("string"),
            2,
            "0"
        )
    )
    .withColumn(
        "INI3",
        F.round("INICIO", 3)
    )
    .withColumn(
        "FIN3",
        F.round("FINAL", 3)
    )
    .withColumn(
        "START_TAB",
        F.least("INICIO", "FINAL")
    )
    .withColumn(
        "END_TAB",
        F.greatest("INICIO", "FINAL")
    )
)

# =========================================================
# 3. Preparar Red Vial Geo
# =========================================================

geo24 = (
    geo
    .withColumn(
        "DEP",
        F.lpad(
            F.col("ID_DEPARTAMENTO")
            .cast("string"),
            2,
            "0"
        )
    )
    .withColumn(
        "INI3",
        F.round("INICIO_KM", 3)
    )
    .withColumn(
        "FIN3",
        F.round("FIN_KM", 3)
    )
    .withColumn(
        "START_GEO",
        F.least(
            "INICIO_KM",
            "FIN_KM"
        )
    )
    .withColumn(
        "END_GEO",
        F.greatest(
            "INICIO_KM",
            "FIN_KM"
        )
    )
)

KEY = [
    "CODIGO_RUTA",
    "DEP",
    "INI3",
    "FIN3"
]

ATRIBUTOS = [
    "DEPARTAMENTO",
    "CLASIFICACION_EJE",
    "JERARQUIA",
    "NRO_CARRILES",
    "ES_CONCES",
    "CODIGO_CONCESION",
    "NOMBRE_CONCESION",
    "CODIGO_LOGISTICO",
    "CORREDOR_LOGISTICO"
]

# =========================================================
# 4. MATCH EXACTO
# =========================================================

g = geo24.alias("g")
r = rv24.alias("r")

exacto = (
    g.join(
        r,
        KEY,
        "inner"
    )
    .select(
        F.col("g.ORDEN").alias("ORDEN"),
        F.col("r.ID_RVN").alias("ID_RVN"),

        *[
            F.col(f"r.{c}").alias(c)
            for c in ATRIBUTOS
        ],

        F.lit("EXACTO")
        .alias("TIPO_MATCH_TABULAR"),

        F.lit(100.0)
        .cast("double")
        .alias("PORC_COBERTURA_MATCH")
    )
)

print("\n======================================")
print("MATCH EXACTO")
print("======================================")
print(
    "SEGMENTOS:",
    exacto.select("ORDEN").distinct().count()
)

# =========================================================
# 5. Detectar segmentos restantes
# =========================================================

ordenes_exactas = (
    exacto
    .select("ORDEN")
    .distinct()
)

geo_unmatched = (
    geo24
    .join(
        ordenes_exactas,
        "ORDEN",
        "left_anti"
    )
)

# Tabular sin correspondencia exacta
geo_exact_keys = (
    geo24
    .join(
        ordenes_exactas,
        "ORDEN",
        "inner"
    )
    .select(*KEY)
)

rv_unmatched = (
    rv24
    .join(
        geo_exact_keys,
        KEY,
        "left_anti"
    )
)

print(
    "GEO PENDIENTES:",
    geo_unmatched.count()
)

# =========================================================
# 6. MATCH POR SOLAPAMIENTO
# =========================================================

gu = geo_unmatched.alias("g")
ru = rv_unmatched.alias("r")

condicion = (
    (
        F.col("g.CODIGO_RUTA")
        == F.col("r.CODIGO_RUTA")
    )
    &
    (
        F.col("g.DEP")
        == F.col("r.DEP")
    )
    &
    (
        F.greatest(
            F.col("g.START_GEO"),
            F.col("r.START_TAB")
        )
        <
        F.least(
            F.col("g.END_GEO"),
            F.col("r.END_TAB")
        )
    )
)

candidatos = (
    gu.join(
        ru,
        condicion,
        "inner"
    )
    .select(
        F.col("g.ORDEN").alias("ORDEN"),

        F.col("g.START_GEO")
        .alias("START_GEO"),

        F.col("g.END_GEO")
        .alias("END_GEO"),

        F.col("r.START_TAB")
        .alias("START_TAB"),

        F.col("r.END_TAB")
        .alias("END_TAB"),

        F.col("r.ID_RVN")
        .alias("ID_RVN"),

        *[
            F.col(f"r.{c}").alias(c)
            for c in ATRIBUTOS
        ]
    )
    .withColumn(
        "OVERLAP_KM",
        F.greatest(
            F.lit(0.0),
            F.least(
                F.col("END_GEO"),
                F.col("END_TAB")
            )
            -
            F.greatest(
                F.col("START_GEO"),
                F.col("START_TAB")
            )
        )
    )
    .withColumn(
        "SPAN_GEO_KM",
        F.col("END_GEO")
        - F.col("START_GEO")
    )
    .withColumn(
        "PORC_COBERTURA_MATCH",
        F.when(
            F.col("SPAN_GEO_KM") > 0,
            F.round(
                F.col("OVERLAP_KM")
                / F.col("SPAN_GEO_KM")
                * 100,
                2
            )
        )
    )
)

w = (
    Window
    .partitionBy("ORDEN")
    .orderBy(
        F.desc("OVERLAP_KM"),
        F.asc("ID_RVN")
    )
)

solapamiento = (
    candidatos
    .withColumn(
        "RN",
        F.row_number().over(w)
    )
    .filter(
        F.col("RN") == 1
    )
    .select(
        "ORDEN",
        "ID_RVN",
        *ATRIBUTOS,
        F.lit("SOLAPAMIENTO")
        .alias("TIPO_MATCH_TABULAR"),
        "PORC_COBERTURA_MATCH"
    )
)

print("\n======================================")
print("MATCH POR SOLAPAMIENTO")
print("======================================")

print(
    "SEGMENTOS:",
    solapamiento.count()
)

# =========================================================
# 7. Detectar cualquier segmento aún sin resolver
# =========================================================

ordenes_resueltas = (
    exacto
    .select("ORDEN")
    .union(
        solapamiento.select("ORDEN")
    )
    .distinct()
)

sin_resolver = (
    geo24
    .join(
        ordenes_resueltas,
        "ORDEN",
        "left_anti"
    )
)

print(
    "SIN RESOLVER:",
    sin_resolver.count()
)

# =========================================================
# 8. Fallback: atributos estructurales estables del grupo
# =========================================================
#
# Solo se propagan atributos si todos los registros
# tabulares del grupo ruta+departamento tienen el mismo valor.
#
# Estado y superficie NO vienen de aquí.
# Se conservarán siempre desde Red Vial Geo.
# =========================================================

NULL_TXT = "__NULL__"

stats_grupo = (
    rv_unmatched
    .groupBy(
        "CODIGO_RUTA",
        "DEP"
    )
    .agg(
        F.countDistinct(
            F.coalesce(
                F.col("DEPARTAMENTO"),
                F.lit(NULL_TXT)
            )
        ).alias("N_DEP"),

        F.countDistinct(
            F.coalesce(
                F.col("CLASIFICACION_EJE"),
                F.lit(NULL_TXT)
            )
        ).alias("N_CLAS"),

        F.countDistinct(
            F.coalesce(
                F.col("JERARQUIA"),
                F.lit(NULL_TXT)
            )
        ).alias("N_JER"),

        F.countDistinct(
            F.coalesce(
                F.col("NRO_CARRILES")
                .cast("string"),
                F.lit(NULL_TXT)
            )
        ).alias("N_CARR"),

        F.countDistinct(
            F.coalesce(
                F.col("ES_CONCES")
                .cast("string"),
                F.lit(NULL_TXT)
            )
        ).alias("N_CONCES"),

        F.countDistinct(
            F.coalesce(
                F.col("CODIGO_CONCESION"),
                F.lit(NULL_TXT)
            )
        ).alias("N_COD_CON"),

        F.countDistinct(
            F.coalesce(
                F.col("NOMBRE_CONCESION"),
                F.lit(NULL_TXT)
            )
        ).alias("N_NOM_CON"),

        F.countDistinct(
            F.coalesce(
                F.col("CODIGO_LOGISTICO"),
                F.lit(NULL_TXT)
            )
        ).alias("N_COD_LOG"),

        F.countDistinct(
            F.coalesce(
                F.col("CORREDOR_LOGISTICO"),
                F.lit(NULL_TXT)
            )
        ).alias("N_CORREDOR"),

        F.first(
            "DEPARTAMENTO",
            ignorenulls=True
        ).alias("DEPARTAMENTO"),

        F.first(
            "CLASIFICACION_EJE",
            ignorenulls=True
        ).alias("CLASIFICACION_EJE"),

        F.first(
            "JERARQUIA",
            ignorenulls=True
        ).alias("JERARQUIA"),

        F.first(
            "NRO_CARRILES",
            ignorenulls=True
        ).alias("NRO_CARRILES"),

        F.first(
            "ES_CONCES",
            ignorenulls=True
        ).alias("ES_CONCES"),

        F.first(
            "CODIGO_CONCESION",
            ignorenulls=True
        ).alias("CODIGO_CONCESION"),

        F.first(
            "NOMBRE_CONCESION",
            ignorenulls=True
        ).alias("NOMBRE_CONCESION"),

        F.first(
            "CODIGO_LOGISTICO",
            ignorenulls=True
        ).alias("CODIGO_LOGISTICO"),

        F.first(
            "CORREDOR_LOGISTICO",
            ignorenulls=True
        ).alias("CORREDOR_LOGISTICO")
    )
    .filter(
        (F.col("N_DEP") <= 1)
        & (F.col("N_CLAS") <= 1)
        & (F.col("N_JER") <= 1)
        & (F.col("N_CARR") <= 1)
        & (F.col("N_CONCES") <= 1)
        & (F.col("N_COD_CON") <= 1)
        & (F.col("N_NOM_CON") <= 1)
        & (F.col("N_COD_LOG") <= 1)
        & (F.col("N_CORREDOR") <= 1)
    )
)

grupo_estable = (
    sin_resolver.alias("g")
    .join(
        stats_grupo.alias("s"),
        (
            F.col("g.CODIGO_RUTA")
            == F.col("s.CODIGO_RUTA")
        )
        &
        (
            F.col("g.DEP")
            == F.col("s.DEP")
        ),
        "inner"
    )
    .select(
        F.col("g.ORDEN").alias("ORDEN"),

        F.lit(None)
        .cast("long")
        .alias("ID_RVN"),

        *[
            F.col(f"s.{c}").alias(c)
            for c in ATRIBUTOS
        ],

        F.lit("GRUPO_ESTABLE")
        .alias("TIPO_MATCH_TABULAR"),

        F.lit(None)
        .cast("double")
        .alias("PORC_COBERTURA_MATCH")
    )
)

print("\n======================================")
print("MATCH POR GRUPO ESTABLE")
print("======================================")

print(
    "SEGMENTOS:",
    grupo_estable.count()
)

# =========================================================
# 9. Consolidar mapa de enriquecimiento
# =========================================================

mapping = (
    exacto
    .unionByName(solapamiento)
    .unionByName(grupo_estable)
)

print("\n======================================")
print("MAPA FINAL")
print("======================================")

print(
    "MAPEOS:",
    mapping.count()
)

print(
    "ORDEN UNICOS:",
    mapping
    .select("ORDEN")
    .distinct()
    .count()
)

print(
    "ORDEN DUPLICADOS:",
    mapping
    .groupBy("ORDEN")
    .count()
    .filter(
        F.col("count") > 1
    )
    .count()
)

# =========================================================
# 10. Crear MASTER_TRAMOS_2024
# =========================================================

master = (
    geo24.alias("g")
    .join(
        mapping.alias("m"),
        "ORDEN",
        "left"
    )
    .select(
        F.col("ORDEN")
        .alias("ID_TRAMO"),

        F.col("m.ID_RVN")
        .alias("ID_RVN_TABULAR"),

        F.col("g.CODIGO_RUTA"),

        F.col("g.DEP")
        .alias("ID_DEPARTAMENTO"),

        F.col("m.DEPARTAMENTO"),

        F.col("g.TRAYECTORIA"),

        F.col("g.INICIO_KM"),
        F.col("g.FIN_KM"),
        F.col("g.LONGITUD_KM"),

        F.col("g.SUPERFICIE_NORMALIZADA")
        .alias("SUPERFICIE"),

        F.col("g.ESTADO_NORMALIZADO")
        .alias("ESTADO"),

        F.col("m.CLASIFICACION_EJE"),
        F.col("m.JERARQUIA"),
        F.col("m.NRO_CARRILES"),

        F.col("m.ES_CONCES"),
        F.col("m.CODIGO_CONCESION"),
        F.col("m.NOMBRE_CONCESION"),

        F.col("m.CODIGO_LOGISTICO"),
        F.col("m.CORREDOR_LOGISTICO"),

        F.col("g.GEOMETRIA_WKT"),

        F.col("g.LAT_INICIO"),
        F.col("g.LON_INICIO"),
        F.col("g.LAT_FIN"),
        F.col("g.LON_FIN"),

        F.col("g.NRO_PUNTOS"),

        F.col("m.TIPO_MATCH_TABULAR"),
        F.col("m.PORC_COBERTURA_MATCH"),

        F.when(
            (
                F.col("m.TIPO_MATCH_TABULAR")
                == "SOLAPAMIENTO"
            )
            &
            (
                F.col("m.PORC_COBERTURA_MATCH")
                < 90
            ),
            True
        )
        .when(
            F.col("m.TIPO_MATCH_TABULAR")
            == "GRUPO_ESTABLE",
            True
        )
        .otherwise(False)
        .alias("FLAG_MATCH_REVISAR"),

        F.to_date(
            F.lit("2024-12-31")
        ).alias("FECHA_CORTE"),

        F.lit(
            "MTC_RED_VIAL_GEO_DIC2024"
        ).alias("FUENTE_BASE"),

        F.lit(
            "MTC_RED_VIAL_NACIONAL_2024"
        ).alias("FUENTE_ATRIBUTOS_TABULAR"),

        F.current_timestamp()
        .alias("FECHA_PROCESAMIENTO")
    )
)

# =========================================================
# 11. Validaciones previas a escritura
# =========================================================

total = master.count()

unicos = (
    master
    .select("ID_TRAMO")
    .distinct()
    .count()
)

sin_match = (
    master
    .filter(
        F.col("TIPO_MATCH_TABULAR")
        .isNull()
    )
    .count()
)

sin_geometria = (
    master
    .filter(
        F.col("GEOMETRIA_WKT")
        .isNull()
    )
    .count()
)

print("\n======================================")
print("VALIDACION MASTER")
print("======================================")

print("TOTAL:", total)
print("ID_TRAMO UNICOS:", unicos)
print("SIN TIPO MATCH:", sin_match)
print("SIN GEOMETRIA:", sin_geometria)

print("\nDISTRIBUCION MATCH:")

(
    master
    .groupBy(
        "TIPO_MATCH_TABULAR"
    )
    .count()
    .orderBy(
        "TIPO_MATCH_TABULAR"
    )
    .show(
        20,
        False
    )
)

print(
    "FLAG_MATCH_REVISAR:",
    master
    .filter(
        F.col("FLAG_MATCH_REVISAR")
        == True
    )
    .count()
)

# Detener si la integridad no es la esperada
if total != 7058:
    raise Exception(
        f"ERROR: se esperaban 7058 filas y se obtuvieron {total}"
    )

if unicos != 7058:
    raise Exception(
        f"ERROR: ID_TRAMO no es único: {unicos}"
    )

if sin_match != 0:
    raise Exception(
        f"ERROR: existen {sin_match} segmentos sin match"
    )

if sin_geometria != 0:
    raise Exception(
        f"ERROR: existen {sin_geometria} segmentos sin geometría"
    )

# =========================================================
# 12. Escribir Gold
# =========================================================

(
    master
    .repartition(2)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(OUTPUT)
)

print("\n======================================")
print("GOLD ESCRITO CORRECTAMENTE")
print("======================================")
print(OUTPUT)

spark.stop()