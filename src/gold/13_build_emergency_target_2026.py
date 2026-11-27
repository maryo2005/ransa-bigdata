from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
import math

spark = (
    SparkSession.builder
    .appName("build_emergency_target_2026")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

EMERGENCIAS = (
    "gs://ransa-big-data-2026-lake/"
    "silver/emergencias/"
)

MASTER = (
    "gs://ransa-big-data-2026-lake/"
    "gold/master_tramos_2024/"
)

OUT_MAPPING = (
    "gs://ransa-big-data-2026-lake/"
    "gold/emergencias_mapeo_2026/"
)

OUT_SIN_MAPEO = (
    "gs://ransa-big-data-2026-lake/"
    "gold/emergencias_sin_mapeo_2026/"
)

OUT_TARGET = (
    "gs://ransa-big-data-2026-lake/"
    "gold/afectacion_tramo_2026/"
)

R = 6371.0088

e = spark.read.parquet(EMERGENCIAS)
m = spark.read.parquet(MASTER)

# =========================================================
# 1. Normalización
# =========================================================

def norm(c):
    return F.translate(
        F.upper(F.trim(c)),
        "ÁÉÍÓÚÜÑ",
        "AEIOUUN"
    )

e = (
    e
    .withColumn(
        "RUTA_N",
        norm(F.col("RED_AFECTADA_NORMALIZADA"))
    )
    .withColumn(
        "DEP_N",
        norm(F.col("DEPARTAMENTO"))
    )
)

m = (
    m
    .withColumn(
        "RUTA_N",
        norm(F.col("CODIGO_RUTA"))
    )
    .withColumn(
        "DEP_N",
        norm(F.col("DEPARTAMENTO"))
    )
    .withColumn(
        "MID_LAT",
        (
            F.col("LAT_INICIO")
            + F.col("LAT_FIN")
        ) / 2.0
    )
    .withColumn(
        "MID_LON",
        (
            F.col("LON_INICIO")
            + F.col("LON_FIN")
        ) / 2.0
    )
)

# =========================================================
# 2. Haversine para preselección
# =========================================================

def haversine(lat1, lon1, lat2, lon2):

    lat1r = F.radians(lat1)
    lon1r = F.radians(lon1)
    lat2r = F.radians(lat2)
    lon2r = F.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = (
        F.pow(F.sin(dlat / 2.0), 2)
        +
        F.cos(lat1r)
        * F.cos(lat2r)
        * F.pow(F.sin(dlon / 2.0), 2)
    )

    return (
        2.0
        * F.lit(R)
        * F.asin(
            F.sqrt(
                F.least(
                    F.lit(1.0),
                    a
                )
            )
        )
    )

# =========================================================
# 3. Distancia punto -> LINESTRING
# =========================================================

def point_line_distance(lat, lon, wkt):

    if lat is None or lon is None or wkt is None:
        return None

    try:

        texto = wkt.strip()

        if not texto.upper().startswith("LINESTRING"):
            return None

        interior = texto[
            texto.find("(") + 1:
            texto.rfind(")")
        ]

        puntos = []

        lat0 = math.radians(float(lat))
        cos_lat0 = math.cos(lat0)

        for token in interior.split(","):

            partes = token.strip().split()

            if len(partes) < 2:
                continue

            lon2 = float(partes[0])
            lat2 = float(partes[1])

            x = (
                R
                * math.radians(
                    lon2 - float(lon)
                )
                * cos_lat0
            )

            y = (
                R
                * math.radians(
                    lat2 - float(lat)
                )
            )

            puntos.append((x, y))

        if not puntos:
            return None

        if len(puntos) == 1:
            x, y = puntos[0]
            return math.sqrt(x*x + y*y)

        mejor = float("inf")

        for i in range(len(puntos) - 1):

            ax, ay = puntos[i]
            bx, by = puntos[i + 1]

            vx = bx - ax
            vy = by - ay

            denom = vx*vx + vy*vy

            if denom == 0:

                d = math.sqrt(
                    ax*ax + ay*ay
                )

            else:

                t = (
                    -(ax*vx + ay*vy)
                    / denom
                )

                t = max(
                    0.0,
                    min(1.0, t)
                )

                cx = ax + t*vx
                cy = ay + t*vy

                d = math.sqrt(
                    cx*cx + cy*cy
                )

            if d < mejor:
                mejor = d

        return float(mejor)

    except Exception:
        return None


dist_line = F.udf(
    point_line_distance,
    DoubleType()
)

# =========================================================
# 4. MÉTODO DIRECTO
# misma ruta + mismo departamento
# =========================================================

ee = e.alias("e")
mm = m.alias("m")

direct_candidates = (
    ee
    .join(
        mm,
        (
            F.col("e.RUTA_N")
            == F.col("m.RUTA_N")
        )
        &
        (
            F.col("e.DEP_N")
            == F.col("m.DEP_N")
        ),
        "inner"
    )
    .select(
        F.col("e.ID_EMERGENCIA"),
        F.col("e.LATITUD"),
        F.col("e.LONGITUD"),

        F.col("e.RUTA_N")
        .alias("RUTA_REPORTADA"),

        F.col("e.DEP_N")
        .alias("DEP_REPORTADO"),

        F.col("m.ID_TRAMO"),

        F.col("m.RUTA_N")
        .alias("RUTA_MASTER"),

        F.col("m.DEP_N")
        .alias("DEP_MASTER"),

        F.col("m.GEOMETRIA_WKT"),
        F.col("m.MID_LAT"),
        F.col("m.MID_LON")
    )
    .withColumn(
        "DIST_MID_KM",
        haversine(
            F.col("LATITUD"),
            F.col("LONGITUD"),
            F.col("MID_LAT"),
            F.col("MID_LON")
        )
    )
)

w_mid = (
    Window
    .partitionBy("ID_EMERGENCIA")
    .orderBy(
        F.asc("DIST_MID_KM")
    )
)

direct_candidates = (
    direct_candidates
    .withColumn(
        "RN_MID",
        F.row_number().over(w_mid)
    )
    .filter(
        F.col("RN_MID") <= 5
    )
    .drop("RN_MID")
    .withColumn(
        "DISTANCIA_KM",
        dist_line(
            F.col("LATITUD"),
            F.col("LONGITUD"),
            F.col("GEOMETRIA_WKT")
        )
    )
)

w_direct = (
    Window
    .partitionBy("ID_EMERGENCIA")
    .orderBy(
        F.asc_nulls_last("DISTANCIA_KM"),
        F.asc("ID_TRAMO")
    )
)

direct_best = (
    direct_candidates
    .withColumn(
        "RN",
        F.row_number().over(w_direct)
    )
    .filter(
        F.col("RN") == 1
    )
    .drop("RN")
)

direct_ok = (
    direct_best
    .filter(
        F.col("DISTANCIA_KM") <= 5.0
    )
    .withColumn(
        "METODO_MAPEO",
        F.lit("RUTA_DEP_COORD")
    )
    .withColumn(
        "NIVEL_CONFIANZA",
        F.when(
            F.col("DISTANCIA_KM") <= 1.0,
            "ALTA"
        ).otherwise("MEDIA")
    )
)

print("\n======================================")
print("MAPEO DIRECTO")
print("======================================")

print(
    "ACEPTADOS:",
    direct_ok.count()
)

# =========================================================
# 5. Eventos pendientes
# =========================================================

ids_directos = (
    direct_ok
    .select("ID_EMERGENCIA")
    .distinct()
)

pendientes = (
    e
    .join(
        ids_directos,
        "ID_EMERGENCIA",
        "left_anti"
    )
)

print(
    "PENDIENTES:",
    pendientes.count()
)

# =========================================================
# 6. RECUPERACIÓN
# cualquier ruta del mismo departamento
# Solo se acepta <= 1 km
# =========================================================

pp = pendientes.alias("e")

fallback_candidates = (
    pp
    .join(
        mm,
        F.col("e.DEP_N")
        == F.col("m.DEP_N"),
        "inner"
    )
    .select(
        F.col("e.ID_EMERGENCIA"),
        F.col("e.LATITUD"),
        F.col("e.LONGITUD"),

        F.col("e.RUTA_N")
        .alias("RUTA_REPORTADA"),

        F.col("e.DEP_N")
        .alias("DEP_REPORTADO"),

        F.col("m.ID_TRAMO"),

        F.col("m.RUTA_N")
        .alias("RUTA_MASTER"),

        F.col("m.DEP_N")
        .alias("DEP_MASTER"),

        F.col("m.GEOMETRIA_WKT"),
        F.col("m.MID_LAT"),
        F.col("m.MID_LON")
    )
    .withColumn(
        "DIST_MID_KM",
        haversine(
            F.col("LATITUD"),
            F.col("LONGITUD"),
            F.col("MID_LAT"),
            F.col("MID_LON")
        )
    )
)

w_mid2 = (
    Window
    .partitionBy("ID_EMERGENCIA")
    .orderBy(
        F.asc("DIST_MID_KM")
    )
)

fallback_candidates = (
    fallback_candidates
    .withColumn(
        "RN_MID",
        F.row_number().over(w_mid2)
    )
    .filter(
        F.col("RN_MID") <= 10
    )
    .drop("RN_MID")
    .withColumn(
        "DISTANCIA_KM",
        dist_line(
            F.col("LATITUD"),
            F.col("LONGITUD"),
            F.col("GEOMETRIA_WKT")
        )
    )
)

w_fallback = (
    Window
    .partitionBy("ID_EMERGENCIA")
    .orderBy(
        F.asc_nulls_last("DISTANCIA_KM"),
        F.asc("ID_TRAMO")
    )
)

fallback_best = (
    fallback_candidates
    .withColumn(
        "RN",
        F.row_number().over(w_fallback)
    )
    .filter(
        F.col("RN") == 1
    )
    .drop("RN")
)

fallback_ok = (
    fallback_best
    .filter(
        F.col("DISTANCIA_KM") <= 1.0
    )
    .withColumn(
        "METODO_MAPEO",
        F.lit("RECUPERADO_COORD_DEP")
    )
    .withColumn(
        "NIVEL_CONFIANZA",
        F.lit("ALTA")
    )
)

print("\n======================================")
print("RECUPERACION")
print("======================================")

print(
    "RECUPERADOS:",
    fallback_ok.count()
)

# =========================================================
# 7. Mapping final
# =========================================================

cols_mapping = [
    "ID_EMERGENCIA",
    "ID_TRAMO",
    "RUTA_REPORTADA",
    "RUTA_MASTER",
    "DEP_REPORTADO",
    "DEP_MASTER",
    "DISTANCIA_KM",
    "METODO_MAPEO",
    "NIVEL_CONFIANZA"
]

mapping_base = (
    direct_ok
    .select(*cols_mapping)
    .unionByName(
        fallback_ok.select(
            *cols_mapping
        )
    )
    .withColumn(
        "FLAG_CAMBIO_RUTA",
        F.col("RUTA_REPORTADA")
        != F.col("RUTA_MASTER")
    )
)

mapping = (
    mapping_base
    .join(
        e.select(
            "ID_EMERGENCIA",
            "CODIGO_REPORTE",
            "FECHA_OCURRENCIA",
            "EVENTO",
            "CLASE_FENOMENO",
            "FENOMENO",
            "ESTADO_INICIAL",
            "ESTADO_ACTUAL",
            "TRAMO_AFECTADO",
            "SECTOR_AFECTADO",
            "LATITUD",
            "LONGITUD"
        ),
        "ID_EMERGENCIA",
        "left"
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

total_mapping = mapping.count()

unicos_mapping = (
    mapping
    .select("ID_EMERGENCIA")
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
    "ID_EMERGENCIA UNICOS:",
    unicos_mapping
)

print("\nMETODOS:")

(
    mapping
    .groupBy("METODO_MAPEO")
    .count()
    .show(
        20,
        truncate=False
    )
)

print("\nCONFIANZA:")

(
    mapping
    .groupBy("NIVEL_CONFIANZA")
    .count()
    .show(
        20,
        truncate=False
    )
)

print(
    "CAMBIO DE RUTA:",
    mapping
    .filter(
        F.col("FLAG_CAMBIO_RUTA")
    )
    .count()
)

# =========================================================
# 8. Emergencias sin mapeo confiable
# =========================================================

ids_mapeados = (
    mapping
    .select("ID_EMERGENCIA")
    .distinct()
)

rutas_master = (
    m
    .select(
        F.col("RUTA_N")
        .alias("RUTA_MASTER_EXISTENTE")
    )
    .distinct()
)

sin_mapeo = (
    e
    .join(
        ids_mapeados,
        "ID_EMERGENCIA",
        "left_anti"
    )
    .join(
        rutas_master,
        e.RUTA_N
        ==
        rutas_master.RUTA_MASTER_EXISTENTE,
        "left"
    )
    .withColumn(
        "MOTIVO_SIN_MAPEO",
        F.when(
            F.col(
                "RUTA_MASTER_EXISTENTE"
            ).isNull(),
            "RUTA_NO_EXISTE_MASTER_2024"
        )
        .otherwise(
            "DISTANCIA_NO_CONFIABLE"
        )
    )
    .drop(
        "RUTA_MASTER_EXISTENTE"
    )
)

total_sin = sin_mapeo.count()

print("\n======================================")
print("SIN MAPEO CONFIABLE")
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
# 9. Validaciones mapping
# =========================================================

if total_mapping != unicos_mapping:
    raise Exception(
        "ERROR: una emergencia quedó asociada "
        "a más de un tramo."
    )

if total_mapping + total_sin != e.count():
    raise Exception(
        "ERROR: mapping + sin mapping "
        "no coincide con total Emergencias."
    )

# Esperado según diagnósticos previos
if total_mapping != 1301:
    raise Exception(
        f"ERROR: se esperaban 1301 emergencias "
        f"mapeadas y se obtuvieron {total_mapping}"
    )

if total_sin != 61:
    raise Exception(
        f"ERROR: se esperaban 61 emergencias "
        f"sin mapeo y se obtuvieron {total_sin}"
    )

# =========================================================
# 10. Escribir mapping y no mapeados
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
    .parquet(OUT_MAPPING)
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
    .parquet(OUT_SIN_MAPEO)
)

# =========================================================
# 11. Agregar emergencias por tramo
# =========================================================

agg = (
    mapping
    .groupBy("ID_TRAMO")
    .agg(
        F.countDistinct(
            "ID_EMERGENCIA"
        ).alias(
            "N_EMERGENCIAS_OBJETIVO"
        ),

        F.countDistinct(
            "EVENTO"
        ).alias(
            "N_TIPOS_EVENTO_OBJETIVO"
        ),

        F.min(
            "FECHA_OCURRENCIA"
        ).alias(
            "FECHA_PRIMERA_EMERGENCIA"
        ),

        F.max(
            "FECHA_OCURRENCIA"
        ).alias(
            "FECHA_ULTIMA_EMERGENCIA"
        )
    )
)

# =========================================================
# 12. Crear target para los 7058 tramos
# =========================================================

target = (
    m
    .select("ID_TRAMO")
    .join(
        agg,
        "ID_TRAMO",
        "left"
    )
    .fillna(
        0,
        subset=[
            "N_EMERGENCIAS_OBJETIVO",
            "N_TIPOS_EVENTO_OBJETIVO"
        ]
    )
    .withColumn(
        "AFECTACION_TRAMO",
        F.when(
            F.col(
                "N_EMERGENCIAS_OBJETIVO"
            ) > 0,
            F.lit(1)
        ).otherwise(
            F.lit(0)
        )
    )
    .withColumn(
        "FECHA_INICIO_OBJETIVO",
        F.to_date(
            F.lit("2026-01-01")
        )
    )
    .withColumn(
        "FECHA_FIN_OBJETIVO",
        F.to_date(
            F.lit("2026-08-20")
        )
    )
    .withColumn(
        "COBERTURA_EVENTOS_MAPEADOS_PCT",
        F.lit(
            round(
                1301 / 1362 * 100,
                2
            )
        )
    )
    .withColumn(
        "FECHA_PROCESAMIENTO_GOLD",
        F.current_timestamp()
    )
)

# =========================================================
# 13. Validación target
# =========================================================

total_target = target.count()

id_target = (
    target
    .select("ID_TRAMO")
    .distinct()
    .count()
)

print("\n======================================")
print("TARGET AFECTACION_TRAMO")
print("======================================")

print(
    "TOTAL TRAMOS:",
    total_target
)

print(
    "ID_TRAMO UNICOS:",
    id_target
)

print("\nDISTRIBUCION TARGET:")

(
    target
    .groupBy(
        "AFECTACION_TRAMO"
    )
    .count()
    .orderBy(
        "AFECTACION_TRAMO"
    )
    .show(
        20,
        truncate=False
    )
)

print(
    "TRAMOS AFECTADOS:",
    target
    .filter(
        F.col("AFECTACION_TRAMO") == 1
    )
    .count()
)

print(
    "TRAMOS NO AFECTADOS:",
    target
    .filter(
        F.col("AFECTACION_TRAMO") == 0
    )
    .count()
)

print("\nRESUMEN EMERGENCIAS POR TRAMO:")

(
    target
    .agg(
        F.min(
            "N_EMERGENCIAS_OBJETIVO"
        ).alias("MIN"),

        F.avg(
            "N_EMERGENCIAS_OBJETIVO"
        ).alias("PROMEDIO"),

        F.max(
            "N_EMERGENCIAS_OBJETIVO"
        ).alias("MAX")
    )
    .show(
        truncate=False
    )
)

if total_target != 7058:
    raise Exception(
        f"ERROR: se esperaban 7058 tramos "
        f"y se obtuvieron {total_target}"
    )

if id_target != 7058:
    raise Exception(
        "ERROR: ID_TRAMO no es único "
        "en el target."
    )

# =========================================================
# 14. Escribir target
# =========================================================

(
    target
    .repartition(1)
    .write
    .mode("overwrite")
    .option(
        "compression",
        "snappy"
    )
    .parquet(OUT_TARGET)
)

print("\n======================================")
print("GOLD GENERADO CORRECTAMENTE")
print("======================================")

print(
    "MAPPING:",
    OUT_MAPPING
)

print(
    "SIN MAPEO:",
    OUT_SIN_MAPEO
)

print(
    "TARGET:",
    OUT_TARGET
)

spark.stop()