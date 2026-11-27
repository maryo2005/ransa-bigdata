from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("validate_peaje_route_code")
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

f = spark.read.parquet(FLUJO)
m = spark.read.parquet(MASTER)

# =========================================================
# 1. Normalizadores
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
# 2. Extraer estructura del código
#
# Ej:
# 22PE5NACS1
# -> quita 22
# -> PE5NACS1
# -> quita sufijo ACS1
# -> PE5N
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
        "RUTA_DESDE_CODIGO_RAW",
        F.regexp_replace(
            F.col("CODIGO_CORE"),
            r"[A-Z]{3}[0-9]?$",
            ""
        )
    )
    .withColumn(
        "RUTA_DESDE_CODIGO",
        canonical_route(
            F.col("RUTA_DESDE_CODIGO_RAW")
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

rutas_master = (
    m
    .select(
        "RUTA_MASTER"
    )
    .filter(
        F.col("RUTA_MASTER").isNotNull()
    )
    .distinct()
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
# 3. Peajes únicos
# =========================================================

peajes = (
    f
    .select(
        "CODIGO_PEAJE",
        "NOMBRE_PEAJE_NORMALIZADO",
        "DEPARTAMENTO_NORMALIZADO",
        "DEP_N",
        "CODIGO_CORE",
        "RUTA_DESDE_CODIGO_RAW",
        "RUTA_DESDE_CODIGO"
    )
    .distinct()
)

print("\n======================================")
print("PEAJES UNICOS")
print("======================================")

print(
    "TOTAL:",
    peajes.count()
)

print(
    "CON RUTA EXTRAIDA:",
    peajes.filter(
        F.col("RUTA_DESDE_CODIGO").isNotNull()
    ).count()
)

print(
    "SIN RUTA EXTRAIDA:",
    peajes.filter(
        F.col("RUTA_DESDE_CODIGO").isNull()
    ).count()
)

# =========================================================
# 4. Validar ruta contra Master
# =========================================================

peajes_ruta = (
    peajes.alias("p")
    .join(
        rutas_master.alias("m"),
        F.col("p.RUTA_DESDE_CODIGO")
        ==
        F.col("m.RUTA_MASTER"),
        "left"
    )
    .withColumn(
        "FLAG_RUTA_EN_MASTER",
        F.col("m.RUTA_MASTER").isNotNull()
    )
    .select(
        "p.*",
        "FLAG_RUTA_EN_MASTER"
    )
)

print("\n======================================")
print("RUTA CONTRA MASTER")
print("======================================")

(
    peajes_ruta
    .groupBy(
        "FLAG_RUTA_EN_MASTER"
    )
    .count()
    .show(
        truncate=False
    )
)

# =========================================================
# 5. Validar ruta + departamento
# =========================================================

peajes_dep = (
    peajes_ruta.alias("p")
    .join(
        ruta_dep_master.alias("m"),
        (
            F.col("p.RUTA_DESDE_CODIGO")
            ==
            F.col("m.RUTA_MASTER")
        )
        &
        (
            F.col("p.DEP_N")
            ==
            F.col("m.DEP_MASTER_N")
        ),
        "left"
    )
    .withColumn(
        "FLAG_RUTA_DEP_EN_MASTER",
        F.col("m.RUTA_MASTER").isNotNull()
    )
    .select(
        "p.*",
        "FLAG_RUTA_DEP_EN_MASTER"
    )
)

print("\n======================================")
print("RUTA + DEPARTAMENTO CONTRA MASTER")
print("======================================")

(
    peajes_dep
    .groupBy(
        "FLAG_RUTA_DEP_EN_MASTER"
    )
    .count()
    .show(
        truncate=False
    )
)

# =========================================================
# 6. Detalle completo de los 82 peajes
# =========================================================

print("\n======================================")
print("DETALLE PEAJES")
print("======================================")

(
    peajes_dep
    .select(
        "CODIGO_PEAJE",
        "NOMBRE_PEAJE_NORMALIZADO",
        "DEPARTAMENTO_NORMALIZADO",
        "RUTA_DESDE_CODIGO_RAW",
        "RUTA_DESDE_CODIGO",
        "FLAG_RUTA_EN_MASTER",
        "FLAG_RUTA_DEP_EN_MASTER"
    )
    .orderBy(
        "DEPARTAMENTO_NORMALIZADO",
        "NOMBRE_PEAJE_NORMALIZADO"
    )
    .show(
        100,
        truncate=False
    )
)

# =========================================================
# 7. Casos problemáticos
# =========================================================

print("\n======================================")
print("PEAJES SIN VALIDACION RUTA+DEP")
print("======================================")

(
    peajes_dep
    .filter(
        ~F.col(
            "FLAG_RUTA_DEP_EN_MASTER"
        )
    )
    .select(
        "CODIGO_PEAJE",
        "NOMBRE_PEAJE_NORMALIZADO",
        "DEPARTAMENTO_NORMALIZADO",
        "RUTA_DESDE_CODIGO_RAW",
        "RUTA_DESDE_CODIGO",
        "FLAG_RUTA_EN_MASTER"
    )
    .show(
        100,
        truncate=False
    )
)

# =========================================================
# 8. Cobertura histórica de filas
# =========================================================

hist = (
    f
    .filter(
        F.col("ANIO") <= 2025
    )
)

validos = (
    peajes_dep
    .filter(
        F.col(
            "FLAG_RUTA_DEP_EN_MASTER"
        )
    )
    .select(
        "CODIGO_PEAJE"
    )
    .distinct()
)

hist_ok = (
    hist
    .join(
        validos,
        "CODIGO_PEAJE",
        "inner"
    )
)

total_hist = hist.count()
ok_hist = hist_ok.count()

print("\n======================================")
print("COBERTURA HISTORICA")
print("======================================")

print(
    "TOTAL <=2025:",
    total_hist
)

print(
    "VALIDADOS RUTA+DEP:",
    ok_hist
)

print(
    "SIN VALIDAR:",
    total_hist - ok_hist
)

print(
    "COBERTURA_PCT:",
    round(
        ok_hist / total_hist * 100,
        2
    )
)

spark.stop()