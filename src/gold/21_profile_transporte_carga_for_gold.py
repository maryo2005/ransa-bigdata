from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("profile_transporte_carga_for_gold")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

CARGA = (
    "gs://ransa-big-data-2026-lake/"
    "silver/transporte_carga/"
)

df = spark.read.parquet(CARGA)

print("\n======================================")
print("TRANSPORTE CARGA - PERFIL PARA GOLD")
print("======================================")

print("REGISTROS:", df.count())
print("COLUMNAS:", len(df.columns))

print("\n======================================")
print("ESQUEMA")
print("======================================")

df.printSchema()

print("\nCOLUMNAS:")
for c in df.columns:
    print(c)

# =========================================================
# 1. Muestra
# =========================================================

print("\n======================================")
print("MUESTRA")
print("======================================")

df.show(
    20,
    truncate=False
)

# =========================================================
# 2. Campos temporales
# =========================================================

temporales = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "ANIO",
            "AÑO",
            "FECHA",
            "VIGENCIA",
            "PERIODO"
        ]
    )
]

print("\n======================================")
print("CAMPOS TEMPORALES")
print("======================================")

print(temporales)

for c in temporales:
    print(f"\nTOP {c}:")
    (
        df
        .groupBy(c)
        .count()
        .orderBy(
            F.desc("count")
        )
        .show(
            50,
            truncate=False
        )
    )

# =========================================================
# 3. Posibles campos geográficos / ruta
# =========================================================

geo = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "RUTA",
            "DEPART",
            "PROVINC",
            "DISTR",
            "ORIGEN",
            "DESTINO",
            "UBIC",
            "DOMIC",
            "REGION",
            "COORD",
            "LAT",
            "LON"
        ]
    )
]

print("\n======================================")
print("CAMPOS GEOGRAFICOS / RUTA")
print("======================================")

print(geo)

for c in geo:
    print(f"\nTOP {c}:")
    (
        df
        .groupBy(c)
        .count()
        .orderBy(
            F.desc("count")
        )
        .show(
            50,
            truncate=False
        )
    )

# =========================================================
# 4. Campos de carga / vehículo
# =========================================================

carga_cols = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "CARGA",
            "PESO",
            "P_BRUTO",
            "CAPAC",
            "EJE",
            "CLASE",
            "CATEGORIA",
            "MARCA",
            "MODELO",
            "VEHIC"
        ]
    )
]

print("\n======================================")
print("CAMPOS VEHICULO / CARGA")
print("======================================")

for c in carga_cols:
    print(
        c,
        dict(df.dtypes).get(c)
    )

# =========================================================
# 5. Nulos
# =========================================================

print("\n======================================")
print("NULOS POR COLUMNA")
print("======================================")

exprs = []

for c in df.columns:
    exprs.append(
        F.sum(
            F.when(
                F.col(c).isNull(),
                1
            ).otherwise(0)
        ).alias(c)
    )

df.agg(*exprs).show(
    truncate=False
)

# =========================================================
# 6. Hashes / entidades anonimizadas
# =========================================================

hash_cols = [
    c for c in df.columns
    if "HASH" in c.upper()
]

print("\n======================================")
print("CAMPOS HASH")
print("======================================")

print(hash_cols)

for c in hash_cols:
    print(
        c,
        df.select(c)
        .distinct()
        .count()
    )

# =========================================================
# 7. Duplicados exactos
# =========================================================

total = df.count()
distintos = df.distinct().count()

print("\n======================================")
print("DUPLICADOS EXACTOS")
print("======================================")

print("TOTAL:", total)
print("DISTINTOS:", distintos)
print(
    "DUPLICADOS:",
    total - distintos
)

# =========================================================
# 8. Estadísticas numéricas
# =========================================================

numericas = [
    c for c, t in df.dtypes
    if t in [
        "int",
        "bigint",
        "double",
        "float",
        "decimal"
    ]
]

print("\n======================================")
print("COLUMNAS NUMERICAS")
print("======================================")

print(numericas)

if numericas:
    df.select(
        *numericas
    ).summary().show(
        truncate=False
    )

spark.stop()