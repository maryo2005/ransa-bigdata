from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("profile_flujo_peajes_for_gold")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

FLUJO = (
    "gs://ransa-big-data-2026-lake/"
    "silver/flujo_peajes/"
)

df = spark.read.parquet(FLUJO)

print("\n======================================")
print("FLUJO PEAJES - PERFIL PARA GOLD")
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
    15,
    truncate=False
)

# =========================================================
# 2. Posibles campos temporales
# =========================================================

temporales = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "ANIO",
            "AÑO",
            "MES",
            "PERIODO",
            "FECHA",
            "TRIMESTRE"
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
        .orderBy(c)
        .show(
            100,
            truncate=False
        )
    )

# =========================================================
# 3. Posibles campos de peaje / ubicación / ruta
# =========================================================

ubicacion = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "PEAJE",
            "UNIDAD",
            "ESTACION",
            "RUTA",
            "VIA",
            "TRAMO",
            "DEPART",
            "PROVINC",
            "DISTR",
            "UBIC",
            "COD"
        ]
    )
]

print("\n======================================")
print("CAMPOS DE UBICACION / VIA")
print("======================================")

print(ubicacion)

for c in ubicacion:

    print(f"\nTOP {c}:")

    (
        df
        .groupBy(c)
        .count()
        .orderBy(
            F.desc("count")
        )
        .show(
            100,
            truncate=False
        )
    )

# =========================================================
# 4. Posibles coordenadas
# =========================================================

coords = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "LAT",
            "LON",
            "LONGITUD",
            "COORD"
        ]
    )
]

print("\n======================================")
print("POSIBLES COORDENADAS")
print("======================================")

print(coords)

# =========================================================
# 5. Variables vehiculares
# =========================================================

vehiculares = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "VEH",
            "LIGER",
            "PESAD",
            "EJE",
            "TOTAL",
            "FLUJO",
            "TRANSITO"
        ]
    )
]

print("\n======================================")
print("CAMPOS VEHICULARES")
print("======================================")

print(vehiculares)

for c in vehiculares:

    print(
        c,
        dict(df.dtypes).get(c)
    )

# =========================================================
# 6. Nulos por columna
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
# 8. Conteo de estaciones/peajes
# =========================================================

peaje_cols = [
    c for c in df.columns
    if any(
        x in c.upper()
        for x in [
            "PEAJE",
            "ESTACION"
        ]
    )
]

print("\n======================================")
print("CARDINALIDAD PEAJES")
print("======================================")

for c in peaje_cols:

    print(
        c,
        df.select(c)
        .distinct()
        .count()
    )

# =========================================================
# 9. Corte histórico <=2025
# =========================================================

anio_cols = [
    c for c in df.columns
    if c.upper() in [
        "ANIO",
        "AÑO",
        "ANO"
    ]
]

print("\n======================================")
print("CORTE HISTORICO")
print("======================================")

if anio_cols:

    c = anio_cols[0]

    print(
        "CAMPO ANIO:",
        c
    )

    print(
        "HASTA 2025:",
        df.filter(
            F.col(c) <= 2025
        ).count()
    )

    print(
        "2026 EN ADELANTE:",
        df.filter(
            F.col(c) >= 2026
        ).count()
    )

else:

    print(
        "NO SE DETECTO CAMPO ANIO DIRECTO"
    )

spark.stop()