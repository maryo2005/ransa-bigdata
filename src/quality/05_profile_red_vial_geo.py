from pathlib import Path
from collections import Counter
import argparse
import math
import shapefile

parser = argparse.ArgumentParser()

parser.add_argument(
    "--shp",
    required=True
)

args = parser.parse_args()

shp_path = Path(args.shp)

sf = shapefile.Reader(
    str(shp_path),
    encoding="cp1252"
)

campos = [f[0] for f in sf.fields[1:]]
registros = sf.records()
shapes = sf.shapes()

print("=== PERFIL RED VIAL GEOESPACIAL ===")

print("\n=== ESTRUCTURA ===")
print("TOTAL REGISTROS:", len(registros))
print("TOTAL GEOMETRIAS:", len(shapes))
print("TIPO GEOMETRIA:", sf.shapeTypeName)
print("TOTAL CAMPOS:", len(campos))

print("\n=== CAMPOS ===")

for i, campo in enumerate(campos, 1):
    print(f"{i:02d}. {campo}")

print("\n=== NULOS POR CAMPO ===")

for i, campo in enumerate(campos):

    nulos = sum(
        1
        for r in registros
        if r[i] is None
        or (
            isinstance(r[i], str)
            and r[i].strip() == ""
        )
    )

    print(f"{campo}: {nulos}")

idx_orden = campos.index("orden")

ordenes = [
    r[idx_orden]
    for r in registros
]

print("\n=== ORDEN ===")
print("VALORES UNICOS:", len(set(ordenes)))
print(
    "DUPLICADOS:",
    len(ordenes) - len(set(ordenes))
)

idx_ruta = campos.index("codruta")

rutas = [
    str(r[idx_ruta]).strip()
    for r in registros
]

print("\n=== CODRUTA ===")
print("RUTAS UNICAS:", len(set(rutas)))
print(
    "RUTAS VACIAS:",
    sum(x == "" for x in rutas)
)

print("TOP 15 RUTAS:")

for valor, cantidad in Counter(rutas).most_common(15):
    print(valor, cantidad)

idx_carriles = campos.index("carriles")

print("\n=== CARRILES ===")

for valor, cantidad in Counter(
    "[NULL]"
    if r[idx_carriles] is None
    else str(r[idx_carriles])
    for r in registros
).most_common():

    print(valor, cantidad)

idx_superficie = campos.index("superficie")

print("\n=== SUPERFICIE ===")

for valor, cantidad in Counter(
    str(r[idx_superficie]).strip()
    for r in registros
).most_common():

    print(valor, cantidad)

idx_estado = campos.index("estado")

print("\n=== ESTADO ===")

for valor, cantidad in Counter(
    str(r[idx_estado]).strip()
    for r in registros
).most_common():

    print(valor, cantidad)

idx_longitud = campos.index("longitud")

longitudes = []
invalidas = []

for r in registros:

    try:
        valor = float(r[idx_longitud])

        if valor > 0:
            longitudes.append(valor)
        else:
            invalidas.append(valor)

    except (TypeError, ValueError):
        invalidas.append(r[idx_longitud])

print("\n=== LONGITUD ===")
print("INVALIDAS:", len(invalidas))
print("MINIMA:", min(longitudes))
print("MAXIMA:", max(longitudes))
print("SUMA:", round(sum(longitudes), 3))

geometrias_null = sum(
    s.shapeType == 0
    for s in shapes
)

geometrias_sin_puntos = sum(
    not s.points
    for s in shapes
)

cantidad_puntos = [
    len(s.points)
    for s in shapes
]

print("\n=== GEOMETRIAS ===")
print("GEOMETRIAS NULL:", geometrias_null)
print(
    "GEOMETRIAS SIN PUNTOS:",
    geometrias_sin_puntos
)
print(
    "PUNTOS TOTALES:",
    sum(cantidad_puntos)
)
print(
    "MIN PUNTOS/SEGMENTO:",
    min(cantidad_puntos)
)
print(
    "MAX PUNTOS/SEGMENTO:",
    max(cantidad_puntos)
)

lat_min = math.inf
lat_max = -math.inf

lon_min = math.inf
lon_max = -math.inf

puntos_totales = 0
puntos_fuera_rango = 0

for s in shapes:

    for x, y in s.points:

        # Particularidad detectada en la fuente:
        # X contiene latitud
        # Y contiene longitud

        lat = x
        lon = y

        puntos_totales += 1

        lat_min = min(lat_min, lat)
        lat_max = max(lat_max, lat)

        lon_min = min(lon_min, lon)
        lon_max = max(lon_max, lon)

        if not (
            -20 <= lat <= 1
            and -82 <= lon <= -68
        ):
            puntos_fuera_rango += 1

print("\n=== VALIDACION ESPACIAL ===")

print(
    "PUNTOS EVALUADOS:",
    puntos_totales
)

print(
    "PUNTOS FUERA RANGO PERU:",
    puntos_fuera_rango
)

print("LAT MIN:", lat_min)
print("LAT MAX:", lat_max)
print("LON MIN:", lon_min)
print("LON MAX:", lon_max)

print(
    "EJES_INVERTIDOS_FUENTE:",
    1
)

partes = [
    len(s.parts)
    for s in shapes
]

print("\n=== PARTES ===")
print("MIN PARTES:", min(partes))
print("MAX PARTES:", max(partes))

for valor, cantidad in Counter(partes).most_common():
    print(valor, cantidad)

print("\n=== RESULTADO ===")

registro_valido = (
    len(registros) == len(shapes)
    and geometrias_null == 0
    and geometrias_sin_puntos == 0
    and puntos_fuera_rango == 0
)

print(
    "DATASET_GEO_VALIDO:",
    1 if registro_valido else 0
)