import csv
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime


def analizar_dataset(ruta_archivo):
    ruta = Path(ruta_archivo)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    with open(ruta, "r", encoding="cp1252", newline="") as archivo:
        reader = csv.DictReader(archivo, delimiter=";")
        columnas = reader.fieldnames
        filas = list(reader)

    print("=" * 60)
    print("PERFIL DE CALIDAD - EMERGENCIAS MTC")
    print("=" * 60)

    print(f"\nTOTAL REGISTROS: {len(filas)}")
    print(f"TOTAL COLUMNAS: {len(columnas)}")

    # 1. Valores nulos o vacíos

    print("\n=== NULOS POR COLUMNA ===")

    for columna in columnas:
        nulos = sum(
            1
            for fila in filas
            if fila[columna] is None or fila[columna].strip() == ""
        )

        print(f"{columna}: {nulos}")

    # 2. Validación de ID_EMERGENCIA

    ids = [
        fila["ID_EMERGENCIA"].strip()
        for fila in filas
    ]

    conteo_ids = Counter(ids)

    ids_duplicados = [
        identificador
        for identificador, cantidad in conteo_ids.items()
        if cantidad > 1
    ]

    print("\n=== ID_EMERGENCIA ===")
    print(f"IDs únicos: {len(set(ids))}")
    print(f"IDs duplicados: {len(ids) - len(set(ids))}")
    print(f"Ejemplos IDs duplicados: {ids_duplicados[:10]}")

    # 3. Duplicados exactos

    tuplas = [
        tuple(fila[columna] for columna in columnas)
        for fila in filas
    ]

    duplicados_exactos = len(tuplas) - len(set(tuplas))

    print("\n=== DUPLICADOS EXACTOS ===")
    print(f"Filas duplicadas exactas: {duplicados_exactos}")

    # 4. Validación de FECHA_OCURRENCIA

    fechas_invalidas = []

    for fila in filas:
        valor = fila["FECHA_OCURRENCIA"].strip()

        try:
            datetime.strptime(valor, "%d/%m/%Y")
        except ValueError:
            fechas_invalidas.append(valor)

    print("\n=== FECHA_OCURRENCIA ===")
    print(f"Fechas inválidas: {len(fechas_invalidas)}")
    print(f"Ejemplos: {fechas_invalidas[:10]}")

    # 5. Validación de FECHA_CORTE

    fechas_corte_invalidas = []

    for fila in filas:
        valor = fila["FECHA_CORTE"].strip()

        try:
            datetime.strptime(valor, "%Y%m%d")
        except ValueError:
            fechas_corte_invalidas.append(valor)

    print("\n=== FECHA_CORTE ===")
    print(f"Fechas corte inválidas: {len(fechas_corte_invalidas)}")
    print(f"Ejemplos: {fechas_corte_invalidas[:10]}")

    # 6. Validación de coordenadas

    latitudes = []
    longitudes = []
    coordenadas_invalidas = 0

    for fila in filas:

        try:
            latitud = float(fila["LATITUD"].strip())
            longitud = float(fila["LONGITUD"].strip())

            if -90 <= latitud <= 90 and -180 <= longitud <= 180:
                latitudes.append(latitud)
                longitudes.append(longitud)
            else:
                coordenadas_invalidas += 1

        except (ValueError, AttributeError):
            coordenadas_invalidas += 1

    print("\n=== COORDENADAS ===")
    print(f"Coordenadas inválidas o vacías: {coordenadas_invalidas}")

    if latitudes:
        print(f"Latitud min: {min(latitudes)}")
        print(f"Latitud max: {max(latitudes)}")

    if longitudes:
        print(f"Longitud min: {min(longitudes)}")
        print(f"Longitud max: {max(longitudes)}")

    # 7. Estado inicial

    print("\n=== ESTADO INICIAL ===")

    estados_iniciales = Counter(
        fila["ESTADO_INICIAL"].strip()
        for fila in filas
    )

    for valor, cantidad in estados_iniciales.most_common():
        print(f"{valor} -> {cantidad}")

    # 8. Estado actual

    print("\n=== ESTADO ACTUAL ===")

    estados_actuales = Counter(
        fila["ESTADO_ACTUAL"].strip()
        for fila in filas
    )

    for valor, cantidad in estados_actuales.most_common():
        print(f"{valor} -> {cantidad}")

    # 9. Red afectada

    redes = [
        fila["RED_AFECTADA"].strip()
        for fila in filas
    ]

    conteo_redes = Counter(redes)

    print("\n=== RED_AFECTADA ===")
    print(f"Valores únicos: {len(set(redes))}")
    print("Top 10:")

    for valor, cantidad in conteo_redes.most_common(10):
        print(f"{valor} -> {cantidad}")

    print("\n" + "=" * 60)
    print("PERFILADO FINALIZADO")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perfil de calidad del dataset de Emergencias MTC"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del archivo CSV de Emergencias"
    )

    args = parser.parse_args()

    analizar_dataset(args.input)