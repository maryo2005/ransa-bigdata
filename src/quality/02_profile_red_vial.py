import csv
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime


def analizar_red_vial(ruta_archivo):
    ruta = Path(ruta_archivo)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    with open(ruta, "r", encoding="cp1252", newline="") as archivo:
        reader = csv.DictReader(archivo, delimiter=";")
        columnas = reader.fieldnames
        filas = list(reader)

    print("=" * 60)
    print("PERFIL DE CALIDAD - RED VIAL NACIONAL")
    print("=" * 60)

    print(f"\nTOTAL REGISTROS: {len(filas)}")
    print(f"TOTAL COLUMNAS: {len(columnas)}")

    # 1. Valores nulos
    print("\n=== NULOS POR COLUMNA ===")

    for columna in columnas:
        nulos = sum(
            1
            for fila in filas
            if fila[columna] is None or fila[columna].strip() == ""
        )

        print(f"{columna}: {nulos}")

    # 2. ID_RVN
    ids = [
        fila["ID_RVN"].strip()
        for fila in filas
    ]

    print("\n=== ID_RVN ===")
    print(f"IDs únicos: {len(set(ids))}")
    print(f"IDs duplicados: {len(ids) - len(set(ids))}")

    # 3. Duplicados exactos
    tuplas = [
        tuple(fila[columna] for columna in columnas)
        for fila in filas
    ]

    print("\n=== DUPLICADOS EXACTOS ===")
    print(
        f"Filas duplicadas exactas: "
        f"{len(tuplas) - len(set(tuplas))}"
    )

    # 4. CODIGO_RUTA
    rutas = [
        fila["CODIGO_RUTA"].strip()
        for fila in filas
    ]

    print("\n=== CODIGO_RUTA ===")
    print(f"Rutas únicas: {len(set(rutas))}")
    print(f"Rutas vacías: {sum(1 for ruta in rutas if ruta == '')}")

    print("Top 10:")

    for valor, cantidad in Counter(rutas).most_common(10):
        print(f"{valor} -> {cantidad}")

    # 5. LONGITUD
    longitudes_validas = []
    longitudes_invalidas = []

    for fila in filas:
        valor = fila["LONGITUD"].strip()

        try:
            numero = float(valor)

            if numero >= 0:
                longitudes_validas.append(numero)
            else:
                longitudes_invalidas.append(valor)

        except ValueError:
            longitudes_invalidas.append(valor)

    print("\n=== LONGITUD ===")
    print(f"Longitudes inválidas: {len(longitudes_invalidas)}")

    if longitudes_validas:
        print(f"Longitud mínima: {min(longitudes_validas)}")
        print(f"Longitud máxima: {max(longitudes_validas)}")
        print(
            f"Longitud total histórica: "
            f"{round(sum(longitudes_validas), 3)}"
        )

    # 6. NRO_CARRILES
    print("\n=== NRO_CARRILES ===")

    carriles = Counter(
        fila["NRO_CARRILES"].strip()
        for fila in filas
    )

    for valor, cantidad in carriles.most_common():
        etiqueta = valor if valor else "[VACIO]"
        print(f"{etiqueta} -> {cantidad}")

    # 7. SUPERFICIE
    print("\n=== SUPERFICIE_L ===")

    superficies = Counter(
        fila["SUPERFICIE_L"].strip()
        for fila in filas
    )

    for valor, cantidad in superficies.most_common():
        etiqueta = valor if valor else "[VACIO]"
        print(f"{etiqueta} -> {cantidad}")

    # 8. ESTADO
    print("\n=== ESTADO_L ===")

    estados = Counter(
        fila["ESTADO_L"].strip()
        for fila in filas
    )

    for valor, cantidad in estados.most_common():
        etiqueta = valor if valor else "[VACIO]"
        print(f"{etiqueta} -> {cantidad}")

    # 9. FECHA_CORTE
    fechas_invalidas = []

    for fila in filas:
        valor = fila["FECHA_CORTE"].strip()

        try:
            datetime.strptime(valor, "%Y%m%d")
        except ValueError:
            fechas_invalidas.append(valor)

    print("\n=== FECHA_CORTE ===")
    print(f"Fechas inválidas: {len(fechas_invalidas)}")
    print(f"Ejemplos: {fechas_invalidas[:10]}")

    print("Fechas únicas:")

    fechas = Counter(
        fila["FECHA_CORTE"].strip()
        for fila in filas
    )

    for valor, cantidad in sorted(fechas.items()):
        print(f"{valor} -> {cantidad}")

    # 10. ES_CONCES
    print("\n=== ES_CONCES ===")

    concesiones = Counter(
        fila["ES_CONCES"].strip()
        for fila in filas
    )

    for valor, cantidad in concesiones.most_common():
        etiqueta = valor if valor else "[VACIO]"
        print(f"{etiqueta} -> {cantidad}")

    # 11. DEPARTAMENTO
    print("\n=== DEPARTAMENTO ===")

    departamentos = [
        fila["DEPARTAMENTO"].strip()
        for fila in filas
    ]

    print(f"Departamentos únicos: {len(set(departamentos))}")

    for valor, cantidad in Counter(departamentos).most_common():
        print(f"{valor} -> {cantidad}")

    print("\n" + "=" * 60)
    print("PERFILADO FINALIZADO")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perfil de calidad del dataset Red Vial Nacional"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del archivo CSV de Red Vial Nacional"
    )

    args = parser.parse_args()

    analizar_red_vial(args.input)