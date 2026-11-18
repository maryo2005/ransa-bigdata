import csv
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime


def numero(valor):
    return int(valor.strip().replace(" ", ""))


def analizar_flujo_peajes(ruta_archivo):
    ruta = Path(ruta_archivo)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró: {ruta}")

    with open(ruta, "r", encoding="cp1252", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)

    columnas = header[:22]
    registros = [
        dict(zip(columnas, row[:22]))
        for row in rows
    ]

    print("=" * 60)
    print("PERFIL DE CALIDAD - FLUJO VEHICULAR EN PEAJES")
    print("=" * 60)

    print("\nTOTAL REGISTROS:", len(registros))
    print("COLUMNAS FISICAS:", len(header))
    print("COLUMNAS UTILES:", len(columnas))

    print("\n=== COLUMNAS EXTRA ===")

    for idx in [22, 23, 24]:
        no_vacios = sum(
            1
            for row in rows
            if len(row) > idx and row[idx].strip()
        )
        print(
            f"Columna física {idx + 1}: "
            f"{no_vacios} valores no vacíos"
        )

    print("\n=== NULOS / VACIOS ===")

    for col in columnas:
        vacios = sum(
            1
            for r in registros
            if r[col] is None or r[col].strip() == ""
        )
        print(f"{col}: {vacios}")

    tuplas = [
        tuple(r[col] for col in columnas)
        for r in registros
    ]

    print("\n=== DUPLICADOS EXACTOS ===")
    print(
        "Filas duplicadas exactas:",
        len(tuplas) - len(set(tuplas))
    )

    codigos = [
        r["CODIGO_PEAJE"].strip()
        for r in registros
    ]

    print("\n=== CODIGO_PEAJE ===")
    print("Códigos únicos:", len(set(codigos)))

    peajes = [
        r["NOMBRE_PEAJE"].strip()
        for r in registros
    ]

    print("\n=== NOMBRE_PEAJE ===")
    print("Peajes únicos:", len(set(peajes)))

    print("\n=== ADMINIST ===")

    for valor, cantidad in Counter(
        r["ADMINIST"].strip()
        for r in registros
    ).most_common():
        print(valor, "->", cantidad)

    print("\n=== ANIO ===")

    for valor, cantidad in sorted(
        Counter(
            r["ANIO"].strip()
            for r in registros
        ).items()
    ):
        print(valor, "->", cantidad)

    print("\n=== PERIODO ===")

    periodos_invalidos = []

    for r in registros:
        valor = r["PERIODO"].strip()

        try:
            datetime.strptime(valor, "%Y%m")
        except ValueError:
            periodos_invalidos.append(valor)

    periodos = sorted(
        set(r["PERIODO"].strip() for r in registros)
    )

    print("Periodos inválidos:", len(periodos_invalidos))
    print("Periodos únicos:", len(periodos))
    print("Primer periodo:", periodos[0])
    print("Último periodo:", periodos[-1])

    inconsistencias_periodo = 0

    for r in registros:
        esperado = (
            r["ANIO"].strip()
            + r["MES"].strip().zfill(2)
        )

        if esperado != r["PERIODO"].strip():
            inconsistencias_periodo += 1

    print(
        "Inconsistencias ANIO+MES vs PERIODO:",
        inconsistencias_periodo
    )

    campos_numericos = [
        "VEH_LIGEROS_TAR_DIF",
        "VEH_LIGEROS_AUTOMOVILES",
        "VEH_PESADOS_TAR_DIF",
        "VEH_PESADOS_2E",
        "VEH_PESADOS_3E",
        "VEH_PESADOS_4E",
        "VEH_PESADOS_5E",
        "VEH_PESADOS_6E",
        "VEH_PESADOS_7E",
        "VEH_LIGEROS_TOTAL",
        "VEH_LIGEROS_IMD",
        "VEH_PESADOS_TOTAL",
        "VEH_PESADOS_IMD",
        "VEH_TOTAL",
        "VEH_IMD"
    ]

    for campo in campos_numericos:
        validos = []
        invalidos = []

        for r in registros:
            try:
                valor = numero(r[campo])

                if valor >= 0:
                    validos.append(valor)
                else:
                    invalidos.append(r[campo])

            except ValueError:
                invalidos.append(r[campo])

        print(f"\n=== {campo} ===")
        print("Inválidos:", len(invalidos))

        if validos:
            print("Mínimo:", min(validos))
            print("Máximo:", max(validos))

    inconsistencia_ligeros = 0
    inconsistencia_pesados = 0
    inconsistencia_total = 0

    for r in registros:
        ligeros = (
            numero(r["VEH_LIGEROS_TAR_DIF"])
            + numero(r["VEH_LIGEROS_AUTOMOVILES"])
        )

        if ligeros != numero(r["VEH_LIGEROS_TOTAL"]):
            inconsistencia_ligeros += 1

        pesados = (
            numero(r["VEH_PESADOS_TAR_DIF"])
            + numero(r["VEH_PESADOS_2E"])
            + numero(r["VEH_PESADOS_3E"])
            + numero(r["VEH_PESADOS_4E"])
            + numero(r["VEH_PESADOS_5E"])
            + numero(r["VEH_PESADOS_6E"])
            + numero(r["VEH_PESADOS_7E"])
        )

        if pesados != numero(r["VEH_PESADOS_TOTAL"]):
            inconsistencia_pesados += 1

        total = (
            numero(r["VEH_LIGEROS_TOTAL"])
            + numero(r["VEH_PESADOS_TOTAL"])
        )

        if total != numero(r["VEH_TOTAL"]):
            inconsistencia_total += 1

    print("\n=== CONSISTENCIA DE TOTALES ===")
    print(
        "Inconsistencias VEH_LIGEROS_TOTAL:",
        inconsistencia_ligeros
    )
    print(
        "Inconsistencias VEH_PESADOS_TOTAL:",
        inconsistencia_pesados
    )
    print(
        "Inconsistencias VEH_TOTAL:",
        inconsistencia_total
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    args = parser.parse_args()

    analizar_flujo_peajes(args.input)