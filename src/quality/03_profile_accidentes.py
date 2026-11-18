import csv
import argparse
from pathlib import Path
from collections import Counter


def analizar_accidentes(ruta_archivo):
    ruta = Path(ruta_archivo)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}"
        )

    with open(
        ruta,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as archivo:
        reader = csv.DictReader(
            archivo,
            delimiter=";"
        )

        filas = list(reader)
        columnas = reader.fieldnames

    print("=" * 60)
    print("PERFIL DE CALIDAD - ACCIDENTES OSITRAN")
    print("=" * 60)

    print(f"\nTOTAL REGISTROS: {len(filas)}")
    print(f"TOTAL COLUMNAS: {len(columnas)}")

    # 1. Nulos y vacíos
    print("\n=== NULOS / VACIOS ===")

    for columna in columnas:
        vacios = sum(
            1
            for fila in filas
            if fila[columna] is None
            or fila[columna].strip() == ""
        )

        print(f"{columna}: {vacios}")

    # 2. Duplicados exactos
    tuplas = [
        tuple(
            fila[columna]
            for columna in columnas
        )
        for fila in filas
    ]

    duplicados = (
        len(tuplas)
        - len(set(tuplas))
    )

    print("\n=== DUPLICADOS EXACTOS ===")
    print(
        f"Filas duplicadas exactas: {duplicados}"
    )
    print(
        "IMPORTANTE: no se eliminarán "
        "automáticamente."
    )

    # 3. Año
    print("\n=== ANIO ===")

    anios = Counter(
        fila["ANIO"].strip()
        for fila in filas
    )

    for valor, cantidad in sorted(
        anios.items()
    ):
        print(
            f"{valor if valor else '[VACIO]'} "
            f"-> {cantidad}"
        )

    # 4. Mes
    print("\n=== MES ===")

    meses = Counter(
        fila["MES"].strip()
        for fila in filas
    )

    for valor, cantidad in sorted(
        meses.items(),
        key=lambda x: (
            int(x[0])
            if x[0].isdigit()
            else 99
        )
    ):
        print(
            f"{valor if valor else '[VACIO]'} "
            f"-> {cantidad}"
        )

    # 5. Ruta
    rutas = [
        fila["RUTA"].strip()
        for fila in filas
    ]

    print("\n=== RUTA ===")
    print(
        f"Rutas únicas: {len(set(rutas))}"
    )
    print(
        "RUTA vacía:",
        sum(
            1
            for valor in rutas
            if valor == ""
        )
    )
    print(
        "RUTA = SIN INFO:",
        sum(
            1
            for valor in rutas
            if valor.upper() == "SIN INFO"
        )
    )

    print("Top 15:")

    for valor, cantidad in Counter(
        rutas
    ).most_common(15):
        print(
            f"{valor if valor else '[VACIO]'} "
            f"-> {cantidad}"
        )

    # 6. Tramo
    tramos = [
        fila["TRAMO"].strip()
        for fila in filas
    ]

    print("\n=== TRAMO ===")
    print(
        f"Tramos únicos: {len(set(tramos))}"
    )
    print(
        "TRAMO vacío:",
        sum(
            1
            for valor in tramos
            if valor == ""
        )
    )
    print(
        "TRAMO = SIN INFO:",
        sum(
            1
            for valor in tramos
            if valor.upper() == "SIN INFO"
        )
    )

    # 7. Tipo de accidente
    print("\n=== TIPO_ACCIDENTE ===")

    tipos_accidente = Counter(
        fila["TIPO_ACCIDENTE"].strip()
        for fila in filas
    )

    for valor, cantidad in (
        tipos_accidente.most_common(20)
    ):
        print(
            f"{valor if valor else '[VACIO]'} "
            f"-> {cantidad}"
        )

    # 8. Causa del accidente
    print("\n=== CAUSA_ACCIDENTE ===")

    causas = Counter(
        fila["CAUSA_ACCIDENTE"].strip()
        for fila in filas
    )

    for valor, cantidad in (
        causas.most_common(20)
    ):
        print(
            f"{valor if valor else '[VACIO]'} "
            f"-> {cantidad}"
        )

    # 9. Campos numéricos
    campos_numericos = [
        "CANTIDAD VEHINVOLUCRADOS",
        "CANTIDAD AFECTADOS",
        "CANTIDAD ACCIDENTES"
    ]

    for campo in campos_numericos:

        validos = []
        invalidos = []

        for fila in filas:
            valor = fila[campo].strip()

            try:
                numero = int(valor)

                if numero >= 0:
                    validos.append(numero)
                else:
                    invalidos.append(valor)

            except ValueError:
                invalidos.append(valor)

        print(f"\n=== {campo} ===")
        print(
            f"Valores inválidos: "
            f"{len(invalidos)}"
        )

        if validos:
            print(
                f"Mínimo: {min(validos)}"
            )
            print(
                f"Máximo: {max(validos)}"
            )
            print(
                f"Suma: {sum(validos)}"
            )

    # 10. Tipo de vía
    print("\n=== TIPO_VIA ===")

    tipos_via = Counter(
        fila["TIPO_VIA"].strip()
        for fila in filas
    )

    for valor, cantidad in (
        tipos_via.most_common()
    ):
        print(
            f"{valor if valor else '[VACIO]'} "
            f"-> {cantidad}"
        )

    # 11. Clima
    print("\n=== CLIMA ===")

    climas = Counter(
        fila["CLIMA"].strip()
        for fila in filas
    )

    for valor, cantidad in (
        climas.most_common()
    ):
        print(
            f"{valor if valor else '[VACIO]'} "
            f"-> {cantidad}"
        )

    print("\n" + "=" * 60)
    print("PERFILADO FINALIZADO")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Perfil de calidad del dataset "
            "de Accidentes OSITRAN"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del CSV de Accidentes OSITRAN"
    )

    args = parser.parse_args()

    analizar_accidentes(args.input)