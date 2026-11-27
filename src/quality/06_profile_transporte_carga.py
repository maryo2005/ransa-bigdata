from pathlib import Path
from collections import Counter
from datetime import datetime
import csv

ARCHIVOS = [
    {
        "ruta": Path.home() / "ransa_stage/transporte_carga/carga_22_24.csv",
        "nombre": "2022_2024",
        "encoding": "latin1"
    },
    {
        "ruta": Path.home() / "ransa_stage/transporte_carga/carga_25.csv",
        "nombre": "2025",
        "encoding": "cp1252"
    }
]

COLUMNAS_NUMERICAS = [
    "ANIO_FAB",
    "N_ASIENTOS",
    "N_LLANTAS",
    "N_EJES",
    "CARGA_UTIL",
    "P_SECO",
    "P_BRUTO",
    "LARGO",
    "ANCHO",
    "ALTO"
]

COLUMNAS_FECHA = [
    "FECHA_RESOLUCION",
    "VIGENCIA_HASTA",
    "FECHA_CORTE"
]

CATEGORICAS = [
    "PERMISO_OPER",
    "MARCA",
    "SERVICIO",
    "CLASE",
    "COMBUSTIBLE",
    "DEPARTAMENTO"
]

SENSIBLES = [
    "RUC",
    "RAZON_SOCIAL",
    "PLACA",
    "N_CHASIS",
    "N_MOTOR"
]


def vacio(valor):
    return valor is None or valor.strip() == ""


def token_nd(valor):
    return valor.strip().upper() in {
        "ND",
        "N/D",
        "NO DISPONIBLE",
        "SIN INFORMACION",
        "SIN INFORMACIÓN"
    }


def parse_numero(valor):
    valor = valor.strip()

    if vacio(valor) or token_nd(valor):
        return None

    valor = valor.replace(",", ".")

    try:
        return float(valor)
    except ValueError:
        return "INVALIDO"


def tipo_fecha(valor):
    valor = valor.strip()

    if vacio(valor):
        return "VACIO"

    formatos = [
        ("%Y%m%d", "YYYYMMDD"),
        ("%d/%m/%Y", "DD/MM/YYYY"),
        ("%Y-%m-%d", "YYYY-MM-DD")
    ]

    for formato, nombre in formatos:
        try:
            datetime.strptime(valor, formato)
            return nombre
        except ValueError:
            pass

    return "NO_FECHA"


total_general = 0

for archivo in ARCHIVOS:

    ruta = archivo["ruta"]
    encoding = archivo["encoding"]
    nombre = archivo["nombre"]

    print("\n" + "=" * 70)
    print("ARCHIVO:", nombre)
    print("RUTA:", ruta)
    print("ENCODING:", encoding)

    total = 0

    blancos = Counter()
    nd = Counter()

    numericos_validos = Counter()
    numericos_invalidos = Counter()
    numericos_no_disponibles = Counter()

    formatos_fecha = {
        campo: Counter()
        for campo in COLUMNAS_FECHA
    }

    categorias = {
        campo: Counter()
        for campo in CATEGORICAS
    }

    sensibles_blancos = Counter()
    sensibles_nd = Counter()

    filas_c1 = 0

    with open(
        ruta,
        "r",
        encoding=encoding,
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter=";"
        )

        for row in reader:

            total += 1

            for campo, valor in row.items():

                valor = "" if valor is None else valor

                if vacio(valor):
                    blancos[campo] += 1

                if token_nd(valor):
                    nd[campo] += 1

            for campo in COLUMNAS_NUMERICAS:

                resultado = parse_numero(
                    row[campo]
                )

                if resultado is None:
                    numericos_no_disponibles[campo] += 1

                elif resultado == "INVALIDO":
                    numericos_invalidos[campo] += 1

                else:
                    numericos_validos[campo] += 1

            for campo in COLUMNAS_FECHA:

                formatos_fecha[campo][
                    tipo_fecha(row[campo])
                ] += 1

            for campo in CATEGORICAS:

                valor = row[campo].strip().upper()

                if valor == "":
                    valor = "[VACIO]"

                categorias[campo][valor] += 1

            for campo in SENSIBLES:

                valor = row[campo]

                if vacio(valor):
                    sensibles_blancos[campo] += 1

                if token_nd(valor):
                    sensibles_nd[campo] += 1

            texto = "|".join(
                str(v)
                for v in row.values()
                if v is not None
            )

            if any(
                0x80 <= ord(c) <= 0x9F
                for c in texto
            ):
                filas_c1 += 1

    total_general += total

    print("\nTOTAL REGISTROS:", total)

    print("\n=== VACIOS ===")
    for campo, cantidad in blancos.items():
        if cantidad:
            print(campo, cantidad)

    print("\n=== TOKENS ND / NO DISPONIBLE ===")
    for campo, cantidad in nd.items():
        if cantidad:
            print(campo, cantidad)

    print("\n=== NUMERICOS ===")

    for campo in COLUMNAS_NUMERICAS:

        print(
            campo,
            "| validos:",
            numericos_validos[campo],
            "| no disponibles:",
            numericos_no_disponibles[campo],
            "| invalidos:",
            numericos_invalidos[campo]
        )

    print("\n=== FORMATOS DE FECHA ===")

    for campo in COLUMNAS_FECHA:

        print("\n", campo)

        for formato, cantidad in (
            formatos_fecha[campo]
            .most_common()
        ):
            print(
                formato,
                "->",
                cantidad
            )

    print("\n=== IDENTIFICADORES SENSIBLES ===")

    for campo in SENSIBLES:

        print(
            campo,
            "| vacios:",
            sensibles_blancos[campo],
            "| ND:",
            sensibles_nd[campo]
        )

    print("\n=== CATEGORIAS PRINCIPALES ===")

    for campo in CATEGORICAS:

        print("\n", campo)

        for valor, cantidad in (
            categorias[campo]
            .most_common(15)
        ):
            print(
                valor,
                "->",
                cantidad
            )

    print("\n=== ENCODING / TEXTO ===")

    print(
        "FILAS CON CARACTERES C1:",
        filas_c1
    )

print("\n" + "=" * 70)
print("TOTAL GENERAL:", total_general)