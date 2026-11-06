import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


RUTA_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_INFO = (
    RUTA_PROYECTO
    / "datos_lsc"
    / "IMU"
    / "IMU"
    / "INFO"
)

ARCHIVO_TIMESTAMPS = (
    RUTA_INFO
    / "Timestamps.xlsx"
)

RUTA_MODELOS = (
    RUTA_PROYECTO
    / "modelos_entrenados"
)

RUTA_PROCESADO = (
    RUTA_PROYECTO
    / "procesado_lsc50"
)

RUTA_MODELOS.mkdir(
    parents=True,
    exist_ok=True
)

ARCHIVO_MAPA_MODELOS = (
    RUTA_MODELOS
    / "mapa_etiquetas_lsc50.json"
)

ARCHIVO_MAPA_PROCESADO = (
    RUTA_PROCESADO
    / "mapa_etiquetas_lsc50.json"
)


def crear_nombre_tecnico(texto):
    texto = unicodedata.normalize(
        "NFD",
        str(texto)
    )

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    texto = texto.lower().strip()

    texto = re.sub(
        r"[^a-z0-9]+",
        "_",
        texto
    )

    texto = texto.strip("_")

    return texto


print("=" * 78)
print("CREANDO MAPA OFICIAL DE ETIQUETAS LSC50")
print("=" * 78)

if not ARCHIVO_TIMESTAMPS.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{ARCHIVO_TIMESTAMPS}"
    )

libro = pd.ExcelFile(
    ARCHIVO_TIMESTAMPS
)

mapa_referencia = None

for hoja in libro.sheet_names:
    tabla = pd.read_excel(
        ARCHIVO_TIMESTAMPS,
        sheet_name=hoja
    )

    columnas_necesarias = {
        "N",
        "Sign"
    }

    if not columnas_necesarias.issubset(
        tabla.columns
    ):
        raise ValueError(
            f"\nLa hoja '{hoja}' no tiene "
            "las columnas N y Sign."
        )

    tabla = tabla[
        ["N", "Sign"]
    ].dropna()

    mapa_actual = {}

    for _, fila in tabla.iterrows():
        numero = int(fila["N"])
        seña = str(fila["Sign"]).strip()

        mapa_actual[numero] = seña

    print(
        f"\nHoja: {hoja}"
        f"\nSeñas encontradas: "
        f"{len(mapa_actual)}"
    )

    if mapa_referencia is None:
        mapa_referencia = mapa_actual

    elif mapa_actual != mapa_referencia:
        print(
            "\nADVERTENCIA: esta hoja no "
            "coincide exactamente con la "
            "primera hoja."
        )

        diferencias = []

        for numero in sorted(
            set(mapa_referencia)
            | set(mapa_actual)
        ):
            referencia = mapa_referencia.get(numero)
            actual = mapa_actual.get(numero)

            if referencia != actual:
                diferencias.append(
                    (
                        numero,
                        referencia,
                        actual
                    )
                )

        for diferencia in diferencias[:20]:
            print(
                f"N={diferencia[0]} | "
                f"Referencia={diferencia[1]} | "
                f"Hoja actual={diferencia[2]}"
            )

        raise ValueError(
            "\nNo se creará el mapa hasta "
            "resolver las diferencias."
        )

if mapa_referencia is None:
    raise RuntimeError(
        "No se encontró ningún mapa de señas."
    )

clases_esperadas = set(range(50))
clases_encontradas = set(mapa_referencia.keys())

if clases_encontradas != clases_esperadas:
    raise ValueError(
        "\nLas clases encontradas no son "
        "exactamente de 0 a 49."
        f"\nEncontradas: "
        f"{sorted(clases_encontradas)}"
    )

lista_etiquetas = []

print("\n" + "=" * 78)
print("MAPA OFICIAL DE LAS 50 SEÑAS")
print("=" * 78)

for numero in sorted(mapa_referencia):
    seña = mapa_referencia[numero]

    etiqueta_modelo = (
        f"SENA_{numero:04d}"
    )

    nombre_tecnico = crear_nombre_tecnico(
        seña
    )

    registro = {
        "id": numero,
        "etiqueta_modelo": etiqueta_modelo,
        "nombre_lsc": seña,
        "nombre_tecnico": nombre_tecnico
    }

    lista_etiquetas.append(registro)

    print(
        f"{etiqueta_modelo} "
        f"-> {seña} "
        f"-> {nombre_tecnico}"
    )

mapa_json = {
    "dataset": "LSC50",
    "fuente": (
        "Timestamps.xlsx incluido "
        "en el dataset LSC50"
    ),
    "cantidad_clases": len(lista_etiquetas),
    "etiquetas": lista_etiquetas
}

for archivo_salida in [
    ARCHIVO_MAPA_MODELOS,
    ARCHIVO_MAPA_PROCESADO
]:
    with open(
        archivo_salida,
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(
            mapa_json,
            archivo,
            indent=4,
            ensure_ascii=False
        )

print("\n" + "=" * 78)
print("MAPA CREADO CORRECTAMENTE")
print("=" * 78)

print(
    f"\nMapa para modelos:\n"
    f"{ARCHIVO_MAPA_MODELOS}"
)

print(
    f"\nMapa para datos procesados:\n"
    f"{ARCHIVO_MAPA_PROCESADO}"
)