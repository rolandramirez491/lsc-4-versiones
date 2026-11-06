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

print("=" * 78)
print("LECTOR DE METADATA LSC50")
print("=" * 78)

print(f"\nCarpeta INFO:\n{RUTA_INFO}")

if not RUTA_INFO.exists():
    raise FileNotFoundError(
        f"No existe la carpeta:\n{RUTA_INFO}"
    )

print("\n" + "=" * 78)
print("INFORMACION DE VOLUNTARIOS")
print("=" * 78)

archivos_voluntarios = sorted(
    RUTA_INFO.glob("Volunteer_*.txt")
)

if not archivos_voluntarios:
    print("No se encontraron archivos Volunteer_*.txt.")

for archivo in archivos_voluntarios:
    print("\n" + "-" * 78)
    print(f"ARCHIVO: {archivo.name}")
    print("-" * 78)

    contenido = archivo.read_text(
        encoding="utf-8",
        errors="replace"
    )

    print(contenido)

print("\n" + "=" * 78)
print("INSPECCION DE TIMESTAMPS.XLSX")
print("=" * 78)

if not ARCHIVO_TIMESTAMPS.exists():
    raise FileNotFoundError(
        f"No se encontró:\n{ARCHIVO_TIMESTAMPS}"
    )

libro = pd.ExcelFile(
    ARCHIVO_TIMESTAMPS
)

print("\nHojas detectadas:")

for hoja in libro.sheet_names:
    print(f"- {hoja}")

for hoja in libro.sheet_names:
    print("\n" + "=" * 78)
    print(f"HOJA: {hoja}")
    print("=" * 78)

    tabla = pd.read_excel(
        ARCHIVO_TIMESTAMPS,
        sheet_name=hoja,
        nrows=15
    )

    print("\nColumnas:")

    for indice, columna in enumerate(tabla.columns):
        print(f"{indice}: {columna}")

    print("\nPrimeras 15 filas:")

    print(
        tabla.to_string(
            index=False
        )
    )

print("\n" + "=" * 78)
print("LECTURA TERMINADA")
print("=" * 78)