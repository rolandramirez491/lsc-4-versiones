import json
from pathlib import Path

import numpy as np
import pandas as pd


FRAMES_POR_SECUENCIA = 60

RUTA_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_LANDMARKS = (
    RUTA_PROYECTO
    / "datos_lsc"
    / "LANDMARKS"
    / "LANDMARKS"
)

RUTA_CUERPO = (
    RUTA_LANDMARKS
    / "BODY_LANDMARKS"
)

RUTA_CARA = (
    RUTA_LANDMARKS
    / "FACE_LANDMARKS"
)

RUTA_MANO_IZQUIERDA = (
    RUTA_LANDMARKS
    / "HANDS_LANDMARKS"
    / "LEFT_HAND_LANDMARKS"
)

RUTA_MANO_DERECHA = (
    RUTA_LANDMARKS
    / "HANDS_LANDMARKS"
    / "RIGHT_HAND_LANDMARKS"
)

RUTA_SALIDA = (
    RUTA_PROYECTO
    / "procesado_lsc50"
)

ARCHIVO_SALIDA = (
    RUTA_SALIDA
    / "lsc50_landmarks_60frames.npz"
)

ARCHIVO_METADATA = (
    RUTA_SALIDA
    / "metadata_lsc50.json"
)


def leer_landmarks(ruta_csv):
    datos = pd.read_csv(ruta_csv)

    datos = datos.iloc[:, 1:]

    matriz = datos.to_numpy(
        dtype=np.float32
    )

    matriz = np.nan_to_num(
        matriz,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return matriz


def reescalar_secuencia(secuencia, frames_destino):
    cantidad_frames = len(secuencia)

    if cantidad_frames == 0:
        raise ValueError(
            "Se encontró una secuencia sin frames."
        )

    indices = np.linspace(
        0,
        cantidad_frames - 1,
        frames_destino
    ).astype(int)

    return secuencia[indices]


def extraer_ids(nombre_archivo):
    partes = Path(nombre_archivo).stem.split("_")

    if len(partes) != 3:
        raise ValueError(
            f"Nombre de archivo inválido: {nombre_archivo}"
        )

    clase = int(partes[0])
    participante = int(partes[1])
    repeticion = int(partes[2])

    return clase, participante, repeticion


def validar_archivos(archivos_referencia, carpeta, nombre):
    archivos_actuales = {
        archivo.name
        for archivo in carpeta.glob("*.csv")
    }

    faltantes = [
        archivo.name
        for archivo in archivos_referencia
        if archivo.name not in archivos_actuales
    ]

    if faltantes:
        raise FileNotFoundError(
            f"\nFaltan {len(faltantes)} archivos en {nombre}."
            f"\nEjemplo faltante: {faltantes[0]}"
        )

    print(
        f"OK - {nombre}: "
        f"{len(archivos_actuales)} CSV encontrados."
    )


print("=" * 75)
print("PREPARACION DEL DATASET LSC50")
print("=" * 75)

print("\nCarpetas utilizadas:")
print(f"Cuerpo:          {RUTA_CUERPO}")
print(f"Cara:            {RUTA_CARA}")
print(f"Mano izquierda:  {RUTA_MANO_IZQUIERDA}")
print(f"Mano derecha:    {RUTA_MANO_DERECHA}")

rutas_obligatorias = [
    RUTA_CUERPO,
    RUTA_CARA,
    RUTA_MANO_IZQUIERDA,
    RUTA_MANO_DERECHA
]

for ruta in rutas_obligatorias:
    if not ruta.exists():
        raise FileNotFoundError(
            f"\nNo existe esta carpeta:\n{ruta}"
        )

archivos_cuerpo = sorted(
    RUTA_CUERPO.glob("*.csv")
)

if not archivos_cuerpo:
    raise RuntimeError(
        "No se encontraron CSV en BODY_LANDMARKS."
    )

print(f"\nCSV de cuerpo encontrados: {len(archivos_cuerpo)}")

validar_archivos(
    archivos_cuerpo,
    RUTA_CARA,
    "FACE_LANDMARKS"
)

validar_archivos(
    archivos_cuerpo,
    RUTA_MANO_IZQUIERDA,
    "LEFT_HAND_LANDMARKS"
)

validar_archivos(
    archivos_cuerpo,
    RUTA_MANO_DERECHA,
    "RIGHT_HAND_LANDMARKS"
)

RUTA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

secuencias = []
etiquetas = []
participantes = []
repeticiones = []
nombres_archivo = []

total = len(archivos_cuerpo)

print("\n" + "=" * 75)
print("LEYENDO Y UNIENDO CARA + CUERPO + DOS MANOS")
print("=" * 75)

for indice, archivo_cuerpo in enumerate(
    archivos_cuerpo,
    start=1
):
    nombre = archivo_cuerpo.name

    archivo_cara = RUTA_CARA / nombre
    archivo_mano_izquierda = (
        RUTA_MANO_IZQUIERDA / nombre
    )
    archivo_mano_derecha = (
        RUTA_MANO_DERECHA / nombre
    )

    cuerpo = leer_landmarks(archivo_cuerpo)
    cara = leer_landmarks(archivo_cara)
    mano_izquierda = leer_landmarks(
        archivo_mano_izquierda
    )
    mano_derecha = leer_landmarks(
        archivo_mano_derecha
    )

    cuerpo = reescalar_secuencia(
        cuerpo,
        FRAMES_POR_SECUENCIA
    )

    cara = reescalar_secuencia(
        cara,
        FRAMES_POR_SECUENCIA
    )

    mano_izquierda = reescalar_secuencia(
        mano_izquierda,
        FRAMES_POR_SECUENCIA
    )

    mano_derecha = reescalar_secuencia(
        mano_derecha,
        FRAMES_POR_SECUENCIA
    )

    secuencia_completa = np.concatenate(
        [
            cuerpo,
            cara,
            mano_izquierda,
            mano_derecha
        ],
        axis=1
    )

    clase, participante, repeticion = extraer_ids(
        nombre
    )

    secuencias.append(secuencia_completa)
    etiquetas.append(clase)
    participantes.append(participante)
    repeticiones.append(repeticion)
    nombres_archivo.append(nombre)

    if indice == 1:
        print("\nPrimer ejemplo leído:")
        print(f"Archivo: {nombre}")
        print(f"Cuerpo: {cuerpo.shape}")
        print(f"Cara: {cara.shape}")
        print(
            f"Mano izquierda: "
            f"{mano_izquierda.shape}"
        )
        print(
            f"Mano derecha: "
            f"{mano_derecha.shape}"
        )
        print(
            f"Secuencia total: "
            f"{secuencia_completa.shape}"
        )

    if indice % 50 == 0 or indice == total:
        print(
            f"Procesados: {indice}/{total} "
            f"({indice / total * 100:.1f} %)"
        )

X = np.asarray(
    secuencias,
    dtype=np.float32
)

y = np.asarray(
    etiquetas,
    dtype=np.int32
)

p = np.asarray(
    participantes,
    dtype=np.int32
)

r = np.asarray(
    repeticiones,
    dtype=np.int32
)

n = np.asarray(
    nombres_archivo
)

print("\n" + "=" * 75)
print("RESUMEN DEL DATASET PROCESADO")
print("=" * 75)

print(f"X - Secuencias: {X.shape}")
print(f"y - Etiquetas: {y.shape}")
print(f"Participantes: {p.shape}")
print(f"Repeticiones: {r.shape}")

print(f"\nClases encontradas: {len(np.unique(y))}")
print(f"Valores de clase: {np.unique(y)}")

print(
    f"Participantes encontrados: "
    f"{np.unique(p)}"
)

print(
    f"Repeticiones encontradas: "
    f"{np.unique(r)}"
)

print("\nGuardando archivo procesado...")

np.savez_compressed(
    ARCHIVO_SALIDA,
    X=X,
    y=y,
    participantes=p,
    repeticiones=r,
    nombres=n
)

metadata = {
    "dataset": "LSC50",
    "muestras": int(len(X)),
    "frames_por_secuencia": int(
        FRAMES_POR_SECUENCIA
    ),
    "caracteristicas_por_frame": int(
        X.shape[2]
    ),
    "cantidad_clases": int(
        len(np.unique(y))
    ),
    "clases_numericas": [
        int(clase)
        for clase in np.unique(y)
    ],
    "participantes": [
        int(participante)
        for participante in np.unique(p)
    ],
    "repeticiones": [
        int(repeticion)
        for repeticion in np.unique(r)
    ],
    "orden_features": [
        "BODY_LANDMARKS",
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS"
    ]
}

with open(
    ARCHIVO_METADATA,
    "w",
    encoding="utf-8"
) as archivo_metadata:
    json.dump(
        metadata,
        archivo_metadata,
        indent=4,
        ensure_ascii=False
    )

tamano_mb = ARCHIVO_SALIDA.stat().st_size / (
    1024 * 1024
)

print("\n" + "=" * 75)
print("PROCESAMIENTO TERMINADO")
print("=" * 75)

print(f"Archivo creado:\n{ARCHIVO_SALIDA}")
print(f"Tamaño: {tamano_mb:.2f} MB")

print(f"\nMetadata creada:\n{ARCHIVO_METADATA}")

print(
    "\nTodavía no se han asignado palabras "
    "como HOLA o GRACIAS."
)

print(
    "Por ahora las clases se guardan como "
    "SENA_0000 hasta SENA_0049."
)