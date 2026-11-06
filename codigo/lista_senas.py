"""
Tabla de las 90 señas del banco (para revisar a mano): número, nombre, dataset de origen, sus 5 personas
y sus 20 muestras.  Uso:  python lista_senas.py   ->  lista_90_senas.csv
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PAQUETE = Path(__file__).resolve().parent.parent
BANCO = PAQUETE / "dataset 90 senas"

filas = []
for carpeta in sorted(BANCO.iterdir(), key=lambda c: c.name):
    if not carpeta.is_dir():
        continue
    archivos = sorted(carpeta.glob("muestra_*.npz"))
    with np.load(archivos[0], allow_pickle=False) as z:
        clase, nombre = int(z["clase"]), str(z["nombre"])
    personas = defaultdict(int)
    for a in archivos:
        personas[re.match(r"muestra_lsc5[04]_(P\d+|Signer_\d+)_", a.name).group(1)] += 1
    dataset = "LSC50" if archivos[0].name.startswith("muestra_lsc50") else "LSC-54"
    filas.append((clase, nombre, dataset, carpeta.name, len(personas), len(archivos),
                  " ".join(f"{p}({n})" for p, n in sorted(personas.items())), " ".join(a.name for a in archivos)))

with open(PAQUETE / "lista_90_senas.csv", "w", encoding="utf-8-sig") as f:
    f.write("clase;sena;dataset;carpeta;personas;muestras;personas_y_muestras;archivos\n")
    for fila in sorted(filas):
        f.write(";".join(str(x) for x in fila) + "\n")
print(f"{len(filas)} señas -> lista_90_senas.csv  "
      f"({sum(f[5] for f in filas)} muestras; todas con 5 personas y 20 muestras: "
      f"{all(f[4] == 5 and f[5] == 20 for f in filas)})")
