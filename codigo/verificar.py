"""
VERIFICAR que todo es igual al experimento original.

    python verificar.py entorno      -> compara versiones de Python, librerías y GPU con entorno_original.json
    python verificar.py datos        -> huellas SHA-256 de los datos de entrada y de las 1800 muestras del banco
    python verificar.py reconstruido -> el banco rehecho desde los datasets oficiales es IGUAL al incluido
    python verificar.py resultados   -> los resultados nuevos son iguales a resultados_referencia (muestra por muestra)
    python verificar.py crear        -> (solo el autor) guarda las huellas y el entorno originales
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

PAQUETE = Path(__file__).resolve().parent.parent
HUELLAS = PAQUETE / "huellas.json"
ENTORNO = PAQUETE / "entorno_original.json"
BANCO = PAQUETE / "dataset 90 senas"
RECONSTRUIDO = PAQUETE / "dataset 90 senas (reconstruido)"
REFERENCIA = PAQUETE / "resultados_referencia"
RESULTADOS = PAQUETE / "resultados"
ENTRADAS = ["datos_originales/lsc50_landmarks_60frames.npz", "datos_originales/mapa_etiquetas_lsc50.json",
            "datos_originales/lsc54_conteo.json"]


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def huella_muestra(ruta):
    """Huella del CONTENIDO (los números), no del archivo: el .npz guarda la fecha y cambiaría al rehacerlo."""
    h = hashlib.sha256()
    with np.load(ruta, allow_pickle=False) as z:
        for clave in sorted(z.files):
            a = np.ascontiguousarray(z[clave])
            h.update(clave.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def huellas_banco(carpeta):
    return {f"{a.parent.name}/{a.name}": huella_muestra(a) for a in sorted(carpeta.glob("*/muestra_*.npz"))}


def entorno():
    import cv2
    import torch
    import matplotlib
    return {"python": platform.python_version(), "sistema": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "opencv": cv2.__version__, "matplotlib": matplotlib.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU"}


def comparar_banco(a, b, nombre):
    faltan = sorted(set(a) - set(b)); sobran = sorted(set(b) - set(a))
    distintas = [k for k in a if k in b and a[k] != b[k]]
    if not faltan and not sobran and not distintas:
        print(f"OK {nombre}: las {len(a)} muestras son IDÉNTICAS")
        return True
    print(f"DIFERENTE {nombre}: faltan {len(faltan)}, sobran {len(sobran)}, distintas {len(distintas)}")
    for k in (faltan + sobran + distintas)[:10]:
        print("   ", k)
    return False


def main():
    orden = sys.argv[1] if len(sys.argv) > 1 else "datos"
    if orden == "crear":
        datos = {"entradas": {e: sha256_archivo(PAQUETE / e) for e in ENTRADAS}, "banco": huellas_banco(BANCO)}
        HUELLAS.write_text(json.dumps(datos, indent=1), encoding="utf-8")
        ENTORNO.write_text(json.dumps(entorno(), indent=1), encoding="utf-8")
        print(f"Guardadas {len(datos['banco'])} huellas del banco y el entorno.")
        return

    if orden == "entorno":
        original, actual = json.loads(ENTORNO.read_text(encoding="utf-8")), entorno()
        iguales = True
        for k, v in original.items():
            marca = "OK " if str(actual.get(k)) == str(v) else "DISTINTO"
            iguales &= marca == "OK "
            print(f"  {marca} {k}: original {v} | este equipo {actual.get(k)}")
        print("Entorno IDÉNTICO: los resultados deben salir exactamente iguales." if iguales else
              "Hay diferencias de entorno: los resultados pueden variar en décimas (las conclusiones no deberían cambiar).")
        return

    guardadas = json.loads(HUELLAS.read_text(encoding="utf-8"))
    if orden == "datos":
        ok = True
        for e, h in guardadas["entradas"].items():
            if not (PAQUETE / e).exists():
                # el LSC50 procesado (339 MB) va aparte en el Release de GitHub: solo hace falta para REHACER el banco
                print(f"FALTA {e}  (solo se necesita para rehacer el banco; ver README)")
                continue
            igual = sha256_archivo(PAQUETE / e) == h
            ok &= igual
            print(f"{'OK' if igual else 'DIFERENTE'} {e}")
        ok &= comparar_banco(guardadas["banco"], huellas_banco(BANCO), "banco incluido")
        sys.exit(0 if ok else 1)

    if orden == "reconstruido":
        sys.exit(0 if comparar_banco(guardadas["banco"], huellas_banco(RECONSTRUIDO), "banco reconstruido") else 1)

    if orden == "resultados":
        ref = np.load(REFERENCIA / "predicciones_4.npz")
        nuevo = np.load(RESULTADOS / "predicciones_4.npz")
        ok = True
        for clave in sorted(k for k in ref.files if k.startswith("pred_")):
            iguales = int((ref[clave] == nuevo[clave]).sum())
            ok &= iguales == len(ref[clave])
            print(f"  {clave[5:]}: {iguales} de {len(ref[clave])} predicciones iguales")
        r = json.loads((REFERENCIA / "resultados_4.json").read_text(encoding="utf-8"))["configuraciones"]
        n = json.loads((RESULTADOS / "resultados_4.json").read_text(encoding="utf-8"))["configuraciones"]
        for a, b in zip(r, n):
            print(f"  {a['config']}: referencia {a['acierto'] * 100:.1f} %  |  ahora {b['acierto'] * 100:.1f} %")
        print("RESULTADOS IDÉNTICOS a la referencia." if ok else "Hay diferencias con la referencia (ver arriba).")
        sys.exit(0 if ok else 1)
    print(__doc__)


if __name__ == "__main__":
    main()
