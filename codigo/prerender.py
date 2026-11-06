"""
Dibuja de una vez (en TODOS los núcleos del CPU) las imágenes de los trazos:
la original de cada muestra y COPIAS versiones aumentadas (girada, escalada, espejo...).
Así, al entrenar, la GPU no espera a que Python dibuje cada imagen.
"""
import os
from multiprocessing import Pool

import numpy as np

from dibujo_senas import aumentar_muestra, dibujar_trazos

HILOS_SISTEMA = 2        # los 2 primeros hilos del CPU (0 y 1) siempre quedan para Windows


def nucleos_libres():
    """Hilos que usa el entrenamiento: todos menos los del sistema (con 20 hilos: 18)."""
    return max(1, (os.cpu_count() or 4) - HILOS_SISTEMA)


def bajar_prioridad():
    """A TOPE, pero solo en los hilos 2..último: los 2 primeros quedan libres para Windows,
    el mouse y las ventanas (así el PC no se congela). Los procesos hijos heredan esto."""
    total = os.cpu_count() or 4
    hilos = list(range(min(HILOS_SISTEMA, total - 1), total))
    try:
        import ctypes
        mascara = sum(1 << h for h in hilos)
        ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),
                                                      ctypes.c_size_t(mascara))
    except Exception:
        try:
            os.sched_setaffinity(0, hilos)
        except Exception:
            pass


def _lote(args):
    trazos, validos, extras, semilla, copias = args
    rng = np.random.default_rng(semilla)
    n = len(trazos)
    canales = dibujar_trazos(trazos[0], validos[0]).shape[0] if n else 2   # 4 (mano) o 2 (codo)
    orig = np.zeros((n, canales, 64, 64), np.uint8)
    aug = np.zeros((n, copias, canales, 64, 64), np.uint8)
    ext = np.zeros((n, copias, extras.shape[1]), np.float32)
    for i in range(n):
        orig[i] = np.round(dibujar_trazos(trazos[i], validos[i]) * 255)
        for c in range(copias):
            t, v, e = aumentar_muestra(trazos[i], validos[i], extras[i], rng)
            aug[i, c] = np.round(dibujar_trazos(t, v) * 255)
            ext[i, c] = e
    return orig, aug, ext


def prerenderizar(trazos, validos, extras, copias, semilla=0, nucleos=None, al_avanzar=None):
    """Devuelve (originales uint8 (N,C,64,64), aumentadas uint8 (N,copias,C,64,64), extras (N,copias,E))."""
    nucleos = nucleos or nucleos_libres()
    tam = max(1, min(len(trazos) // (nucleos * 8), 64))      # lotes chicos: no se atasca al enviarlos
    trabajos = [(trazos[i:i + tam], validos[i:i + tam], extras[i:i + tam], semilla + i, copias)
                for i in range(0, len(trazos), tam)]
    n = len(trazos)
    canales = dibujar_trazos(trazos[0], validos[0]).shape[0] if n else 2
    orig = np.zeros((n, canales, 64, 64), np.uint8)              # se reserva una sola vez
    aug = np.zeros((n, copias, canales, 64, 64), np.uint8)       # (antes se juntaban partes: RAM x2)
    ext = np.zeros((n, copias, extras.shape[1]), np.float32)
    with Pool(nucleos, initializer=bajar_prioridad) as grupo:
        for k, (o, a, e) in enumerate(grupo.imap(_lote, trabajos)):
            i = k * tam
            orig[i:i + len(o)], aug[i:i + len(o)], ext[i:i + len(o)] = o, a, e
            if al_avanzar:
                al_avanzar((k + 1) / len(trabajos))
    return orig, aug, ext


def prerenderizar_aparte(trazos, validos, extras, copias, carpeta, nombre, semilla=0):
    """
    Lo mismo, pero en un proceso de Python aparte (en Windows los núcleos extra no pueden
    nacer desde el script de entrenamiento porque lo volverían a ejecutar).
    Guarda el resultado en carpeta/nombre.npz para reutilizarlo si los datos no cambiaron.
    """
    import hashlib
    import subprocess
    import sys
    from pathlib import Path

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    huella = hashlib.md5(np.ascontiguousarray(trazos).tobytes() + np.ascontiguousarray(validos).tobytes()
                         + np.ascontiguousarray(extras).tobytes() + str(copias).encode()).hexdigest()[:12]
    salida = carpeta / f"{nombre}_{huella}.npz"
    if not salida.exists():
        entrada = carpeta / f"{nombre}_entrada.npz"
        np.savez(entrada, trazos=trazos, validos=validos, extras=extras, copias=copias, semilla=semilla)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(entrada), str(salida)], check=True)
        entrada.unlink(missing_ok=True)
    d = np.load(salida)
    return d["orig"], d["aug"], d["ext"]


if __name__ == "__main__":
    import sys
    import time

    bajar_prioridad()
    entrada, salida = sys.argv[1], sys.argv[2]
    d = np.load(entrada)
    inicio = time.time()

    def avance(f):
        print(f"\r   Dibujando imágenes en {nucleos_libres()} núcleos... {f * 100:5.1f}%  "
              f"({time.time() - inicio:.0f} s)", end="", flush=True)

    o, a, e = prerenderizar(d["trazos"], d["validos"], d["extras"], int(d["copias"]), int(d["semilla"]),
                            al_avanzar=avance)
    np.savez(salida, orig=o, aug=a, ext=e)
    print()
