"""
El dibujo del trazo EXACTAMENTE como lo pinta el celular (Muestra.kt: dibujarTrazos), en numpy.
OpenCV (cv2.line con LINE_AA) no existe en el celular: allá se imita. Con esto se mide cuánto cambia.
"""
import numpy as np

import dibujo_senas as D

ALCANCE_LINEA = 2.3
PUNTA_LINEA = 0.45


def dibujar_trazos_celular(trazos, validos, tamano=D.TAMANO_IMAGEN):
    trazos = np.asarray(trazos)
    grupos = [[0], [1]]
    imagen = np.zeros((len(grupos), tamano, tamano), np.float32)
    ys, xs = np.mgrid[0:tamano, 0:tamano].astype(np.float64)
    for canal, indices in enumerate(grupos):
        lienzo = np.zeros((tamano, tamano), np.float64)
        hay = False
        for i in indices:
            if not validos[i]:
                continue
            hay = True
            p = np.clip(np.rint((trazos[i] / D.MEDIA_VENTANA * 0.5 + 0.5) * (tamano - 1)), 0, tamano - 1)
            cant = len(p)
            for k in range(cant - 1):
                intensidad = float(int(90 + 165 * (k + 1) / (cant - 1)))
                a, b = p[k], p[k + 1]
                d = b - a
                l2 = float(d @ d)
                largo = np.sqrt(l2)
                m = ALCANCE_LINEA + 1.0
                x0, x1 = max(0, int(np.floor(min(a[0], b[0]) - m))), min(tamano - 1, int(np.ceil(max(a[0], b[0]) + m)))
                y0, y1 = max(0, int(np.floor(min(a[1], b[1]) - m))), min(tamano - 1, int(np.ceil(max(a[1], b[1]) + m)))
                X, Y = xs[y0:y1 + 1, x0:x1 + 1], ys[y0:y1 + 1, x0:x1 + 1]
                tt = ((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / l2 if l2 > 0 else np.zeros_like(X)
                t = np.clip(tt, 0.0, 1.0)
                dist = np.hypot(X - (a[0] + t * d[0]), Y - (a[1] + t * d[1]))
                fuera = np.maximum(0.0, np.maximum(-tt, tt - 1)) * largo if l2 > 0 else dist
                cob = np.clip(ALCANCE_LINEA - dist - PUNTA_LINEA * fuera, 0.0, 1.0)
                zona = lienzo[y0:y1 + 1, x0:x1 + 1]
                nuevo = np.rint(zona * (1 - cob) + intensidad * cob)
                lienzo[y0:y1 + 1, x0:x1 + 1] = np.where(cob > 0, nuevo, zona)
        if hay:
            imagen[canal] = (lienzo / 255.0).astype(np.float32)
    return imagen


if __name__ == "__main__":
    # ¿qué tanto cambia la respuesta de la versión 3 si se pinta como el celular?
    import experimento_versiones as E
    import versiones as V
    import json
    casos = json.loads((V.PAQUETE / "android/app/src/test/resources/casos.json").read_text(encoding="utf-8"))
    for c in casos:
        t, v = np.asarray(c["trazos"]), np.asarray(c["validos"])
        print(c["nombre"], "dif. con el celular (debe ser ~0):", float(np.abs(dibujar_trazos_celular(t, v) - np.asarray(c["imagen"])).max()))
    R = V.Reconocedor()
    muestras, y, _, _, nombres = E.cargar_banco()
    iguales = acierta_cv = acierta_cel = 0
    original = D.dibujar_trazos
    for m, clase in zip(muestras, y):
        D.dibujar_trazos = original
        p_cv = R.probabilidades(m, 3)[0].argmax()
        D.dibujar_trazos = dibujar_trazos_celular
        p_cel = R.probabilidades(m, 3)[0].argmax()
        iguales += p_cv == p_cel
        acierta_cv += p_cv == clase
        acierta_cel += p_cel == clase
    D.dibujar_trazos = original
    n = len(y)
    print(f"Versión 3 en {n} muestras: misma respuesta {iguales}/{n} ({iguales / n * 100:.1f} %) | "
          f"acierto OpenCV {acierta_cv / n * 100:.1f} % vs celular {acierta_cel / n * 100:.1f} %")
