"""
EXPERIMENTO PRINCIPAL (para documentar): 4 configuraciones, de la más sencilla a la más completa.

  B0  Línea base: solo el Hand Landmarker (21 puntos de cada mano), red GRU
  M1  + el cuerpo como referencia: muñecas y codos, todo medido desde los hombros (= V4)
  M2  + la seña como dibujo, su lugar (cuerpo y cara) y la forma de los dedos en 3 momentos (= V14, cascada)
  M3  Fusión M1 + M2: las dos redes votan juntas

Todo igual para las 4: banco "dataset 90 senas" (90 señas x 5 personas x 4 muestras), semilla 42,
5 vueltas dejando una persona por fuera (entrena 1260 / valida 180 / prueba 360 en cada vuelta).
La fusión usa un peso fijo de 0.5 (las dos redes valen igual), decidido antes de ver resultados.

Además se guarda la predicción de cada muestra de prueba (cada una se prueba una sola vez en las 5 vueltas)
para comparar configuraciones con la prueba de McNemar (pareada, la misma muestra en las dos).

Uso:  python experimento_4.py            (completo)
      python experimento_4.py --rapido   (1 vuelta, pocas épocas: solo para ver que corre)
"""
import json
import sys
import time
from math import comb, sqrt

import experimento_versiones as E      # primero: deja la GPU en modo reproducible antes de usar CUDA
import dibujo_senas as D
import numpy as np
import torch
from prerender import bajar_prioridad, nucleos_libres

CONFIG_B0 = {**E.VERSIONES[0][2], "normalizar": False, "estabilizar": False, "cara_seq": False, "solo_manos": True}
CONFIG_M1 = E.VERSIONES[3][2]          # V4
CONFIG_M2 = E.VERSIONES[13][2]         # V14
PESO_FUSION = 0.5                      # peso de M1 en la fusión (el resto es de M2)

CONFIGURACIONES = [
    ("B0", "Línea base: solo Hand Landmarker (21 puntos por mano)"),
    ("M1", "+ cuerpo como referencia (muñecas, codos, medido desde los hombros)"),
    ("M2", "+ dibujo de la seña, lugar respecto al cuerpo y la cara, dedos en 3 momentos"),
    ("M3", "Fusión M1 + M2"),
]


# =============================================================================
# CADA RAMA: probabilidades en validación y en prueba de una vuelta
# =============================================================================

def preparar_secuencia(cfg, muestras):
    X = np.stack([E.secuencia(m, cfg) for m in muestras])
    rng = np.random.default_rng(E.SEMILLA)
    X_aug = np.stack([np.stack([E.aumentar_secuencia(x, rng) for _ in range(E.COPIAS_AUMENTADAS)]) for x in X])
    return X, X_aug


def rama_secuencia(datos_seq, y, tr, val, prueba, clases, disp, epocas, etiqueta):
    X, X_aug = datos_seq
    media = X[tr].reshape(-1, X.shape[-1]).mean(0)
    desv = X[tr].reshape(-1, X.shape[-1]).std(0)
    desv[desv < 1e-6] = 1.0
    norm = lambda a: ((a - media) / desv).astype(np.float32)
    m = E.entrenar(lambda: E.ModeloSecuencia(X.shape[-1], clases), (E.a_gpu(norm(X_aug[tr]), disp),),
                   (E.a_gpu(norm(X[val]), disp),), y[tr], y[val], clases, disp, E.SEMILLA, etiqueta, epocas)
    pv = E.probabilidades(m, (E.a_gpu(norm(X[val]), disp),))
    pp = E.probabilidades(m, (E.a_gpu(norm(X[prueba]), disp),))
    del m
    torch.cuda.empty_cache()
    return pv, pp


def rama_dibujo(datos_dib, y, tr, val, prueba, clases, disp, epocas, cfg):
    """La cascada de la V14 (coordenadas + dibujo 1 + dibujo 2), igual que en experimento_versiones."""
    img, ext, aug_img, aug_ext, img_b, aug_b = datos_dib
    mi, di = img[tr].mean((0, 2, 3)), img[tr].std((0, 2, 3))
    me, de = ext[tr].mean(0), ext[tr].std(0)
    di[di < 1e-6], de[de < 1e-6] = 1.0, 1.0
    ni = lambda a: ((a - mi[:, None, None]) / di[:, None, None]).astype(np.float32)
    ne = lambda a: ((a - me) / de).astype(np.float32)
    g = lambda a: E.a_gpu(a, disp)
    ent_aug = (g(ni(aug_img[tr].astype(np.float32) / 255.0)), g(ne(aug_ext[tr])))
    ent = {k: (g(ni(img[idx])), g(ne(ext[idx]))) for k, idx in (("val", val), ("prueba", prueba))}
    args = (y[tr], y[val], clases, disp)
    coord = E.entrenar(lambda: D.ModeloCoordenadas(clases), ent_aug, ent["val"], *args, E.SEMILLA, "M2 COORD", epocas)
    dib = E.entrenar(lambda: D.ModeloDibujo(clases, D.CANALES_DIBUJO), ent_aug, ent["val"], *args, E.SEMILLA + 1,
                     "M2 DIBUJO 1", epocas)
    extra = {"val": None, "prueba": None}
    if cfg["dibujo2"]:
        mb, db = img_b[tr].mean((0, 2, 3)), img_b[tr].std((0, 2, 3))
        db[db < 1e-6] = 1.0
        nb = lambda a: ((a - mb[:, None, None]) / db[:, None, None]).astype(np.float32)
        cero = lambda forma: g(np.zeros(forma, np.float32))
        b2 = E.entrenar(lambda: D.ModeloDibujo(clases),
                        (g(nb(aug_b[tr].astype(np.float32) / 255.0)), cero((len(tr), E.COPIAS_AUMENTADAS, D.CANTIDAD_EXTRAS))),
                        (g(nb(img_b[val])), cero((len(val), D.CANTIDAD_EXTRAS))), *args, E.SEMILLA + 2, "M2 DIBUJO 2", epocas)
        for k, idx in (("val", val), ("prueba", prueba)):
            extra[k] = E.probabilidades(b2, (g(nb(img_b[idx])), cero((len(idx), D.CANTIDAD_EXTRAS))))
    salida = []
    for k in ("val", "prueba"):
        p, _ = D.combinar_cascada(E.probabilidades(coord, ent[k]), E.probabilidades(dib, ent[k]),
                                  k=E.CANDIDATAS_K, minimo=0.0, prob_dibujo2=extra[k])
        salida.append(p)
    torch.cuda.empty_cache()
    return salida[0], salida[1]


def fusionar(p1, p2, w):
    """Media geométrica pesada de las dos redes (w = peso de M1)."""
    l = w * np.log(np.clip(p1, 1e-9, 1)) + (1 - w) * np.log(np.clip(p2, 1e-9, 1))
    l -= l.max(1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(1, keepdims=True)


# =============================================================================
# ESTADÍSTICA
# =============================================================================

def wilson(aciertos, n, z=1.96):
    """Intervalo de confianza del 95 % de una proporción."""
    p = aciertos / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    r = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - r, c + r


def mcnemar(a_ok, b_ok):
    """Prueba de McNemar exacta (binomial): ¿la diferencia entre A y B sobre las MISMAS muestras es real?"""
    b = int(np.sum(a_ok & ~b_ok))       # A acierta, B no
    c = int(np.sum(~a_ok & b_ok))       # B acierta, A no
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, p


# =============================================================================
# PROGRAMA
# =============================================================================

def main():
    rapido = "--rapido" in sys.argv
    bajar_prioridad()
    disp = E.modo_reproducible()
    muestras, y, personas, vuelta, nombres = E.cargar_banco()
    clases = len(nombres)
    vueltas, epocas = (1, 15) if rapido else (E.VUELTAS, E.EPOCAS)
    print(f"Banco: {len(y)} muestras, {clases} señas | semilla {E.SEMILLA} | {vueltas} vueltas" + (" (RÁPIDO)" if rapido else ""))

    t0 = time.time()
    seq_b0 = preparar_secuencia(CONFIG_B0, muestras)
    seq_m1 = preparar_secuencia(CONFIG_M1, muestras)
    dib_m2 = E.preparar_dibujos(muestras, CONFIG_M2)
    print(f"Datos listos ({time.time() - t0:.0f} s): B0 {seq_b0[0].shape[-1]} valores por cuadro, "
          f"M1 {seq_m1[0].shape[-1]}, M2 dibujo + {D.CANTIDAD_EXTRAS} extras")

    # predicción de cada muestra (-1 = no probada en modo rápido)
    pred = {c: np.full(len(y), -1) for c, _ in CONFIGURACIONES}
    top3 = {c: np.zeros(len(y), bool) for c, _ in CONFIGURACIONES}
    por_vuelta = {c: [] for c, _ in CONFIGURACIONES}
    pesos = []
    for k in range(vueltas):
        prueba = np.where(vuelta == k)[0]
        tr, val = E.separar_validacion(np.where(vuelta != k)[0], y, E.SEMILLA + k)
        print(f"\n--- Vuelta {k + 1}/{vueltas}: entrena {len(tr)}, valida {len(val)}, prueba {len(prueba)} ---")
        _, p_b0 = rama_secuencia(seq_b0, y, tr, val, prueba, clases, disp, epocas, "B0 GRU")
        v_m1, p_m1 = rama_secuencia(seq_m1, y, tr, val, prueba, clases, disp, epocas, "M1 GRU")
        v_m2, p_m2 = rama_dibujo(dib_m2, y, tr, val, prueba, clases, disp, epocas, CONFIG_M2)
        # peso de la fusión FIJO (decidido antes de ver resultados): las dos redes valen igual.
        # No se ajusta con la validación porque esa sale de las MISMAS personas del entrenamiento
        # y favorece a la red que más se aprende a esas personas.
        w = PESO_FUSION
        pesos.append(float(w))
        p_m3 = fusionar(p_m1, p_m2, w)
        for c, p in (("B0", p_b0), ("M1", p_m1), ("M2", p_m2), ("M3", p_m3)):
            orden = np.argsort(-p, 1)
            pred[c][prueba] = orden[:, 0]
            top3[c][prueba] = (orden[:, :3] == y[prueba][:, None]).any(1)
            por_vuelta[c].append(float((orden[:, 0] == y[prueba]).mean()))
        print("   " + "  ".join(f"{c} {por_vuelta[c][-1] * 100:.1f} %" for c, _ in CONFIGURACIONES)
              + f"   (peso fusión M1 = {w})")

    # ---- resultados ----
    probadas = pred["B0"] >= 0
    n = int(probadas.sum())
    filas = []
    for c, nombre in CONFIGURACIONES:
        ok = pred[c][probadas] == y[probadas]
        bajo, alto = wilson(int(ok.sum()), n)
        filas.append({"config": c, "nombre": nombre, "acierto": float(ok.mean()), "ic95": [bajo, alto],
                      "media_vueltas": float(np.mean(por_vuelta[c])), "desv_vueltas": float(np.std(por_vuelta[c])),
                      "top3": float(top3[c][probadas].mean()), "vueltas": por_vuelta[c]})
    comparaciones = []
    for a, b in (("B0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M1", "M3"), ("B0", "M3")):
        sa, sb, p = mcnemar(pred[a][probadas] == y[probadas], pred[b][probadas] == y[probadas])
        comparaciones.append({"a": a, "b": b, "solo_a_acierta": sa, "solo_b_acierta": sb, "p_valor": p})

    E.SALIDA.mkdir(exist_ok=True)
    sufijo = "_rapido" if rapido else ""
    info = {"semilla": E.SEMILLA, "vueltas": vueltas, "epocas": epocas, "paciencia": E.PACIENCIA,
            "muestras_probadas": n, "pesos_fusion_por_vuelta": pesos,
            "banco": "dataset 90 senas (LSC50 + LSC-54): 90 senas x 5 personas x 4 muestras",
            "particion": "5 vueltas dejando una persona por fuera; en cada vuelta 1260 entrenamiento / 180 validacion / 360 prueba",
            "configuraciones": filas, "comparaciones_mcnemar": comparaciones}
    (E.SALIDA / f"resultados_4{sufijo}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(E.SALIDA / f"predicciones_4{sufijo}.npz", y=y, personas=personas, vuelta=vuelta,
             **{f"pred_{c}": pred[c] for c, _ in CONFIGURACIONES})
    with open(E.SALIDA / f"resultados_4{sufijo}.csv", "w", encoding="utf-8-sig") as f:
        f.write("config;descripcion;acierto_%;ic95_bajo_%;ic95_alto_%;desv_entre_personas_%;top3_%\n")
        for r in filas:
            f.write(f"{r['config']};{r['nombre']};{r['acierto'] * 100:.1f};{r['ic95'][0] * 100:.1f};"
                    f"{r['ic95'][1] * 100:.1f};{r['desv_vueltas'] * 100:.1f};{r['top3'] * 100:.1f}\n".replace(".", ","))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        x = [r["config"] for r in filas]
        v = [r["acierto"] * 100 for r in filas]
        err = [[(r["acierto"] - r["ic95"][0]) * 100 for r in filas], [(r["ic95"][1] - r["acierto"]) * 100 for r in filas]]
        barras = ax.bar(x, v, yerr=err, capsize=6, color=["#9aa5b1", "#5b8def", "#2a6fdb", "#123f8c"])
        for b_, val_ in zip(barras, v):
            ax.text(b_.get_x() + b_.get_width() / 2, val_ + 5, f"{val_:.1f} %", ha="center", fontsize=11, weight="bold")
        ax.set_ylabel("Acierto con personas no vistas (%)")
        ax.set_title("90 señas LSC, 5 personas por seña, semilla 42 (barras: IC 95 %)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(E.SALIDA / f"resultados_4{sufijo}.png", dpi=150)
    except Exception as e:                                  # noqa: BLE001
        print(f"(sin gráfica: {e})")

    print(f"\nRESULTADO ({n} muestras de prueba, cada una probada una vez)")
    for r in filas:
        print(f"  {r['config']}  {r['acierto'] * 100:5.1f} %  IC95 [{r['ic95'][0] * 100:.1f}-{r['ic95'][1] * 100:.1f}]  "
              f"± {r['desv_vueltas'] * 100:.1f} entre personas   top3 {r['top3'] * 100:.1f} %")
    print("McNemar (misma muestra en las dos):")
    for c in comparaciones:
        print(f"  {c['a']} vs {c['b']}: solo {c['a']} {c['solo_a_acierta']}, solo {c['b']} {c['solo_b_acierta']}, p = {c['p_valor']:.4f}")
    print(f"Pesos de la fusión (M1) por vuelta: {pesos}")


if __name__ == "__main__":
    main()
