"""
EXPERIMENTO DE VERSIONES (para la divulgación): ¿cuánto mejoró el reconocimiento con cada cosa que hicimos?

Las 17 versiones se entrenan y se prueban EXACTAMENTE igual, con el banco "dataset 90 senas"
(LSC50 + LSC-54: 90 señas x 5 personas x 4 muestras = 1800). Lo único que cambia entre versiones
es la mejora que se va sumando (estudio de ablación).

- Semilla fija: 42 (la misma en todas las versiones y vueltas).
- Evaluación: dejando una persona por fuera, 5 vueltas. En la vuelta k se prueba con la persona k de
  cada seña (4 muestras por seña = 360) y se entrena con las otras 4 personas.
- Mismas épocas, lote, tasa de aprendizaje, aumento de datos y parada temprana para todas.

Uso (desde la carpeta codigo):
    python experimento_versiones.py todas        -> V1..V17
    python experimento_versiones.py 1 5 17       -> solo esas
    python experimento_versiones.py 17 --rapido  -> prueba corta (1 vuelta, pocas épocas)
Resultados: "resultados versiones/resultados.json", resultados.csv y precision_por_version.png
"""
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

# para que la GPU dé SIEMPRE los mismos números (tiene que ir antes de usar CUDA)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

import dibujo_senas as D
from prerender import bajar_prioridad, nucleos_libres

SEMILLA = 42
VUELTAS = 5
EPOCAS = 150
PACIENCIA = 25
TAMANO_LOTE = 32
TASA_APRENDIZAJE = 1e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.10
COPIAS_AUMENTADAS = 24       # versiones aumentadas de cada muestra (dibujo y extras)
CANDIDATAS_K = 6             # como en el traductor
FRACCION_VALIDACION = 0.15   # de las muestras de entrenamiento, para la parada temprana

PROYECTO = Path(__file__).resolve().parent.parent
BANCO = PROYECTO / "dataset 90 senas"
SALIDA = PROYECTO / "resultados"
ANCHO, ALTO = 1280, 720


# =============================================================================
# LAS 17 VERSIONES (cada una suma una cosa a la anterior)
# =============================================================================

def _v(**cambios):
    base = VERSIONES[-1][2].copy() if VERSIONES else {}
    base.update(cambios)
    return base


VERSIONES = []
def agregar(numero, nombre, **cambios):
    VERSIONES.append((numero, nombre, _v(**cambios)))


# Etapa 1: la red mira los puntos (landmarks) cuadro a cuadro
agregar(1, "Landmarks crudos (GRU)", tipo="secuencia", normalizar=False, estabilizar=False, cara_seq=True)
agregar(2, "Landmarks normalizados por los hombros", normalizar=True)
agregar(3, "Landmarks estabilizados (filtro y huecos)", estabilizar=True)
agregar(4, "Landmarks limitados (manos, muñecas y codos)", cara_seq=False)
# Etapa 2: la seña como DIBUJO
agregar(5, "Trazo de la muñeca como dibujo (CNN)", tipo="dibujo", activa=False, extras=False, cascada=False,
        dibujo2=False, cara=False, limpiar=False, mano_pegada=False, inicio_codo=False, fuente="mano",
        forma="mitad", brazo=None, quitar_reposo=False)
agregar(6, "Solo la parte activa del trazo", activa=True)
agregar(7, "Extras: dónde ocurre y forma de la mano", extras=True)
agregar(8, "Cascada: coordenadas eligen, dibujo decide", cascada=True)
agregar(9, "Trazo del codo (DIBUJO 2)", dibujo2=True)
agregar(10, "La mano respecto a la cara y el pecho", cara=True)
agregar(11, "Trazo estabilizado (saltos, huecos, pincel)", limpiar=True)
agregar(12, "Mano pegada al brazo cuando se pierde", mano_pegada=True)
agregar(13, "La seña empieza al subir la mano sobre el codo", inicio_codo=True)
agregar(14, "Trazo de la palma y 5 dedos en 3 momentos", fuente="palma", forma="tres")
agregar(15, "Ángulo del codo aparte (sin DIBUJO 2)", dibujo2=False, brazo="tres")
agregar(16, "Curva del ángulo del codo en el tiempo", brazo="curva")
agregar(17, "Trazo de la muñeca del esqueleto, sin mano en reposo", fuente="esqueleto", quitar_reposo=True)


# =============================================================================
# DATOS
# =============================================================================

def cargar_banco():
    muestras, y, personas, nombres = [], [], [], {}
    for carpeta in sorted(BANCO.iterdir(), key=lambda c: c.name):
        if not carpeta.is_dir():
            continue
        for archivo in sorted(carpeta.glob("muestra_*.npz")):
            m = dict(np.load(archivo, allow_pickle=True))
            clase = int(m["clase"])
            nombres[clase] = str(m["nombre"])
            muestras.append(m)
            y.append(clase)
            personas.append(re.match(r"muestra_lsc5[04]_(P\d+|Signer_\d+)_", archivo.name).group(1))
    y = np.asarray(y)
    # vuelta de cada muestra: la persona 0..4 de SU seña (en orden), así cada vuelta prueba 1 persona por seña
    vuelta = np.zeros(len(y), dtype=int)
    for clase in np.unique(y):
        idx = np.where(y == clase)[0]
        orden = sorted(set(personas[i] for i in idx))
        for i in idx:
            vuelta[i] = orden.index(personas[i])
    return muestras, y, np.asarray(personas), vuelta, [nombres[c] for c in sorted(nombres)]


# ---------------- Etapa 1: secuencias ----------------
def secuencia(m, cfg):
    """(99, F): puntos de cada cuadro (y si se vio) según la versión."""
    fps = len(m["muneca_izq"]) / D.SEGUNDOS_MUESTRA
    manos = [np.asarray(m["mano_izq"], np.float64), np.asarray(m["mano_der"], np.float64)]
    ok = [np.asarray(m["mano_izq_ok"], bool), np.asarray(m["mano_der_ok"], bool)]
    munecas = [np.asarray(m["muneca_izq"], np.float64), np.asarray(m["muneca_der"], np.float64)]
    codos = [np.asarray(m["codo_izq"], np.float64), np.asarray(m["codo_der"], np.float64)]
    cara = np.asarray(m["cara"], np.float64)
    if cfg["estabilizar"]:
        manos = [D.suavizar_manos(manos[l], ok[l], fps) for l in range(2)]
        munecas = [_suave(p) for p in munecas]
        codos = [_suave(p) for p in codos]
        cara = _suave(cara.reshape(len(cara), -1)).reshape(cara.shape)
    centro = np.median(m["centro_hombros"], axis=0)
    ancho = float(np.median(m["ancho_hombros"]))
    hombros = np.stack([centro + [0.5 * ancho, 0], centro - [0.5 * ancho, 0]])
    partes = [munecas[0][:, None], munecas[1][:, None], codos[0][:, None], codos[1][:, None],
              np.repeat(hombros[None], len(cara), 0), manos[0], manos[1]]
    if cfg.get("solo_manos"):                            # línea base: solo el Hand Landmarker (21 puntos por mano)
        partes = [manos[0], manos[1]]
    elif cfg["cara_seq"]:
        partes.append(cara)
    p = np.concatenate(partes, axis=1)                               # (T, puntos, 2)
    if cfg["normalizar"]:
        p = (p - centro) / max(ancho, 1e-6)
    else:
        p = p / [ANCHO, ALTO]
    banderas = np.stack([ok[0], ok[1]], 1).astype(np.float32)
    return np.concatenate([p.reshape(len(p), -1), banderas], 1).astype(np.float32)


def _suave(p, ventana=5):
    p = np.asarray(p, np.float64)
    k = np.ones(ventana) / ventana
    rell = np.concatenate([np.repeat(p[:1], ventana // 2, 0), p, np.repeat(p[-1:], ventana // 2, 0)])
    return np.stack([np.convolve(rell[:, j], k, mode="valid") for j in range(p.shape[1])], 1)


# ---------------- Etapa 2: dibujo + extras ----------------
def construir(m, cfg):
    """Si ninguna mano queda (seña hecha abajo), se repite sin quitar la de reposo (como el programa)."""
    r = _construir(m, cfg)
    if cfg.get("quitar_reposo") and not r[1].any():
        r = _construir(m, {**cfg, "quitar_reposo": False})
    return r


def _construir(m, cfg):
    """Igual que D._construir_muestra, pero cada mejora se prende o apaga según la versión."""
    trazos = np.zeros((D.CANTIDAD_TRAZOS, D.PUNTOS_TRAZO, 2), np.float32)
    validos = np.zeros(D.CANTIDAD_TRAZOS, bool)
    extras = np.zeros(D.CANTIDAD_EXTRAS, np.float32)
    trazos_b = np.zeros((2, D.PUNTOS_TRAZO, 2), np.float32)
    validos_b = np.zeros(2, bool)
    ancho = float(np.median(m["ancho_hombros"]))
    centro_h = np.median(m["centro_hombros"], axis=0).astype(np.float64)
    munecas = [m["muneca_izq"], m["muneca_der"]]
    validas = [m["valida_izq"], m["valida_der"]]
    codos = [m["codo_izq"], m["codo_der"]]
    codos_ok = [m["codo_izq_ok"], m["codo_der_ok"]]
    caras, caras_ok = m["cara"], m["cara_ok"]
    manos, manos_ok = D.corregir_manos_grabacion([m["mano_izq"], m["mano_der"]], [m["mano_izq_ok"], m["mano_der_ok"]],
                                                  munecas, validas)
    for l in range(2):
        brazo = np.asarray(munecas[l], np.float64)
        bok = np.asarray(validas[l], bool)
        fps = len(brazo) / D.SEGUNDOS_MUESTRA
        mano = np.asarray(manos[l], np.float64)
        mok = np.asarray(manos_ok[l], bool)
        if cfg["mano_pegada"]:
            mok = D.filtrar_mano(mano, mok, ancho, fps, caras if cfg["cara"] else None, caras_ok if cfg["cara"] else None)
            mano = D.suavizar_manos(mano, mok, fps)

        # de dónde sale el trazo
        if cfg["fuente"] == "esqueleto":
            centro, valida = D.trazo_del_brazo(brazo, bok, ancho, fps)
        else:
            if cfg["fuente"] == "palma":
                base = np.where(mok[:, None], D.centro_palma(mano), brazo)
            else:                                           # la muñeca de la mano (o la del brazo si no se ve)
                base = np.where(mok[:, None], mano[:, 0], brazo)
            if cfg["mano_pegada"]:
                centro, valida = D.limpiar_recorrido(brazo, bok, ancho, fps, mano, mok)
                if cfg["fuente"] == "palma":
                    centro = np.where(mok[:, None], base, centro)
            elif cfg["limpiar"]:
                centro, valida = D.limpiar_recorrido(base, bok, ancho, fps)
            else:
                centro, valida = base.copy(), bok.copy()

        # qué cuadros forman la seña
        if cfg["activa"]:
            D._QUITAR_REPOSO[0] = bool(cfg["quitar_reposo"])
            indices = D.indices_activos(centro, valida, centro_h, ancho,
                                        codos[l] if cfg["inicio_codo"] else None,
                                        codos_ok[l] if cfg["inicio_codo"] else None)
            D._QUITAR_REPOSO[0] = True
        else:
            indices = np.where(valida)[0]
        if len(indices) < D.MIN_FRAMES_MANO:
            continue

        recorrido = centro[indices]
        trazo = D.remuestrear_por_longitud(D.suavizar(recorrido), D.PUNTOS_TRAZO)
        trazos[l] = (trazo - trazo.mean(axis=0)) / ancho
        validos[l] = True

        if cfg["dibujo2"]:
            pb, vb = D.recorrido_antebrazo(brazo, bok, codos[l], codos_ok[l], ancho, fps)
            ib = indices[vb[indices]]
            if len(ib) >= D.MIN_FRAMES_MANO:
                t = D.remuestrear_por_longitud(D.suavizar(pb[ib]), D.PUNTOS_TRAZO)
                trazos_b[l], validos_b[l] = (t - t.mean(axis=0)) / ancho, True

        if not cfg["extras"]:
            continue
        b = extras[l * D.EXTRAS_POR_MANO:(l + 1) * D.EXTRAS_POR_MANO]
        rel = (recorrido - centro_h) / ancho
        b[D.E_ACTIVA] = 1.0
        b[D.E_INICIO], b[D.E_FINAL] = rel[0], rel[-1]
        b[D.E_CENTRO], b[D.E_TAMANO] = rel.mean(axis=0), rel.max(axis=0) - rel.min(axis=0)

        if cfg["mano_pegada"]:                             # igual que el programa: la mano tapada sigue a la muñeca limpia
            limpio, limpio_ok = D.limpiar_recorrido(brazo, bok, ancho, fps, mano, mok)
            mano_c, ok_c = D.completar_manos_tapadas(mano, mok, limpio, limpio_ok, fps)
        else:
            mano_c, ok_c = mano, mok
        forma, forma_ok = D.forma_de_mano(mano_c, ok_c, indices)
        if cfg["forma"] == "mitad":                        # solo la mitad de la seña (antes de v14)
            n = len(D.DEDOS_PUNTAS) * 2
            mitad = forma[(D.MOMENTOS // 2) * n:(D.MOMENTOS // 2 + 1) * n].copy()
            forma[:D.MOMENTOS * n] = 0.0
            forma[(D.MOMENTOS // 2) * n:(D.MOMENTOS // 2 + 1) * n] = mitad
        b[D.E_FORMA_VALIDA] = float(forma_ok)
        b[D.E_FORMA] = forma

        if cfg["cara"]:
            refs = D._referencias_cara(caras, caras_ok, indices, centro_h, ancho)
            if refs is not None:
                c = np.zeros(len(D.LUGARES) * 3, np.float32)
                for j, ref in enumerate(refs):
                    d = (recorrido - ref) / ancho
                    c[j * 3:j * 3 + 2] = d.mean(axis=0)
                    c[j * 3 + 2] = np.linalg.norm(d, axis=1).min()
                b[D.E_CARA_VALIDA], b[D.E_CARA] = 1.0, c

        if cfg["brazo"]:
            br, br_ok = D.angulos_brazo(brazo, bok, codos[l], codos_ok[l], indices, l, centro_h, ancho)
            if cfg["brazo"] == "tres":                     # inicio, mitad y final (antes de v16)
                n = D.PUNTOS_ANGULO
                guardar = np.zeros_like(br)
                for j in (0, n // 2, n - 1):
                    guardar[[j, n + j, 2 * n + j]] = br[[j, n + j, 2 * n + j]]
                br = guardar
            b[D.E_BRAZO_VALIDO], b[D.E_BRAZO] = float(br_ok), br
    return trazos, validos, extras, trazos_b, validos_b


# =============================================================================
# MODELOS
# =============================================================================

class ModeloSecuencia(nn.Module):
    """GRU bidireccional (como el primer modelo del proyecto)."""
    def __init__(self, entradas, clases):
        super().__init__()
        self.proyeccion = nn.Sequential(nn.Linear(entradas, 128), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.salida = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, clases))

    def forward(self, x, _extras=None):
        h, _ = self.gru(self.proyeccion(x))
        return self.salida(torch.cat([h.mean(1), h.max(1).values], 1))


class ModeloCascadaHibrido(D.ModeloHibrido):
    pass


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def modo_reproducible():
    """Todo determinista: mismos datos + misma semilla = mismos números (en la misma GPU y versiones)."""
    torch.set_num_threads(nucleos_libres())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(semilla):
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def separar_validacion(indices, y, semilla):
    rng = np.random.default_rng(semilla)
    val, tr = [], []
    for c in np.unique(y[indices]):
        de = rng.permutation(indices[y[indices] == c])
        corte = max(1, int(round(len(de) * FRACCION_VALIDACION)))
        val.extend(de[:corte]); tr.extend(de[corte:])
    return np.asarray(tr), np.asarray(val)


def sin_pool_adaptativo(modelo):
    """AdaptiveAvgPool2d(4) sobre 8x8 hace la misma cuenta que AvgPool2d(2), pero esta sí es determinista en la GPU."""
    for nombre, hijo in modelo.named_children():
        if isinstance(hijo, nn.AdaptiveAvgPool2d) and hijo.output_size in (4, (4, 4)):
            setattr(modelo, nombre, nn.AvgPool2d(D.TAMANO_IMAGEN // 8 // 4))
        else:
            sin_pool_adaptativo(hijo)
    return modelo


def entrenar(crear, entradas_aug, entradas_val, y_tr, y_val, clases, disp, semilla, etiqueta, epocas):
    """entradas_aug: tupla de tensores (n, copias, ...) ya normalizados en la GPU; entradas_val: (n_val, ...)."""
    fijar_semilla(semilla)
    modelo = sin_pool_adaptativo(crear()).to(disp)
    opt = torch.optim.AdamW(modelo.parameters(), lr=TASA_APRENDIZAJE, weight_decay=WEIGHT_DECAY)
    prog = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    perdida = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    gen = torch.Generator(device=disp).manual_seed(semilla)
    yt = torch.from_numpy(y_tr).long().to(disp)
    yv = torch.from_numpy(y_val).long()
    n, copias = len(y_tr), entradas_aug[0].shape[1]
    mejor, mejor_estado, sin_mejora = float("inf"), copy.deepcopy(modelo.state_dict()), 0
    for epoca in range(1, epocas + 1):
        modelo.train()
        orden = torch.randperm(n, device=disp, generator=gen)
        copia = torch.randint(0, copias, (n,), device=disp, generator=gen)
        for i in range(0, n, TAMANO_LOTE):
            sel = orden[i:i + TAMANO_LOTE]
            lote = [e[sel, copia[sel]] for e in entradas_aug]
            opt.zero_grad(set_to_none=True)
            p = perdida(modelo(*lote), yt[sel])
            p.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
        modelo.eval()
        with torch.no_grad():
            pv = perdida(modelo(*entradas_val).cpu(), yv).item()
        prog.step(pv)
        if pv < mejor:
            mejor, mejor_estado, sin_mejora = pv, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    modelo.load_state_dict(mejor_estado)
    print(f"      [{etiqueta}] {epoca} épocas, pérdida val {mejor:.3f}")
    return modelo


def probabilidades(modelo, entradas):
    modelo.eval()
    with torch.no_grad():
        return torch.softmax(modelo(*entradas), 1).cpu().numpy()


def aumentar_secuencia(x, rng):
    """Rotación, escala, proporción, ruido y corrimiento en el tiempo (igual idea que el dibujo)."""
    pts = x[:, :-2].reshape(len(x), -1, 2).copy()
    centro = pts.reshape(-1, 2).mean(0)
    a = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (pts - centro) @ rot.T * rng.uniform(0.85, 1.15)
    pts[..., 0] *= rng.uniform(0.85, 1.18)
    pts = pts + centro + rng.normal(0, 0.01 * np.abs(pts).mean(), pts.shape)
    corrimiento = int(rng.integers(-8, 9))
    salida = np.concatenate([pts.reshape(len(x), -1), x[:, -2:]], 1)
    return np.roll(salida, corrimiento, axis=0).astype(np.float32)


def _trozo_dibujos(args):
    """En un proceso aparte: arma, dibuja y aumenta un trozo de muestras. La semilla de cada muestra es
    SEMILLA + su número: da lo mismo cuántos procesos se usen."""
    muestras, cfg, desde = args
    salida = []
    for j, m in enumerate(muestras):
        t, v, e, tb, vb = construir(m, cfg)
        rng = np.random.default_rng(SEMILLA + desde + j)
        img, ext = D.dibujar_trazos(t, v), e
        img_b = D.dibujar_trazos(tb, vb) if cfg["dibujo2"] else None
        ai, ae, ab = [], [], []
        for _ in range(COPIAS_AUMENTADAS):
            ta, va, ea = D.aumentar_muestra(t, v, e, rng)
            ai.append(np.round(D.dibujar_trazos(ta, va) * 255).astype(np.uint8))
            ae.append(ea)
            if cfg["dibujo2"]:
                tba, vba, _ = D.aumentar_muestra(tb, vb, np.zeros(D.CANTIDAD_EXTRAS, np.float32), rng)
                ab.append(np.round(D.dibujar_trazos(tba, vba) * 255).astype(np.uint8))
        salida.append((img, ext, np.stack(ai), np.stack(ae), img_b, np.stack(ab) if ab else None))
    return salida


def preparar_dibujos(muestras, cfg):
    """Imágenes y extras: originales (n, ...) y aumentadas (n, copias, ...), en varios procesos."""
    from concurrent.futures import ProcessPoolExecutor
    tam = 50
    trozos = [(muestras[i:i + tam], cfg, i) for i in range(0, len(muestras), tam)]
    with ProcessPoolExecutor(max_workers=nucleos_libres()) as grupo:
        todo = [r for parte in grupo.map(_trozo_dibujos, trozos) for r in parte]
    orig_img = np.stack([r[0] for r in todo]).astype(np.float32)
    orig_ext = np.stack([r[1] for r in todo]).astype(np.float32)
    aug_img = np.stack([r[2] for r in todo])
    aug_ext = np.stack([r[3] for r in todo]).astype(np.float32)
    orig_b = np.stack([r[4] for r in todo]).astype(np.float32) if cfg["dibujo2"] else None
    aug_b = np.stack([r[5] for r in todo]) if cfg["dibujo2"] else None
    # columnas que esta versión NO usa (siempre 0): el aumento no les puede meter ruido
    usadas = (np.abs(orig_ext) > 0).any(axis=0).astype(np.float32)
    aug_ext *= usadas
    return orig_img, orig_ext, aug_img, aug_ext, orig_b, aug_b


def a_gpu(arr, disp):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(disp)


def correr_version(numero, nombre, cfg, datos, disp, vueltas, epocas):
    muestras, y, personas, vuelta, nombres = datos
    clases = len(nombres)
    print(f"\n=== V{numero}: {nombre} ===")
    t0 = time.time()

    if cfg["tipo"] == "secuencia":
        X = np.stack([secuencia(m, cfg) for m in muestras])
        rng = np.random.default_rng(SEMILLA)
        X_aug = np.stack([np.stack([aumentar_secuencia(x, rng) for _ in range(COPIAS_AUMENTADAS)]) for x in X])
    else:
        img, ext, aug_img, aug_ext, img_b, aug_b = preparar_dibujos(muestras, cfg)
    print(f"   datos listos ({time.time() - t0:.0f} s)")

    filas = []
    for k in range(vueltas):
        prueba = np.where(vuelta == k)[0]
        resto = np.where(vuelta != k)[0]
        tr, val = separar_validacion(resto, y, SEMILLA + k)
        print(f"   vuelta {k + 1}/{vueltas}: entrena {len(tr)}, valida {len(val)}, prueba {len(prueba)}")

        if cfg["tipo"] == "secuencia":
            media = X[tr].reshape(-1, X.shape[-1]).mean(0)
            desv = X[tr].reshape(-1, X.shape[-1]).std(0)
            desv[desv < 1e-6] = 1.0
            norm = lambda a: ((a - media) / desv).astype(np.float32)
            m = entrenar(lambda: ModeloSecuencia(X.shape[-1], clases), (a_gpu(norm(X_aug[tr]), disp),),
                         (a_gpu(norm(X[val]), disp),), y[tr], y[val], clases, disp, SEMILLA, "GRU", epocas)
            p = probabilidades(m, (a_gpu(norm(X[prueba]), disp),))
        else:
            # como el Normalizador del programa: si no varía, la desviación es 1
            mi, di = img[tr].mean((0, 2, 3)), img[tr].std((0, 2, 3))
            me, de = ext[tr].mean(0), ext[tr].std(0)
            di[di < 1e-6], de[de < 1e-6] = 1.0, 1.0
            ni = lambda a: ((a - mi[:, None, None]) / di[:, None, None]).astype(np.float32)
            ne = lambda a: ((a - me) / de).astype(np.float32)
            aug_i = ni(aug_img[tr].astype(np.float32) / 255.0)
            aug_e = ne(aug_ext[tr])
            ent_aug = (a_gpu(aug_i, disp), a_gpu(aug_e, disp))
            ent_val = (a_gpu(ni(img[val]), disp), a_gpu(ne(ext[val]), disp))
            ent_pr = (a_gpu(ni(img[prueba]), disp), a_gpu(ne(ext[prueba]), disp))
            args = (y[tr], y[val], clases, disp)

            if not cfg["extras"]:
                m = entrenar(lambda: D.ModeloDibujo(clases, D.CANALES_DIBUJO), ent_aug, ent_val, *args, SEMILLA + 1, "DIBUJO", epocas)
                p = probabilidades(m, ent_pr)
            elif not cfg["cascada"]:
                m = entrenar(lambda: D.ModeloHibrido(clases), ent_aug, ent_val, *args, SEMILLA, "HIBRIDO", epocas)
                p = probabilidades(m, ent_pr)
            else:
                # máscaras: desde v15 el ángulo del codo va en un modelo aparte
                mascara_ang = np.zeros(D.CANTIDAD_EXTRAS, np.float32)
                if cfg["brazo"]:
                    for l in range(2):
                        mascara_ang[l * D.EXTRAS_POR_MANO + D.E_BRAZO_VALIDO:l * D.EXTRAS_POR_MANO + D.E_BRAZO.stop] = 1
                mc = a_gpu(1 - mascara_ang, disp)
                coord = entrenar(lambda: D.ModeloCoordenadas(clases), (ent_aug[0], ent_aug[1] * mc), (ent_val[0], ent_val[1] * mc),
                                 *args, SEMILLA, "COORD", epocas)
                dib = entrenar(lambda: D.ModeloDibujo(clases, D.CANALES_DIBUJO), ent_aug, ent_val, *args, SEMILLA + 1, "DIBUJO 1", epocas)
                pc = probabilidades(coord, (ent_pr[0], ent_pr[1] * mc))
                pd = probabilidades(dib, ent_pr)
                extra = None
                if cfg["dibujo2"]:
                    mb, db = img_b[tr].mean((0, 2, 3)), img_b[tr].std((0, 2, 3))
                    db[db < 1e-6] = 1.0
                    nb = lambda a: ((a - mb[:, None, None]) / db[:, None, None]).astype(np.float32)
                    cero = lambda n_: a_gpu(np.zeros((n_, D.CANTIDAD_EXTRAS), np.float32), disp)
                    aug_bn = a_gpu(nb(aug_b[tr].astype(np.float32) / 255.0), disp)
                    cero_aug = a_gpu(np.zeros((len(tr), COPIAS_AUMENTADAS, D.CANTIDAD_EXTRAS), np.float32), disp)
                    b2 = entrenar(lambda: D.ModeloDibujo(clases), (aug_bn, cero_aug), (a_gpu(nb(img_b[val]), disp), cero(len(val))),
                                  *args, SEMILLA + 2, "DIBUJO 2", epocas)
                    extra = probabilidades(b2, (a_gpu(nb(img_b[prueba]), disp), cero(len(prueba))))
                elif cfg["brazo"]:
                    ma = a_gpu(mascara_ang, disp)
                    ang = entrenar(lambda: D.ModeloCoordenadas(clases), (ent_aug[0], ent_aug[1] * ma), (ent_val[0], ent_val[1] * ma),
                                   *args, SEMILLA + 3, "ANGULO CODO", epocas)
                    extra = probabilidades(ang, (ent_pr[0], ent_pr[1] * ma))
                p, _ = D.combinar_cascada(pc, pd, k=CANDIDATAS_K, minimo=0.0, prob_dibujo2=extra)
            torch.cuda.empty_cache()

        orden = np.argsort(-p, 1)
        top1 = float((orden[:, 0] == y[prueba]).mean())
        top3 = float((orden[:, :3] == y[prueba][:, None]).any(1).mean())
        filas.append({"vuelta": k + 1, "n": int(len(prueba)), "top1": top1, "top3": top3})
        print(f"   -> vuelta {k + 1}: acierto {top1 * 100:.1f} %  (entre las 3 primeras {top3 * 100:.1f} %)")

    top1 = np.array([f["top1"] for f in filas])
    top3 = np.array([f["top3"] for f in filas])
    r = {"version": numero, "nombre": nombre, "config": cfg, "vueltas": filas,
         "top1_media": float(top1.mean()), "top1_desv": float(top1.std()),
         "top3_media": float(top3.mean()), "top3_desv": float(top3.std()),
         "segundos": round(time.time() - t0)}
    print(f"   V{numero}: {r['top1_media'] * 100:.1f} % ± {r['top1_desv'] * 100:.1f}   ({r['segundos']} s)")
    return r


# =============================================================================
# PROGRAMA
# =============================================================================

def guardar_resultados(resultados, rapido):
    SALIDA.mkdir(exist_ok=True)
    nombre = "resultados_rapido" if rapido else "resultados"
    ruta = SALIDA / f"{nombre}.json"
    todos = {}
    if ruta.exists():
        todos = {r["version"]: r for r in json.loads(ruta.read_text(encoding="utf-8"))["versiones"]}
    todos.update({r["version"]: r for r in resultados})
    lista = [todos[k] for k in sorted(todos)]
    info = {"semilla": SEMILLA, "vueltas": VUELTAS, "epocas": EPOCAS, "paciencia": PACIENCIA, "lote": TAMANO_LOTE,
            "tasa_aprendizaje": TASA_APRENDIZAJE, "copias_aumentadas": COPIAS_AUMENTADAS, "candidatas_k": CANDIDATAS_K,
            "banco": "dataset 90 senas (LSC50 + LSC-54): 90 senas x 5 personas x 4 muestras",
            "evaluacion": "dejando una persona por fuera (la persona k de cada sena en la vuelta k)", "versiones": lista}
    ruta.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(SALIDA / f"{nombre}.csv", "w", encoding="utf-8-sig") as f:
        f.write("version;nombre;acierto_%;desviacion_%;top3_%;" + ";".join(f"vuelta{k + 1}_%" for k in range(VUELTAS)) + "\n")
        for r in lista:
            vs = ";".join(f"{v['top1'] * 100:.1f}" for v in r["vueltas"])
            f.write(f"V{r['version']};{r['nombre']};{r['top1_media'] * 100:.1f};{r['top1_desv'] * 100:.1f};"
                    f"{r['top3_media'] * 100:.1f};{vs}\n".replace(".", ","))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [f"V{r['version']}" for r in lista]
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.errorbar(x, [r["top1_media"] * 100 for r in lista], yerr=[r["top1_desv"] * 100 for r in lista],
                    marker="o", capsize=4, color="#2a6fdb")
        for i, r in enumerate(lista):
            ax.annotate(f"{r['top1_media'] * 100:.1f}", (i, r["top1_media"] * 100), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        ax.set_ylabel("Acierto con personas no vistas (%)")
        ax.set_title("Precisión por versión - 90 señas, 5 personas por seña, semilla 42")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(SALIDA / f"precision_por_version{'_rapido' if rapido else ''}.png", dpi=150)
    except Exception as e:                                  # noqa: BLE001
        print(f"(sin gráfica: {e})")
    print(f"\nResultados en: {SALIDA}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rapido = "--rapido" in sys.argv
    if not args or args == ["todas"]:
        elegidas = [v[0] for v in VERSIONES]
    else:
        elegidas = [int(a.lstrip("vV")) for a in args]
    bajar_prioridad()
    disp = modo_reproducible()
    print(f"Dispositivo: {disp}  |  semilla {SEMILLA}  |  versiones {elegidas}" + ("  (RÁPIDO)" if rapido else ""))
    datos = cargar_banco()
    print(f"Banco: {len(datos[1])} muestras, {len(datos[4])} señas")
    vueltas, epocas = (1, 15) if rapido else (VUELTAS, EPOCAS)
    resultados = []
    for numero, nombre, cfg in VERSIONES:
        if numero in elegidas:
            resultados.append(correr_version(numero, nombre, cfg, datos, disp, vueltas, epocas))
            guardar_resultados(resultados, rapido)          # se guarda después de cada versión
    print("\nRESUMEN")
    for r in resultados:
        print(f"  V{r['version']:>2}  {r['top1_media'] * 100:5.1f} % ± {r['top1_desv'] * 100:4.1f}   {r['nombre']}")


if __name__ == "__main__":
    main()
