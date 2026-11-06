"""
LAS 4 VERSIONES (para usar en vivo): entrenar con las 1800 muestras y reconocer una seña nueva.

  Versión 1  Solo manos                (Hand Landmarker, red GRU)                         = B0
  Versión 2  Manos + cuerpo            (muñecas y codos medidos desde los hombros, GRU)   = M1
  Versión 3  Dibujo + lugar + dedos    (cascada: coordenadas, dibujo de la mano, del codo) = M2
  Versión 4  Fusión 2 + 3              (las dos votan con el mismo peso)                   = M3

Uso:  python versiones.py entrenar     -> modelos/cuatro_versiones.pt  (unos 6 min con GPU)
Desde otro programa:
      R = Reconocedor();  r = R.reconocer(registro, version=4)
      r["top"]          -> [(seña, %), ...] las 4 más parecidas
      r["partes"]       -> {"Coordenadas": {seña: %}, ...} lo que opina cada parte de esa versión
"""
import sys
from pathlib import Path

import experimento_versiones as E       # primero: GPU en modo reproducible
import experimento_4 as X
import dibujo_senas as D
import numpy as np
import torch

PAQUETE = Path(__file__).resolve().parent.parent
RUTA_MODELOS = PAQUETE / "modelos" / "cuatro_versiones.pt"

VERSIONES = {
    1: ("Versión 1: solo manos", "21 puntos de cada mano (Hand Landmarker), red GRU"),
    2: ("Versión 2: manos + cuerpo", "manos, muñecas y codos medidos desde los hombros, red GRU"),
    3: ("Versión 3: dibujo + lugar + dedos", "trazo de la seña, dónde ocurre (cuerpo y cara) y forma de los dedos"),
    4: ("Versión 4: fusión 2 + 3", "la versión 2 y la 3 votan juntas (la mejor)"),
}
TOP = 4


# =============================================================================
# ENTRENAR (todas las 1800 muestras; 15 % para la parada temprana)
# =============================================================================

def entrenar_todo():
    disp = E.modo_reproducible()
    muestras, y, _, _, nombres = E.cargar_banco()
    clases = len(nombres)
    tr, val = E.separar_validacion(np.arange(len(y)), y, E.SEMILLA)
    print(f"Entrenando las 4 versiones con {len(tr)} muestras (+{len(val)} para validar), {clases} señas")
    guardado = {"nombres": nombres, "semilla": E.SEMILLA, "peso_fusion": X.PESO_FUSION}

    for clave, cfg in (("v1", X.CONFIG_B0), ("v2", X.CONFIG_M1)):
        Xs, X_aug = X.preparar_secuencia(cfg, muestras)
        media = Xs[tr].reshape(-1, Xs.shape[-1]).mean(0)
        desv = Xs[tr].reshape(-1, Xs.shape[-1]).std(0)
        desv[desv < 1e-6] = 1.0
        norm = lambda a: ((a - media) / desv).astype(np.float32)
        m = E.entrenar(lambda: E.ModeloSecuencia(Xs.shape[-1], clases), (E.a_gpu(norm(X_aug[tr]), disp),),
                       (E.a_gpu(norm(Xs[val]), disp),), y[tr], y[val], clases, disp, E.SEMILLA, clave.upper(), E.EPOCAS)
        guardado[clave] = {"estado": {k: v.cpu() for k, v in m.state_dict().items()}, "entradas": Xs.shape[-1],
                           "media": media.astype(np.float32), "desv": desv.astype(np.float32), "config": cfg}

    img, ext, aug_img, aug_ext, img_b, aug_b = E.preparar_dibujos(muestras, X.CONFIG_M2)
    mi, di = img[tr].mean((0, 2, 3)), img[tr].std((0, 2, 3))
    me, de = ext[tr].mean(0), ext[tr].std(0)
    mb, db = img_b[tr].mean((0, 2, 3)), img_b[tr].std((0, 2, 3))
    di[di < 1e-6], de[de < 1e-6], db[db < 1e-6] = 1.0, 1.0, 1.0
    ni = lambda a: ((a - mi[:, None, None]) / di[:, None, None]).astype(np.float32)
    ne = lambda a: ((a - me) / de).astype(np.float32)
    nb = lambda a: ((a - mb[:, None, None]) / db[:, None, None]).astype(np.float32)
    g = lambda a: E.a_gpu(a, disp)
    ent_aug = (g(ni(aug_img[tr].astype(np.float32) / 255.0)), g(ne(aug_ext[tr])))
    ent_val = (g(ni(img[val])), g(ne(ext[val])))
    args = (y[tr], y[val], clases, disp)
    coord = E.entrenar(lambda: D.ModeloCoordenadas(clases), ent_aug, ent_val, *args, E.SEMILLA, "V3 COORD", E.EPOCAS)
    dib = E.entrenar(lambda: D.ModeloDibujo(clases, D.CANALES_DIBUJO), ent_aug, ent_val, *args, E.SEMILLA + 1, "V3 DIBUJO 1", E.EPOCAS)
    cero = lambda forma: g(np.zeros(forma, np.float32))
    b2 = E.entrenar(lambda: D.ModeloDibujo(clases),
                    (g(nb(aug_b[tr].astype(np.float32) / 255.0)), cero((len(tr), E.COPIAS_AUMENTADAS, D.CANTIDAD_EXTRAS))),
                    (g(nb(img_b[val])), cero((len(val), D.CANTIDAD_EXTRAS))), *args, E.SEMILLA + 2, "V3 DIBUJO 2", E.EPOCAS)
    estado = lambda m: {k: v.cpu() for k, v in m.state_dict().items()}
    guardado["v3"] = {"coord": estado(coord), "dibujo": estado(dib), "dibujo2": estado(b2), "config": X.CONFIG_M2,
                      "media_imagen": mi, "desv_imagen": di, "media_extras": me, "desv_extras": de,
                      "media_imagen2": mb, "desv_imagen2": db}
    RUTA_MODELOS.parent.mkdir(exist_ok=True)
    torch.save(guardado, RUTA_MODELOS)
    print(f"Guardado: {RUTA_MODELOS}")


# =============================================================================
# RECONOCER
# =============================================================================

def registro_a_muestra(registro, ancho, alto):
    """Del registro en vivo (listas por cuadro, en píxeles de la cámara) a una muestra como las del banco:
    99 cuadros y la misma escala de 1280x720."""
    reg = D.normalizar_registro({k: list(v) for k, v in registro.items()})
    escala = np.array([E.ANCHO / ancho, E.ALTO / alto], np.float32)
    m = {}
    for k in ("muneca_izq", "muneca_der", "codo_izq", "codo_der", "mano_izq", "mano_der", "cara"):
        m[k] = np.asarray(reg[k], np.float32) * escala
    for k in ("valida_izq", "valida_der", "codo_izq_ok", "codo_der_ok", "mano_izq_ok", "mano_der_ok", "cara_ok"):
        m[k] = np.asarray(reg[k], bool)
    anchos = np.asarray(registro["ancho_hombros"], np.float32) * escala[0]
    centros = np.asarray(registro["centro_hombros"], np.float32) * escala
    m["ancho_hombros"] = np.full(len(m["muneca_izq"]), np.median(anchos) if len(anchos) else 0.0, np.float32)
    m["centro_hombros"] = np.repeat((np.median(centros, 0) if len(centros) else np.zeros(2, np.float32))[None],
                                    len(m["muneca_izq"]), 0)
    return m


class Reconocedor:
    def __init__(self, ruta=RUTA_MODELOS):
        if not Path(ruta).exists():
            raise SystemExit(f"Falta {ruta}\nPrimero:  python versiones.py entrenar")
        self.g = torch.load(ruta, map_location="cpu", weights_only=False)
        self.nombres = self.g["nombres"]
        n = len(self.nombres)
        self.gru = {}
        for v in ("v1", "v2"):
            m = E.ModeloSecuencia(self.g[v]["entradas"], n)
            m.load_state_dict(self.g[v]["estado"]); m.eval()
            self.gru[v] = m
        v3 = self.g["v3"]
        self.coord = D.ModeloCoordenadas(n); self.coord.load_state_dict(v3["coord"])
        self.dib = E.sin_pool_adaptativo(D.ModeloDibujo(n, D.CANALES_DIBUJO)); self.dib.load_state_dict(v3["dibujo"])
        self.dib2 = E.sin_pool_adaptativo(D.ModeloDibujo(n)); self.dib2.load_state_dict(v3["dibujo2"])
        for m in (self.coord, self.dib, self.dib2):
            m.eval()

    @torch.no_grad()
    def _gru(self, v, muestra):
        d = self.g[v]
        x = (E.secuencia(muestra, d["config"]) - d["media"]) / d["desv"]
        return torch.softmax(self.gru[v](torch.from_numpy(x[None].astype(np.float32))), 1)[0].numpy()

    @torch.no_grad()
    def _v3(self, muestra):
        v3 = self.g["v3"]
        t, val, e, tb, vb = E.construir(muestra, v3["config"])
        img = (D.dibujar_trazos(t, val) - v3["media_imagen"][:, None, None]) / v3["desv_imagen"][:, None, None]
        ext = (e - v3["media_extras"]) / v3["desv_extras"]
        img2 = (D.dibujar_trazos(tb, vb) - v3["media_imagen2"][:, None, None]) / v3["desv_imagen2"][:, None, None]
        f = lambda a: torch.from_numpy(np.asarray(a, np.float32)[None])
        cero = torch.zeros(1, D.CANTIDAD_EXTRAS)
        pc = torch.softmax(self.coord(f(img), f(ext)), 1)[0].numpy()
        pd = torch.softmax(self.dib(f(img), f(ext)), 1)[0].numpy()
        p2 = torch.softmax(self.dib2(f(img2), cero), 1)[0].numpy()
        final, _ = D.combinar_cascada(pc, pd, k=E.CANDIDATAS_K, minimo=0.0, prob_dibujo2=p2)
        return final[0], {"Coordenadas (lugar y dedos)": pc, "Dibujo de la mano": pd, "Dibujo del codo": p2}

    def probabilidades(self, muestra, version):
        """(probabilidad final, {parte: probabilidades}) de una muestra (formato del banco)."""
        if version == 1:
            p = self._gru("v1", muestra)
            return p, {"Manos (GRU)": p}
        if version == 2:
            p = self._gru("v2", muestra)
            return p, {"Manos + cuerpo (GRU)": p}
        p3, partes = self._v3(muestra)
        if version == 3:
            return p3, partes
        p2 = self._gru("v2", muestra)
        p4 = X.fusionar(p2[None], p3[None], self.g["peso_fusion"])[0]
        return p4, {"Versión 2": p2, "Versión 3": p3}

    def reconocer(self, registro, version, ancho, alto, top=TOP, objetivo=None):
        """registro en vivo -> las `top` señas más parecidas y lo que opina cada parte (en %).
        Con `objetivo` (la seña que se intentaba) dice también en qué puesto quedó y con cuánto."""
        p, partes = self.probabilidades(registro_a_muestra(registro, ancho, alto), version)
        todos = np.argsort(-p)
        orden = todos[:top]
        r = {"top": [(self.nombres[i], float(p[i] * 100)) for i in orden],
             "partes": {k: {self.nombres[i]: float(v[i] * 100) for i in orden} for k, v in partes.items()}}
        if objetivo is not None:
            o = next((i for i, n in enumerate(self.nombres) if D.texto_ascii(n) == D.texto_ascii(objetivo)), None)
            if o is not None:
                r["objetivo"] = {"nombre": self.nombres[o], "puesto": int(np.where(todos == o)[0][0]) + 1,
                                 "pct": float(p[o] * 100), "partes": {k: float(v[o] * 100) for k, v in partes.items()}}
        return r


if __name__ == "__main__":
    if sys.argv[1:] == ["entrenar"]:
        entrenar_todo()
    else:
        print(__doc__)
