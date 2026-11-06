"""
BANCO DE PRUEBAS para la divulgación: 90 señas de los datasets oficiales LSC50 + LSC-54,
todas IGUALES: 5 personas por seña x 4 muestras de cada una = 20 muestras por seña (1800 en total).

- LSC50 (50 señas, 5 personas, 4 repeticiones): se usan todas tal cual.
- LSC-54 (54 señas): las 14 que también están en LSC50 se toman de LSC50 (ya traen 5 x 4).
  De las otras 40 se escogen al azar (semilla 42) 5 personas que tengan al menos 4 muestras,
  y de cada una 4 muestras al azar (misma semilla).
- Todo queda en el formato del proyecto (muestra_*.npz, 99 cuadros, cara con los 11 puntos de CARA_REF).
- Las manos se asignan por cercanía a la muñeca del esqueleto (así no importa cómo las nombró cada dataset).

Cada archivo lleva la persona en el nombre, para evaluar DEJANDO UNA PERSONA POR FUERA.
Queda también  seleccion.json  con qué se escogió y la semilla (para repetir el experimento).

Necesita:  pip install ijson
Uso:       python crear_dataset_90.py
"""
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import ijson

from dibujo_senas import CARA_REF, CANTIDAD_CARA_REF
from senas import nombre_carpeta

SEMILLA = 42
PERSONAS_POR_SENA = 5
MUESTRAS_POR_PERSONA = 4
ANCHO, ALTO = 1280, 720          # de 0..1 a píxeles como la cámara
FRAMES = 99                      # como todas las muestras del proyecto

# todo relativo a la carpeta "reproducir experimento" (funciona donde se copie)
PAQUETE = Path(__file__).resolve().parent.parent
BASE = PAQUETE / "datos_originales"
LSC50 = BASE / "lsc50_landmarks_60frames.npz"
MAPA50 = BASE / "mapa_etiquetas_lsc50.json"
LSC54 = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "datos.json"     # datos.json de LSC-54 (52 GB, no viene incluido)
DESTINO = PAQUETE / "dataset 90 senas (reconstruido)"

# nombres distintos que son la MISMA seña (para encontrar las 14 comunes)
IGUALES = {"comoestas": "comoesta", "maso menos": "masomenos", "masomenos": "masomenos"}
BONITO = {"cafe": "Café", "si": "Sí", "bano": "Baño", "muchasgracias": "Muchas gracias", "muybien": "Muy bien",
          "muymal": "Muy mal", "losiento": "Lo siento", "masomenos": "Más o menos", "comoesta": "Cómo está",
          "dies": "Diez", "denada": "De nada", "porfavor": "Por favor", "congusto": "Con gusto",
          "buenosdias": "Buenos días", "buenastardes": "Buenas tardes", "buenasnoches": "Buenas noches"}


def clave(s):
    s = unicodedata.normalize("NFKD", s.lower())
    c = "".join(ch for ch in s if ch.isalnum())
    return IGUALES.get(c, c)


def remuestrear(n):
    return np.round(np.linspace(0, n - 1, FRAMES)).astype(int)


def armar(cuerpo, cuerpo_ok, mano_a, a_ok, mano_b, b_ok, cara, cara_ok):
    """De puntos en píxeles (T cuadros) al formato muestra_*.npz. Las dos manos se asignan
    a la muñeca del esqueleto más cercana (izq = punto 15, der = punto 16)."""
    T = len(cuerpo)
    mi = np.zeros((T, 21, 2), np.float32); md = np.zeros((T, 21, 2), np.float32)
    mio = np.zeros(T, bool); mdo = np.zeros(T, bool)
    for t in range(T):
        wi, wd = cuerpo[t, 15], cuerpo[t, 16]
        for mano, ok in ((mano_a[t], a_ok[t]), (mano_b[t], b_ok[t])):
            if not ok:
                continue
            izq = np.linalg.norm(mano[0] - wi) <= np.linalg.norm(mano[0] - wd)
            if izq and not mio[t]:
                mi[t], mio[t] = mano, True
            elif not mdo[t]:
                md[t], mdo[t] = mano, True
            else:
                mi[t], mio[t] = mano, True
    h11, h12 = cuerpo[:, 11], cuerpo[:, 12]
    return dict(
        ancho_hombros=np.linalg.norm(h11 - h12, axis=1).astype(np.float32),
        centro_hombros=((h11 + h12) / 2).astype(np.float32),
        muneca_izq=cuerpo[:, 15].astype(np.float32), muneca_der=cuerpo[:, 16].astype(np.float32),
        valida_izq=cuerpo_ok.copy(), valida_der=cuerpo_ok.copy(),
        mano_izq=mi, mano_der=md, mano_izq_ok=mio, mano_der_ok=mdo,
        codo_izq=cuerpo[:, 13].astype(np.float32), codo_der=cuerpo[:, 14].astype(np.float32),
        codo_izq_ok=cuerpo_ok.copy(), codo_der_ok=cuerpo_ok.copy(),
        cara=cara.astype(np.float32), cara_ok=cara_ok,
    )


def guardar(datos, clase, nombre, archivo):
    carpeta = DESTINO / f"{clase:02d}_{nombre_carpeta(nombre)}"
    carpeta.mkdir(parents=True, exist_ok=True)
    datos["clase"] = np.int64(clase)
    datos["nombre"] = np.asarray(nombre)
    np.savez_compressed(carpeta / archivo, **datos)


# ---------------- LSC50 ----------------
def leer_lsc50():
    """(clave, nombre, persona, repeticion, datos) de las 1000 muestras."""
    etq = {e["id"]: e["nombre_lsc"] for e in json.loads(MAPA50.read_text(encoding="utf-8"))["etiquetas"]}
    m = np.load(LSC50)
    X, y, part, rep = m["X"], m["y"], m["participantes"], m["repeticiones"]
    px = lambda a: (a[..., :2] * [ANCHO, ALTO]).astype(np.float32)
    ok = lambda p: np.abs(p[..., :2]).sum(-1) > 0
    for s in range(len(X)):
        f = X[s][remuestrear(X.shape[1])]
        cuerpo = f[:, :99].reshape(FRAMES, 33, 3)
        cara = f[:, 99:1503].reshape(FRAMES, 468, 3)[:, list(CARA_REF)]
        li = f[:, 1503:1566].reshape(FRAMES, 21, 3)
        ld = f[:, 1566:1629].reshape(FRAMES, 21, 3)
        nombre = etq[int(y[s])].replace("¿", "").replace("?", "").replace("Jucioso", "Juicioso")
        datos = armar(px(cuerpo), ok(cuerpo).all(1), px(li), ok(li).all(1), px(ld), ok(ld).all(1),
                      px(cara), ok(cara).all(1))
        yield clave(nombre), nombre, f"P{int(part[s])}", int(rep[s]), datos


# ---------------- LSC-54 ----------------
def a_arreglo(parte, n):
    if not parte or "x" not in parte or len(parte["x"]) < n:
        return np.zeros((n, 2), np.float32), False
    p = np.stack([np.asarray(parte["x"][:n], np.float64), np.asarray(parte["y"][:n], np.float64)], 1)
    if not np.isfinite(p).all() or np.abs(p).sum() == 0:
        return np.zeros((n, 2), np.float32), False
    return (p * [ANCHO, ALTO]).astype(np.float32), True


def convertir_lsc54(frames):
    claves = sorted(frames, key=lambda k: int(k.split("_")[1]))
    if len(claves) < 5:
        return None
    cuerpo, cok, ma, mao, mb, mbo, cara, carok = [], [], [], [], [], [], [], []
    for i in remuestrear(len(claves)):
        f = frames[claves[i]]
        p, o = a_arreglo(f.get("pose"), 33); cuerpo.append(p); cok.append(o)
        a, o = a_arreglo(f.get("l_hand"), 21); ma.append(a); mao.append(o)
        b, o = a_arreglo(f.get("r_hand"), 21); mb.append(b); mbo.append(o)
        c, o = a_arreglo(f.get("face"), 468)
        cara.append(c[list(CARA_REF)] if o else np.zeros((CANTIDAD_CARA_REF, 2), np.float32)); carok.append(o)
    cok = np.asarray(cok)
    if not cok.any():
        return None
    return armar(np.stack(cuerpo), cok, np.stack(ma), np.asarray(mao), np.stack(mb), np.asarray(mbo),
                 np.stack(cara), np.asarray(carok))


def repeticiones_lsc54(quiero):
    """Recorre datos.json (52 GB, poco a poco) y entrega solo las repeticiones de las personas elegidas:
    ((persona, categoria, seña, video, repeticion), frames)."""
    with open(LSC54, "rb") as f:
        constructor, camino = None, None
        for prefijo, evento, valor in ijson.parse(f, use_float=True):
            if constructor is None:
                if evento == "start_map" and prefijo.count(".") == 4:
                    partes = prefijo.split(".")
                    if (clave(partes[2]), partes[0]) in quiero:
                        constructor, camino = ijson.ObjectBuilder(), partes
                        constructor.event(evento, valor)
                continue
            constructor.event(evento, valor)
            if evento == "end_map" and prefijo == ".".join(camino):
                yield camino, constructor.value
                constructor = None


def main():
    if DESTINO.exists() and any(DESTINO.iterdir()):
        raise SystemExit(f"Ya existe '{DESTINO.name}': bórrala para volver a crearla.")
    rng = np.random.default_rng(SEMILLA)
    inicio = time.time()
    seleccion = {"semilla": SEMILLA, "personas_por_sena": PERSONAS_POR_SENA,
                 "muestras_por_persona": MUESTRAS_POR_PERSONA, "cuadros": FRAMES, "senas": {}}

    # 1) LSC50 completo: 50 señas x 5 personas x 4
    print("1) LSC50...")
    claves_lsc50, nombres = {}, {}
    for c, nombre, persona, r, datos in leer_lsc50():
        if c not in claves_lsc50:
            claves_lsc50[c] = len(claves_lsc50)
            nombres[c] = nombre
        guardar(datos, claves_lsc50[c], nombre, f"muestra_lsc50_{persona}_r{r}.npz")
        seleccion["senas"].setdefault(nombre, {"dataset": "LSC50", "muestras": []})["muestras"].append(f"{persona}_r{r}")
    print(f"   {len(claves_lsc50)} señas")

    # 2) LSC-54: las 40 que no están en LSC50 -> 5 personas al azar (con >= 4 muestras)
    conteo = json.loads((BASE / "lsc54_conteo.json").read_text(encoding="utf-8"))
    quiero, clase_de = set(), {}
    for sena in sorted(conteo):
        c = clave(sena)
        if c in claves_lsc50:
            continue
        aptas = sorted(p for p, n in conteo[sena]["por_persona"].items() if n >= MUESTRAS_POR_PERSONA)
        if len(aptas) < PERSONAS_POR_SENA:
            raise SystemExit(f"{sena}: solo {len(aptas)} personas con {MUESTRAS_POR_PERSONA}+ muestras")
        elegidas = sorted(rng.choice(aptas, PERSONAS_POR_SENA, replace=False).tolist())
        clase_de[c] = len(claves_lsc50) + len(clase_de)
        nombres[c] = BONITO.get(c, sena[:1].upper() + sena[1:])
        quiero.update((c, p) for p in elegidas)
        seleccion["senas"][nombres[c]] = {"dataset": "LSC-54", "personas": elegidas, "muestras": []}
    print(f"2) LSC-54: {len(clase_de)} señas nuevas -> leyendo datos.json (unos 10-20 min)...")

    candidatas = {}                   # (clave, persona) -> lista de (nombre de archivo, datos)
    for k, ((persona, _cat, sena, video, rep), frames) in enumerate(repeticiones_lsc54(quiero), 1):
        datos = convertir_lsc54(frames)
        if datos is not None:
            candidatas.setdefault((clave(sena), persona), []).append((f"{persona}_{video}_{rep}", datos))
        if k % 200 == 0:
            print(f"   {k} repeticiones leídas ({time.time() - inicio:.0f} s)")

    # 4 muestras al azar de cada persona elegida (orden fijo antes de sortear: se puede repetir igual)
    for (c, persona) in sorted(quiero):
        lista = sorted(candidatas.get((c, persona), []), key=lambda x: x[0])
        if len(lista) < MUESTRAS_POR_PERSONA:
            raise SystemExit(f"{nombres[c]} / {persona}: solo {len(lista)} muestras válidas")
        for i in sorted(rng.choice(len(lista), MUESTRAS_POR_PERSONA, replace=False).tolist()):
            nombre_archivo, datos = lista[i]
            guardar(datos, clase_de[c], nombres[c], f"muestra_lsc54_{nombre_archivo}.npz")
            seleccion["senas"][nombres[c]]["muestras"].append(nombre_archivo)

    # 3) revisión: 90 señas, 20 muestras y 5 personas cada una
    total = 0
    for carpeta in sorted(DESTINO.iterdir()):
        if not carpeta.is_dir():
            continue
        archivos = list(carpeta.glob("muestra_*.npz"))
        personas = {re.match(r"muestra_lsc5[04]_(P\d+|Signer_\d+)_", a.name).group(1) for a in archivos}
        if len(archivos) != PERSONAS_POR_SENA * MUESTRAS_POR_PERSONA or len(personas) != PERSONAS_POR_SENA:
            print(f"   OJO {carpeta.name}: {len(archivos)} muestras, {len(personas)} personas")
        total += len(archivos)
    (DESTINO / "mis_senas.txt").write_text(
        "\n".join(nombres[c] for c in sorted(nombres, key=lambda c: {**claves_lsc50, **clase_de}[c])) + "\n", encoding="utf-8")
    (DESTINO / "seleccion.json").write_text(json.dumps(seleccion, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nListo en {time.time() - inicio:.0f} s: {len(nombres)} señas, {total} muestras -> {DESTINO}")


if __name__ == "__main__":
    main()
