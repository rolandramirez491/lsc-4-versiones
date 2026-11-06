"""
MOSTRAR SEÑAS (celular): el muñeco de cada una de las 90 señas y, en las de LSC50, el video real.
    android/app/src/main/assets/aprender/<SEÑA>/avatar.json  (el muñeco: brazos, manos y trazo)
    android/app/src/main/assets/aprender/<SEÑA>/clip.mp4     (solo LSC50: el video de esa misma muestra)
Se usa la PRIMERA muestra de cada seña (persona 0, repetición 0).

Necesita el proyecto completo (usa aprenderY.py del traductor) y los videos de LSC50 (COLOR_BODY).
Uso:  python preparar_mostrar.py "C:\\ruta\\proyecto\\codigo" "C:\\ruta\\LSC50\\VIDEOS\\COLOR_BODY"
"""
import json
import sys
from pathlib import Path

import numpy as np

PAQUETE = Path(__file__).resolve().parent.parent
BANCO = PAQUETE / "dataset 90 senas"
DESTINO = PAQUETE / "android" / "app" / "src" / "main" / "assets" / "aprender"
MAPA50 = PAQUETE / "datos_originales" / "mapa_etiquetas_lsc50.json"
ALTO_CLIP = 240


def puntos_por_version(archivo):
    """Lo que ve cada versión, cuadro a cuadro, para dibujarlo ENCIMA DEL VIDEO (escala 1280x720 = el video):
      manos (21 puntos), muñecas, codos y hombros (versión 2) y, de la versión 3, el trazo de la palma,
      el del codo (solo en los cuadros de la seña) y los 11 puntos de la cara."""
    import dibujo_senas as D
    m = dict(np.load(archivo, allow_pickle=False))
    T = len(m["muneca_izq"])
    fps = T / D.SEGUNDOS_MUESTRA
    ancho = float(np.median(m["ancho_hombros"]))
    centro = np.median(m["centro_hombros"], axis=0)
    r = lambda a: np.round(np.asarray(a, np.float64), 1).tolist()
    salida = {"T": T, "ancho_ref": 1280, "alto_ref": 720,
              "hombros": r([centro + [0.5 * ancho, 0], centro - [0.5 * ancho, 0]]),
              "cara": r(m["cara"]), "cara_ok": np.asarray(m["cara_ok"], bool).tolist(), "lados": []}
    manos, manos_ok = D.corregir_manos_grabacion([m["mano_izq"], m["mano_der"]], [m["mano_izq_ok"], m["mano_der_ok"]],
                                                 [m["muneca_izq"], m["muneca_der"]], [m["valida_izq"], m["valida_der"]])
    for l, s in enumerate(("izq", "der")):
        brazo, bok = np.asarray(m[f"muneca_{s}"], np.float64), np.asarray(m[f"valida_{s}"], bool)
        # versión 3 (igual que experimento_versiones con la configuración V14)
        mok = D.filtrar_mano(manos[l], manos_ok[l], ancho, fps, m["cara"], m["cara_ok"])
        mano = D.suavizar_manos(manos[l], mok, fps)
        limpio, valida = D.limpiar_recorrido(brazo, bok, ancho, fps, mano, mok)
        palma = np.where(mok[:, None], D.centro_palma(mano), limpio)
        D._QUITAR_REPOSO[0] = False
        indices = D.indices_activos(palma, valida, centro, ancho, m[f"codo_{s}"], m[f"codo_{s}_ok"])
        D._QUITAR_REPOSO[0] = True
        activo = np.zeros(T, bool)
        activo[indices] = True
        pb, vb = D.recorrido_antebrazo(brazo, bok, m[f"codo_{s}"], m[f"codo_{s}_ok"], ancho, fps)
        salida["lados"].append({
            "mano": r(m[f"mano_{s}"]), "mano_ok": np.asarray(m[f"mano_{s}_ok"], bool).tolist(),
            "muneca": r(brazo), "codo": r(m[f"codo_{s}"]),
            "palma": r(palma), "palma_ok": (activo & valida).tolist(), "trazo_codo": r(pb), "trazo_codo_ok": (activo & vb).tolist(),
        })
    return salida


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.path.insert(0, sys.argv[1])
    videos = Path(sys.argv[2])
    import cv2
    import imageio_ffmpeg
    import aprenderY as A
    from senas import nombre_carpeta

    id_lsc50 = {e["nombre_lsc"].replace("¿", "").replace("?", "").replace("Jucioso", "Juicioso"): e["id"]
                for e in json.loads(MAPA50.read_text(encoding="utf-8"))["etiquetas"]}
    redondo = lambda a: None if a is None else np.round(np.asarray(a, np.float64), 3).tolist()
    DESTINO.mkdir(parents=True, exist_ok=True)
    indice = []
    for carpeta in sorted(BANCO.iterdir(), key=lambda c: c.name):
        if not carpeta.is_dir():
            continue
        archivo = sorted(carpeta.glob("muestra_*.npz"))[0]
        with np.load(archivo, allow_pickle=False) as z:
            nombre, clase = str(z["nombre"]), int(z["clase"])
        sena = A.preparar(archivo)
        # SIN ESPEJO: el muñeco se ve como la persona del video (de frente, no como en un espejo):
        # x al revés y cada brazo al otro lado
        sin_espejo = lambda a: None if a is None else (np.asarray(a, np.float64) * [-1.0, 1.0])
        lados = {}
        for clave, d in zip(("der", "izq"), sena["lados"]):
            if d is None:
                continue
            lados[clave] = {k: redondo(sin_espejo(d.get(k))) for k in ("muneca", "codo", "mano", "trazo")}
            lados[clave]["trazo_ok"] = np.asarray(d.get("trazo_ok", []), bool).tolist()
        giro = sena.get("giro")
        avatar = {"muestra": archivo.name, "T": int(sena["T"]), "fps": float(sena["fps"]), "lados": lados,
                  "giro": redondo(None if giro is None else -np.asarray(giro, np.float64)),
                  "nariz": redondo(sin_espejo(sena.get("nariz"))), "clip": False}
        destino = DESTINO / nombre_carpeta(nombre)
        destino.mkdir(exist_ok=True)

        # video real: solo LSC50 (muestra_lsc50_P<persona>_r<rep>)
        if archivo.name.startswith("muestra_lsc50") and nombre in id_lsc50:
            persona, rep = archivo.stem.split("_")[2][1:], archivo.stem.split("_")[3][1:]
            avi = videos / f"{id_lsc50[nombre]:04d}_{int(persona):04d}_{int(rep):04d}.avi"
            lector = cv2.VideoCapture(str(avi))
            cuadros = []
            while True:
                ok, c = lector.read()
                if not ok:
                    break
                cuadros.append(c)
            lector.release()
            if cuadros:
                T = avatar["T"]
                h, w = cuadros[0].shape[:2]
                tam = (int(round(w * ALTO_CLIP / h / 2)) * 2, ALTO_CLIP)
                escritor = imageio_ffmpeg.write_frames(
                    str(destino / "clip.mp4"), tam, fps=avatar["fps"], codec="libx264", quality=None, macro_block_size=2,
                    output_params=["-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"])
                escritor.send(None)
                for i in range(T):                       # el cuadro i del video va con el cuadro i del muñeco
                    c = cuadros[int(round(i * (len(cuadros) - 1) / max(1, T - 1)))]   # sin espejo
                    escritor.send(np.ascontiguousarray(cv2.cvtColor(cv2.resize(c, tam, interpolation=cv2.INTER_AREA),
                                                                    cv2.COLOR_BGR2RGB)))
                escritor.close()
                avatar["clip"] = True
        (destino / "avatar.json").write_text(json.dumps(avatar, separators=(",", ":")), encoding="utf-8")
        (destino / "puntos.json").write_text(json.dumps(puntos_por_version(archivo), separators=(",", ":")), encoding="utf-8")
        indice.append({"clase": clase, "nombre": nombre, "carpeta": nombre_carpeta(nombre), "video": avatar["clip"]})
        print(f"  {clase:02d} {nombre}: muñeco" + (" + video" if avatar["clip"] else ""))
    (DESTINO.parent / "senas_90.json").write_text(json.dumps(indice, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo: {len(indice)} señas, {sum(i['video'] for i in indice)} con video -> {DESTINO}")


if __name__ == "__main__":
    main()
