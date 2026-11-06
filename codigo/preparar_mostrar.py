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
        lados = {}
        for clave, d in zip(("izq", "der"), sena["lados"]):
            if d is None:
                continue
            lados[clave] = {k: redondo(d.get(k)) for k in ("muneca", "codo", "mano", "trazo")}
            lados[clave]["trazo_ok"] = np.asarray(d.get("trazo_ok", []), bool).tolist()
        avatar = {"muestra": archivo.name, "T": int(sena["T"]), "fps": float(sena["fps"]), "lados": lados,
                  "giro": redondo(sena.get("giro")), "nariz": redondo(sena.get("nariz")), "clip": False}
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
                    c = cuadros[int(round(i * (len(cuadros) - 1) / max(1, T - 1)))]
                    c = cv2.flip(c, 1)                    # en espejo, como en APRENDER
                    escritor.send(np.ascontiguousarray(cv2.cvtColor(cv2.resize(c, tam, interpolation=cv2.INTER_AREA),
                                                                    cv2.COLOR_BGR2RGB)))
                escritor.close()
                avatar["clip"] = True
        (destino / "avatar.json").write_text(json.dumps(avatar, separators=(",", ":")), encoding="utf-8")
        indice.append({"clase": clase, "nombre": nombre, "carpeta": nombre_carpeta(nombre), "video": avatar["clip"]})
        print(f"  {clase:02d} {nombre}: muñeco" + (" + video" if avatar["clip"] else ""))
    (DESTINO.parent / "senas_90.json").write_text(json.dumps(indice, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo: {len(indice)} señas, {sum(i['video'] for i in indice)} con video -> {DESTINO}")


if __name__ == "__main__":
    main()
