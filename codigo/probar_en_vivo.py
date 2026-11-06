"""
PROBAR EN VIVO las 4 versiones con la cámara, como una persona nueva.

  1. Ponte frente a la cámara (que se vean la cara, los hombros y las manos).
  2. ABRE LA BOCA: empieza a grabar (sale un destello).
  3. Haz la seña y BAJA LAS MANOS: termina (o a los 5 s).
  4. A la izquierda salen las 4 señas más parecidas, con su % y lo que opina cada parte de la versión.

Teclas: 1, 2, 3, 4 = cambiar de versión (o clic en los botones de arriba) | ESC = salir
Uso:    python probar_en_vivo.py        (cámara 0)
        python probar_en_vivo.py 1      (otra cámara)
"""
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import dibujo_senas as D
import versiones as V

RUTA_HOLISTIC = V.PAQUETE / "modelos" / "holistic_landmarker.task"
VISIBILIDAD_MINIMA = 0.5
UMBRAL_BOCA_ABIERTA, UMBRAL_BOCA_CERRADA, CUADROS_BOCA = 0.12, 0.06, 3
ANCHO_PANEL = 470
VENTANA = "LSC 4 versiones"
COLORES = {1: (90, 160, 90), 2: (200, 140, 60), 3: (60, 120, 220), 4: (40, 40, 200)}
CLAVES = ("muneca_izq", "muneca_der", "valida_izq", "valida_der", "mano_izq", "mano_der", "mano_izq_ok",
          "mano_der_ok", "codo_izq", "codo_der", "codo_izq_ok", "codo_der_ok", "cara", "cara_ok",
          "ancho_hombros", "centro_hombros")


def leer_cuadro(r, ancho, alto):
    """Los puntos de un cuadro en píxeles (igual que el traductor), o None si no se ve el cuerpo."""
    if len(r.pose_landmarks) == 0:
        return None
    pose = r.pose_landmarks
    px = lambda i: np.array([pose[i].x * ancho, pose[i].y * alto], np.float32)
    vis = lambda i: getattr(pose[i], "visibility", 1.0) > VISIBILIDAD_MINIMA

    def mano(puntos):
        if len(puntos) == 0:
            return np.zeros((21, 2), np.float32), False
        return np.array([[p.x * ancho, p.y * alto] for p in puntos], np.float32), True

    mi, mio = mano(r.left_hand_landmarks)
    md, mdo = mano(r.right_hand_landmarks)
    mi, mio, md, mdo = D.asignar_manos(mi, mio, md, mdo, px(D.POSE_MUNECA_IZQ) if vis(D.POSE_MUNECA_IZQ) else None,
                                       px(D.POSE_MUNECA_DER) if vis(D.POSE_MUNECA_DER) else None)
    cara, cara_ok = D.cara_ref_pixeles(r, ancho, alto)
    hi, hd = px(D.POSE_HOMBRO_IZQ), px(D.POSE_HOMBRO_DER)
    return {"muneca_izq": px(D.POSE_MUNECA_IZQ), "muneca_der": px(D.POSE_MUNECA_DER),
            "valida_izq": vis(D.POSE_MUNECA_IZQ), "valida_der": vis(D.POSE_MUNECA_DER),
            "mano_izq": np.asarray(mi, np.float32), "mano_der": np.asarray(md, np.float32),
            "mano_izq_ok": mio, "mano_der_ok": mdo,
            "codo_izq": px(D.POSE_CODO_IZQ), "codo_der": px(D.POSE_CODO_DER),
            "codo_izq_ok": vis(D.POSE_CODO_IZQ), "codo_der_ok": vis(D.POSE_CODO_DER),
            "cara": cara, "cara_ok": cara_ok,
            "ancho_hombros": float(np.linalg.norm(hi - hd)), "centro_hombros": (hi + hd) / 2.0}


def agregar(registro, datos):
    if datos is None:                              # cuadro sin cuerpo: todo "no visto"
        for k in CLAVES[:14]:
            ejemplo = registro[k][-1] if registro[k] else None
            if k.endswith("ok") or k.startswith("valida"):
                registro[k].append(False)
            elif k == "cara":
                registro[k].append(np.zeros((D.CANTIDAD_CARA_REF, 2), np.float32))
            elif k.startswith("mano"):
                registro[k].append(np.zeros((21, 2), np.float32))
            else:
                registro[k].append(np.zeros(2, np.float32) if ejemplo is None else ejemplo * 0)
        return
    for k in CLAVES:
        registro[k].append(datos[k])


def apertura_boca(r, ancho, alto):
    c = r.face_landmarks
    if len(c) <= 263:
        return None
    px = lambda i: np.array([c[i].x * ancho, c[i].y * alto])
    ref = np.linalg.norm(px(33) - px(263))
    return None if ref < 1e-6 else float(np.linalg.norm(px(13) - px(14)) / ref)


def texto(img, t, pos, escala, color, grosor=2):
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def sin_tildes(t):
    return D.texto_ascii(t)


def dibujar_puntos(img, r, ancho):
    """Puntos de las manos y la cara (en espejo, como un espejo de verdad)."""
    for puntos, color in ((r.left_hand_landmarks, (0, 255, 0)), (r.right_hand_landmarks, (255, 0, 255))):
        for p in puntos:
            cv2.circle(img, (int(ancho - p.x * ancho), int(p.y * img.shape[0])), 3, color, -1, cv2.LINE_AA)
    pose = r.pose_landmarks
    if len(pose):
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12)):
            pa = (int(ancho - pose[a].x * ancho), int(pose[a].y * img.shape[0]))
            pb = (int(ancho - pose[b].x * ancho), int(pose[b].y * img.shape[0]))
            cv2.line(img, pa, pb, (220, 220, 220), 2, cv2.LINE_AA)


def panel(alto, version, resultado, estado, segundos, subio, listo):
    p = np.full((alto, ANCHO_PANEL, 3), (34, 32, 30), np.uint8)
    titulo, detalle = V.VERSIONES[version]
    cv2.rectangle(p, (0, 0), (ANCHO_PANEL, 92), COLORES[version], -1)
    texto(p, sin_tildes(titulo), (14, 38), 0.8, (255, 255, 255))
    texto(p, sin_tildes(detalle)[:52], (14, 72), 0.45, (255, 255, 255), 1)
    y = 130
    if estado == "GRABANDO":
        texto(p, f"GRABANDO {segundos:.1f} s", (14, y), 0.9, (0, 0, 255))
        texto(p, "BAJA LAS MANOS PARA TERMINAR" if subio else "HAZ LA SENA", (14, y + 36), 0.6, (255, 255, 255), 1)
        return p
    if resultado is None:
        texto(p, "ABRE LA BOCA Y HAZ LA SENA", (14, y), 0.65, (0, 255, 0) if listo else (180, 180, 180))
        if not listo:
            texto(p, "(primero cierra la boca)", (14, y + 32), 0.55, (180, 180, 180), 1)
        return p
    texto(p, "LAS 4 MAS PARECIDAS", (14, y), 0.65, (255, 255, 255))
    y += 40
    for i, (nombre, pct) in enumerate(resultado["top"]):
        color = (0, 255, 0) if i == 0 else (220, 220, 220)
        texto(p, f"{i + 1}. {sin_tildes(nombre)}", (14, y), 0.75 if i == 0 else 0.65, color)
        texto(p, f"{pct:.0f} %", (ANCHO_PANEL - 90, y), 0.7, color)
        cv2.rectangle(p, (14, y + 8), (14 + int((ANCHO_PANEL - 28) * pct / 100), y + 14), color, -1)
        y += 26
        for parte, valores in resultado["partes"].items():
            texto(p, f"   {sin_tildes(parte)}: {valores[nombre]:.0f} %", (14, y), 0.45, (190, 190, 190), 1)
            y += 20
        y += 14
    texto(p, "ABRE LA BOCA PARA OTRA VEZ", (14, alto - 20), 0.55, (255, 255, 255), 1)
    return p


def main():
    camara_n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    print("Cargando las 4 versiones...")
    R = V.Reconocedor()
    opciones = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(RUTA_HOLISTIC)), running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5, min_face_landmarks_confidence=0.5, min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5, min_hand_landmarks_confidence=0.5)
    detector = vision.HolisticLandmarker.create_from_options(opciones)
    camara = cv2.VideoCapture(camara_n, cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camara.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    version, estado, resultado = 4, "PREPARACION", None
    listo, boca, registro, fin, inicio, destello = False, 0, None, None, 0.0, 0.0
    botones = {}
    pedido = [None]

    def clic(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            for v, (x0, y0, x1, y1) in botones.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    pedido[0] = v

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(VENTANA, clic)
    reloj = time.monotonic()
    while True:
        ok, frame = camara.read()
        if not ok:
            continue
        alto, ancho = frame.shape[:2]
        ahora = time.monotonic()
        r = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                      int((ahora - reloj) * 1000))
        datos = leer_cuadro(r, ancho, alto)
        a = apertura_boca(r, ancho, alto)
        if a is not None and a < UMBRAL_BOCA_CERRADA:
            listo = True
        boca = boca + 1 if (a is not None and a >= UMBRAL_BOCA_ABIERTA) else 0
        if estado != "GRABANDO" and listo and boca >= CUADROS_BOCA:
            estado, registro, fin, inicio = "GRABANDO", {k: [] for k in CLAVES}, D.FinDeSena(ahora), ahora
            listo, boca, destello = False, 0, ahora + 0.45

        if estado == "GRABANDO":
            agregar(registro, datos)
            if fin.actualizar(datos, ahora):
                estado = "PREPARACION"
                destello = ahora + 0.45
                if registro["ancho_hombros"]:
                    resultado = R.reconocer(registro, version, ancho, alto)
                    resultado["registro"] = registro
                else:
                    resultado = None

        if pedido[0] is not None:                      # al cambiar de versión se re-evalúa la última seña
            version, pedido[0] = pedido[0], None
            if resultado is not None:
                reg = resultado["registro"]
                resultado = R.reconocer(reg, version, ancho, alto)
                resultado["registro"] = reg
            cv2.setWindowTitle(VENTANA, f"LSC - {V.VERSIONES[version][0]}")

        vista = cv2.flip(frame, 1)
        dibujar_puntos(vista, r, ancho)
        if ahora < destello:
            cv2.rectangle(vista, (0, 0), (ancho - 1, alto - 1), (255, 255, 255), 14)
        # botones de versión arriba de la cámara
        botones.clear()
        x = 12
        for v in (1, 2, 3, 4):
            t = f"VERSION {v}"
            (w, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vista, (x, 10), (x + w + 24, 48), COLORES[v] if v == version else (60, 60, 60), -1)
            cv2.rectangle(vista, (x, 10), (x + w + 24, 48), (255, 255, 255), 2 if v == version else 1)
            cv2.putText(vista, t, (x + 12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            botones[v] = (ANCHO_PANEL + x, 10, ANCHO_PANEL + x + w + 24, 48)
            x += w + 34
        pantalla = np.hstack([panel(alto, version, resultado, estado, ahora - inicio,
                                    fin.subio if fin else False, listo), vista])
        cv2.imshow(VENTANA, pantalla)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        if ord("1") <= tecla <= ord("4"):
            pedido[0] = tecla - ord("0")
    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
