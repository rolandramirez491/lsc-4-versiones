"""
PROBAR EN VIVO las 4 versiones con la cámara, como una persona nueva (igual que la app del celular).

  - A la derecha: la seña que hay que intentar (video real de LSC50 o, si no hay, solo el nombre),
    con lo que ve la versión dibujado encima.
  - Encima de la cámara: SOLO lo que ve la versión elegida
      1 = los 21 puntos de cada mano · 2 = + hombros, codos y muñecas
      3 = manos + trazo de la palma + trazo del codo + lugares de la cara y el pecho · 4 = todo
  - ABRE LA BOCA, haz la seña y BAJA LAS MANOS: la cámara se va y queda el resultado
    (si acertaste, en qué puesto quedó tu seña y las 4 más parecidas con el % de cada parte).

Teclas: 1-4 versión | ← → (o A / D) cambiar la seña | ESPACIO intentar otra vez | ESC salir
Uso:    python probar_en_vivo.py        (cámara 0)
        python probar_en_vivo.py 1      (otra cámara)
"""
import json
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
SENAS = V.PAQUETE / "android" / "app" / "src" / "main" / "assets"     # las 90 señas (video y puntos) de la app
VISIBILIDAD_MINIMA = 0.5
UMBRAL_BOCA_ABIERTA, UMBRAL_BOCA_CERRADA, CUADROS_BOCA = 0.12, 0.06, 3
ANCHO_PANEL, ANCHO_SENA = 470, 380
VENTANA = "LSC 4 versiones"
COLORES = {1: (90, 160, 90), 2: (200, 140, 60), 3: (60, 120, 220), 4: (40, 40, 200)}
VERDE, MAGENTA, DORADO, CELESTE = (0, 255, 0), (255, 0, 255), (0, 215, 255), (255, 200, 0)
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
CLAVES = ("muneca_izq", "muneca_der", "valida_izq", "valida_der", "mano_izq", "mano_der", "mano_izq_ok",
          "mano_der_ok", "codo_izq", "codo_der", "codo_izq_ok", "codo_der_ok", "cara", "cara_ok",
          "ancho_hombros", "centro_hombros")


# =============================================================================
# CÁMARA -> REGISTRO (igual que el traductor)
# =============================================================================

def leer_cuadro(r, ancho, alto):
    """Los puntos de un cuadro en píxeles, o None si no se ve el cuerpo."""
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
            if k.endswith("ok") or k.startswith("valida"):
                registro[k].append(False)
            elif k == "cara":
                registro[k].append(np.zeros((D.CANTIDAD_CARA_REF, 2), np.float32))
            elif k.startswith("mano"):
                registro[k].append(np.zeros((21, 2), np.float32))
            else:
                registro[k].append(np.zeros(2, np.float32))
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


# =============================================================================
# DIBUJO: solo lo que ve cada versión
# =============================================================================

def texto(img, t, pos, escala, color, grosor=2):
    t = D.texto_ascii(t)
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def dibujar_version_camara(img, r, version, palmas, codos):
    """Encima de la cámara (en espejo) con los puntos CRUDOS de MediaPipe, sin filtros ni siluetas."""
    alto, ancho = img.shape[:2]
    e = lambda p: (int(ancho - p.x * ancho), int(p.y * alto))
    ep = lambda q: (int(ancho - q[0]), int(q[1]))
    pose = r.pose_landmarks
    if version in (2, 4) and len(pose) > 16:
        for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
            cv2.line(img, e(pose[a]), e(pose[b]), (255, 255, 255), 3, cv2.LINE_AA)
        for i in range(11, 17):
            cv2.circle(img, e(pose[i]), 7, CELESTE, -1, cv2.LINE_AA)
    if version >= 3:
        if len(r.face_landmarks) > max(D.CARA_REF):
            for i in D.CARA_REF:
                cv2.circle(img, e(r.face_landmarks[i]), 4, DORADO, -1, cv2.LINE_AA)
        for t in codos:
            if len(t) > 1:
                cv2.polylines(img, [np.array([ep(q) for q in t], np.int32)], False, (50, 175, 205), 3, cv2.LINE_AA)
        for t, color in zip(palmas, (VERDE, MAGENTA)):
            if len(t) > 1:
                cv2.polylines(img, [np.array([ep(q) for q in t], np.int32)], False, color, 4, cv2.LINE_AA)
    for puntos, color in ((r.left_hand_landmarks, VERDE), (r.right_hand_landmarks, MAGENTA)):
        if len(puntos) == 21:
            q = [e(p) for p in puntos]
            for a, b in CONEXIONES:
                cv2.line(img, q[a], q[b], color, 2, cv2.LINE_AA)
            for o in q:
                cv2.circle(img, o, 4, (255, 255, 255), -1, cv2.LINE_AA)


def dibujar_version_video(img, p, f, version):
    """Encima del video de la seña (sin espejo), con los puntos de la muestra (puntos.json)."""
    alto, ancho = img.shape[:2]
    a = lambda q: (int(q[0] / p["ancho_ref"] * ancho), int(q[1] / p["alto_ref"] * alto))
    for l, lado in enumerate(p["lados"]):
        color = MAGENTA if l == 0 else VERDE                # igual que el muñeco del celular (sin espejo)
        if version in (2, 4):
            h, c, m = a(p["hombros"][l]), a(lado["codo"][f]), a(lado["muneca"][f])
            cv2.line(img, h, c, (255, 255, 255), 2, cv2.LINE_AA); cv2.line(img, c, m, (255, 255, 255), 2, cv2.LINE_AA)
            for q in (h, c, m):
                cv2.circle(img, q, 4, CELESTE, -1, cv2.LINE_AA)
        if version >= 3:
            for clave, ok, col, g in (("trazo_codo", "trazo_codo_ok", (50, 175, 205), 2), ("palma", "palma_ok", color, 3)):
                tramo = []
                for i in range(f + 1):
                    if lado[ok][i]:
                        tramo.append(a(lado[clave][i]))
                    elif tramo:
                        cv2.polylines(img, [np.array(tramo, np.int32)], False, col, g, cv2.LINE_AA); tramo = []
                if len(tramo) > 1:
                    cv2.polylines(img, [np.array(tramo, np.int32)], False, col, g, cv2.LINE_AA)
        if lado["mano_ok"][f]:
            q = [a(x) for x in lado["mano"][f]]
            for i, j in CONEXIONES:
                cv2.line(img, q[i], q[j], color, 1, cv2.LINE_AA)
            for o in q:
                cv2.circle(img, o, 2, (255, 255, 255), -1, cv2.LINE_AA)
    if version >= 3 and p["cara_ok"][f]:
        for q in p["cara"][f]:
            cv2.circle(img, a(q), 3, DORADO, -1, cv2.LINE_AA)


# =============================================================================
# LAS 90 SEÑAS (video y puntos de la app)
# =============================================================================

class Sena:
    def __init__(self, info):
        self.nombre, self.carpeta, self.tiene_video = info["nombre"], info["carpeta"], info["video"]
        self._cuadros, self._puntos = None, None

    def cargar(self):
        if self._cuadros is None:
            self._cuadros = []
            if self.tiene_video:
                v = cv2.VideoCapture(str(SENAS / "aprender" / self.carpeta / "clip.mp4"))
                while True:
                    ok, c = v.read()
                    if not ok:
                        break
                    self._cuadros.append(c)
                v.release()
                self._puntos = json.loads((SENAS / "aprender" / self.carpeta / "puntos.json").read_text(encoding="utf-8"))
        return self._cuadros, self._puntos


def panel_sena(sena, version, alto, t):
    p = np.full((alto, ANCHO_SENA, 3), (34, 32, 30), np.uint8)
    texto(p, "INTENTA ESTA SENA", (12, 32), 0.6, (200, 200, 200), 1)
    texto(p, sena.nombre, (12, 72), 1.0, VERDE)
    texto(p, "<- / ->  cambiar sena", (12, alto - 16), 0.5, (200, 200, 200), 1)
    cuadros, puntos = sena.cargar()
    if not cuadros:
        texto(p, "(sin video: LSC-54 solo trae puntos)", (12, 120), 0.5, (200, 200, 200), 1)
        return p
    f = int(t * 29) % (len(cuadros) + 20)                    # se repite sin fin, con una pausa al final
    f = min(f, len(cuadros) - 1)
    c = cuadros[f].copy()
    h = int(c.shape[0] * ANCHO_SENA / c.shape[1])
    c = cv2.resize(c, (ANCHO_SENA, h))
    if puntos:
        dibujar_version_video(c, puntos, min(f, puntos["T"] - 1), version)
    y0 = 100
    p[y0:y0 + min(h, alto - y0 - 30)] = c[:alto - y0 - 30]
    return p


# =============================================================================
# PANELES
# =============================================================================

def panel(alto, version, estado, segundos, subio, listo):
    p = np.full((alto, ANCHO_PANEL, 3), (34, 32, 30), np.uint8)
    titulo, detalle = V.VERSIONES[version]
    cv2.rectangle(p, (0, 0), (ANCHO_PANEL, 92), COLORES[version], -1)
    texto(p, titulo, (14, 38), 0.8, (255, 255, 255))
    texto(p, detalle[:52], (14, 72), 0.45, (255, 255, 255), 1)
    y = 140
    if estado == "GRABANDO":
        texto(p, f"GRABANDO {segundos:.1f} s", (14, y), 0.9, (0, 0, 255))
        texto(p, "BAJA LAS MANOS PARA TERMINAR" if subio else "HAZ LA SENA", (14, y + 36), 0.6, (255, 255, 255), 1)
    else:
        texto(p, "1. ABRE LA BOCA", (14, y), 0.7, VERDE if listo else (180, 180, 180))
        texto(p, "2. HAZ LA SENA", (14, y + 34), 0.7, (220, 220, 220))
        texto(p, "3. BAJA LAS MANOS", (14, y + 68), 0.7, (220, 220, 220))
        if not listo:
            texto(p, "(primero cierra la boca)", (14, y + 104), 0.55, (180, 180, 180), 1)
    texto(p, "Teclas 1-4: version  |  ESC: salir", (14, alto - 16), 0.5, (200, 200, 200), 1)
    return p


def pantalla_resultado(alto, ancho, resultado, version, objetivo):
    """En lugar de la cámara: si acertó, en qué puesto quedó y las 4 más parecidas con el % de cada parte."""
    p = np.full((alto, ancho, 3), (24, 20, 20), np.uint8)
    o = resultado.get("objetivo")
    y = 60
    if o and o["puesto"] == 1:
        texto(p, "ACERTASTE!", (30, y), 1.5, VERDE, 3)
        texto(p, f"{V.VERSIONES[version][0]} reconocio {o['nombre']} con {o['pct']:.0f} %", (30, y + 50), 0.8, (255, 255, 255))
    else:
        texto(p, "NO LA RECONOCIO", (30, y), 1.5, (0, 165, 255), 3)
        if o:
            texto(p, f"{o['nombre']} quedo en el puesto {o['puesto']} con {o['pct']:.1f} %", (30, y + 50), 0.8, (255, 255, 255))
    y += 84
    if o and len(o["partes"]) > 1:
        for parte, v in o["partes"].items():
            texto(p, f"   {parte}: {v:.0f} %", (30, y), 0.55, (200, 200, 200), 1)
            y += 24
    y += 16
    texto(p, "LAS 4 QUE MAS SE PARECIERON", (30, y), 0.7, (220, 220, 220))
    y += 40
    for i, (nombre, pct) in enumerate(resultado["top"]):
        mia = objetivo is not None and D.texto_ascii(nombre) == D.texto_ascii(objetivo)
        if mia:
            cv2.rectangle(p, (20, y - 30), (ancho - 20, y + 14 + 22 * len(resultado["partes"])), (30, 80, 30), -1)
        texto(p, f"{i + 1}. {nombre}", (30, y), 0.85, (255, 255, 255))
        texto(p, f"{pct:.0f} %", (ancho - 130, y), 0.85, (255, 255, 255))
        cv2.rectangle(p, (30, y + 8), (30 + int((ancho - 60) * pct / 100), y + 14), COLORES[version], -1)
        y += 24
        for parte, valores in resultado["partes"].items():
            texto(p, f"     {parte}: {valores[nombre]:.0f} %", (30, y + 8), 0.5, (190, 190, 190), 1)
            y += 22
        y += 22
    texto(p, "ESPACIO: INTENTAR OTRA VEZ", (30, alto - 24), 0.8, (0, 0, 255), 2)
    return p


# =============================================================================
# PROGRAMA
# =============================================================================

def main():
    camara_n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    print("Cargando las 4 versiones...")
    R = V.Reconocedor()
    senas = [Sena(i) for i in json.loads((SENAS / "senas_90.json").read_text(encoding="utf-8"))]
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

    version, indice, estado = 4, 0, "PREPARACION"
    resultado, grabacion = None, None
    listo, boca, registro, fin, inicio, destello = False, 0, None, None, 0.0, 0.0
    palmas, codos = [[], []], [[], []]
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    reloj = time.monotonic()
    while True:
        ok, frame = camara.read()
        if not ok:
            continue
        alto, ancho = frame.shape[:2]
        ahora = time.monotonic()
        sena = senas[indice]

        if estado == "RESULTADO":                      # la cámara se va: queda el resultado
            centro = pantalla_resultado(alto, ancho, resultado, version, sena.nombre)
        else:
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
                palmas, codos = [[], []], [[], []]
            if estado == "GRABANDO":
                agregar(registro, datos)
                # seguimiento de la palma y el codo (para dibujar en las versiones 3 y 4)
                for l, puntos in enumerate((r.left_hand_landmarks, r.right_hand_landmarks)):
                    if len(puntos) == 21:
                        palmas[l].append(np.mean([[puntos[j].x * ancho, puntos[j].y * alto] for j in (0, 1, 5, 9, 13, 17)], 0))
                for l, i in enumerate((13, 14)):
                    if len(r.pose_landmarks) > i and getattr(r.pose_landmarks[i], "visibility", 1.0) > VISIBILIDAD_MINIMA:
                        codos[l].append((r.pose_landmarks[i].x * ancho, r.pose_landmarks[i].y * alto))
                if fin.actualizar(datos, ahora):
                    if registro["ancho_hombros"]:
                        grabacion = registro
                        resultado = R.reconocer(registro, version, ancho, alto, objetivo=sena.nombre)
                        estado = "RESULTADO"
                    else:
                        estado = "PREPARACION"
            centro = cv2.flip(frame, 1)
            dibujar_version_camara(centro, r, version, palmas, codos)
            if ahora < destello:
                cv2.rectangle(centro, (0, 0), (ancho - 1, alto - 1), (255, 255, 255), 14)
        pantalla = np.hstack([panel(alto, version, estado, ahora - inicio, fin.subio if fin else False, listo),
                              centro, panel_sena(sena, version, alto, ahora)])
        cv2.imshow(VENTANA, pantalla)
        cv2.setWindowTitle(VENTANA, f"LSC - {V.VERSIONES[version][0]}")
        tecla = cv2.waitKeyEx(1)
        if tecla == 27:
            break
        if tecla in (ord("1"), ord("2"), ord("3"), ord("4")):
            version = tecla - ord("0")
            if estado == "RESULTADO" and grabacion is not None:     # la misma seña con otra versión
                resultado = R.reconocer(grabacion, version, ancho, alto, objetivo=sena.nombre)
        elif tecla in (2424832, ord("a"), ord("A")) and estado != "GRABANDO":
            indice, estado = (indice - 1) % len(senas), "PREPARACION"
        elif tecla in (2555904, ord("d"), ord("D")) and estado != "GRABANDO":
            indice, estado = (indice + 1) % len(senas), "PREPARACION"
        elif tecla == 32 and estado == "RESULTADO":
            estado = "PREPARACION"
    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
