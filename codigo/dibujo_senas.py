

import unicodedata

import cv2
import numpy as np
import torch
import torch.nn as nn


CANTIDAD_CUERPO = 14
CANTIDAD_MANO = 21
CANTIDAD_CARA = 71

INICIO_CUERPO = 0
INICIO_MI = CANTIDAD_CUERPO * 3
INICIO_MD = INICIO_MI + CANTIDAD_MANO * 3
INICIO_CARA = INICIO_MD + CANTIDAD_MANO * 3
TOTAL_BASE = INICIO_CARA + CANTIDAD_CARA * 3  # 381

# Los 14 puntos de cuerpo son los puntos 11..24 de MediaPipe Pose:
# 0-1 hombros, 2-3 codos, 4-5 muñecas, 6-7 meñique, 8-9 índice,
# 10-11 pulgar, 12-13 caderas
CUERPO_HOMBRO_IZQ = 0
CUERPO_HOMBRO_DER = 1
CUERPO_MUNECA_IZQ = 4
CUERPO_MUNECA_DER = 5
CUERPO_CADERA_IZQ = 12
CUERPO_CADERA_DER = 13

# Mismos puntos en MediaPipe Pose completo (webcam)
POSE_HOMBRO_IZQ = 11
POSE_HOMBRO_DER = 12
POSE_MUNECA_IZQ = 15
POSE_MUNECA_DER = 16

# Puntos de la mano (MediaPipe Hands)
MANO_MUNECA = 0
MANO_BASE_MEDIO = 9


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PUNTOS_TRAZO = 48        # puntos por trazo después de remuestrear
TAMANO_IMAGEN = 64       # imagen de 64x64
MEDIA_VENTANA = 1.5      # el lienzo cubre ±1.5 anchos de hombros
GROSOR_LINEA = 2
MIN_FRAMES_MANO = 5      # menos frames que esto = mano no usada

# Línea de "mano activa": hombros + 0.8 anchos de hombro hacia abajo.
# Por debajo de esa línea la mano está subiendo, bajando o en descanso.
UMBRAL_TORSO = 1.8      # antes 0.8: ahora también cuentan las señas bajas (casi hasta la cintura)

# Corrección de la escala vertical del dataset para que coincida con la cámara.
# Se calcula con comparar_escala.py. 1.0 = sin corrección.
ESCALA_Y_DATASET = 1.0

# Extras por mano:
# [activa, inicio x/y, final x/y, centro x/y, tamaño ancho/alto,
#  forma_valida, 21 puntos x/y]
# ---- Puntos de referencia de la CARA (índices de MediaPipe) que se guardan en cada muestra ----
# La cara casi no se mueve durante la seña: sirve de "mapa" para saber DÓNDE está la mano
CARA_REF = (1, 13, 14, 152, 10, 159, 145, 386, 374, 234, 454)
CANTIDAD_CARA_REF = len(CARA_REF)
# lugares con los que se compara la mano: nariz, boca, mentón, frente, ojo A, ojo B, oreja A, oreja B, pecho
LUGARES = ("nariz", "boca", "menton", "frente", "ojo_a", "ojo_b", "oreja_a", "oreja_b", "pecho")
PARES_ESPEJO = ((4, 5), (6, 7))          # al hacer espejo se cambian ojo A <-> B y oreja A <-> B
DEDOS_PUNTAS = (4, 8, 12, 16, 20)        # pulgar, índice, medio, anular, meñique
CARA_POR_MANO = 1 + len(LUGARES) * 3     # válida + (dx, dy, distancia mínima) por lugar = 28

# v14: la palma (sin dedos) hace el trazo; los dedos van como numeros
PALMA_PUNTOS = (0, 1, 5, 9, 13, 17)      # muñeca y bases de los dedos
DEDOS_BASES = (1, 5, 9, 13, 17)          # donde empieza cada dedo
MOMENTOS = 3                             # inicio, mitad y final de la seña
DEDOS_POR_MANO = MOMENTOS * len(DEDOS_PUNTAS) * 2 + len(DEDOS_PUNTAS) * 2   # (largo, estirado) x 3 + puntas x/y = 40

# v16: el brazo como una CURVA en el tiempo (para saber si se dobla, se estira, dos veces...)
PUNTOS_ANGULO = 12                       # puntos de la curva del ángulo a lo largo de la seña
DOBLEZ_MINIMO = 20.0                     # grados que tiene que bajar el ángulo para contar como "se dobló"
# válido + ángulo x 12 + dir x x 12 + dir y x 12 + mínimo + máximo + veces que se dobla = 40
BRAZO_POR_MANO = 1 + PUNTOS_ANGULO * 3 + 3
LADO_HOMBRO = (0.5, -0.5)                # hombro = centro de hombros + esto x ancho (en x); izq queda a la derecha en la imagen

EXTRAS_POR_MANO = 1 + 2 + 2 + 2 + 2 + 1 + DEDOS_POR_MANO + CARA_POR_MANO + BRAZO_POR_MANO   # 78 + 10 = 88
CANTIDAD_EXTRAS = EXTRAS_POR_MANO * 2                          # 176

# Posiciones dentro del bloque de una mano
E_ACTIVA = 0
E_INICIO = slice(1, 3)
E_FINAL = slice(3, 5)
E_CENTRO = slice(5, 7)
E_TAMANO = slice(7, 9)
E_FORMA_VALIDA = 9                                            # se vieron los dedos
E_DEDOS = slice(10, 10 + MOMENTOS * len(DEDOS_PUNTAS) * 2)    # [momento][dedo][largo, estirado] = 30
E_PUNTAS = slice(E_DEDOS.stop, E_DEDOS.stop + len(DEDOS_PUNTAS) * 2)   # puntas x/y respecto a la palma = 10
E_FORMA = slice(10, E_PUNTAS.stop)                            # todo lo de los dedos
E_CARA_VALIDA = E_PUNTAS.stop                                 # 50
E_CARA = slice(E_CARA_VALIDA + 1, E_CARA_VALIDA + CARA_POR_MANO)
E_BRAZO_VALIDO = E_CARA.stop                                  # 78
E_BRAZO = slice(E_BRAZO_VALIDO + 1, E_BRAZO_VALIDO + BRAZO_POR_MANO)   # [momento][ángulo, dir x, dir y]
TRAZOS_POR_MANO = 1                                          # solo la palma
CANTIDAD_TRAZOS = 2 * TRAZOS_POR_MANO                         # 2 (más los 2 del codo aparte)
CANALES_DIBUJO = 2                                            # palma izq, palma der

# Limpieza del recorrido de la muñeca (antes de dibujar el trazo)
SEGUNDOS_MUESTRA = 3.4     # duración de cada grabación (igual a SEGUNDOS_GRABACION)
SEGUNDOS_HUECO = 1.0       # si la muñeca se pierde hasta 1 s, se une suave
SALTO_MAXIMO = 0.25        # anchos de hombro por frame (a 30 FPS) = movimiento imposible
BRAZO_BORDE = 0.3          # s: al perder/encontrar la mano, el brazo cubre hasta esto
MANO_FANTASMA_SEGUNDOS = 1.5  # mano tapada: se mantiene pegada al antebrazo hasta esto
FRACCION_ANTEBRAZO = 1.0   # punto del DIBUJO 2: 1.0 = el CODO (0.5 sería la mitad del antebrazo)
RADIO_PINCEL = 0.06        # anchos de hombro: el trazo solo avanza si la mano se aleja más que esto
SEGUNDOS_SUAVIZADO = 0.15  # ventana del suavizado
ANGULO_MAXIMO_HUECO = 45   # la curva del hueco nunca se abre más que esto

# Suavizado de los 21 puntos de la mano (filtro "One Euro"):
# quieta o lenta -> muy suave; rápida -> sigue el movimiento sin retraso.
MANO_CORTE_MINIMO = 1.0    # Hz. Más bajo = más suave cuando la mano va lenta
MANO_BETA = 0.5            # cuánto se "suelta" el filtro cuando la mano va rápido
DEDOS_CORTE_MINIMO = 0.5   # Hz para la FORMA de los dedos (más bajo = dedos más quietos)
DEDOS_BETA = 0.3
RADIO_DEDOS = 0.06         # en tamaños de mano: temblor de los dedos menor a esto se ignora

VERSION = "hibrido_15_v17"   # v17: trazo de la muñeca del esqueleto, 5 dedos, curva del ángulo del codo, cara como mapa


def configuracion():
    """Lo que debe coincidir entre entrenamiento y cámara."""
    return {
        "version": VERSION,
        "puntos_trazo": PUNTOS_TRAZO,
        "tamano_imagen": TAMANO_IMAGEN,
        "media_ventana": MEDIA_VENTANA,
        "grosor_linea": GROSOR_LINEA,
        "min_frames_mano": MIN_FRAMES_MANO,
        "umbral_torso": UMBRAL_TORSO,
        "escala_y_dataset": ESCALA_Y_DATASET,
        "cantidad_extras": CANTIDAD_EXTRAS,
        "segundos_hueco": SEGUNDOS_HUECO,
        "salto_maximo": SALTO_MAXIMO,
        "segundos_suavizado": SEGUNDOS_SUAVIZADO,
        "angulo_maximo_hueco": ANGULO_MAXIMO_HUECO,
        "radio_pincel": RADIO_PINCEL,
        "brazo_borde": BRAZO_BORDE,
        "mano_fantasma_segundos": MANO_FANTASMA_SEGUNDOS,
        "fraccion_antebrazo": FRACCION_ANTEBRAZO,
        "mano_corte_minimo": MANO_CORTE_MINIMO,
        "mano_beta": MANO_BETA,
        "dedos": [DEDOS_CORTE_MINIMO, DEDOS_BETA, RADIO_DEDOS],
        "cantidad_trazos": CANTIDAD_TRAZOS,
        "canales_dibujo": CANALES_DIBUJO,
        "cara_ref": list(CARA_REF),
    }


def texto_ascii(texto):
    """Quita tildes y signos para poder escribir con cv2.putText."""
    texto = unicodedata.normalize("NFD", str(texto))
    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    texto = texto.replace("¿", "").replace("¡", "")
    return texto.encode("ascii", "ignore").decode().upper()


# =============================================================================
# TRAZO
# =============================================================================

def suavizar(puntos, ventana=3):
    if len(puntos) < ventana:
        return puntos

    kernel = np.ones(ventana) / ventana
    x = np.convolve(puntos[:, 0], kernel, mode="valid")
    y = np.convolve(puntos[:, 1], kernel, mode="valid")

    return np.stack([x, y], axis=1)


def remuestrear_por_longitud(puntos, cantidad):
    """Reparte 'cantidad' puntos a distancias iguales sobre el trazo."""
    puntos = np.asarray(puntos, dtype=np.float64)

    segmentos = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
    acumulado = np.concatenate([[0.0], np.cumsum(segmentos)])
    total = acumulado[-1]

    if total < 1e-8:
        centro = puntos.mean(axis=0, keepdims=True)
        return np.repeat(centro, cantidad, axis=0)

    objetivos = np.linspace(0.0, total, cantidad)

    x = np.interp(objetivos, acumulado, puntos[:, 0])
    y = np.interp(objetivos, acumulado, puntos[:, 1])

    return np.stack([x, y], axis=1)


# =============================================================================
# SUAVIZADO DE LOS 21 PUNTOS DE LA MANO
# =============================================================================

def _alfa(corte, dt):
    tau = 1.0 / (2.0 * np.pi * corte)
    return 1.0 / (1.0 + tau / dt)


class FiltroUnoEuro:
    """
    Filtro "One Euro": cuanto más lento se mueve algo, más lo suaviza
    (quita el temblor); cuando se mueve rápido, casi no lo frena.
    La velocidad se mide en "tamaños de mano por segundo".
    """

    def __init__(self, corte_minimo, beta, corte_velocidad=1.0):
        self.corte_minimo = corte_minimo
        self.beta = beta
        self.corte_velocidad = corte_velocidad
        self.reiniciar()

    def reiniciar(self):
        self.valor = None
        self.velocidad = None

    def __call__(self, x, dt, escala):
        x = np.asarray(x, dtype=np.float64)

        if self.valor is None:
            self.valor = x.copy()
            self.velocidad = np.zeros_like(x)
            return x.copy()

        dt = max(dt, 1e-3)
        a_v = _alfa(self.corte_velocidad, dt)
        self.velocidad = a_v * (x - self.valor) / dt + (1 - a_v) * self.velocidad

        rapidez = np.linalg.norm(self.velocidad.reshape(-1, 2), axis=1).max() / max(escala, 1e-6)
        corte = self.corte_minimo + self.beta * rapidez

        a = _alfa(corte, dt)
        self.valor = a * x + (1 - a) * self.valor
        return self.valor.copy()


class FiltroMano:
    """
    Suaviza los 21 puntos de una mano en dos partes:
    - la POSICIÓN (muñeca), y
    - la FORMA (dedos medidos desde la muñeca),
    así los dedos no tiemblan pero la mano no se deforma al moverse.
    """

    def __init__(self):
        self.posicion = FiltroUnoEuro(MANO_CORTE_MINIMO, MANO_BETA)
        self.forma = FiltroUnoEuro(DEDOS_CORTE_MINIMO, DEDOS_BETA)
        self.dedos = None   # "pincel perezoso" de cada punto de la forma

    def reiniciar(self):
        self.posicion.reiniciar()
        self.forma.reiniciar()
        self.dedos = None

    def __call__(self, puntos, ok, dt):
        puntos = np.asarray(puntos, dtype=np.float64)

        if not ok:
            self.reiniciar()
            return puntos

        escala = float(np.linalg.norm(puntos[MANO_BASE_MEDIO] - puntos[MANO_MUNECA]))
        muneca = self.posicion(puntos[MANO_MUNECA], dt, escala)
        forma = self.forma(puntos - puntos[MANO_MUNECA], dt, escala)

        # Cada dedo solo se mueve si cambia de verdad (más que RADIO_DEDOS):
        # el temblor rapidito de MediaPipe se ignora
        if self.dedos is None:
            self.dedos = forma.copy()
        else:
            radio = RADIO_DEDOS * max(escala, 1.0)
            diferencia = forma - self.dedos
            distancia = np.linalg.norm(diferencia, axis=1, keepdims=True)
            avance = np.clip(distancia - radio, 0.0, None) / np.maximum(distancia, 1e-9)
            self.dedos = self.dedos + diferencia * avance

        return self.dedos + muneca


def _girar(vectores, angulo):
    c, s = np.cos(angulo), np.sin(angulo)
    return np.stack([
        c * vectores[..., 0] - s * vectores[..., 1],
        s * vectores[..., 0] + c * vectores[..., 1]
    ], axis=-1)


ANTEBRAZO_CORTE_MINIMO = 0.4   # Hz: más bajo = antebrazo más tranquilo
ANTEBRAZO_BETA = 0.3
RADIO_ANTEBRAZO = 0.04         # en largos de antebrazo: temblor menor a esto se ignora
GIRO_MAXIMO = 300.0            # grados por segundo que puede girar el antebrazo


class FiltroAntebrazo:
    """
    Suaviza muñeca y codo del brazo como un objeto rígido:
    - filtro One Euro (quieto -> muy suave, rápido -> sigue sin retraso),
    - zona muerta: el temblor pequeño no lo mueve,
    - no puede girar más rápido que GIRO_MAXIMO (no hay giros imposibles).
    """

    def __init__(self):
        self.filtro = FiltroUnoEuro(ANTEBRAZO_CORTE_MINIMO, ANTEBRAZO_BETA)
        self.actual = None

    def reiniciar(self):
        self.filtro.reiniciar()
        self.actual = None

    def __call__(self, muneca, codo, dt):
        extremos = np.array([muneca, codo], dtype=np.float64)
        largo = max(float(np.linalg.norm(extremos[0] - extremos[1])), 1.0)
        suave = self.filtro(extremos, dt, largo / 3.0)

        if self.actual is None:
            self.actual = suave.copy()
            return self.actual[0].copy(), self.actual[1].copy()

        # Zona muerta en cada extremo
        radio = RADIO_ANTEBRAZO * largo
        diferencia = suave - self.actual
        distancia = np.linalg.norm(diferencia, axis=1, keepdims=True)
        avance = np.clip(distancia - radio, 0.0, None) / np.maximum(distancia, 1e-9)
        nuevo = self.actual + diferencia * avance

        # Límite de giro (alrededor del codo)
        antes = self.actual[0] - self.actual[1]
        ahora = nuevo[0] - nuevo[1]
        if np.linalg.norm(antes) > 1e-6 and np.linalg.norm(ahora) > 1e-6:
            angulo = np.arctan2(antes[0] * ahora[1] - antes[1] * ahora[0], float(np.dot(antes, ahora)))
            maximo = np.radians(GIRO_MAXIMO) * dt
            if abs(angulo) > maximo:
                nuevo[0] = nuevo[1] + _girar(antes / np.linalg.norm(antes), np.sign(angulo) * maximo) * np.linalg.norm(ahora)

        self.actual = nuevo
        return nuevo[0].copy(), nuevo[1].copy()


def suavizar_manos(puntos_mano, mano_ok, fps):
    """Aplica FiltroMano a toda una grabación (T, 21, 2) -> (T, 21, 2)."""
    puntos_mano = np.asarray(puntos_mano, dtype=np.float64)
    mano_ok = np.asarray(mano_ok, dtype=bool).reshape(-1)

    if puntos_mano.ndim != 3 or len(puntos_mano) != len(mano_ok):
        return puntos_mano

    filtro = FiltroMano()
    dt = 1.0 / max(float(fps), 1.0)
    salida = puntos_mano.copy()

    for i in range(len(salida)):
        salida[i] = filtro(puntos_mano[i], mano_ok[i], dt)

    return salida


POSE_CODO_IZQ = 13
POSE_CODO_DER = 14


# ---- Manos en la cámara ----
SALTO_MANO_VIVO = 1.2      # salto (en tamaños de mano) que se considera error de detección
RECHAZOS_MAXIMOS = 5       # tras tantos frames seguidos lejos, se acepta (sí se movió)
TRANSICION_MANO = 0.25     # s para pasar suave de la mano quieta a la nueva posición
MANO_QUIETA_SEGUNDOS = 0.3 # cuánto se queda quieta la mano perdida antes de desaparecer
CRUCE_MARGEN = 0.7         # una mano se pasa al otro brazo solo si está claramente más cerca
MANOS_DUPLICADAS = 0.45    # si las dos manos quedan casi encima (en tamaños de mano) es la misma

# Doblez de la muñeca (mano respecto al antebrazo), en grados.
# Mano DERECHA: hacia su derecha 87°, hacia su izquierda 70°.
# Mano IZQUIERDA: hacia su izquierda 87°, hacia su derecha 70°.
DOBLEZ_HACIA_AFUERA = 87
DOBLEZ_HACIA_ADENTRO = 70
INVERTIR_DOBLEZ = False    # si las detecciones buenas salen rechazadas, prueba True


def asignar_manos(mano_izq, izq_ok, mano_der, der_ok, pose_izq, pose_der):
    """
    Evita que la mano de un lado aparezca en el antebrazo del otro.
    Cada mano detectada se asigna a la muñeca del brazo (pose) más cercana:
    - si MediaPipe las cruzó, se intercambian;
    - si puso las dos en el mismo brazo, se queda la más cercana y la otra se descarta;
    - si una sola está claramente en el brazo contrario, se pasa a ese lado.
    pose_izq / pose_der: (x, y) de la muñeca del brazo, o None si no se ve.
    Devuelve (mano_izq, izq_ok, mano_der, der_ok).
    """
    if pose_izq is None or pose_der is None:
        return mano_izq, izq_ok, mano_der, der_ok

    pose = [np.asarray(pose_izq, dtype=np.float64), np.asarray(pose_der, dtype=np.float64)]
    manos = [np.asarray(mano_izq, dtype=np.float64), np.asarray(mano_der, dtype=np.float64)]
    ok = [bool(izq_ok), bool(der_ok)]

    def distancia(mano, lado):
        return float(np.linalg.norm(mano[MANO_MUNECA] - pose[lado]))

    # Lado al que pertenece cada mano detectada
    destino = [None, None]
    for lado in (0, 1):
        if not ok[lado]:
            continue
        propia, otra = distancia(manos[lado], lado), distancia(manos[lado], 1 - lado)
        destino[lado] = (1 - lado) if otra < CRUCE_MARGEN * propia else lado

    # Las dos "manos" casi en el mismo lugar = MediaPipe duplicó una mano:
    # se queda solo la que está más cerca de su propio brazo
    if ok[0] and ok[1]:
        tamano = max(
            float(np.linalg.norm(manos[0][MANO_BASE_MEDIO] - manos[0][MANO_MUNECA])),
            float(np.linalg.norm(manos[1][MANO_BASE_MEDIO] - manos[1][MANO_MUNECA])), 1.0
        )
        separacion = float(np.linalg.norm(manos[0].mean(axis=0) - manos[1].mean(axis=0)))
        if separacion < MANOS_DUPLICADAS * tamano:
            quitar = 0 if distancia(manos[0], 0) > distancia(manos[1], 1) else 1
            destino[quitar] = None

    nuevas = [None, None]
    for lado in (0, 1):
        d = destino[lado]
        if d is None:
            continue
        if nuevas[d] is None or distancia(manos[lado], d) < distancia(nuevas[d], d):
            nuevas[d] = manos[lado]

    vacia = np.zeros_like(manos[0])
    return (
        nuevas[0] if nuevas[0] is not None else vacia, nuevas[0] is not None,
        nuevas[1] if nuevas[1] is not None else vacia, nuevas[1] is not None
    )


def corregir_manos_grabacion(puntos_manos, manos_ok, munecas, validas):
    """asignar_manos frame por frame sobre una grabación completa."""
    try:
        mi, md = np.array(puntos_manos[0], dtype=np.float64), np.array(puntos_manos[1], dtype=np.float64)
        oi, od = np.array(manos_ok[0], dtype=bool), np.array(manos_ok[1], dtype=bool)
        pi, pd = np.asarray(munecas[0], dtype=np.float64), np.asarray(munecas[1], dtype=np.float64)
        vi, vd = np.asarray(validas[0], dtype=bool), np.asarray(validas[1], dtype=bool)
    except Exception:
        return puntos_manos, manos_ok

    if mi.ndim != 3 or md.ndim != 3 or not (len(mi) == len(md) == len(oi) == len(pi)):
        return puntos_manos, manos_ok

    for t in range(len(oi)):
        if (oi[t] or od[t]) and vi[t] and vd[t]:
            mi[t], oi[t], md[t], od[t] = asignar_manos(mi[t], oi[t], md[t], od[t], pi[t], pd[t])

    return [mi, md], [oi, od]


# ---- Tamaño de la mano y de cada dedo ----
CRECIMIENTO_MAXIMO = 1.15   # un hueso del dedo no puede verse más largo que esto x su largo conocido
SALTO_TAMANO = 1.35         # si toda la mano "crece" más que esto de golpe -> detección mala
APRENDIZAJE_LARGOS = 0.05   # qué tan rápido se ajustan los largos conocidos (lento = estable)
CUADROS_PARA_CRECER = 8     # si un hueso se ve más largo tantos cuadros seguidos, de verdad mide eso (no es error)

PADRE = {j: (0 if j in (1, 5, 9, 13, 17) else j - 1) for j in range(1, 21)}


class LargosMano:
    """
    Recuerda el largo de cada hueso de la mano (20 segmentos: palma y falanges).
    - Un dedo puede verse más CORTO (si apunta a la cámara), pero nunca más
      LARGO de lo que mide: si MediaPipe lo estira, se recorta a su largo.
    - Si TODA la mano aparece mucho más grande de golpe, es una detección mala.
    - Los largos se actualizan despacio con las detecciones buenas
      (así se adapta si acercas la mano a la cámara poco a poco).
    """

    def __init__(self):
        self.largos = None
        self.rechazos = 0
        self.largo_seguidos = np.zeros(20, dtype=int)   # cuadros seguidos que cada hueso se vio más largo

    def reiniciar(self):
        self.largos = None
        self.rechazos = 0
        self.largo_seguidos = np.zeros(20, dtype=int)

    @staticmethod
    def _medir(puntos):
        return np.array([np.linalg.norm(puntos[j] - puntos[PADRE[j]]) for j in range(1, 21)])

    def corregir(self, puntos):
        """Devuelve (puntos corregidos, es_buena)."""
        medidos = self._medir(puntos)

        if self.largos is None:
            self.largos = medidos.copy()
            return puntos, True

        # ¿Toda la mano creció de golpe? -> mala
        if np.median(medidos / np.maximum(self.largos, 1e-6)) > SALTO_TAMANO:
            self.rechazos += 1
            if self.rechazos < 10:
                return puntos, False
            # Lleva 10 frames así: de verdad está más cerca -> aprender el nuevo tamaño
            self.largos = medidos.copy()
            self.rechazos = 0
            return puntos, True

        self.rechazos = 0

        # Un hueso que se ve más largo VARIOS cuadros seguidos de verdad mide eso (p. ej. el dedo se aprendió
        # doblado al aparecer la mano): se acepta su largo. Un error de MediaPipe dura 1 o 2 cuadros y se sigue recortando.
        largo_ahora = medidos > self.largos * CRECIMIENTO_MAXIMO
        self.largo_seguidos = np.where(largo_ahora, self.largo_seguidos + 1, 0)
        crecer = self.largo_seguidos >= CUADROS_PARA_CRECER
        if crecer.any():
            self.largos[crecer] = medidos[crecer]
            self.largo_seguidos[crecer] = 0

        # Reconstruir desde la muñeca respetando direcciones, sin estirar huesos
        nuevos = puntos.copy()
        for j in range(1, 21):
            padre = PADRE[j]
            direccion = puntos[j] - puntos[padre]
            largo = float(np.linalg.norm(direccion))
            maximo = self.largos[j - 1] * CRECIMIENTO_MAXIMO
            if largo > maximo and largo > 1e-6:
                direccion = direccion * (maximo / largo)
            nuevos[j] = nuevos[padre] + direccion

        # Aprender despacio (solo con lo que no se pasó del largo conocido)
        sanos = medidos <= self.largos * CRECIMIENTO_MAXIMO
        self.largos[sanos] += APRENDIZAJE_LARGOS * (medidos[sanos] - self.largos[sanos])
        return nuevos, True


def doblez_posible(mano, codo, lado):
    """
    False si la mano está doblada respecto al antebrazo más de lo que una
    muñeca real puede (detección equivocada). lado: 0 izquierda, 1 derecha.
    """
    antebrazo = mano[MANO_MUNECA] - codo
    palma = mano[MANO_BASE_MEDIO] - mano[MANO_MUNECA]
    if np.linalg.norm(antebrazo) < 1e-6 or np.linalg.norm(palma) < 1e-6:
        return True

    angulo = np.degrees(np.arctan2(
        antebrazo[0] * palma[1] - antebrazo[1] * palma[0],
        float(np.dot(antebrazo, palma))
    ))
    # En la imagen (sin espejo) el ángulo negativo va hacia la derecha de la persona
    hacia_su_derecha = -angulo
    if INVERTIR_DOBLEZ:
        hacia_su_derecha = -hacia_su_derecha

    if lado == 1:   # mano derecha
        limite_der, limite_izq = DOBLEZ_HACIA_AFUERA, DOBLEZ_HACIA_ADENTRO
    else:           # mano izquierda
        limite_der, limite_izq = DOBLEZ_HACIA_ADENTRO, DOBLEZ_HACIA_AFUERA

    return -limite_izq <= hacia_su_derecha <= limite_der


def _pose_pixel(pose, indice, ancho, alto, minimo=0.5):
    if len(pose) <= indice or getattr(pose[indice], "visibility", 1.0) < minimo:
        return None
    return np.array([pose[indice].x * ancho, pose[indice].y * alto])


class ManosEnVivo:
    """
    Para la cámara: los 21 puntos suavizados de cada mano, en píxeles.

    - Cada mano se asigna a SU antebrazo (nunca al del otro lado).
    - Si una detección salta lejísimos de golpe (error de MediaPipe), se ignora
      unos frames: la mano se queda QUIETA en el promedio de sus dos últimas
      posiciones estables.
    - Si la mano se pierde, también se queda quieta ahí (hasta MANO_QUIETA_SEGUNDOS).
    - Al volver a encontrarla, pasa SUAVE de la posición quieta a la nueva.
    self.fantasma dice qué mano se está mostrando quieta (estimada).
    """

    def __init__(self):
        self.filtros = [FiltroMano(), FiltroMano()]
        self.anterior = None
        self.fantasma = [False, False]
        self.estables = [[], []]          # dos últimas posiciones estables
        self.ultima_vista = [None, None]
        self.rechazos = [0, 0]
        self.perdida = [False, False]
        self.transicion = [None, None]    # (inicio, puntos de partida)
        self.vista_antes = [False, False] # ¿el frame anterior tuvo detección real?
        self.pegada = [None, None]        # forma de la mano respecto a su antebrazo
        self.mostrada = [None, None]      # lo último que se dibujó
        self.largos = [LargosMano(), LargosMano()]
        self.brazos = [FiltroAntebrazo(), FiltroAntebrazo()]

    def _brazo(self, pose, lado, ancho, alto, dt):
        """(muñeca del brazo, ángulo del antebrazo), suavizados, o None."""
        codo_i, muneca_i = (
            (POSE_CODO_IZQ, POSE_MUNECA_IZQ) if lado == 0 else (POSE_CODO_DER, POSE_MUNECA_DER)
        )
        codo = _pose_pixel(pose, codo_i, ancho, alto)
        muneca = _pose_pixel(pose, muneca_i, ancho, alto)
        if codo is None or muneca is None:
            self.brazos[lado].reiniciar()
            return None
        muneca, codo = self.brazos[lado](muneca, codo, dt)
        d = muneca - codo
        if np.linalg.norm(d) < 1e-6:
            return None
        return muneca, float(np.arctan2(d[1], d[0]))

    def _ancla(self, lado):
        if not self.estables[lado]:
            return None
        return np.mean(self.estables[lado], axis=0)

    def actualizar(self, deteccion, ancho, alto, ahora):
        dt = 1.0 / 30.0 if self.anterior is None else max(ahora - self.anterior, 1e-3)
        self.anterior = ahora

        # Detecciones crudas, ya asignadas a su brazo
        crudas = []
        for puntos in (deteccion.left_hand_landmarks, deteccion.right_hand_landmarks):
            if len(puntos) > 0:
                crudas.append((np.array([[p.x * ancho, p.y * alto] for p in puntos]), True))
            else:
                crudas.append((np.zeros((CANTIDAD_MANO, 2)), False))

        pose = deteccion.pose_landmarks
        mi, oi, md, od = asignar_manos(
            crudas[0][0], crudas[0][1], crudas[1][0], crudas[1][1],
            _pose_pixel(pose, POSE_MUNECA_IZQ, ancho, alto),
            _pose_pixel(pose, POSE_MUNECA_DER, ancho, alto)
        )

        resultado = []
        for lado, (pixeles, visto) in enumerate(((mi, oi), (md, od))):
            filtro = self.filtros[lado]
            ancla = self._ancla(lado)
            brazo = self._brazo(pose, lado, ancho, alto, dt)

            # Muñeca doblada más de lo posible = detección equivocada
            if visto:
                codo = _pose_pixel(pose, POSE_CODO_IZQ if lado == 0 else POSE_CODO_DER, ancho, alto)
                if codo is not None and not doblez_posible(pixeles, codo, lado):
                    visto = False

            # Tamaño de la mano y de los dedos: no se agranda raro
            if visto:
                pixeles, buena = self.largos[lado].corregir(pixeles)
                if not buena:
                    visto = False

            # ¿Salto imposible? Solo si la mano se veía en el frame anterior.
            # Si estaba perdida y reaparece en otro lugar, se acepta de inmediato.
            detectada = visto
            if visto and ancla is not None and self.vista_antes[lado]:
                tamano = max(float(np.linalg.norm(ancla[MANO_BASE_MEDIO] - ancla[MANO_MUNECA])), 1.0)
                salto = float(np.linalg.norm(pixeles[MANO_MUNECA] - ancla[MANO_MUNECA]))
                if salto > SALTO_MANO_VIVO * tamano and self.rechazos[lado] < RECHAZOS_MAXIMOS:
                    self.rechazos[lado] += 1
                    visto = False

            if not detectada:
                self.vista_antes[lado] = False    # perdida de verdad (no hubo detección)

            if visto:
                if not self.vista_antes[lado]:
                    filtro.reiniciar()            # reaparece: sin arrastrar la posición vieja
                    self.estables[lado] = []
                self.vista_antes[lado] = True
                self.rechazos[lado] = 0
                suave = filtro(pixeles, True, dt)

                if self.perdida[lado] and self.mostrada[lado] is not None:
                    self.transicion[lado] = (ahora, self.mostrada[lado].copy())
                self.perdida[lado] = False

                salida = suave
                if self.transicion[lado] is not None:
                    inicio, desde = self.transicion[lado]
                    k = min(1.0, (ahora - inicio) / TRANSICION_MANO)
                    k = k * k * (3 - 2 * k)               # entrada y salida suaves
                    salida = desde + (suave - desde) * k
                    if k >= 1.0:
                        self.transicion[lado] = None

                self.estables[lado] = (self.estables[lado] + [suave.copy()])[-2:]
                self.ultima_vista[lado] = ahora
                self.fantasma[lado] = False

                # Cómo va pegada la mano a su antebrazo (para cuando se pierda)
                if brazo is not None:
                    base = np.mean(self.estables[lado], axis=0)
                    self.pegada[lado] = {
                        "forma": base - base[MANO_MUNECA],
                        "desfase": base[MANO_MUNECA] - brazo[0],
                        "angulo": brazo[1]
                    }

                self.mostrada[lado] = salida
                resultado.append(salida)
                continue

            # ---- Perdida, pero se ve el antebrazo: la mano va PEGADA a él ----
            if brazo is not None and self.pegada[lado] is not None:
                pegada = self.pegada[lado]
                giro = brazo[1] - pegada["angulo"]
                mano = brazo[0] + _girar(pegada["desfase"], giro) + _girar(pegada["forma"], giro)
                self.perdida[lado] = True
                self.fantasma[lado] = True
                self.mostrada[lado] = mano
                resultado.append(mano)
                continue

            # ---- Sin mano ni antebrazo: quieta un momento en sus dos últimas posiciones ----
            if ancla is not None and ahora - self.ultima_vista[lado] <= MANO_QUIETA_SEGUNDOS:
                self.perdida[lado] = True
                self.fantasma[lado] = True
                self.mostrada[lado] = ancla
                resultado.append(ancla)
                continue

            filtro.reiniciar()
            self.estables[lado] = []
            self.pegada[lado] = None
            self.mostrada[lado] = None
            self.perdida[lado] = False
            self.transicion[lado] = None
            self.fantasma[lado] = False
            resultado.append(None)

        return resultado


# Giro del antebrazo: si sale al revés (negro con la palma hacia la cámara),
# cambia esto a True.
INVERTIR_GIRO = False

HUESO_ALFA = 0.45            # transparencia del hueso (0 = invisible, 1 = sólido)
HUESO_GROSOR = 0.16          # grosor del hueso, en fracción del largo del antebrazo
COLOR_HUESO_CLARO = (255, 255, 255)
COLOR_HUESO_OSCURO = (40, 40, 40)   # "ahumado" cuando está girado 180°

MANO_ALFA = 0.45             # transparencia del relleno de la mano (igual que el hueso)
DEDOS = [
    (0, 1, 2, 3, 4),         # pulgar
    (5, 6, 7, 8),            # índice
    (9, 10, 11, 12),         # medio
    (13, 14, 15, 16),        # anular
    (17, 18, 19, 20),        # meñique
]
PALMA = (0, 1, 5, 9, 13, 17)
CONEXIONES_DEDOS_DIBUJO = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]


def giro_de_mano(puntos, lado):
    """
    Cuánto está girada la mano (y el antebrazo), con los 21 puntos:
    0 = palma hacia la cámara ... 1 = dorso hacia la cámara (180°).
    lado: 0 = mano izquierda, 1 = mano derecha.
    """
    a = puntos[5] - puntos[0]      # muñeca -> base del índice
    b = puntos[17] - puntos[0]     # muñeca -> base del meñique
    normas = np.linalg.norm(a) * np.linalg.norm(b)
    if normas < 1e-6:
        return None

    cruz = (a[0] * b[1] - a[1] * b[0]) / normas      # -1 .. 1
    signo = 1.0 if lado == 0 else -1.0
    if INVERTIR_GIRO:
        signo = -signo

    return float(np.clip((1.0 - cruz * signo) / 2.0, 0.0, 1.0))


class AntebrazosEnVivo:
    """
    Para la cámara: muñeca y codo de cada antebrazo (suavizados) y su giro.
    """

    def __init__(self):
        self.filtros = [FiltroAntebrazo(), FiltroAntebrazo()]
        self.giro = [0.0, 0.0]
        self.anterior = None

    def actualizar(self, deteccion, ancho, alto, ahora, manos=None):
        """Devuelve [(muñeca, codo) o None, (muñeca, codo) o None]."""
        dt = 1.0 / 30.0 if self.anterior is None else max(ahora - self.anterior, 1e-3)
        self.anterior = ahora

        resultado = []
        pose = deteccion.pose_landmarks

        for lado, filtro in enumerate(self.filtros):
            # Giro: sale de la mano; si no se ve, se queda el último
            if manos is not None and manos[lado] is not None:
                giro = giro_de_mano(manos[lado], lado)
                if giro is not None:
                    # El color del giro cambia despacio (nada de saltos)
                    cambio = np.clip(giro - self.giro[lado], -GIRO_MAXIMO / 180.0 * dt, GIRO_MAXIMO / 180.0 * dt)
                    self.giro[lado] = 0.85 * self.giro[lado] + 0.15 * (self.giro[lado] + cambio)

            codo_i, muneca_i = (
                (POSE_CODO_IZQ, POSE_MUNECA_IZQ) if lado == 0 else (POSE_CODO_DER, POSE_MUNECA_DER)
            )
            if len(pose) <= POSE_MUNECA_DER or min(
                getattr(pose[codo_i], "visibility", 1.0),
                getattr(pose[muneca_i], "visibility", 1.0)
            ) < 0.5:
                filtro.reiniciar()
                resultado.append(None)
                continue

            extremos = np.array([
                [pose[muneca_i].x * ancho, pose[muneca_i].y * alto],
                [pose[codo_i].x * ancho, pose[codo_i].y * alto]
            ])

            # Si se ve la mano, el hueso empieza en la muñeca de la MANO (más precisa)
            if manos is not None and manos[lado] is not None:
                extremos[0] = manos[lado][MANO_MUNECA]

            muneca, codo = filtro(extremos[0], extremos[1], dt)
            resultado.append((muneca, codo))

        return resultado


def color_por_giro(giro):
    """Blanco (palma hacia la cámara) -> gris ahumado (girado 180°)."""
    return tuple(
        int(round(cl + (os - cl) * giro))
        for cl, os in zip(COLOR_HUESO_CLARO, COLOR_HUESO_OSCURO)
    )


def _mezclar(pantalla, capa, mascara, alfa):
    """Pone 'capa' encima de 'pantalla' con transparencia, solo donde mascara > 0."""
    zona = mascara > 0
    if zona.any():
        pantalla[zona] = (
            pantalla[zona].astype(np.float32) * (1.0 - alfa)
            + capa[zona].astype(np.float32) * alfa
        ).astype(np.uint8)


def dibujar_antebrazos(pantalla, antebrazos, ancho, colores, giros=(0.0, 0.0)):
    """
    Hueso del antebrazo (en espejo): grueso, semitransparente, de la muñeca al codo.
    Color: blanco (palma hacia la cámara) -> gris ahumado (girado 180°).
    Puntos: uno al inicio (muñeca); en la cabeza del hueso (codo) uno en el centro
    y dos a los lados, que se cruzan cuando el antebrazo gira.
    El DIBUJO 2 sale del punto central de la cabeza (el codo).
    """
    for datos, color, giro in zip(antebrazos, colores, giros):
        if datos is None:
            continue

        muneca, codo = datos
        m = np.array([ancho - 1 - muneca[0], muneca[1]])
        c = np.array([ancho - 1 - codo[0], codo[1]])

        largo = float(np.linalg.norm(c - m))
        if largo < 5:
            continue

        grosor = max(8, int(round(largo * HUESO_GROSOR)))
        radio_cabeza = int(round(grosor * 0.75))
        radio_pie = int(round(grosor * 0.55))

        color_hueso = color_por_giro(giro)

        # ---- Hueso semitransparente (solo en el recuadro del hueso: rápido) ----
        alto_p, ancho_p = pantalla.shape[:2]
        borde = radio_cabeza + 3
        x0 = int(max(0, min(m[0], c[0]) - borde))
        y0 = int(max(0, min(m[1], c[1]) - borde))
        x1 = int(min(ancho_p, max(m[0], c[0]) + borde))
        y1 = int(min(alto_p, max(m[1], c[1]) + borde))

        if x1 - x0 > 2 and y1 - y0 > 2:
            mascara = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            pm = tuple(np.round(m - [x0, y0]).astype(int))
            pc = tuple(np.round(c - [x0, y0]).astype(int))
            cv2.line(mascara, pm, pc, 255, grosor, cv2.LINE_AA)
            cv2.circle(mascara, pm, radio_pie, 255, -1, cv2.LINE_AA)
            cv2.circle(mascara, pc, radio_cabeza, 255, -1, cv2.LINE_AA)

            zona = pantalla[y0:y1, x0:x1]
            capa = np.empty_like(zona)
            capa[:] = color_hueso
            _mezclar(zona, capa, mascara, HUESO_ALFA)

        # ---- Puntos de seguimiento ----
        direccion = (c - m) / largo
        perpendicular = np.array([-direccion[1], direccion[0]])

        # Los dos puntos laterales se acercan al girar 90° y se cruzan a 180°
        apertura = np.cos(giro * np.pi)
        lado_a = c + perpendicular * radio_cabeza * apertura
        lado_b = c - perpendicular * radio_cabeza * apertura

        claro = color_antebrazo(color)
        for punto, radio, relleno in (
            (m, 5, claro),            # inicio del hueso (muñeca)
            (lado_a, 4, (255, 255, 255)),
            (lado_b, 4, (60, 60, 60)),
            (c, 6, claro),            # centro de la cabeza (codo) -> DIBUJO 2
        ):
            centro = tuple(np.round(punto).astype(int))
            cv2.circle(pantalla, centro, radio, relleno, -1, cv2.LINE_AA)
            cv2.circle(pantalla, centro, radio, color, 1, cv2.LINE_AA)


def dibujar_siluetas_manos(pantalla, manos, ancho, giros=(0.0, 0.0), fantasmas=(False, False)):
    """
    Dibuja la forma de cada mano (palma + 5 dedos) con un trazo negro delgado
    y un relleno semitransparente que cambia de color con el giro, igual que
    el hueso del antebrazo: blanco (palma) -> gris ahumado (dorso).
    """
    alto_p, ancho_p = pantalla.shape[:2]

    for puntos, giro, fantasma in zip(manos, giros, fantasmas):
        if puntos is None:
            continue

        p = np.stack([ancho - 1 - puntos[:, 0], puntos[:, 1]], axis=1)
        tamano = float(np.linalg.norm(p[MANO_BASE_MEDIO] - p[MANO_MUNECA]))
        if tamano < 5:
            continue

        grosor_dedo = max(4, int(round(tamano * 0.28)))
        margen = grosor_dedo + 4

        x0 = int(max(0, np.floor(p[:, 0].min()) - margen))
        y0 = int(max(0, np.floor(p[:, 1].min()) - margen))
        x1 = int(min(ancho_p, np.ceil(p[:, 0].max()) + margen))
        y1 = int(min(alto_p, np.ceil(p[:, 1].max()) + margen))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue

        local = np.round(p - [x0, y0]).astype(np.int32)
        mascara = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)

        cv2.fillConvexPoly(mascara, cv2.convexHull(local[list(PALMA)]), 255)
        for dedo in DEDOS:
            for a, b in zip(dedo[:-1], dedo[1:]):
                cv2.line(mascara, tuple(local[a]), tuple(local[b]), 255, grosor_dedo, cv2.LINE_AA)
            cv2.circle(mascara, tuple(local[dedo[-1]]), grosor_dedo // 2, 255, -1, cv2.LINE_AA)

        zona = pantalla[y0:y1, x0:x1]
        capa = np.empty_like(zona)
        capa[:] = color_por_giro(giro)
        _mezclar(zona, capa, mascara, MANO_ALFA)

        contornos, _ = cv2.findContours(
            (mascara > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # contorno que sigue la mano y cada dedo: negro si se ve la mano;
        # blanco brillante (con resplandor) si se perdió el seguimiento y la mano es estimada
        if fantasma:
            brillo = np.zeros_like(zona)
            cv2.drawContours(brillo, contornos, -1, (255, 255, 255), 6, cv2.LINE_AA)
            brillo = cv2.GaussianBlur(brillo, (0, 0), 3)
            np.copyto(zona, np.maximum(zona, brillo))
            cv2.drawContours(zona, contornos, -1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.drawContours(zona, contornos, -1, (0, 0, 0), 2, cv2.LINE_AA)


def completar_manos_tapadas(puntos_mano, mano_ok, munecas, munecas_ok, fps):
    """
    Para el modelo: en los frames donde la mano no se vio pero su muñeca sí se
    sabe (por el brazo), pone la ÚLTIMA forma conocida de la mano en esa
    muñeca, hasta MANO_FANTASMA_SEGUNDOS. Devuelve (puntos_mano, mano_ok) nuevos.
    """
    puntos_mano = np.array(puntos_mano, dtype=np.float64)
    mano_ok = np.array(mano_ok, dtype=bool).reshape(-1)

    if puntos_mano.ndim != 3 or len(puntos_mano) != len(mano_ok) or not mano_ok.any():
        return puntos_mano, mano_ok

    limite = max(1, int(round(MANO_FANTASMA_SEGUNDOS * fps)))
    salida_ok = mano_ok.copy()
    ultima_forma, ultimo_i = None, None

    for i in range(len(mano_ok)):
        if mano_ok[i]:
            ultima_forma = puntos_mano[i] - puntos_mano[i, MANO_MUNECA]
            ultimo_i = i
        elif (
            ultima_forma is not None
            and munecas_ok[i]
            and i - ultimo_i <= limite
        ):
            puntos_mano[i] = munecas[i] + ultima_forma
            salida_ok[i] = True

    return puntos_mano, salida_ok


# =============================================================================
# CARA EN PANTALLA (solo visual: el modelo no usa la cara)
# =============================================================================

COLOR_DORADO = (0, 215, 255)     # BGR de #FFD700
PUNTOS_POR_ARISTA = 1            # puntos extra en cada línea de la malla (más = más denso)
PUNTOS_DETALLE = 6               # puntos extra en cada línea de la BOCA y las CEJAS
PUNTOS_NARIZ = 1                 # puntos extra en el contorno de la NARIZ (disimulado)
PUNTOS_OREJA = 26                # puntos de cada OREJA (dibujada, MediaPipe no la detecta)
CARA_CORTE_MINIMO = 0.3          # más bajo = cara más quieta
CARA_BETA = 0.3

CARA_NARIZ = 1
CARA_OJO_A = 33
CARA_OJO_B = 263

_aristas_cara = None
_aristas_detalle = None
_aristas_nariz = None


def _aristas_malla():
    """Pares (a, b) de la malla de la cara de MediaPipe, sin repetir."""
    global _aristas_cara

    if _aristas_cara is None:
        try:
            from mediapipe.tasks.python import vision
            conexiones = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
            pares = {tuple(sorted((c.start, c.end))) for c in conexiones}
            _aristas_cara = np.array(sorted(pares), dtype=np.int64)
        except Exception:
            _aristas_cara = np.zeros((0, 2), dtype=np.int64)

    return _aristas_cara


def _aristas_boca_cejas():
    """Líneas del contorno de los labios y de las dos cejas."""
    global _aristas_detalle

    if _aristas_detalle is None:
        try:
            from mediapipe.tasks.python import vision
            c = vision.FaceLandmarksConnections
            conexiones = (
                list(c.FACE_LANDMARKS_LIPS)
                + list(c.FACE_LANDMARKS_LEFT_EYEBROW)
                + list(c.FACE_LANDMARKS_RIGHT_EYEBROW)
            )
            pares = {tuple(sorted((x.start, x.end))) for x in conexiones}
            _aristas_detalle = np.array(sorted(pares), dtype=np.int64)
        except Exception:
            _aristas_detalle = np.zeros((0, 2), dtype=np.int64)

    return _aristas_detalle


def _aristas_de_nariz():
    global _aristas_nariz

    if _aristas_nariz is None:
        try:
            from mediapipe.tasks.python import vision
            conexiones = vision.FaceLandmarksConnections.FACE_LANDMARKS_NOSE
            pares = {tuple(sorted((x.start, x.end))) for x in conexiones}
            _aristas_nariz = np.array(sorted(pares), dtype=np.int64)
        except Exception:
            _aristas_nariz = np.zeros((0, 2), dtype=np.int64)

    return _aristas_nariz


# Puntos de la cara que usa la oreja
CARA_FRENTE = 10
CARA_MENTON = 152
CARA_LADO_A = 234      # borde de la cara de un lado (a la altura de la oreja)
CARA_LADO_B = 454      # borde del otro lado


def puntos_orejas(puntos):
    """
    MediaPipe no detecta las orejas: se DIBUJAN al lado de la cara, con la
    forma de una "C" (borde de la oreja) y un pliegue interior, a la altura
    de los ojos-nariz. Si la cabeza gira, la oreja del lado que se esconde
    se angosta y desaparece.
    """
    if len(puntos) <= max(CARA_LADO_B, CARA_MENTON):
        return []

    arriba = puntos[CARA_FRENTE] - puntos[CARA_MENTON]
    alto_cara = float(np.linalg.norm(arriba))
    if alto_cara < 1e-6:
        return []
    arriba = arriba / alto_cara

    medio = (puntos[CARA_LADO_A] + puntos[CARA_LADO_B]) / 2.0
    nariz = puntos[CARA_NARIZ]
    d_a = float(np.linalg.norm(nariz - puntos[CARA_LADO_A]))
    d_b = float(np.linalg.norm(nariz - puntos[CARA_LADO_B]))

    resultado = []
    for lado, d_lado in ((CARA_LADO_A, d_a), (CARA_LADO_B, d_b)):
        visible = d_lado / max(d_a + d_b, 1e-6)          # 0.5 = de frente
        if visible < 0.28:
            continue                                      # oreja escondida
        apertura = min(1.0, visible / 0.5)

        afuera = puntos[lado] - medio
        norma = float(np.linalg.norm(afuera))
        if norma < 1e-6:
            continue
        afuera = afuera / norma

        alto_oreja = alto_cara * 0.30
        ancho_oreja = alto_oreja * 0.42 * apertura
        centro = puntos[lado] + afuera * ancho_oreja * 0.45 + arriba * alto_oreja * 0.10

        # Borde exterior: "C" abierta hacia la cara
        t = np.linspace(-np.pi / 2 - 0.35, np.pi / 2 + 0.35, PUNTOS_OREJA)[:, None]
        borde = centro + afuera * np.cos(t) * ancho_oreja / 2 + arriba * np.sin(t) * alto_oreja / 2

        # Pliegue interior, más pequeño
        t2 = np.linspace(-np.pi / 2 + 0.2, np.pi / 2 - 0.1, max(6, PUNTOS_OREJA // 2))[:, None]
        interior = (
            centro + afuera * np.cos(t2) * ancho_oreja * 0.22
            + arriba * (np.sin(t2) * alto_oreja * 0.28 + alto_oreja * 0.04)
        )
        resultado += [borde, interior]

    return resultado


def _puntos_en_aristas(puntos, aristas, cantidad):
    if len(aristas) == 0 or cantidad <= 0 or len(puntos) <= aristas.max():
        return []
    a = puntos[aristas[:, 0]]
    b = puntos[aristas[:, 1]]
    return [a + (b - a) * (k / (cantidad + 1)) for k in range(1, cantidad + 1)]


def puntos_densos_cara(puntos):
    """
    Los 478 puntos de la cara + puntos intermedios sobre cada línea de la malla
    + muchos más sobre el contorno de la BOCA y de las CEJAS.
    """
    extras = (
        _puntos_en_aristas(puntos, _aristas_malla(), PUNTOS_POR_ARISTA)
        + _puntos_en_aristas(puntos, _aristas_boca_cejas(), PUNTOS_DETALLE)
        + _puntos_en_aristas(puntos, _aristas_de_nariz(), PUNTOS_NARIZ)
        + puntos_orejas(puntos)
    )
    if not extras:
        return puntos
    return np.concatenate([puntos] + extras, axis=0)


class FiltroCara:
    """Suaviza la cara igual que la mano: posición (nariz) + forma por separado."""

    def __init__(self):
        self.posicion = FiltroUnoEuro(CARA_CORTE_MINIMO, CARA_BETA)
        self.forma = FiltroUnoEuro(CARA_CORTE_MINIMO, CARA_BETA)
        self.anterior = None

    def actualizar(self, deteccion, ancho, alto, ahora):
        """Devuelve los puntos de la cara suavizados (en píxeles) o None."""
        dt = 1.0 / 30.0 if self.anterior is None else max(ahora - self.anterior, 1e-3)
        self.anterior = ahora

        cara = deteccion.face_landmarks
        if len(cara) <= CARA_OJO_B:
            self.posicion.reiniciar()
            self.forma.reiniciar()
            return None

        puntos = np.array([[p.x * ancho, p.y * alto] for p in cara])
        escala = float(np.linalg.norm(puntos[CARA_OJO_A] - puntos[CARA_OJO_B]))

        nariz = self.posicion(puntos[CARA_NARIZ], dt, escala)
        forma = self.forma(puntos - puntos[CARA_NARIZ], dt, escala)

        return forma + nariz


def nivel_boca(apertura, cerrada, abierta):
    """0 = boca cerrada ... 1 = abierta lo suficiente para disparar."""
    if apertura is None:
        return 0.0
    return float(np.clip((apertura - cerrada) / max(abierta - cerrada, 1e-6), 0.0, 1.0))


def dibujar_cara_dorada(pantalla, puntos, ancho, nivel=0.0):
    """
    Dibuja la cara (en espejo) con puntos densos.
    nivel 0 = dorado ... nivel 1 = blanco (según cuánto se abre la boca).
    """
    if puntos is None:
        return

    color = tuple(
        int(round(d + (255 - d) * nivel)) for d in COLOR_DORADO
    )

    densos = puntos_densos_cara(puntos)
    alto_p, ancho_p = pantalla.shape[:2]

    x = np.round(ancho - 1 - densos[:, 0]).astype(np.int64)
    y = np.round(densos[:, 1]).astype(np.int64)

    # Puntos de 2x2 píxeles, dibujados de una sola vez (rápido)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = x + dx, y + dy
        dentro = (xx >= 0) & (xx < ancho_p) & (yy >= 0) & (yy < alto_p)
        pantalla[yy[dentro], xx[dentro]] = color


def destello_desde_cara(pantalla, puntos, ancho, progreso):
    """
    Destello que NACE de la cara: un brillo blanco que sale de la cara
    y se expande mientras se apaga. progreso: 0 (inicio) -> 1 (fin).
    Si no se ve la cara, destello suave en toda la pantalla.
    """
    progreso = float(np.clip(progreso, 0.0, 1.0))
    intensidad = (1.0 - progreso) ** 1.5

    if puntos is None or len(puntos) < 3:
        blanco = np.full_like(pantalla, 255)
        return cv2.addWeighted(pantalla, 1.0 - 0.5 * intensidad, blanco, 0.5 * intensidad, 0)

    alto_p, ancho_p = pantalla.shape[:2]
    escala = 4
    pequena = np.zeros((alto_p // escala, ancho_p // escala), np.float32)

    x = (ancho - 1 - puntos[:, 0]) / escala
    y = puntos[:, 1] / escala
    casco = cv2.convexHull(np.stack([x, y], axis=1).astype(np.int32))
    cv2.fillConvexPoly(pequena, casco, 1.0)

    tamano = max(float(np.ptp(x)), float(np.ptp(y)), 4.0)
    sigma = tamano * (0.2 + 1.3 * progreso)
    halo = cv2.GaussianBlur(pequena, (0, 0), sigma)
    halo = np.maximum(halo / max(float(halo.max()), 1e-6), pequena)

    alfa = np.clip(halo * 1.3, 0.0, 1.0) * intensidad
    alfa = cv2.resize(alfa, (ancho_p, alto_p), interpolation=cv2.INTER_LINEAR)[:, :, None]

    return (pantalla * (1.0 - alfa) + 255.0 * alfa).astype(np.uint8)


def _quitar_saltos(puntos, valida, ancho_hombros, fps):
    """Marca como no válidos los puntos que saltan imposiblemente lejos."""
    salto = SALTO_MAXIMO * ancho_hombros * max(1.0, 30.0 / fps)
    aceptar_tras = max(2, int(round(0.2 * fps)))   # tras 0.2 s rechazando, se acepta

    ultimo = None
    rechazados = 0

    for i in range(len(puntos)):
        if not valida[i]:
            continue

        if ultimo is not None:
            pasos = i - ultimo
            distancia = np.linalg.norm(puntos[i] - puntos[ultimo])

            if distancia > salto * pasos and rechazados < aceptar_tras:
                valida[i] = False
                rechazados += 1
                continue

        ultimo = i
        rechazados = 0


def _direccion(puntos, valida, i, hacia, cantidad):
    """
    Dirección del movimiento en el punto i, promediada sobre 'cantidad'
    frames válidos hacia atrás (hacia=-1) o hacia adelante (hacia=+1).
    Devuelve un vector unitario apuntando en el sentido del movimiento, o None.
    """
    vecinos = []
    j = i + hacia
    while 0 <= j < len(puntos) and len(vecinos) < cantidad:
        if valida[j]:
            vecinos.append(j)
        j += hacia

    if not vecinos:
        return None

    lejano = vecinos[-1]
    vector = (puntos[i] - puntos[lejano]) if hacia < 0 else (puntos[lejano] - puntos[i])
    norma = np.linalg.norm(vector)

    return vector / norma if norma > 1e-6 else None


def _limitar_angulo(direccion, cuerda_unitaria, maximo_grados):
    """Si la dirección se aleja demasiado de la cuerda, la gira hacia ella."""
    if direccion is None:
        return cuerda_unitaria

    coseno = float(np.clip(np.dot(direccion, cuerda_unitaria), -1.0, 1.0))
    if coseno >= np.cos(np.radians(maximo_grados)):
        return direccion

    # Gira la cuerda 'maximo_grados' hacia el lado al que apuntaba la dirección
    lado = np.sign(cuerda_unitaria[0] * direccion[1] - cuerda_unitaria[1] * direccion[0]) or 1.0
    angulo = np.radians(maximo_grados) * lado
    c, s = np.cos(angulo), np.sin(angulo)

    return np.array([
        c * cuerda_unitaria[0] - s * cuerda_unitaria[1],
        s * cuerda_unitaria[0] + c * cuerda_unitaria[1]
    ])


def _rellenar_huecos(puntos, valida, fps):
    """
    Une los huecos cortos (mano perdida hasta SEGUNDOS_HUECO):
    - Sale en la dirección (y curvatura: subiendo o bajando) que traía la mano
      y llega en la dirección en que la mano sigue después.
    - Usa la curva más suave posible (Bézier con brazos de 1/3 de la distancia)
      y nunca se desvía más de ANGULO_MAXIMO_HUECO de la línea recta.
    - Los frames del hueco se reparten a VELOCIDAD CONSTANTE sobre la curva,
      para que no "acelere" al final para alcanzar el trazo.
    """
    max_hueco = max(1, int(round(SEGUNDOS_HUECO * fps)))
    cantidad = max(2, int(round(0.15 * fps)))   # ~0.15 s para medir la dirección
    indices = np.where(valida)[0]

    for a, b in zip(indices[:-1], indices[1:]):
        largo = b - a
        if largo < 2 or largo > max_hueco + 1:
            continue

        p0, p3 = puntos[a].copy(), puntos[b].copy()
        cuerda = p3 - p0
        distancia = np.linalg.norm(cuerda)

        if distancia < 1e-6:
            puntos[a + 1:b] = p0
            valida[a + 1:b] = True
            continue

        unitaria = cuerda / distancia

        salida = _limitar_angulo(
            _direccion(puntos, valida, a, -1, cantidad), unitaria, ANGULO_MAXIMO_HUECO
        )
        llegada = _limitar_angulo(
            _direccion(puntos, valida, b, +1, cantidad), unitaria, ANGULO_MAXIMO_HUECO
        )

        brazo = distancia / 3.0
        p1 = p0 + salida * brazo
        p2 = p3 - llegada * brazo

        # Curva muy densa y luego puntos a distancias iguales (velocidad constante)
        t = np.linspace(0.0, 1.0, 200)[:, None]
        curva = (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3
        )
        segmentos = np.linalg.norm(np.diff(curva, axis=0), axis=1)
        acumulado = np.concatenate([[0.0], np.cumsum(segmentos)])
        objetivos = np.linspace(0.0, acumulado[-1], largo + 1)[1:-1]

        puntos[a + 1:b, 0] = np.interp(objetivos, acumulado, curva[:, 0])
        puntos[a + 1:b, 1] = np.interp(objetivos, acumulado, curva[:, 1])
        valida[a + 1:b] = True


def _suavizar_tramos(puntos, valida, fps):
    """Promedio móvil centrado, solo dentro de cada tramo continuo válido."""
    ventana = max(3, int(round(SEGUNDOS_SUAVIZADO * fps)) | 1)
    mitad = ventana // 2
    kernel = np.ones(ventana) / ventana

    i = 0
    n = len(puntos)
    while i < n:
        if not valida[i]:
            i += 1
            continue

        j = i
        while j < n and valida[j]:
            j += 1

        tramo = puntos[i:j]
        if len(tramo) >= 3:
            relleno = np.pad(tramo, ((mitad, mitad), (0, 0)), mode="edge")
            puntos[i:j, 0] = np.convolve(relleno[:, 0], kernel, mode="valid")
            puntos[i:j, 1] = np.convolve(relleno[:, 1], kernel, mode="valid")

        i = j


def _usar_mano(puntos, valida, puntos_mano, mano_ok, fps):
    """
    Dos fuentes para la muñeca:
    - MANO (21 puntos): la más precisa. Se usa siempre que se ve.
    - BRAZO (muñeca del cuerpo, que MediaPipe calcula siguiendo el antebrazo):
      sigue funcionando aunque se pierdan los dedos. Pero está corrida unos
      píxeles respecto a la de la mano, así que se CORRIGE con la diferencia
      medida justo antes y justo después de perder la mano.
      Solo se usa en huecos cortos (hasta SEGUNDOS_HUECO) entre detecciones
      de la mano: así no inventa nada cuando la mano está quieta abajo.
    Lo que no se pueda cubrir con el brazo se une después con una curva suave.
    """
    if mano_ok is None:
        return

    mano_ok = np.array(mano_ok, dtype=bool).reshape(-1)
    n = len(mano_ok)
    if n != len(puntos):
        return

    tiene_mano = (
        puntos_mano is not None
        and np.asarray(puntos_mano).ndim == 3
        and len(puntos_mano) == n
    )

    if not tiene_mano:
        valida[:] = mano_ok
        return

    puntos_mano = np.asarray(puntos_mano, dtype=np.float64)
    brazo = puntos.copy()
    brazo_ok = valida.copy()

    puntos[mano_ok] = puntos_mano[mano_ok, MANO_MUNECA]
    nueva_valida = mano_ok.copy()

    # Diferencia mano - brazo en los frames donde se ven las dos
    ambos = mano_ok & brazo_ok
    max_hueco = max(1, int(round(SEGUNDOS_HUECO * fps)))
    borde = max(1, int(round(BRAZO_BORDE * fps)))
    indices_ambos = np.where(ambos)[0]

    if len(indices_ambos) > 0:
        diferencia = np.zeros((n, 2))
        diferencia[ambos] = puntos[ambos] - brazo[ambos]

        for i in np.where(~mano_ok & brazo_ok)[0]:
            antes = indices_ambos[indices_ambos < i]
            despues = indices_ambos[indices_ambos > i]
            a = antes[-1] if len(antes) else None
            b = despues[0] if len(despues) else None

            if a is not None and b is not None and b - a <= max_hueco + 1:
                # Hueco entre dos detecciones: corrección interpolada
                t = (i - a) / (b - a)
                correccion = (1 - t) * diferencia[a] + t * diferencia[b]
            elif a is not None and b is None and i - a <= borde:
                correccion = diferencia[a]          # justo después de perderla
            elif b is not None and a is None and b - i <= borde:
                correccion = diferencia[b]          # justo antes de encontrarla
            else:
                continue

            puntos[i] = brazo[i] + correccion
            nueva_valida[i] = True

    valida[:] = nueva_valida


def _pincel_perezoso(puntos, valida, ancho_hombros):
    """
    El trazo es un "pincel" atado a la mano con una cuerda de RADIO_PINCEL:
    - mientras la mano tiembla dentro de ese radio, el pincel NO se mueve;
    - cuando la mano se aleja, el pincel la sigue suave.
    Así desaparecen los garabatos con la mano casi quieta y los giros
    muy cerrados se vuelven curvas (radio mínimo de giro = RADIO_PINCEL).
    """
    radio = RADIO_PINCEL * ancho_hombros
    pincel = None

    for i in range(len(puntos)):
        if not valida[i]:
            pincel = None
            continue

        if pincel is None:
            pincel = puntos[i].copy()
        else:
            diferencia = puntos[i] - pincel
            distancia = np.linalg.norm(diferencia)
            if distancia > radio:
                pincel = pincel + diferencia * (distancia - radio) / distancia

        puntos[i] = pincel


# ---- Al RECONOCER (LSC y PRACTICAR): grabar hasta bajar las manos (en vez de 3.4 s fijos) ----
# Las muestras y los videos del entrenamiento NO cambian.
SEGUNDOS_MAXIMO_GRABACION = 5.0   # por si nunca se detecta que bajó las manos
SEGUNDOS_MINIMO_GRABACION = 0.8
SEGUNDOS_MANOS_ABAJO = 0.35       # tanto tiempo con las manos abajo = terminó la seña
CUADROS_MUESTRA = 99              # las muestras de entrenamiento tienen 99 cuadros (3.4 s)


class FinDeSena:
    """Decide cuándo termina la grabación: después de SUBIR alguna mano por encima de su codo, cuando
    las dos manos quedan abajo (muñeca por debajo del codo, o no se ven) un momento. Máximo 7 s."""

    def __init__(self, inicio):
        self.inicio, self.subio, self.abajo_desde = inicio, False, None

    def actualizar(self, datos, ahora):
        """True cuando la seña terminó."""
        t = ahora - self.inicio
        if t >= SEGUNDOS_MAXIMO_GRABACION:
            return True
        arriba, abajo = False, True
        if datos is not None:
            ancho = float(datos["ancho_hombros"])
            for lado in ("izq", "der"):
                if not (datos[f"valida_{lado}"] and datos[f"codo_{lado}_ok"]):
                    continue                              # no se ve: cuenta como abajo
                y_m, y_c = float(datos[f"muneca_{lado}"][1]), float(datos[f"codo_{lado}"][1])
                if y_m < y_c:
                    arriba = True
                if y_m < y_c + 0.10 * ancho:
                    abajo = False
        if arriba:
            self.subio = True
        if not (self.subio and abajo and t >= SEGUNDOS_MINIMO_GRABACION):
            self.abajo_desde = None
            return False
        if self.abajo_desde is None:
            self.abajo_desde = ahora
        return ahora - self.abajo_desde >= SEGUNDOS_MANOS_ABAJO


CLAVES_POR_CUADRO = ("muneca_izq", "muneca_der", "valida_izq", "valida_der", "mano_izq", "mano_der",
                     "mano_izq_ok", "mano_der_ok", "codo_izq", "codo_der", "codo_izq_ok", "codo_der_ok",
                     "cara", "cara_ok")


def normalizar_registro(registro, cuadros=CUADROS_MUESTRA):
    """La grabación (dure lo que dure) se lleva a 99 cuadros, como las muestras con que se entrenó
    (los videos también se ajustaron a 99 cuadros). Así los filtros calculan bien la velocidad.
    Se toma el cuadro más cercano (no se mezclan puntos vistos con no vistos)."""
    n = len(registro["muneca_izq"])
    if n == 0 or n == cuadros:
        return registro
    elegidos = np.clip(np.round(np.linspace(0, n - 1, cuadros)).astype(int), 0, n - 1)
    salida = dict(registro)
    for clave in CLAVES_POR_CUADRO:
        if clave in registro and len(registro[clave]) == n:
            valores = registro[clave]
            salida[clave] = [valores[i] for i in elegidos]
    return salida


MOVIMIENTO_MINIMO_REPOSO = 0.35  # anchos de hombro: una mano abajo que se mueve menos que esto está en reposo
_QUITAR_REPOSO = [True]           # si en una grabación NINGUNA mano queda, se vuelve a armar sin quitar la de reposo


def trazo_del_brazo(munecas, validas, ancho_hombros, fps):
    """
    v17: el recorrido que se usa como TRAZO de la seña: la muñeca del ESQUELETO (donde el antebrazo se une
    con la mano, puntos 15 y 16 del cuerpo), limpio y suavizado. No se mezcla con la muñeca de la mano:
    el cuerpo casi nunca se pierde, así el trazo no salta cuando la mano sale borrosa.
    """
    return limpiar_recorrido(munecas, validas, ancho_hombros, fps)


def limpiar_recorrido(puntos, valida, ancho_hombros, fps, puntos_mano=None, mano_ok=None):
    """
    Recorrido de una muñeca más estable:
    0. Usa la muñeca de la mano (21 puntos); si se pierden los dedos poco
       tiempo, sigue con el brazo corregido.
    1. Quita movimientos imposibles (más rápidos que una mano real).
    2. Si la mano se pierde hasta 1 s, une los dos extremos con una curva suave.
    3. "Pincel perezoso": ignora el temblor con la mano casi quieta y
       redondea los giros demasiado cerrados.
    4. Suaviza lo que quede.
    Devuelve copias nuevas (puntos, valida).
    """
    puntos = np.array(puntos, dtype=np.float64).reshape(-1, 2)
    valida = np.array(valida, dtype=bool).reshape(-1)
    fps = max(float(fps), 1.0)

    _usar_mano(puntos, valida, puntos_mano, mano_ok, fps)

    if len(puntos) == 0 or not valida.any() or ancho_hombros < 1e-6:
        return puntos, valida

    _quitar_saltos(puntos, valida, ancho_hombros, fps)
    _rellenar_huecos(puntos, valida, fps)
    _pincel_perezoso(puntos, valida, ancho_hombros)
    _suavizar_tramos(puntos, valida, fps)

    return puntos, valida


def punto_antebrazo(munecas, codos):
    """Punto del DIBUJO 2: con FRACCION_ANTEBRAZO = 1.0 es el codo."""
    munecas = np.asarray(munecas, dtype=np.float64)
    codos = np.asarray(codos, dtype=np.float64)
    return munecas + FRACCION_ANTEBRAZO * (codos - munecas)


def recorrido_antebrazo(munecas, munecas_ok, codos, codos_ok, ancho_hombros, fps):
    """Recorrido limpio del codo (DIBUJO 2)."""
    puntos = punto_antebrazo(munecas, codos).reshape(-1, 2)
    valida = (
        np.asarray(munecas_ok, dtype=bool).reshape(-1)
        & np.asarray(codos_ok, dtype=bool).reshape(-1)
    )
    return limpiar_recorrido(puntos, valida, ancho_hombros, fps)


def _dibujar_tramos(pantalla, puntos, valida, ancho, color, grosor):
    pantalla_x = ancho - 1 - puntos[:, 0]
    total = len(puntos)
    i = 0
    while i < total:
        if not valida[i]:
            i += 1
            continue
        j = i
        while j < total and valida[j]:
            j += 1
        linea = np.stack([pantalla_x[i:j], puntos[i:j, 1]], axis=1)
        cv2.polylines(pantalla, [np.round(linea).astype(np.int32)], False, color, grosor, cv2.LINE_AA)
        i = j


COLOR_TRAZO_ANTEBRAZO = (50, 175, 205)   # amarillo opaco (BGR)


def color_antebrazo(color):
    """Color de los puntos del antebrazo: el de la mano, más claro."""
    return tuple(int(c + (255 - c) * 0.55) for c in color)


def dibujar_recorrido_en_vivo(pantalla, registro, ancho, segundos, colores):
    """
    Dibuja en pantalla (en espejo) el recorrido ya limpio de cada muñeca,
    igual al que usa el modelo. Si la mano se perdió hace menos de 1 s,
    deja un círculo en su última posición ("en el aire").
    """
    if len(registro["ancho_hombros"]) == 0:
        return

    ancho_hombros = float(np.median(registro["ancho_hombros"]))
    total = len(registro["muneca_izq"])
    fps = total / max(segundos, 0.1)
    max_hueco = max(1, int(round(SEGUNDOS_HUECO * fps)))

    for lado, color in (("izq", colores[0]), ("der", colores[1])):
        # ---- DIBUJO 2: mitad del antebrazo (línea delgada, más clara) ----
        if f"codo_{lado}" in registro and len(registro[f"codo_{lado}"]) == total:
            p_brazo, v_brazo = recorrido_antebrazo(
                registro[f"muneca_{lado}"], registro[f"valida_{lado}"],
                registro[f"codo_{lado}"], registro[f"codo_{lado}_ok"],
                ancho_hombros, fps
            )
            if v_brazo.any():
                _dibujar_tramos(pantalla, p_brazo, v_brazo, ancho, COLOR_TRAZO_ANTEBRAZO, 3)

        # ---- DIBUJO 1: CENTRO de la mano (promedio de los 21 puntos) ----
        ok_m = filtrar_mano(registro[f"mano_{lado}"], registro[f"mano_{lado}_ok"], ancho_hombros, fps,
                            registro.get("cara"), registro.get("cara_ok"))
        mano_suave = suavizar_manos(registro[f"mano_{lado}"], ok_m, fps)
        puntos, valida = limpiar_recorrido(
            registro[f"muneca_{lado}"], registro[f"valida_{lado}"], ancho_hombros, fps,
            mano_suave, ok_m
        )
        mano_arr = np.asarray(mano_suave, dtype=np.float64)
        if mano_arr.ndim == 3 and len(mano_arr) == total and ok_m.any():
            puntos = puntos.copy()
            puntos[ok_m] += mano_arr[ok_m].mean(axis=1) - mano_arr[ok_m, MANO_MUNECA]
        # no se dibuja nada hasta que la mano sube por encima de su codo
        if f"codo_{lado}" in registro and len(registro[f"codo_{lado}"]) == total:
            codo = np.asarray(registro[f"codo_{lado}"], dtype=np.float64)
            codo_ok = np.asarray(registro[f"codo_{lado}_ok"], dtype=bool)
            arriba = valida & codo_ok & (puntos[:, 1] < codo[:, 1])
            valida = valida.copy()
            if arriba.any():
                valida[:int(np.argmax(arriba))] = False
            else:
                valida[:] = False

        if not valida.any():
            continue

        pantalla_x = ancho - 1 - puntos[:, 0]

        i = 0
        while i < total:
            if not valida[i]:
                i += 1
                continue
            j = i
            while j < total and valida[j]:
                j += 1
            linea = np.stack([pantalla_x[i:j], puntos[i:j, 1]], axis=1)
            cv2.polylines(
                pantalla, [np.round(linea).astype(np.int32)], False,
                color, 3, cv2.LINE_AA
            )
            i = j

        ultimo = int(np.where(valida)[0][-1])
        punto = (int(round(pantalla_x[ultimo])), int(round(puntos[ultimo, 1])))
        perdidos = total - 1 - ultimo

        if perdidos == 0:
            cv2.circle(pantalla, punto, 6, color, -1, cv2.LINE_AA)
        elif perdidos <= max_hueco:
            cv2.circle(pantalla, punto, 10, color, 2, cv2.LINE_AA)


def cara_ref_pixeles(resultado, ancho, alto):
    """Los 11 puntos de referencia de la cara (CARA_REF) en pixeles, y si se vio la cara."""
    cara = getattr(resultado, "face_landmarks", None) or []
    if len(cara) <= max(CARA_REF):
        return np.zeros((CANTIDAD_CARA_REF, 2), np.float32), False
    return np.array([[cara[j].x * ancho, cara[j].y * alto] for j in CARA_REF], np.float32), True


def indices_activos(puntos, valida, centro_hombros, ancho_hombros, codos=None, codos_ok=None):
    """
    0. (v13) La seña EMPIEZA cuando la mano sube por encima de su codo: antes de eso
       (manos abajo, preparándose) no se guarda nada. Después se guarda todo.
    Frames que forman la seña de una mano:
    1. Desde el primero hasta el último frame con la muñeca por encima
       de la mitad del torso.
    2. Sin la llegada (sube casi vertical) ni la salida (baja casi vertical)
       mientras la mano está por debajo de los hombros.
    """
    altura_hombros = centro_hombros[1]
    linea = altura_hombros + UMBRAL_TORSO * ancho_hombros

    # En las imágenes y en MediaPipe, y crece hacia ABAJO
    activa = valida & (puntos[:, 1] < linea)
    indices = np.where(activa)[0]

    if len(indices) == 0:
        return indices

    tramo = np.arange(indices[0], indices[-1] + 1)
    tramo = tramo[valida[tramo]]

    if len(tramo) < 2:
        return tramo

    p = puntos[tramo]

    def es_viaje(a, b):
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        return dy > 2.0 * dx and max(a[1], b[1]) > altura_hombros

    inicio = 0
    while (
        inicio < len(p) - 1
        and p[inicio + 1][1] < p[inicio][1]
        and es_viaje(p[inicio], p[inicio + 1])
    ):
        inicio += 1

    fin = len(p) - 1
    while (
        fin > inicio
        and p[fin][1] > p[fin - 1][1]
        and es_viaje(p[fin - 1], p[fin])
    ):
        fin -= 1

    tramo = tramo[inicio:fin + 1]
    if codos is not None and codos_ok is not None and len(tramo):
        codos = np.asarray(codos, dtype=np.float64)
        codos_ok = np.asarray(codos_ok, dtype=bool)
        arriba = codos_ok[tramo] & (puntos[tramo, 1] < codos[tramo, 1])
        if arriba.any():
            tramo = tramo[int(np.argmax(arriba)):]
        elif codos_ok[tramo].any() and _QUITAR_REPOSO[0]:
            # v17: la mano nunca subió por encima de su codo y casi no se movió -> es la mano en reposo
            # (antes eso lo resolvía que la mano abajo casi no se detecta; la muñeca del esqueleto sí se ve).
            # Si se movió, cuenta aunque esté abajo (ESTÓMAGO, FALDA, PANTALÓN...)
            recorrido = puntos[tramo]
            movimiento = float(np.max(np.ptp(recorrido, axis=0))) / max(ancho_hombros, 1e-6)
            if movimiento < MOVIMIENTO_MINIMO_REPOSO:
                return tramo[:0]
    return tramo


def centro_palma(mano):
    """Centro de la palma: promedio de la muñeca y las 5 bases de los dedos (sin las puntas)."""
    return np.asarray(mano, dtype=np.float64)[..., PALMA_PUNTOS, :].mean(axis=-2)


def medidas_dedos(mano):
    """
    De una mano (21, 2): por cada dedo
      largo    = distancia de la punta al centro de la palma / tamaño de la mano
      estirado = distancia base->punta / suma de los huesos del dedo (1 = recto, menos = recogido)
    y las puntas x/y respecto a la palma (en tamaños de mano).
    """
    p = np.asarray(mano, dtype=np.float64)
    tam = max(float(np.linalg.norm(p[MANO_BASE_MEDIO] - p[MANO_MUNECA])), 1e-6)
    palma = centro_palma(p)
    largo, estirado, puntas = [], [], []
    for base, punta in zip(DEDOS_BASES, DEDOS_PUNTAS):
        huesos = sum(np.linalg.norm(p[j + 1] - p[j]) for j in range(base, punta))
        largo.append(np.linalg.norm(p[punta] - palma) / tam)
        estirado.append(np.linalg.norm(p[punta] - p[base]) / max(huesos, 1e-6))
        puntas.append((p[punta] - palma) / tam)
    return np.array(largo), np.array(estirado), np.array(puntas)


def angulos_brazo(munecas, munecas_ok, codos, codos_ok, indices, lado, centro_hombros, ancho_hombros):
    """
    v16: el brazo como una curva a lo largo de la seña (PUNTOS_ANGULO puntos):
      ángulo del codo (entre hombro->codo y codo->muñeca) / 180°: 1 = estirado, 0 = doblado del todo
      dirección del antebrazo (codo->muñeca) como x, y
      + ángulo mínimo, máximo y cuántas veces se dobla
    El hombro sale del centro de hombros +- medio ancho (las muestras no guardan cada hombro).
    Devuelve (BRAZO_POR_MANO - 1 valores, válido True/False).
    """
    vacio = np.zeros(BRAZO_POR_MANO - 1, dtype=np.float32)
    if codos is None or codos_ok is None or len(indices) == 0:
        return vacio, False
    munecas = np.asarray(munecas, dtype=np.float64)
    codos = np.asarray(codos, dtype=np.float64)
    ok = np.asarray(munecas_ok, dtype=bool) & np.asarray(codos_ok, dtype=bool)
    todos = indices[ok[indices]]
    if len(todos) == 0:
        return vacio, False
    hombro = np.asarray(centro_hombros, dtype=np.float64) + [LADO_HOMBRO[lado] * ancho_hombros, 0.0]

    # ángulo y dirección en cada cuadro que se vio (de todos los cuadros de la seña)
    arriba = hombro - codos[todos]
    abajo = munecas[todos] - codos[todos]
    la = np.maximum(np.linalg.norm(arriba, axis=1), 1e-6)
    lb = np.maximum(np.linalg.norm(abajo, axis=1), 1e-6)
    coseno = np.clip((arriba * abajo).sum(axis=1) / (la * lb), -1.0, 1.0)
    angulo = np.degrees(np.arccos(coseno))
    dx, dy = abajo[:, 0] / lb, abajo[:, 1] / lb

    # serie cuadro a cuadro desde el inicio hasta el final de la seña (los huecos se unen en línea recta)
    i0, i1 = int(indices[0]), int(indices[-1])
    cuadros = np.arange(i0, i1 + 1, dtype=np.float64)
    xp = todos.astype(np.float64)

    def serie(valores):
        s = np.interp(cuadros, xp, valores)
        # suavizado: promedio de 5 cuadros (en los bordes se repite el extremo)
        n = len(s)
        return np.array([np.mean([s[min(max(i + k, 0), n - 1)] for k in (-2, -1, 0, 1, 2)]) for i in range(n)])

    # la curva: PUNTOS_ANGULO puntos repartidos igual del inicio al final
    posiciones = np.array([i0 + j * (i1 - i0) / (PUNTOS_ANGULO - 1) for j in range(PUNTOS_ANGULO)])
    curva_a = np.interp(posiciones, cuadros, serie(angulo))
    curva_x = np.interp(posiciones, cuadros, serie(dx))
    curva_y = np.interp(posiciones, cuadros, serie(dy))

    # cuántas veces se dobla: el ángulo baja DOBLEZ_MINIMO grados desde lo más estirado (y luego se cuenta otra
    # vez solo si antes se volvió a estirar esos mismos grados)
    dobleces, bajando = 0, False
    alto = bajo = float(curva_a[0])
    for v in curva_a:
        v = float(v)
        if not bajando:
            alto = max(alto, v)
            if alto - v >= DOBLEZ_MINIMO:
                bajando, bajo = True, v
                dobleces += 1
        else:
            bajo = min(bajo, v)
            if v - bajo >= DOBLEZ_MINIMO:
                bajando, alto = False, v

    salida = np.concatenate([
        curva_a / 180.0, curva_x, curva_y,
        [curva_a.min() / 180.0, curva_a.max() / 180.0, float(dobleces)]
    ]).astype(np.float32)
    return salida, True


def forma_de_mano(puntos_mano, mano_ok, indices):
    """
    v14: los 5 dedos al inicio, a la mitad y al final de la seña (largo y estirado)
    y las puntas de los dedos en la mitad.
    Devuelve (DEDOS_POR_MANO valores, válida True/False).
    """
    vacio = np.zeros(DEDOS_POR_MANO, dtype=np.float32)
    if len(indices) == 0:
        return vacio, False
    todos = indices[mano_ok[indices]]
    if len(todos) == 0:
        return vacio, False

    n = len(indices)
    salida = np.zeros(DEDOS_POR_MANO, dtype=np.float32)
    puntas_mitad = None
    for m in range(MOMENTOS):
        tramo = indices[m * n // MOMENTOS: max(m * n // MOMENTOS + 1, (m + 1) * n // MOMENTOS)]
        usar = tramo[mano_ok[tramo]]
        if len(usar) == 0:
            usar = todos
        medidas = [medidas_dedos(puntos_mano[i]) for i in usar]
        largo = np.median([a for a, _, _ in medidas], axis=0)
        estirado = np.median([b for _, b, _ in medidas], axis=0)
        bloque = np.stack([largo, estirado], axis=1).reshape(-1)    # [dedo][largo, estirado]
        salida[m * len(DEDOS_PUNTAS) * 2:(m + 1) * len(DEDOS_PUNTAS) * 2] = bloque
        if m == MOMENTOS // 2:
            puntas_mitad = np.median([c for _, _, c in medidas], axis=0).reshape(-1)
    salida[E_PUNTAS.start - E_FORMA.start:] = puntas_mitad
    return salida, True


def construir_muestra(munecas, validas, puntos_manos, manos_ok, ancho_hombros, centro_hombros):
    """DIBUJO 1 + coordenadas (ver construir_muestra_completa)."""
    trazos, validos, extras, _, _ = construir_muestra_completa(
        munecas, validas, puntos_manos, manos_ok, ancho_hombros, centro_hombros
    )
    return trazos, validos, extras


SALTO_MANO_CUADRO = 0.30     # anchos de hombro por cuadro (a 30 FPS): más que esto = la mano "saltó" (error)
RECHAZOS_SEGUIDOS = 5        # tras tantos cuadros seguidos lejos, se acepta (la mano sí se movió)
TAMANO_MANO_MIN, TAMANO_MANO_MAX = 0.55, 1.6   # la mano no puede encogerse/crecer tanto (detección mala)
RADIO_CARA = 0.45            # anchos de hombro alrededor de la nariz: zona donde la mano se confunde con la cara
CAMBIO_TAMANO_CARA = 1.30    # frente a la cara, un cambio de tamaño así de brusco = error


def filtrar_mano(puntos_mano, mano_ok, ancho_hombros, fps, caras=None, caras_ok=None):
    """
    Descarta los cuadros donde la detección de la mano es imposible (se marcan como "no se vio";
    luego se completan con los vecinos):
      1. la mano salta más de lo que una mano real puede moverse en ese tiempo
         (si se queda lejos varios cuadros seguidos, sí se movió y se acepta)
      2. la mano se ve mucho más grande o más pequeña de lo normal
      3. frente a la CARA (que está quieta y bien detectada): si la mano cambia de tamaño de golpe,
         MediaPipe la está mezclando con la cara
    """
    ok = np.asarray(mano_ok, dtype=bool).copy()
    puntos = np.asarray(puntos_mano, dtype=np.float64)
    if puntos.ndim != 3 or ok.sum() < 3 or ancho_hombros <= 0:
        return ok
    centros = puntos.mean(axis=1)
    tamanos = np.linalg.norm(puntos[:, MANO_BASE_MEDIO] - puntos[:, MANO_MUNECA], axis=1)
    normal = float(np.median(tamanos[ok]))
    if normal <= 1e-6:
        return ok
    ok &= (tamanos > TAMANO_MANO_MIN * normal) & (tamanos < TAMANO_MANO_MAX * normal)

    nariz = None
    if caras is not None and caras_ok is not None:
        caras = np.asarray(caras, dtype=np.float64)
        caras_ok = np.asarray(caras_ok, dtype=bool)
        if caras.ndim == 3 and len(caras) == len(ok) and caras_ok.any():
            nariz = caras[:, 0]                                  # CARA_REF[0] = nariz
            mediana = np.median(nariz[caras_ok], axis=0)
            nariz = np.where(caras_ok[:, None], nariz, mediana)

    salto = SALTO_MANO_CUADRO * ancho_hombros * 30.0 / max(fps, 1.0)
    ultimo, rechazos = None, 0
    for i in range(len(ok)):
        if not ok[i]:
            continue
        if ultimo is None:
            ultimo = i
            continue
        lejos = np.linalg.norm(centros[i] - centros[ultimo]) > salto * (i - ultimo)
        cara_mal = False
        if nariz is not None and np.linalg.norm(centros[i] - nariz[i]) < RADIO_CARA * ancho_hombros:
            r = tamanos[i] / max(tamanos[ultimo], 1e-6)
            cara_mal = r > CAMBIO_TAMANO_CARA or r < 1 / CAMBIO_TAMANO_CARA
        if (lejos or cara_mal) and rechazos < RECHAZOS_SEGUIDOS:
            ok[i] = False
            rechazos += 1
            continue
        rechazos = 0
        ultimo = i
    return ok


def _referencias_cara(caras, caras_ok, indices, centro_hombros, ancho_hombros):
    """
    Los lugares de la cara (y el pecho) durante la seña, en pixeles: (9, 2) o None si no se vio la cara.
    La cara casi no se mueve: se toma la mediana de los cuadros de la seña.
    """
    if caras is None or caras_ok is None:
        return None
    caras = np.asarray(caras, dtype=np.float64)
    caras_ok = np.asarray(caras_ok, dtype=bool)
    if caras.ndim != 3 or caras.shape[1] < CANTIDAD_CARA_REF:
        return None
    usar = indices[caras_ok[indices]] if len(indices) else indices
    if len(usar) == 0:
        usar = np.where(caras_ok)[0]
    if len(usar) == 0:
        return None
    c = np.median(caras[usar], axis=0)            # (11, 2): 1, 13, 14, 152, 10, 159, 145, 386, 374, 234, 454
    pecho = np.asarray(centro_hombros, dtype=np.float64) + [0.0, 0.45 * ancho_hombros]
    return np.array([
        c[0],                       # nariz
        (c[1] + c[2]) / 2,          # boca
        c[3],                       # mentón
        c[4],                       # frente
        (c[5] + c[6]) / 2,          # ojo A
        (c[7] + c[8]) / 2,          # ojo B
        c[9],                       # oreja A
        c[10],                      # oreja B
        pecho,
    ])


def trazo_final_mano(*args, **kwargs):
    """Igual que _trazo_final_mano; si ninguna mano queda (seña hecha abajo), se repite sin quitar la de reposo."""
    return _sin_quedar_vacio(_trazo_final_mano, lambda r: any(x is not None for x in r), *args, **kwargs)


def _sin_quedar_vacio(funcion, hay_algo, *args, **kwargs):
    _QUITAR_REPOSO[0] = True
    r = funcion(*args, **kwargs)
    if not hay_algo(r):
        _QUITAR_REPOSO[0] = False
        try:
            r = funcion(*args, **kwargs)
        finally:
            _QUITAR_REPOSO[0] = True
    return r


def _trazo_final_mano(munecas, validas, puntos_manos, manos_ok, ancho_hombros, centro_hombros,
                     codos=None, codos_ok=None, caras=None, caras_ok=None):
    """
    Para VER en pantalla el trazo tal como lo recibe el modelo (ya rellenado donde se perdió la
    mano). Mismos pasos que construir_muestra_completa. Devuelve por mano (centro (T,2), indices)
    o None: centro = trazo del centro de la mano en pixeles, indices = cuadros de la seña.
    """
    salida = [None, None]
    if not np.isfinite(ancho_hombros) or ancho_hombros < 1e-6:
        return salida
    centro_hombros = np.asarray(centro_hombros, dtype=np.float64)
    puntos_manos, manos_ok = corregir_manos_grabacion(puntos_manos, manos_ok, munecas, validas)
    for mano in range(2):
        fps = len(munecas[mano]) / SEGUNDOS_MUESTRA
        # v17: el trazo sale de la muñeca del esqueleto (igual que en construir_muestra_completa)
        centro, valida = trazo_del_brazo(munecas[mano], validas[mano], ancho_hombros, fps)
        codo_m = None if codos is None else codos[mano]
        codo_ok_m = None if codos_ok is None else codos_ok[mano]
        indices = indices_activos(centro, valida, centro_hombros, ancho_hombros, codo_m, codo_ok_m)
        if len(indices) >= MIN_FRAMES_MANO:
            # suavizar() quita un punto a cada lado: se repiten los extremos para que el punto i sea el cuadro i
            s = suavizar(centro)
            if len(s) == len(centro) - 2:
                s = np.vstack([s[:1], s, s[-1:]])
            salida[mano] = (s, np.asarray(indices))
    return salida


def construir_muestra_completa(*args, **kwargs):
    """Arma la muestra (ver _construir_muestra). Si ninguna mano queda (seña hecha abajo con poco
    movimiento), se arma otra vez sin quitar la mano en reposo: ninguna seña se queda sin trazo."""
    return _sin_quedar_vacio(_construir_muestra, lambda r: bool(np.asarray(r[1]).any()), *args, **kwargs)


def _construir_muestra(
    munecas,
    validas,
    puntos_manos,
    manos_ok,
    ancho_hombros,
    centro_hombros,
    codos=None,
    codos_ok=None,
    caras=None,
    caras_ok=None
):
    """
    v13. Por cada mano:
      - trazo del CENTRO de la mano (promedio de los 21 puntos; si no se ve la mano, la muñeca)
      - 5 trazos de las PUNTAS de los dedos (no se ven en pantalla, pero el modelo los compara)
      - trazo del CODO (aparte, para el DIBUJO 2)
      - extras: dónde ocurre, forma de la mano y DÓNDE está la mano respecto a la cara y el pecho
    La seña empieza cuando la mano sube por encima de su codo.
    trazos: (12, PUNTOS_TRAZO, 2) = [centro izq, centro der, 5 dedos izq, 5 dedos der]
    """
    trazos = np.zeros((CANTIDAD_TRAZOS, PUNTOS_TRAZO, 2), dtype=np.float32)
    validos = np.zeros(CANTIDAD_TRAZOS, dtype=bool)
    extras = np.zeros(CANTIDAD_EXTRAS, dtype=np.float32)
    trazos_brazo = np.zeros((2, PUNTOS_TRAZO, 2), dtype=np.float32)
    validos_brazo = np.zeros(2, dtype=bool)

    if not np.isfinite(ancho_hombros) or ancho_hombros < 1e-6:
        return trazos, validos, extras, trazos_brazo, validos_brazo

    centro_hombros = np.asarray(centro_hombros, dtype=np.float64)

    # Cada mano en SU brazo (corrige grabaciones donde MediaPipe las cruzó)
    puntos_manos, manos_ok = corregir_manos_grabacion(puntos_manos, manos_ok, munecas, validas)

    for mano in range(2):
        puntos = np.asarray(munecas[mano], dtype=np.float64)
        valida = np.asarray(validas[mano], dtype=bool)
        fps = len(puntos) / SEGUNDOS_MUESTRA
        # la cara (quieta) ayuda a descartar detecciones malas de la mano: saltos imposibles,
        # o la mano que "se pega" a la cara cuando pasa por delante
        ok_mano = filtrar_mano(puntos_manos[mano], manos_ok[mano], ancho_hombros, fps, caras, caras_ok)
        mano_suave = suavizar_manos(puntos_manos[mano], ok_mano, fps)
        puntos, valida = limpiar_recorrido(
            puntos, valida, ancho_hombros, fps,
            mano_suave, ok_mano
        )

        mano_arr = np.asarray(mano_suave, dtype=np.float64)
        # v17: el trazo sale SIEMPRE de la muñeca del esqueleto (la punta del antebrazo): casi nunca se
        # pierde, así no hay saltos al pasar de la mano al brazo cuando la mano se ve borrosa
        centro, valida_t = trazo_del_brazo(munecas[mano], validas[mano], ancho_hombros, fps)

        codo_m = None if codos is None else codos[mano]
        codo_ok_m = None if codos_ok is None else codos_ok[mano]
        indices = indices_activos(centro, valida_t, centro_hombros, ancho_hombros, codo_m, codo_ok_m)

        if len(indices) < MIN_FRAMES_MANO:
            continue

        recorrido = centro[indices]
        media = recorrido.mean(axis=0)

        # ---- Trazo de la palma ----
        trazo = suavizar(recorrido)
        trazo = remuestrear_por_longitud(trazo, PUNTOS_TRAZO)
        trazo = (trazo - trazo.mean(axis=0)) / ancho_hombros
        trazos[mano] = trazo
        validos[mano] = True

        # si se pierde la mano (tapada, borrosa) se ESTIMA: su última forma sigue pegada a la muñeca
        # del brazo (hasta MANO_FANTASMA_SEGUNDOS); con eso se miden los dedos
        mano_completa, ok_completo = completar_manos_tapadas(
            mano_suave, ok_mano, puntos, valida, fps
        )

        # v15: ya no hay trazo del codo (DIBUJO 2); el brazo va como ángulos en los extras

        # ---- Dónde ocurre (respecto a los hombros) ----
        relativo = (recorrido - centro_hombros) / ancho_hombros
        bloque = extras[mano * EXTRAS_POR_MANO:(mano + 1) * EXTRAS_POR_MANO]

        bloque[E_ACTIVA] = 1.0
        bloque[E_INICIO] = relativo[0]
        bloque[E_FINAL] = relativo[-1]
        bloque[E_CENTRO] = relativo.mean(axis=0)
        bloque[E_TAMANO] = relativo.max(axis=0) - relativo.min(axis=0)

        # ---- Forma de la mano (con las manos tapadas completadas) ----
        forma, forma_valida = forma_de_mano(mano_completa, ok_completo, indices)

        bloque[E_FORMA_VALIDA] = 1.0 if forma_valida else 0.0
        bloque[E_FORMA] = forma

        # ---- Dónde está la mano respecto a la CARA y el pecho (el "mapa" quieto) ----
        refs = _referencias_cara(caras, caras_ok, indices, centro_hombros, ancho_hombros)
        if refs is not None:
            cara = np.zeros(len(LUGARES) * 3, dtype=np.float32)
            for j, ref in enumerate(refs):
                d = (recorrido - ref) / ancho_hombros
                cara[j * 3:j * 3 + 2] = d.mean(axis=0)
                cara[j * 3 + 2] = np.linalg.norm(d, axis=1).min()
            bloque[E_CARA_VALIDA] = 1.0
            bloque[E_CARA] = cara

        # ---- El brazo: ángulo del codo y dirección del antebrazo (inicio, mitad, final) ----
        brazo, brazo_valido = angulos_brazo(
            munecas[mano], validas[mano],
            None if codos is None else codos[mano], None if codos_ok is None else codos_ok[mano],
            indices, mano, centro_hombros, ancho_hombros
        )
        bloque[E_BRAZO_VALIDO] = 1.0 if brazo_valido else 0.0
        bloque[E_BRAZO] = brazo

    return trazos, validos, extras, trazos_brazo, validos_brazo


# =============================================================================
# DIBUJO
# =============================================================================

def _linea_trazo(lienzo, trazo, tamano):
    pixeles = (trazo / MEDIA_VENTANA * 0.5 + 0.5) * (tamano - 1)
    pixeles = np.clip(np.round(pixeles), 0, tamano - 1).astype(np.int32)
    cantidad = len(pixeles)
    for k in range(cantidad - 1):
        # Más oscuro al inicio, más claro al final = dirección del trazo
        intensidad = int(90 + 165 * (k + 1) / (cantidad - 1))
        cv2.line(lienzo, (int(pixeles[k][0]), int(pixeles[k][1])),
                 (int(pixeles[k + 1][0]), int(pixeles[k + 1][1])), intensidad, GROSOR_LINEA, cv2.LINE_AA)


def dibujar_trazos(trazos, validos, tamano=TAMANO_IMAGEN):
    """
    Convierte los trazos en una imagen con valores 0..1:
      - 2 trazos (codos):            (2, tamano, tamano)
      - 12 trazos (centro + dedos):  (4, tamano, tamano) = centro izq, centro der, dedos izq, dedos der
    """
    trazos = np.asarray(trazos)
    if len(trazos) == 2:
        grupos = [[0], [1]]
    else:
        n = len(DEDOS_PUNTAS)
        grupos = [[0], [1], list(range(2, 2 + n)), list(range(2 + n, 2 + 2 * n))]
    imagen = np.zeros((len(grupos), tamano, tamano), dtype=np.float32)
    for canal, indices in enumerate(grupos):
        lienzo = np.zeros((tamano, tamano), dtype=np.uint8)
        hay = False
        for indice in indices:
            if validos[indice]:
                _linea_trazo(lienzo, trazos[indice], tamano)
                hay = True
        if hay:
            imagen[canal] = lienzo.astype(np.float32) / 255.0
    return imagen


# =============================================================================
# AUMENTO DE DATOS (dibujo y extras juntos)
# =============================================================================

def espejar_extras(extras):
    """Espejo: invierte x y cambia mano izquierda <-> derecha (y ojo/oreja A <-> B)."""
    extras = extras.copy()

    for mano in range(2):
        bloque = extras[mano * EXTRAS_POR_MANO:(mano + 1) * EXTRAS_POR_MANO]
        bloque[E_INICIO.start] *= -1
        bloque[E_FINAL.start] *= -1
        bloque[E_CENTRO.start] *= -1
        bloque[E_PUNTAS.start:E_PUNTAS.stop:2] *= -1     # solo las x de las puntas; largo y estirado no cambian
        cara = bloque[E_CARA].reshape(len(LUGARES), 3).copy()
        cara[:, 0] *= -1
        for a, b in PARES_ESPEJO:
            cara[[a, b]] = cara[[b, a]]
        bloque[E_CARA] = cara.reshape(-1)
        bloque[E_BRAZO.start + PUNTOS_ANGULO:E_BRAZO.start + 2 * PUNTOS_ANGULO] *= -1     # dirección x del antebrazo

    return np.concatenate(
        [extras[EXTRAS_POR_MANO:], extras[:EXTRAS_POR_MANO]]
    ).astype(np.float32)


def aumentar_muestra(trazos, validos, extras, rng):
    """Espejo, rotación, escala, proporción, recorte y ruido."""
    trazos = trazos.astype(np.float32).copy()
    validos = validos.copy()
    extras = extras.astype(np.float32).copy()

    if len(trazos) == 2:
        orden_espejo = [1, 0]
    else:
        n = len(DEDOS_PUNTAS)
        orden_espejo = [1, 0] + list(range(2 + n, 2 + 2 * n)) + list(range(2, 2 + n))
    # sin espejo: todos hacen la seña con la misma mano (no se simulan zurdos)
    ESPEJO = False
    if ESPEJO and rng.random() < 0.5:
        trazos[:, :, 0] *= -1
        trazos = trazos[orden_espejo].copy()
        validos = validos[orden_espejo].copy()
        extras = espejar_extras(extras)

    angulo = np.deg2rad(rng.uniform(-12, 12))
    coseno, seno = np.cos(angulo), np.sin(angulo)
    rotacion = np.array([[coseno, -seno], [seno, coseno]], dtype=np.float32)
    trazos = trazos @ rotacion.T

    escala = rng.uniform(0.85, 1.15)
    proporcion = rng.uniform(0.85, 1.18)

    trazos *= escala
    trazos[:, :, 0] *= proporcion

    cantidad = trazos.shape[1]
    recorte_max = int(cantidad * 0.12)

    for indice in range(len(trazos)):
        if validos[indice]:
            trazos[indice] += rng.normal(0.0, 0.02, size=trazos[indice].shape)

    for mano in range(2):
        if not validos[mano]:
            continue

        if rng.random() < 0.5:
            inicio = int(rng.integers(0, recorte_max + 1))
            fin = cantidad - int(rng.integers(0, recorte_max + 1))

            if fin - inicio >= 8:
                nuevo = remuestrear_por_longitud(trazos[mano, inicio:fin], cantidad)
                trazos[mano] = nuevo - nuevo.mean(axis=0)

        bloque = extras[mano * EXTRAS_POR_MANO:(mano + 1) * EXTRAS_POR_MANO]

        for parte in (E_INICIO, E_FINAL, E_CENTRO):
            bloque[parte] += rng.normal(0.0, 0.06, size=2)
            bloque[parte.start] *= proporcion

        bloque[E_TAMANO] *= escala
        bloque[E_TAMANO.start] *= proporcion

        if bloque[E_FORMA_VALIDA] > 0.5:
            bloque[E_DEDOS] += rng.normal(0.0, 0.03, size=E_DEDOS.stop - E_DEDOS.start)
            bloque[E_PUNTAS] += rng.normal(0.0, 0.04, size=E_PUNTAS.stop - E_PUNTAS.start)
            bloque[E_PUNTAS.start:E_PUNTAS.stop:2] *= proporcion

        if len(extras) >= CANTIDAD_EXTRAS and bloque[E_CARA_VALIDA] > 0.5:
            bloque[E_CARA] += rng.normal(0.0, 0.03, size=len(LUGARES) * 3)

        if len(extras) >= CANTIDAD_EXTRAS and bloque[E_BRAZO_VALIDO] > 0.5:
            # toda la curva del ángulo se corre ±0.03 (unos 5°) y la dirección del antebrazo gira igual que el trazo
            brazo = bloque[E_BRAZO].copy()
            n = PUNTOS_ANGULO
            corrimiento = rng.normal(0.0, 0.03)
            brazo[:n] += corrimiento + rng.normal(0.0, 0.01, size=n)
            brazo[3 * n:3 * n + 2] += corrimiento              # mínimo y máximo
            direccion = np.stack([brazo[n:2 * n], brazo[2 * n:3 * n]], axis=1) @ rotacion.T
            brazo[n:2 * n], brazo[2 * n:3 * n] = direccion[:, 0], direccion[:, 1]
            bloque[E_BRAZO] = brazo

    return trazos.astype(np.float32), validos, extras.astype(np.float32)


# =============================================================================
# LECTURA DEL DATASET
# =============================================================================

def extraer_de_secuencia_dataset(secuencia):
    """Saca todo lo necesario de una secuencia de dataset_14_completo.npz."""
    secuencia = np.nan_to_num(
        np.asarray(secuencia, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    cuerpo = secuencia[:, INICIO_CUERPO:INICIO_MI].reshape(-1, CANTIDAD_CUERPO, 3).copy()
    mano_izq = secuencia[:, INICIO_MI:INICIO_MD].reshape(-1, CANTIDAD_MANO, 3).copy()
    mano_der = secuencia[:, INICIO_MD:INICIO_CARA].reshape(-1, CANTIDAD_MANO, 3).copy()

    # Corrección de escala vertical (ver comparar_escala.py).
    # Los ceros (punto no detectado) siguen siendo ceros.
    cuerpo[:, :, 1] *= ESCALA_Y_DATASET
    mano_izq[:, :, 1] *= ESCALA_Y_DATASET
    mano_der[:, :, 1] *= ESCALA_Y_DATASET

    cuerpo_ok = np.abs(cuerpo[:, :, :2]).sum(axis=(1, 2)) > 1e-6
    izq_ok = np.abs(mano_izq[:, :, :2]).sum(axis=(1, 2)) > 1e-6
    der_ok = np.abs(mano_der[:, :, :2]).sum(axis=(1, 2)) > 1e-6

    hombro_izq = cuerpo[:, CUERPO_HOMBRO_IZQ, :2]
    hombro_der = cuerpo[:, CUERPO_HOMBRO_DER, :2]

    distancias = np.linalg.norm(hombro_izq - hombro_der, axis=1)
    distancias = distancias[cuerpo_ok & (distancias > 1e-6)]
    ancho_hombros = float(np.median(distancias)) if len(distancias) else 0.0

    centros = (hombro_izq + hombro_der) / 2.0
    centros = centros[cuerpo_ok]
    centro_hombros = (
        np.median(centros, axis=0) if len(centros) else np.zeros(2)
    )

    return {
        "munecas": [
            cuerpo[:, CUERPO_MUNECA_IZQ, :2],
            cuerpo[:, CUERPO_MUNECA_DER, :2]
        ],
        "validas": [
            cuerpo_ok & izq_ok,
            cuerpo_ok & der_ok
        ],
        "puntos_manos": [mano_izq[:, :, :2], mano_der[:, :, :2]],
        "manos_ok": [izq_ok, der_ok],
        "ancho_hombros": ancho_hombros,
        "centro_hombros": centro_hombros
    }


# =============================================================================
# VISTA
# =============================================================================

def imagen_a_color(imagen, escala=3):
    """Imagen (2, H, W) -> BGR. Izquierda = verde, derecha = magenta."""
    izquierda = (imagen[0] * 255).astype(np.uint8)
    derecha = (imagen[1] * 255).astype(np.uint8)

    color = np.zeros((imagen.shape[1], imagen.shape[2], 3), dtype=np.uint8)
    color[:, :, 0] = derecha
    color[:, :, 1] = izquierda
    color[:, :, 2] = derecha

    return cv2.resize(color, None, fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST)


# =============================================================================
# MODELO
# =============================================================================

def _rama_cnn(canales=2):
    """CNN pequeña que convierte el dibujo (canales, 64, 64) en 96 números."""
    def bloque(entrada, salida):
        return nn.Sequential(
            nn.Conv2d(entrada, salida, 3, padding=1),
            nn.BatchNorm2d(salida),
            nn.ReLU(),
            nn.Conv2d(salida, salida, 3, padding=1),
            nn.BatchNorm2d(salida),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    return nn.Sequential(
        bloque(canales, 16),
        bloque(16, 32),
        bloque(32, 48),
        nn.Dropout2d(0.10),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Dropout(0.40),
        nn.Linear(48 * 4 * 4, 96),
        nn.ReLU()
    )


class ModeloCoordenadas(nn.Module):
    """ETAPA 1 de la cascada: solo coordenadas (lugar + 21 puntos de la mano)."""

    def __init__(self, cantidad_clases, cantidad_extras=CANTIDAD_EXTRAS):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(cantidad_extras, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, cantidad_clases)
        )

    def forward(self, imagen, extras):
        return self.red(extras)


class ModeloDibujo(nn.Module):
    """ETAPA 2 de la cascada: solo el dibujo."""

    def __init__(self, cantidad_clases, canales=2):
        super().__init__()

        self.rama_dibujo = _rama_cnn(canales)
        self.clasificador = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(96, cantidad_clases)
        )

    def forward(self, imagen, extras):
        return self.clasificador(self.rama_dibujo(imagen))


def combinar_cascada(prob_coordenadas, prob_dibujo, k=3, minimo=0.05, prob_dibujo2=None):
    """
    Etapa 1: las coordenadas eligen hasta k señas candidatas
             (las de mayor probabilidad, con al menos 'minimo').
    Etapa 2: el dibujo de la mano (y, si hay, el del antebrazo) deciden
             entre esas candidatas: gana la que mejor coincide en los tres.

    Funciona con un vector (clases,) o un lote (n, clases).
    Devuelve (puntaje final, máscara de candidatas).
    """
    prob_coordenadas = np.atleast_2d(prob_coordenadas)
    prob_dibujo = np.atleast_2d(prob_dibujo)

    orden = np.argsort(-prob_coordenadas, axis=1)
    candidatas = np.zeros_like(prob_coordenadas, dtype=bool)

    filas = np.arange(len(prob_coordenadas))
    for posicion in range(k):
        columna = orden[:, posicion]
        suficiente = prob_coordenadas[filas, columna] >= minimo
        if posicion == 0:
            suficiente[:] = True
        candidatas[filas[suficiente], columna[suficiente]] = True

    producto = prob_coordenadas * prob_dibujo
    if prob_dibujo2 is not None:
        producto = producto * np.atleast_2d(prob_dibujo2)

    puntaje = np.where(candidatas, producto, 0.0)
    puntaje = puntaje / np.maximum(puntaje.sum(axis=1, keepdims=True), 1e-12)

    return puntaje, candidatas


class ModeloHibrido(nn.Module):
    """
    Rama 1: CNN pequeña que mira el DIBUJO.
    Rama 2: red densa que mira los EXTRAS (lugar + forma de la mano).
    Las dos se unen para decidir la seña.
    """

    def __init__(self, cantidad_clases, cantidad_extras=CANTIDAD_EXTRAS):
        super().__init__()

        def bloque(entrada, salida):
            return nn.Sequential(
                nn.Conv2d(entrada, salida, 3, padding=1),
                nn.BatchNorm2d(salida),
                nn.ReLU(),
                nn.Conv2d(salida, salida, 3, padding=1),
                nn.BatchNorm2d(salida),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.rama_dibujo = nn.Sequential(
            bloque(2, 16),
            bloque(16, 32),
            bloque(32, 48),
            nn.Dropout2d(0.10),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(48 * 4 * 4, 96),
            nn.ReLU()
        )

        self.rama_extras = nn.Sequential(
            nn.Linear(cantidad_extras, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.clasificador = nn.Sequential(
            nn.Dropout(0.40),
            nn.Linear(96 + 64, 96),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(96, cantidad_clases)
        )

    def forward(self, imagen, extras):
        unidos = torch.cat(
            [self.rama_dibujo(imagen), self.rama_extras(extras)],
            dim=1
        )
        return self.clasificador(unidos)
