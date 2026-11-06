package co.edu.itm.manosquehablan.vivo

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt

// Seguimiento en vivo de manos, antebrazos y cara (misma logica de dibujo_senas.py).
// Las coordenadas estan en pixeles de la imagen sin espejo, el espejo se hace al dibujar.

data class P(val x: Double, val y: Double) {
    operator fun plus(o: P) = P(x + o.x, y + o.y)
    operator fun minus(o: P) = P(x - o.x, y - o.y)
    operator fun times(k: Double) = P(x * k, y * k)
    operator fun div(k: Double) = P(x / k, y / k)
    fun norma() = sqrt(x * x + y * y)
    fun punto(o: P) = x * o.x + y * o.y
    fun cruz(o: P) = x * o.y - y * o.x
}

typealias Mano = Array<P>

const val CANTIDAD_MANO = 21
const val MANO_MUNECA = 0
const val MANO_BASE_MEDIO = 9

const val POSE_HOMBRO_IZQ = 11
const val POSE_HOMBRO_DER = 12
const val POSE_CODO_IZQ = 13
const val POSE_CODO_DER = 14
const val POSE_MUNECA_IZQ = 15
const val POSE_MUNECA_DER = 16

const val UMBRAL_TORSO = 1.8

const val MANO_CORTE_MINIMO = 1.0
const val MANO_BETA = 0.5
const val DEDOS_CORTE_MINIMO = 0.5
const val DEDOS_BETA = 0.3
const val RADIO_DEDOS = 0.06

const val ANTEBRAZO_CORTE_MINIMO = 0.4
const val ANTEBRAZO_BETA = 0.3
const val RADIO_ANTEBRAZO = 0.04
const val GIRO_MAXIMO = 300.0

const val SALTO_MANO_VIVO = 1.2
const val RECHAZOS_MAXIMOS = 5
const val TRANSICION_MANO = 0.25
const val MANO_QUIETA_SEGUNDOS = 0.3
const val CRUCE_MARGEN = 0.7
const val MANOS_DUPLICADAS = 0.45

const val DOBLEZ_HACIA_AFUERA = 87.0
const val DOBLEZ_HACIA_ADENTRO = 70.0
const val INVERTIR_DOBLEZ = false
const val INVERTIR_GIRO = false

const val CRECIMIENTO_MAXIMO = 1.15
const val SALTO_TAMANO = 1.35
const val APRENDIZAJE_LARGOS = 0.05

const val CARA_CORTE_MINIMO = 0.3
const val CARA_BETA = 0.3
const val CARA_NARIZ = 1
const val CARA_OJO_A = 33
const val CARA_OJO_B = 263

val PADRE = IntArray(21) { j -> if (j == 0) 0 else if (j in intArrayOf(1, 5, 9, 13, 17)) 0 else j - 1 }

// puntos de mediapipe de un cuadro (normalizados de 0 a 1)
class Deteccion(
    val pose: List<FloatArray>,      // (x, y, visibilidad)
    val manoIzq: List<FloatArray>,
    val manoDer: List<FloatArray>,
    val cara: List<FloatArray>
)

fun posePixel(pose: List<FloatArray>, i: Int, ancho: Int, alto: Int, minimo: Double = 0.5): P? {
    if (pose.size <= i || pose[i][2] < minimo) return null
    return P(pose[i][0].toDouble() * ancho, pose[i][1].toDouble() * alto)
}

fun aPixeles(puntos: List<FloatArray>, ancho: Int, alto: Int): Mano =
    Array(puntos.size) { P(puntos[it][0].toDouble() * ancho, puntos[it][1].toDouble() * alto) }

private fun plano(m: Array<P>) = DoubleArray(m.size * 2) { if (it % 2 == 0) m[it / 2].x else m[it / 2].y }
private fun desplano(d: DoubleArray) = Array(d.size / 2) { P(d[2 * it], d[2 * it + 1]) }

private fun alfa(corte: Double, dt: Double): Double {
    val tau = 1.0 / (2.0 * PI * corte)
    return 1.0 / (1.0 + tau / dt)
}

fun girar(v: P, angulo: Double): P {
    val c = cos(angulo); val s = sin(angulo)
    return P(c * v.x - s * v.y, s * v.x + c * v.y)
}

private fun promedio(lista: List<Mano>): Mano =
    Array(lista[0].size) { j -> lista.fold(P(0.0, 0.0)) { a, m -> a + m[j] } / lista.size.toDouble() }

// =============================================================================
// FILTROS
// =============================================================================

class FiltroUnoEuro(private val corteMinimo: Double, private val beta: Double, private val corteVelocidad: Double = 1.0) {
    private var valor: DoubleArray? = null
    private var velocidad: DoubleArray? = null

    fun reiniciar() { valor = null; velocidad = null }

    fun filtrar(x: DoubleArray, dt0: Double, escala: Double): DoubleArray {
        val v0 = valor
        if (v0 == null) {
            valor = x.copyOf(); velocidad = DoubleArray(x.size)
            return x.copyOf()
        }
        val dt = max(dt0, 1e-3)
        val aV = alfa(corteVelocidad, dt)
        val vel = velocidad!!
        for (i in x.indices) vel[i] = aV * (x[i] - v0[i]) / dt + (1 - aV) * vel[i]

        var rapidez = 0.0
        for (i in 0 until x.size / 2) rapidez = max(rapidez, sqrt(vel[2 * i] * vel[2 * i] + vel[2 * i + 1] * vel[2 * i + 1]))
        rapidez /= max(escala, 1e-6)
        val a = alfa(corteMinimo + beta * rapidez, dt)
        for (i in x.indices) v0[i] = a * x[i] + (1 - a) * v0[i]
        return v0.copyOf()
    }
}

class FiltroMano {
    private val posicion = FiltroUnoEuro(MANO_CORTE_MINIMO, MANO_BETA)
    private val forma = FiltroUnoEuro(DEDOS_CORTE_MINIMO, DEDOS_BETA)
    private var dedos: Mano? = null

    fun reiniciar() { posicion.reiniciar(); forma.reiniciar(); dedos = null }

    fun filtrar(puntos: Mano, ok: Boolean, dt: Double): Mano {
        if (!ok) { reiniciar(); return puntos }
        val escala = (puntos[MANO_BASE_MEDIO] - puntos[MANO_MUNECA]).norma()
        val m = posicion.filtrar(plano(arrayOf(puntos[MANO_MUNECA])), dt, escala)
        val muneca = P(m[0], m[1])
        val f = desplano(forma.filtrar(plano(Array(puntos.size) { puntos[it] - puntos[MANO_MUNECA] }), dt, escala))

        val d = dedos
        dedos = if (d == null) f else {
            val radio = RADIO_DEDOS * max(escala, 1.0)
            Array(f.size) { j ->
                val dif = f[j] - d[j]
                val dist = dif.norma()
                val avance = max(dist - radio, 0.0) / max(dist, 1e-9)
                d[j] + dif * avance
            }
        }
        return Array(f.size) { dedos!![it] + muneca }
    }
}

class FiltroAntebrazo {
    private val filtro = FiltroUnoEuro(ANTEBRAZO_CORTE_MINIMO, ANTEBRAZO_BETA)
    private var actual: Array<P>? = null

    fun reiniciar() { filtro.reiniciar(); actual = null }

    // devuelve muñeca y codo suavizados
    fun filtrar(muneca: P, codo: P, dt: Double): Pair<P, P> {
        val largo = max((muneca - codo).norma(), 1.0)
        val suave = desplano(filtro.filtrar(plano(arrayOf(muneca, codo)), dt, largo / 3.0))

        val act = actual
        if (act == null) { actual = suave; return suave[0] to suave[1] }

        val radio = RADIO_ANTEBRAZO * largo
        val nuevo = Array(2) { i ->
            val dif = suave[i] - act[i]
            val dist = dif.norma()
            act[i] + dif * (max(dist - radio, 0.0) / max(dist, 1e-9))
        }

        val antes = act[0] - act[1]
        val ahora = nuevo[0] - nuevo[1]
        if (antes.norma() > 1e-6 && ahora.norma() > 1e-6) {
            val angulo = atan2(antes.cruz(ahora), antes.punto(ahora))
            val maximo = Math.toRadians(GIRO_MAXIMO) * dt
            if (abs(angulo) > maximo) {
                nuevo[0] = nuevo[1] + girar(antes / antes.norma(), sign(angulo) * maximo) * ahora.norma()
            }
        }
        actual = nuevo
        return nuevo[0] to nuevo[1]
    }
}

// =============================================================================
// MANOS
// =============================================================================

// asigna cada mano a la muñeca de su brazo para que no se crucen
fun asignarManos(manoIzq: Mano?, manoDer: Mano?, poseIzq: P?, poseDer: P?): Pair<Mano?, Mano?> {
    if (poseIzq == null || poseDer == null) return manoIzq to manoDer
    val pose = arrayOf(poseIzq, poseDer)
    val manos = arrayOf(manoIzq, manoDer)
    fun distancia(m: Mano, lado: Int) = (m[MANO_MUNECA] - pose[lado]).norma()

    val destino = arrayOfNulls<Int>(2)
    for (lado in 0..1) {
        val m = manos[lado] ?: continue
        val propia = distancia(m, lado); val otra = distancia(m, 1 - lado)
        destino[lado] = if (otra < CRUCE_MARGEN * propia) 1 - lado else lado
    }

    val a = manos[0]; val b = manos[1]
    if (a != null && b != null) {
        val tamano = maxOf((a[MANO_BASE_MEDIO] - a[MANO_MUNECA]).norma(), (b[MANO_BASE_MEDIO] - b[MANO_MUNECA]).norma(), 1.0)
        fun centro(m: Mano) = m.fold(P(0.0, 0.0)) { s, p -> s + p } / m.size.toDouble()
        val separacion = (centro(a) - centro(b)).norma()
        if (separacion < MANOS_DUPLICADAS * tamano) {
            val quitar = if (distancia(a, 0) > distancia(b, 1)) 0 else 1
            destino[quitar] = null
        }
    }

    val nuevas = arrayOfNulls<Mano>(2)
    for (lado in 0..1) {
        val d = destino[lado] ?: continue
        val m = manos[lado]!!
        val actual = nuevas[d]
        if (actual == null || distancia(m, d) < distancia(actual, d)) nuevas[d] = m
    }
    return nuevas[0] to nuevas[1]
}

// si un hueso se ve mas largo tantos cuadros seguidos, de verdad mide eso (no es error de deteccion)
const val CUADROS_PARA_CRECER = 8

class LargosMano {
    private var largos: DoubleArray? = null
    private var rechazos = 0
    private val largoSeguidos = IntArray(20)    // cuadros seguidos que cada hueso se vio mas largo

    fun reiniciar() { largos = null; rechazos = 0; largoSeguidos.fill(0) }

    private fun medir(p: Mano) = DoubleArray(20) { (p[it + 1] - p[PADRE[it + 1]]).norma() }

    // devuelve los puntos corregidos y si la deteccion es buena
    fun corregir(puntos: Mano): Pair<Mano, Boolean> {
        val medidos = medir(puntos)
        val l = largos
        if (l == null) { largos = medidos.copyOf(); return puntos to true }

        val razones = DoubleArray(20) { medidos[it] / max(l[it], 1e-6) }.sorted()
        val mediana = (razones[9] + razones[10]) / 2.0
        if (mediana > SALTO_TAMANO) {
            rechazos += 1
            if (rechazos < 10) return puntos to false
            largos = medidos.copyOf(); rechazos = 0
            return puntos to true
        }
        rechazos = 0

        // un dedo que se aprendio corto (estaba doblado al aparecer la mano) recupera su largo real
        // si se ve mas largo varios cuadros seguidos; un error de MediaPipe dura 1 o 2 cuadros y se recorta
        for (i in 0 until 20) {
            if (medidos[i] > l[i] * CRECIMIENTO_MAXIMO) largoSeguidos[i]++ else largoSeguidos[i] = 0
            if (largoSeguidos[i] >= CUADROS_PARA_CRECER) { l[i] = medidos[i]; largoSeguidos[i] = 0 }
        }

        val nuevos = puntos.copyOf()
        for (j in 1 until 21) {
            val padre = PADRE[j]
            var dir = puntos[j] - puntos[padre]
            val largo = dir.norma()
            val maximo = l[j - 1] * CRECIMIENTO_MAXIMO
            if (largo > maximo && largo > 1e-6) dir = dir * (maximo / largo)
            nuevos[j] = nuevos[padre] + dir
        }
        for (i in 0 until 20) if (medidos[i] <= l[i] * CRECIMIENTO_MAXIMO) l[i] += APRENDIZAJE_LARGOS * (medidos[i] - l[i])
        return nuevos to true
    }
}

fun doblezPosible(mano: Mano, codo: P, lado: Int): Boolean {
    val antebrazo = mano[MANO_MUNECA] - codo
    val palma = mano[MANO_BASE_MEDIO] - mano[MANO_MUNECA]
    if (antebrazo.norma() < 1e-6 || palma.norma() < 1e-6) return true
    val angulo = Math.toDegrees(atan2(antebrazo.cruz(palma), antebrazo.punto(palma)))
    var haciaSuDerecha = -angulo
    if (INVERTIR_DOBLEZ) haciaSuDerecha = -haciaSuDerecha
    val (limDer, limIzq) = if (lado == 1) DOBLEZ_HACIA_AFUERA to DOBLEZ_HACIA_ADENTRO else DOBLEZ_HACIA_ADENTRO to DOBLEZ_HACIA_AFUERA
    return haciaSuDerecha in -limIzq..limDer
}

private class Pegada(val forma: Mano, val desfase: P, val angulo: Double)

class ManosEnVivo {
    private val filtros = arrayOf(FiltroMano(), FiltroMano())
    private var anterior: Double? = null
    val fantasma = booleanArrayOf(false, false)
    private val estables = arrayOf(mutableListOf<Mano>(), mutableListOf())
    private val ultimaVista = arrayOf(0.0, 0.0)
    private val rechazos = intArrayOf(0, 0)
    private val perdida = booleanArrayOf(false, false)
    private val transicion = arrayOfNulls<Pair<Double, Mano>>(2)
    private val vistaAntes = booleanArrayOf(false, false)
    private val pegada = arrayOfNulls<Pegada>(2)
    private val mostrada = arrayOfNulls<Mano>(2)
    private val largos = arrayOf(LargosMano(), LargosMano())
    private val brazos = arrayOf(FiltroAntebrazo(), FiltroAntebrazo())

    private fun brazo(pose: List<FloatArray>, lado: Int, ancho: Int, alto: Int, dt: Double): Pair<P, Double>? {
        val codo = posePixel(pose, if (lado == 0) POSE_CODO_IZQ else POSE_CODO_DER, ancho, alto)
        val muneca = posePixel(pose, if (lado == 0) POSE_MUNECA_IZQ else POSE_MUNECA_DER, ancho, alto)
        if (codo == null || muneca == null) { brazos[lado].reiniciar(); return null }
        val (m, c) = brazos[lado].filtrar(muneca, codo, dt)
        val d = m - c
        if (d.norma() < 1e-6) return null
        return m to atan2(d.y, d.x)
    }

    private fun ancla(lado: Int): Mano? = if (estables[lado].isEmpty()) null else promedio(estables[lado])

    fun actualizar(det: Deteccion, ancho: Int, alto: Int, ahora: Double): Array<Mano?> {
        val dt = anterior?.let { max(ahora - it, 1e-3) } ?: (1.0 / 30.0)
        anterior = ahora

        val crudaIzq = if (det.manoIzq.isNotEmpty()) aPixeles(det.manoIzq, ancho, alto) else null
        val crudaDer = if (det.manoDer.isNotEmpty()) aPixeles(det.manoDer, ancho, alto) else null
        val pose = det.pose
        val (mi, md) = asignarManos(
            crudaIzq, crudaDer,
            posePixel(pose, POSE_MUNECA_IZQ, ancho, alto), posePixel(pose, POSE_MUNECA_DER, ancho, alto)
        )

        val resultado = arrayOfNulls<Mano>(2)
        for ((lado, entrada) in listOf(mi, md).withIndex()) {
            var pixeles = entrada
            var visto = pixeles != null
            val filtro = filtros[lado]
            val ancla = ancla(lado)
            val brazo = brazo(pose, lado, ancho, alto, dt)

            if (visto) {
                val codo = posePixel(pose, if (lado == 0) POSE_CODO_IZQ else POSE_CODO_DER, ancho, alto)
                if (codo != null && !doblezPosible(pixeles!!, codo, lado)) visto = false
            }
            if (visto) {
                val (corregidos, buena) = largos[lado].corregir(pixeles!!)
                pixeles = corregidos
                if (!buena) visto = false
            }

            val detectada = visto
            if (visto && ancla != null && vistaAntes[lado]) {
                val tamano = max((ancla[MANO_BASE_MEDIO] - ancla[MANO_MUNECA]).norma(), 1.0)
                val salto = (pixeles!![MANO_MUNECA] - ancla[MANO_MUNECA]).norma()
                if (salto > SALTO_MANO_VIVO * tamano && rechazos[lado] < RECHAZOS_MAXIMOS) {
                    rechazos[lado] += 1
                    visto = false
                }
            }
            if (!detectada) vistaAntes[lado] = false

            if (visto) {
                if (!vistaAntes[lado]) { filtro.reiniciar(); estables[lado].clear() }
                vistaAntes[lado] = true
                rechazos[lado] = 0
                val suave = filtro.filtrar(pixeles!!, true, dt)

                val mos = mostrada[lado]
                if (perdida[lado] && mos != null) transicion[lado] = ahora to mos.copyOf()
                perdida[lado] = false

                var salida = suave
                transicion[lado]?.let { (inicio, desde) ->
                    var k = min(1.0, (ahora - inicio) / TRANSICION_MANO)
                    k = k * k * (3 - 2 * k)
                    salida = Array(suave.size) { desde[it] + (suave[it] - desde[it]) * k }
                    if (k >= 1.0) transicion[lado] = null
                }

                estables[lado].add(suave.copyOf())
                while (estables[lado].size > 2) estables[lado].removeAt(0)
                ultimaVista[lado] = ahora
                fantasma[lado] = false

                if (brazo != null) {
                    val base = promedio(estables[lado])
                    pegada[lado] = Pegada(
                        Array(base.size) { base[it] - base[MANO_MUNECA] },
                        base[MANO_MUNECA] - brazo.first, brazo.second
                    )
                }
                mostrada[lado] = salida
                resultado[lado] = salida
                continue
            }

            val peg = pegada[lado]
            if (brazo != null && peg != null) {
                val giro = brazo.second - peg.angulo
                val desfase = girar(peg.desfase, giro)
                val mano = Array(peg.forma.size) { brazo.first + desfase + girar(peg.forma[it], giro) }
                perdida[lado] = true; fantasma[lado] = true
                mostrada[lado] = mano; resultado[lado] = mano
                continue
            }

            if (ancla != null && ahora - ultimaVista[lado] <= MANO_QUIETA_SEGUNDOS) {
                perdida[lado] = true; fantasma[lado] = true
                mostrada[lado] = ancla; resultado[lado] = ancla
                continue
            }

            filtro.reiniciar()
            estables[lado].clear()
            pegada[lado] = null; mostrada[lado] = null
            perdida[lado] = false; transicion[lado] = null; fantasma[lado] = false
            resultado[lado] = null
        }
        return resultado
    }
}

// =============================================================================
// ANTEBRAZOS
// =============================================================================

// giro de la mano: 0 palma hacia la camara, 1 dorso
fun giroDeMano(p: Mano, lado: Int): Double? {
    val a = p[5] - p[0]
    val b = p[17] - p[0]
    val normas = a.norma() * b.norma()
    if (normas < 1e-6) return null
    val cruz = a.cruz(b) / normas
    var signo = if (lado == 0) 1.0 else -1.0
    if (INVERTIR_GIRO) signo = -signo
    return ((1.0 - cruz * signo) / 2.0).coerceIn(0.0, 1.0)
}

class AntebrazosEnVivo {
    private val filtros = arrayOf(FiltroAntebrazo(), FiltroAntebrazo())
    val giro = doubleArrayOf(0.0, 0.0)
    private var anterior: Double? = null

    // devuelve (muñeca, codo) de cada brazo o null si no se ve
    fun actualizar(det: Deteccion, ancho: Int, alto: Int, ahora: Double, manos: Array<Mano?>): Array<Pair<P, P>?> {
        val dt = anterior?.let { max(ahora - it, 1e-3) } ?: (1.0 / 30.0)
        anterior = ahora
        val pose = det.pose
        val resultado = arrayOfNulls<Pair<P, P>>(2)

        for (lado in 0..1) {
            manos[lado]?.let { m ->
                giroDeMano(m, lado)?.let { g ->
                    val lim = GIRO_MAXIMO / 180.0 * dt
                    val cambio = (g - giro[lado]).coerceIn(-lim, lim)
                    giro[lado] = 0.85 * giro[lado] + 0.15 * (giro[lado] + cambio)
                }
            }
            val codoI = if (lado == 0) POSE_CODO_IZQ else POSE_CODO_DER
            val munecaI = if (lado == 0) POSE_MUNECA_IZQ else POSE_MUNECA_DER
            if (pose.size <= POSE_MUNECA_DER || min(pose[codoI][2], pose[munecaI][2]) < 0.5f) {
                filtros[lado].reiniciar()
                continue
            }
            var muneca = P(pose[munecaI][0].toDouble() * ancho, pose[munecaI][1].toDouble() * alto)
            val codo = P(pose[codoI][0].toDouble() * ancho, pose[codoI][1].toDouble() * alto)
            manos[lado]?.let { muneca = it[MANO_MUNECA] }
            resultado[lado] = filtros[lado].filtrar(muneca, codo, dt)
        }
        return resultado
    }
}

// =============================================================================
// CARA
// =============================================================================

class FiltroCara {
    private val posicion = FiltroUnoEuro(CARA_CORTE_MINIMO, CARA_BETA)
    private val forma = FiltroUnoEuro(CARA_CORTE_MINIMO, CARA_BETA)
    private var anterior: Double? = null

    fun actualizar(det: Deteccion, ancho: Int, alto: Int, ahora: Double): Array<P>? {
        val dt = anterior?.let { max(ahora - it, 1e-3) } ?: (1.0 / 30.0)
        anterior = ahora
        if (det.cara.size <= CARA_OJO_B) { posicion.reiniciar(); forma.reiniciar(); return null }
        val p = aPixeles(det.cara, ancho, alto)
        val escala = (p[CARA_OJO_A] - p[CARA_OJO_B]).norma()
        val n = posicion.filtrar(plano(arrayOf(p[CARA_NARIZ])), dt, escala)
        val nariz = P(n[0], n[1])
        val f = desplano(forma.filtrar(plano(Array(p.size) { p[it] - p[CARA_NARIZ] }), dt, escala))
        return Array(f.size) { f[it] + nariz }
    }
}

const val LABIO_SUPERIOR = 13
const val LABIO_INFERIOR = 14
const val UMBRAL_BOCA_ABIERTA = 0.12
const val UMBRAL_BOCA_CERRADA = 0.06

fun aperturaBoca(det: Deteccion, ancho: Int, alto: Int): Double? {
    if (det.cara.size <= CARA_OJO_B) return null
    val p = aPixeles(det.cara, ancho, alto)
    val ref = (p[CARA_OJO_A] - p[CARA_OJO_B]).norma()
    if (ref < 1e-6) return null
    return (p[LABIO_SUPERIOR] - p[LABIO_INFERIOR]).norma() / ref
}

fun nivelBoca(apertura: Double?, cerrada: Double, abierta: Double): Double =
    if (apertura == null) 0.0 else ((apertura - cerrada) / max(abierta - cerrada, 1e-6)).coerceIn(0.0, 1.0)

// ---- puntos extra de la cara (malla, boca, cejas, nariz y orejas) ----

const val PUNTOS_POR_ARISTA = 1
const val PUNTOS_DETALLE = 6
const val PUNTOS_NARIZ = 1
const val PUNTOS_OREJA = 26
const val CARA_FRENTE = 10
const val CARA_MENTON = 152
const val CARA_LADO_A = 234
const val CARA_LADO_B = 454

class AristasCara(val malla: Array<IntArray>, val detalle: Array<IntArray>, val nariz: Array<IntArray>)

private fun puntosEnAristas(p: Array<P>, aristas: Array<IntArray>, cantidad: Int, salida: MutableList<P>) {
    if (aristas.isEmpty() || cantidad <= 0) return
    for (k in 1..cantidad) {
        val t = k / (cantidad + 1.0)
        for (ar in aristas) {
            if (ar[0] >= p.size || ar[1] >= p.size) return
            val a = p[ar[0]]; val b = p[ar[1]]
            salida.add(a + (b - a) * t)
        }
    }
}


fun puntosOrejas(p: Array<P>, salida: MutableList<P>) {
    if (p.size <= max(CARA_LADO_B, CARA_MENTON)) return
    var arriba = p[CARA_FRENTE] - p[CARA_MENTON]
    val altoCara = arriba.norma()
    if (altoCara < 1e-6) return
    arriba = arriba / altoCara
    val medio = (p[CARA_LADO_A] + p[CARA_LADO_B]) / 2.0
    val nariz = p[CARA_NARIZ]
    val dA = (nariz - p[CARA_LADO_A]).norma()
    val dB = (nariz - p[CARA_LADO_B]).norma()

    for ((lado, dLado) in listOf(CARA_LADO_A to dA, CARA_LADO_B to dB)) {
        val visible = dLado / max(dA + dB, 1e-6)
        if (visible < 0.28) continue
        val apertura = min(1.0, visible / 0.5)
        var afuera = p[lado] - medio
        val norma = afuera.norma()
        if (norma < 1e-6) continue
        afuera = afuera / norma
        val altoOreja = altoCara * 0.30
        val anchoOreja = altoOreja * 0.42 * apertura
        val centro = p[lado] + afuera * (anchoOreja * 0.45) + arriba * (altoOreja * 0.10)
        for (t in linspace(-PI / 2 - 0.35, PI / 2 + 0.35, PUNTOS_OREJA))
            salida.add(centro + afuera * (cos(t) * anchoOreja / 2) + arriba * (sin(t) * altoOreja / 2))
        for (t in linspace(-PI / 2 + 0.2, PI / 2 - 0.1, max(6, PUNTOS_OREJA / 2)))
            salida.add(centro + afuera * (cos(t) * anchoOreja * 0.22) + arriba * (sin(t) * altoOreja * 0.28 + altoOreja * 0.04))
    }
}

fun puntosDensosCara(p: Array<P>, aristas: AristasCara): List<P> {
    val salida = p.toMutableList()
    puntosEnAristas(p, aristas.malla, PUNTOS_POR_ARISTA, salida)
    puntosEnAristas(p, aristas.detalle, PUNTOS_DETALLE, salida)
    puntosEnAristas(p, aristas.nariz, PUNTOS_NARIZ, salida)
    puntosOrejas(p, salida)
    return salida
}
