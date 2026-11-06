package co.edu.itm.manosquehablan.aprender

import co.edu.itm.manosquehablan.vivo.MANO_MUNECA
import co.edu.itm.manosquehablan.vivo.Mano
import co.edu.itm.manosquehablan.vivo.P
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

// Escenas y gestos del muñeco en APRENDER.
// Coordenadas en anchos de hombro, dx positivo = hacia la derecha de la pantalla.

const val FPS_ESCENA = 30.0
// tiempos y probabilidades de los eventos cuando esta quieto
const val SEGUNDOS_QUIETO_PARA_DESPEDIRSE = 12.0
const val MAXIMO_SIN_CHICA = 4
const val PROBABILIDAD_DESPEDIRSE = 0.10
const val PROBABILIDAD_ASOMARSE = 0.10
const val PROBABILIDAD_RELEVO_AL_TOCAR = 0.10
const val CUADROS_ASOMO = 84

enum class TipoEscena { DESPEDIDA, EMPUJON, BESO }

// brazo en un cuadro: muñeca, codo, angulo de los dedos y forma de la mano
private class Brazo(val muneca: P, val codo: P, val angulo: Double, val forma: Mano?)

private fun anguloDe(forma: Mano?) = forma?.let { atan2(it[9].y, it[9].x) } ?: (PI / 2)

private fun girar(v: P, a: Double): P { val c = cos(a); val s = sin(a); return P(c * v.x - s * v.y, s * v.x + c * v.y) }

private fun diferenciaAngulo(a: Double, b: Double): Double {
    var d = b - a
    while (d > PI) d -= 2 * PI
    while (d < -PI) d += 2 * PI
    return d
}

private fun suave(t: Double) = t.coerceIn(0.0, 1.0).let { it * it * (3 - 2 * it) }

private fun mezclar(a: Brazo, b: Brazo, t: Double): Brazo {
    val k = suave(t)
    return Brazo(a.muneca + (b.muneca - a.muneca) * k, a.codo + (b.codo - a.codo) * k,
        a.angulo + diferenciaAngulo(a.angulo, b.angulo) * k, mezclarForma(a.forma, b.forma, k))
}

// mezcla dos formas de mano
private fun mezclarForma(a: Mano?, b: Mano?, k: Double): Mano? {
    if (a == null || b == null || a.size != b.size || k <= 0.0) return a ?: b
    if (k >= 1.0) return b
    // se alinean antes de mezclar
    val giro = diferenciaAngulo(anguloDe(b), anguloDe(a))
    return Array(a.size) { a[it] + (girar(b[it], giro) - a[it]) * k }
}

// mano relajada
private fun manoNeutral(lado: Int): Mano? =
    manoPlantilla[lado] ?: manoPlantilla[1 - lado]?.let { h -> Array(h.size) { P(-h[it].x, h[it].y) } }

// brazo en el ultimo cuadro de una seña
private fun brazoFinal(s: AvatarSena, lado: Int): Brazo? {
    val d = s.lados[lado] ?: return null
    val f = d.mano?.last()?.let { h -> Array(h.size) { h[it] - h[MANO_MUNECA] } }
    return Brazo(d.muneca.last(), d.codo.last(), anguloDe(f), f)
}

private fun signo(lado: Int) = if (lado == 0) 1.0 else -1.0

// brazo abajo con la mano relajada
@Suppress("UNUSED_PARAMETER")
private fun reposo(lado: Int, forma: Mano?) = Brazo(P(signo(lado) * 0.55, 1.12), P(signo(lado) * 0.64, 0.56), PI / 2, manoNeutral(lado) ?: forma)

// mano derecha arriba para saludar
private fun arriba(forma: Mano?) = Brazo(P(-0.88, -0.52), P(-0.92, 0.18), -PI / 2, forma)

// mano derecha adelante
private fun llamando(forma: Mano?) = Brazo(P(-0.78, -0.30), P(-0.90, 0.25), -PI / 2, forma)

// mano izquierda estirada para empujar
private fun empujando(forma: Mano?) = Brazo(P(1.08, -0.02), P(0.80, 0.12), 0.0, forma)

// convierte los brazos de cada cuadro en una AvatarSena
private fun aSena(cuadros: List<Pair<Brazo?, Brazo?>>): AvatarSena {
    val lados = Array(2) { l ->
        if (cuadros.any { (if (l == 0) it.first else it.second) == null }) return@Array null
        val brazos = cuadros.map { if (l == 0) it.first!! else it.second!! }
        LadoAvatar(
            brazos.map { it.muneca }.toTypedArray(),
            brazos.map { it.codo }.toTypedArray(),
            if (brazos.all { it.forma != null }) brazos.map { b ->
                val giro = diferenciaAngulo(anguloDe(b.forma), b.angulo)
                Array(b.forma!!.size) { b.muneca + girar(b.forma[it], giro) }
            }.toTypedArray() else null
        )
    }
    return AvatarSena(lados, cuadros.size, FPS_ESCENA, sinTrazo = true)
}

// personaje de una escena con sus datos por cuadro
class Actor(val apariencia: Apariencia, val sena: AvatarSena, val dx: FloatArray, val sonrie: BooleanArray, val beso: FloatArray)

// posicion, tamaño y transparencia del corazon
class CorazonEscena(val dx: Float, val y: Float, val tam: Float, val alfa: Float)

// escena completa, al final queda el personaje 'nueva'
class Escena(
    val total: Int, val actores: List<Actor>, val corazon: (Int) -> CorazonEscena?,
    val nueva: Apariencia, val poseFinal: AvatarSena
)

// Cambio de personaje:
// DESPEDIDA: llega el otro saludando, le da un beso y el primero se va
// EMPUJON: llega el otro, lo empuja y el primero se va
// BESO: llega el otro, le da un beso y el primero se va
fun escenaRelevo(tipo: TipoEscena, actual: Apariencia, pose: AvatarSena): Escena {
    val nueva = if (actual.mujer) HOMBRE else MUJER
    val formaIzq = brazoFinal(pose, 0)?.forma
    val formaDer = brazoFinal(pose, 1)?.forma
    val sale0 = Array(2) { l -> brazoFinal(pose, l) ?: reposo(l, if (l == 0) formaIzq else formaDer) }
    val reposoNueva = Array(2) { l -> reposo(l, if (l == 0) formaIzq else formaDer) }

    fun s(seg: Double) = (seg * FPS_ESCENA).toInt()
    fun tramo(f: Int, a: Int, b: Int) = ((f - a).toDouble() / maxOf(1, b - a)).coerceIn(0.0, 1.0)

    val total: Int
    val brazosSale = ArrayList<Pair<Brazo?, Brazo?>>(); val brazosEntra = ArrayList<Pair<Brazo?, Brazo?>>()
    val dxSale = ArrayList<Float>(); val dxEntra = ArrayList<Float>()
    val sonrieSale = ArrayList<Boolean>()
    // inclinacion de las cabezas para el beso
    val besoSale = ArrayList<Float>(); val besoEntra = ArrayList<Float>()
    fun besar(k: Float) { besoSale += k; besoEntra += -k }
    var corazon: (Int) -> CorazonEscena? = { null }
    val fuera = 4.6f

    when (tipo) {
        TipoEscena.DESPEDIDA -> {
            total = s(8.0)
            for (f in 0 until total) {
                // el que se va baja los brazos, recibe el beso y sale por la izquierda
                val izq = mezclar(sale0[0], reposo(0, sale0[0].forma), tramo(f, 0, s(0.8)))
                val der = mezclar(sale0[1], reposo(1, sale0[1].forma), tramo(f, 0, s(0.8)))
                brazosSale += izq to der
                dxSale += (-fuera * suave(tramo(f, s(5.6), s(7.4)))).toFloat()
                sonrieSale += f >= s(2.4)

                // el que llega entra por la derecha, saluda, da el beso y pasa al centro
                val entra = 1.45f + (fuera - 1.45f) * (1 - suave(tramo(f, s(0.9), s(2.3))).toFloat())
                val acerca = (1.45f - 0.62f) * suave(tramo(f, s(4.0), s(4.6))).toFloat()
                besar(suave(tramo(f, s(4.6), s(4.9)) - tramo(f, s(5.4), s(5.7))).toFloat())
                val alCentro = suave(tramo(f, s(6.2), s(7.8))).toFloat()
                dxEntra += (entra - acerca) * (1 - alCentro)
                // saludo
                var derN = reposoNueva[1]
                if (f in s(1.6) until s(4.6)) {
                    val subirN = tramo(f, s(1.6), s(2.1))
                    val baja = tramo(f, s(4.0), s(4.6))
                    val v = sin((f - s(1.6)) * 2 * PI / (FPS_ESCENA / 2.5)) * 0.09
                    val a = arriba(reposoNueva[1].forma)
                    val saluda = Brazo(a.muneca + P(v, 0.0), a.codo + P(v * 0.3, 0.0), a.angulo + v * 2.5, a.forma)
                    derN = mezclar(mezclar(reposoNueva[1], saluda, subirN), reposoNueva[1], baja)
                }
                brazosEntra += reposoNueva[0] to derN
            }
            corazon = { f ->
                if (f in s(4.85) until s(6.5)) {
                    val t = tramo(f, s(4.85), s(6.5)).toFloat()
                    CorazonEscena(0.31f, -0.95f - 0.5f * t, 0.14f + 0.06f * t, 1f - t * t)
                } else null
            }
        }
        TipoEscena.EMPUJON -> {
            total = s(3.8)
            for (f in 0 until total) {
                val entra = 1.2f + (fuera - 1.2f) * (1 - suave(tramo(f, 0, s(1.3))).toFloat())
                val alCentro = suave(tramo(f, s(2.4), s(3.6))).toFloat()
                dxEntra += entra * (1 - alCentro)
                // estira la mano para empujar
                val estira = tramo(f, s(1.1), s(1.6)) - tramo(f, s(2.1), s(2.6))
                brazosEntra += mezclar(reposoNueva[0], empujando(reposoNueva[0].forma), estira) to reposoNueva[1]
                // el empujado se corre y sale
                val golpe = suave(tramo(f, s(1.5), s(1.8))) * 0.35
                dxSale += (-golpe - (fuera - 0.35) * suave(tramo(f, s(1.9), s(3.4)))).toFloat()
                brazosSale += mezclar(sale0[0], reposo(0, sale0[0].forma), tramo(f, s(1.5), s(2.2))) to
                    mezclar(sale0[1], reposo(1, sale0[1].forma), tramo(f, s(1.5), s(2.2)))
                sonrieSale += false
                besar(0f)
            }
        }
        TipoEscena.BESO -> {
            total = s(4.6)
            for (f in 0 until total) {
                // llega cerca, beso y pasa al centro
                val entra = 0.62f + (fuera - 0.62f) * (1 - suave(tramo(f, 0, s(1.4))).toFloat())
                besar(suave(tramo(f, s(1.4), s(1.7)) - tramo(f, s(2.3), s(2.6))).toFloat())
                val alCentro = suave(tramo(f, s(3.0), s(4.4))).toFloat()
                dxEntra += entra * (1 - alCentro)
                brazosEntra += reposoNueva[0] to reposoNueva[1]
                dxSale += (-fuera * suave(tramo(f, s(2.6), s(4.2)))).toFloat()
                brazosSale += mezclar(sale0[0], reposo(0, sale0[0].forma), tramo(f, s(1.6), s(2.4))) to
                    mezclar(sale0[1], reposo(1, sale0[1].forma), tramo(f, s(1.6), s(2.4)))
                sonrieSale += f >= s(1.6)
            }
            // corazon
            corazon = { f ->
                if (f in s(1.65) until s(3.3)) {
                    val t = tramo(f, s(1.65), s(3.3)).toFloat()
                    CorazonEscena(0.31f, -0.95f - 0.5f * t, 0.14f + 0.06f * t, 1f - t * t)
                } else null
            }
        }
    }

    if (besoSale.isEmpty()) repeat(total) { besar(0f) }
    val actorSale = Actor(actual, aSena(brazosSale), dxSale.toFloatArray(), sonrieSale.toBooleanArray(), besoSale.toFloatArray())
    val actorEntra = Actor(nueva, aSena(brazosEntra), dxEntra.toFloatArray(), BooleanArray(total), besoEntra.toFloatArray())
    val poseFinal = aSena(listOf(brazosEntra.last()))
    return Escena(total, listOf(actorSale, actorEntra), corazon, nueva, poseFinal)
}

// personaje a dibujar en un cuadro
class Figura(val apariencia: Apariencia, val sena: AvatarSena, val cuadro: Int, val dx: Float = 0f, val sonrie: Boolean = false, val beso: Float = 0f)

// un cuadro de una seña como pose
fun poseEn(s: AvatarSena, cuadro: Int): AvatarSena {
    val c = cuadro.coerceIn(0, s.T - 1)
    val lados = Array(2) { l ->
        val d = s.lados[l] ?: return@Array null
        LadoAvatar(arrayOf(d.muneca[minOf(c, d.muneca.size - 1)]), arrayOf(d.codo[minOf(c, d.codo.size - 1)]),
            d.mano?.let { h -> arrayOf(h[minOf(c, h.size - 1)]) })
    }
    return AvatarSena(lados, 1, s.fps, sinTrazo = true)
}

// =============================================================================
// GESTOS CUANDO ESTA QUIETO
// =============================================================================

enum class GestoQuieto { RASCARSE, BOSTEZAR, ESTIRARSE, RELOJ, PENSAR, SALUDAR, APLAUDIR }

private fun hombro(lado: Int) = P(signo(lado) * 0.5, 0.0)

// brazo con la punta del indice en el objetivo
private fun tocar(lado: Int, forma: Mano?, objetivo: P, angulo: Double): Brazo {
    val f = forma ?: manoNeutral(lado)
    val punta = f?.let { girar(it[8], diferenciaAngulo(anguloDe(it), angulo)) } ?: P(cos(angulo) * 0.2, sin(angulo) * 0.2)
    val muneca = objetivo - punta
    val codo = hombro(lado) + (muneca - hombro(lado)) * 0.5 + P(signo(lado) * 0.36, 0.30)
    return Brazo(muneca, codo, angulo, f)
}

private fun cara(sonrisa: Float = 0f, cejas: FloatArray = floatArrayOf(0f, 0f), mx: Float = 0f, my: Float = 0f,
                 bostezo: Float = 0f, ojos: Float = 0f) = Gestos(ojos, sonrisa, cejas, mx, my, 0f, 0f, bostezo)

// arma la animacion de un gesto desde la pose actual y vuelve a ella
fun gestoQuieto(tipo: GestoQuieto, pose: AvatarSena, rng: kotlin.random.Random): AvatarSena? {
    if (tipo == GestoQuieto.RASCARSE) return gestoRascarse(pose, rng)
    val inicio = Array(2) { l -> brazoFinal(pose, l) ?: reposo(l, null) }
    fun s(seg: Double) = (seg * FPS_ESCENA).toInt()
    fun tramo(f: Int, a: Int, b: Int) = ((f - a).toDouble() / maxOf(1, b - a)).coerceIn(0.0, 1.0)
    // 0..1: sube, se mantiene y baja
    fun nivel(f: Int, ir: Int, volver: Int, total: Int) = tramo(f, 0, ir) - tramo(f, volver, total)

    val total: Int
    val brazos = ArrayList<Pair<Brazo, Brazo>>()
    val caras = ArrayList<Gestos>()
    when (tipo) {
        GestoQuieto.BOSTEZAR -> {
            total = s(3.2)
            for (f in 0 until total) {
                val k = nivel(f, s(0.6), s(2.5), total)
                val tapa = tocar(1, inicio[1].forma, P(-0.03, -0.60), -PI / 2 - 0.5)
                brazos += inicio[0] to mezclar(inicio[1], tapa, k)
                val b = suave(tramo(f, s(0.3), s(1.0)) - tramo(f, s(2.0), s(2.6))).toFloat()
                caras += cara(bostezo = b, cejas = floatArrayOf(0.5f * b, 0.5f * b))
            }
        }
        GestoQuieto.ESTIRARSE -> {
            total = s(3.4)
            for (f in 0 until total) {
                val k = nivel(f, s(0.8), s(2.5), total)
                val arribaI = Brazo(P(0.30, -1.15), P(0.62, -0.55), -PI / 2, manoNeutral(0) ?: inicio[0].forma)
                val arribaD = Brazo(P(-0.30, -1.15), P(-0.62, -0.55), -PI / 2, manoNeutral(1) ?: inicio[1].forma)
                brazos += mezclar(inicio[0], arribaI, k) to mezclar(inicio[1], arribaD, k)
                val kk = suave(k).toFloat()
                caras += cara(bostezo = 0.45f * kk, ojos = 0.9f * kk)
            }
        }
        GestoQuieto.RELOJ -> {
            total = s(3.0)
            for (f in 0 until total) {
                val k = nivel(f, s(0.6), s(2.3), total)
                // brazo izquierdo delante del pecho
                val reloj = Brazo(P(0.05, 0.30), P(0.62, 0.42), PI, manoNeutral(0) ?: inicio[0].forma)
                brazos += mezclar(inicio[0], reloj, k) to inicio[1]
                val kk = suave(k).toFloat()
                caras += cara(mx = 0.5f * kk, my = 1f * kk, cejas = floatArrayOf(0.3f * kk, 0f))
            }
        }
        GestoQuieto.PENSAR -> {
            total = s(3.4)
            for (f in 0 until total) {
                val k = nivel(f, s(0.6), s(2.7), total)
                // mano en la barbilla
                val golpe = if (f in s(0.6) until s(2.7)) sin((f - s(0.6)) * 2 * PI / 12.0) * 0.015 else 0.0
                val barbilla = tocar(1, inicio[1].forma, P(-0.02, -0.48 + golpe), -PI / 2 + 0.25)
                brazos += inicio[0] to mezclar(inicio[1], barbilla, k)
                val kk = suave(k).toFloat()
                caras += cara(mx = -0.7f * kk, my = -0.8f * kk, cejas = floatArrayOf(0f, 0.8f * kk))
            }
        }
        GestoQuieto.SALUDAR -> {
            total = s(2.8)
            for (f in 0 until total) {
                val k = nivel(f, s(0.5), s(2.2), total)
                val v = sin(f * 2 * PI / (FPS_ESCENA / 2.5)) * 0.09
                val a = arriba(manoNeutral(1) ?: inicio[1].forma)
                val hola = Brazo(a.muneca + P(v, 0.0), a.codo + P(v * 0.3, 0.0), a.angulo + v * 2.5, a.forma)
                brazos += inicio[0] to mezclar(inicio[1], hola, k)
                caras += cara(sonrisa = suave(k).toFloat(), cejas = floatArrayOf(0.4f * suave(k).toFloat(), 0.4f * suave(k).toFloat()))
            }
        }
        GestoQuieto.APLAUDIR -> {
            total = s(3.0)
            for (f in 0 until total) {
                val k = nivel(f, s(0.5), s(2.4), total)
                // manos juntandose y separandose
                val abre = 0.06 + 0.07 * (1 + cos(f * 2 * PI / 10.0)) / 2
                val izq = Brazo(P(abre, 0.18), P(0.62, 0.45), PI / 2 + 0.9, manoNeutral(0) ?: inicio[0].forma)
                val der = Brazo(P(-abre, 0.18), P(-0.62, 0.45), PI / 2 - 0.9, manoNeutral(1) ?: inicio[1].forma)
                brazos += mezclar(inicio[0], izq, k) to mezclar(inicio[1], der, k)
                caras += cara(sonrisa = suave(k).toFloat())
            }
        }
        GestoQuieto.RASCARSE -> return null
    }
    val sena = aSena(brazos.map { it.first to it.second })
    return AvatarSena(sena.lados, sena.T, sena.fps, sinTrazo = true, caras = caras.toTypedArray())
}

// baja los dos brazos hasta el reposo
fun bajarBrazos(pose: AvatarSena): AvatarSena {
    val desde = Array(2) { l -> brazoFinal(pose, l) }
    val total = (0.7 * FPS_ESCENA).toInt()
    return aSena(List(total) { f ->
        val t = (f + 1).toDouble() / total
        val b = Array(2) { l -> desde[l]?.let { mezclar(it, reposo(l, it.forma), t) } ?: reposo(l, null) }
        b[0] to b[1]
    })
}

// pose con los brazos abajo
fun poseReposo(desde: AvatarSena): AvatarSena =
    aSena(listOf(reposo(0, brazoFinal(desde, 0)?.forma) to reposo(1, brazoFinal(desde, 1)?.forma)))
