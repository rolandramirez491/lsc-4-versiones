package co.edu.itm.manosquehablan.vivo

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sin

// Arma la muestra de una grabacion: trazos, extras y trazo del codo (construir_muestra_completa).

const val PUNTOS_TRAZO = 48
const val TAMANO_IMAGEN = 64
const val MEDIA_VENTANA = 1.5
const val GROSOR_LINEA = 2
const val MIN_FRAMES_MANO = 5
const val SEGUNDOS_MUESTRA = 3.4
const val SEGUNDOS_HUECO = 1.0
const val SALTO_MAXIMO = 0.25
const val BRAZO_BORDE = 0.3
const val MANO_FANTASMA_SEGUNDOS = 1.5
const val FRACCION_ANTEBRAZO = 1.0
const val RADIO_PINCEL = 0.06
const val SEGUNDOS_SUAVIZADO = 0.15
const val ANGULO_MAXIMO_HUECO = 45.0

// v13: centro de la mano + 5 dedos, la cara como "mapa" y la seña empieza sobre el codo
val CARA_REF = intArrayOf(1, 13, 14, 152, 10, 159, 145, 386, 374, 234, 454)
val CANTIDAD_CARA_REF = CARA_REF.size
const val CANTIDAD_LUGARES = 9            // nariz, boca, menton, frente, ojo a, ojo b, oreja a, oreja b, pecho
val DEDOS_PUNTAS = intArrayOf(4, 8, 12, 16, 20)
const val CARA_POR_MANO = 1 + CANTIDAD_LUGARES * 3                       // 28
// v14: la palma (sin dedos) hace el trazo; los dedos van como numeros
val PALMA_PUNTOS = intArrayOf(0, 1, 5, 9, 13, 17)   // muñeca y bases de los dedos
val DEDOS_BASES = intArrayOf(1, 5, 9, 13, 17)
const val MOMENTOS = 3                              // inicio, mitad y final de la seña
const val DEDOS_POR_MANO = MOMENTOS * 5 * 2 + 5 * 2  // (largo, estirado) x 3 momentos + puntas x/y = 40
// v16: el brazo como una CURVA en el tiempo (si se dobla, se estira, dos veces...)
const val PUNTOS_ANGULO = 12
const val DOBLEZ_MINIMO = 20.0
const val BRAZO_POR_MANO = 1 + PUNTOS_ANGULO * 3 + 3                      // 40
val LADO_HOMBRO = doubleArrayOf(0.5, -0.5)                               // hombro = centro + esto x ancho (en x)
const val EXTRAS_POR_MANO = 1 + 2 + 2 + 2 + 2 + 1 + DEDOS_POR_MANO + CARA_POR_MANO + BRAZO_POR_MANO   // 88
const val CANTIDAD_EXTRAS = EXTRAS_POR_MANO * 2                          // 176
const val E_CARA_VALIDA = 10 + DEDOS_POR_MANO                             // 50
const val E_BRAZO_VALIDO = E_CARA_VALIDA + CARA_POR_MANO                  // 78
const val TRAZOS_POR_MANO = 1
const val CANTIDAD_TRAZOS = 2 * TRAZOS_POR_MANO                          // 2: palma izq y der
const val CANALES_DIBUJO = 2

const val SALTO_MANO_CUADRO = 0.30
const val RECHAZOS_SEGUIDOS = 5
const val TAMANO_MANO_MIN = 0.55
const val TAMANO_MANO_MAX = 1.6
const val RADIO_CARA = 0.45
const val CAMBIO_TAMANO_CARA = 1.30

// redondeo al par, igual que round() de python
fun redondear(x: Double): Int = Math.rint(x).toInt()

// datos de un brazo en toda la grabacion
class Brazo(
    val munecas: Array<P>, val validas: BooleanArray,
    val manos: Array<Mano>, val manosOk: BooleanArray,
    val codos: Array<P>, val codosOk: BooleanArray
)

class Muestra(
    val trazos: Array<Array<P>>, val validos: BooleanArray, val extras: FloatArray,
    val trazosBrazo: Array<Array<P>>, val validosBrazo: BooleanArray
)

private fun vacia() = Array(CANTIDAD_MANO) { P(0.0, 0.0) }

// ---------------------------------------------------------------- utilidades

fun suavizar(p: Array<P>, ventana: Int = 3): Array<P> {
    if (p.size < ventana) return p
    val k = 1.0 / ventana
    return Array(p.size - ventana + 1) { i ->
        var x = 0.0; var y = 0.0
        for (j in 0 until ventana) { x += p[i + ventana - 1 - j].x * k; y += p[i + ventana - 1 - j].y * k }
        P(x, y)
    }
}

// como np.linspace
fun linspace(a: Double, b: Double, n: Int): DoubleArray {
    if (n == 1) return doubleArrayOf(a)
    val paso = (b - a) / (n - 1)
    return DoubleArray(n) { if (it == n - 1) b else a + it * paso }
}

// como np.interp
fun interp(x: Double, xp: DoubleArray, fp: DoubleArray): Double {
    val n = xp.size
    if (x <= xp[0]) return fp[0]
    if (x >= xp[n - 1]) return fp[n - 1]
    var lo = 0; var hi = n - 1
    while (hi - lo > 1) { val m = (lo + hi) / 2; if (xp[m] <= x) lo = m else hi = m }
    val pendiente = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo])
    return pendiente * (x - xp[lo]) + fp[lo]
}

fun remuestrearPorLongitud(p: Array<P>, cantidad: Int): Array<P> {
    val acumulado = DoubleArray(p.size)
    for (i in 1 until p.size) acumulado[i] = acumulado[i - 1] + (p[i] - p[i - 1]).norma()
    val total = acumulado.last()
    if (total < 1e-8) {
        val c = p.fold(P(0.0, 0.0)) { s, q -> s + q } / p.size.toDouble()
        return Array(cantidad) { c }
    }
    val xs = DoubleArray(p.size) { p[it].x }; val ys = DoubleArray(p.size) { p[it].y }
    return linspace(0.0, total, cantidad).map { P(interp(it, acumulado, xs), interp(it, acumulado, ys)) }.toTypedArray()
}

fun suavizarManos(manos: Array<Mano>, ok: BooleanArray, fps: Double): Array<Mano> {
    val filtro = FiltroMano()
    val dt = 1.0 / max(fps, 1.0)
    return Array(manos.size) { filtro.filtrar(manos[it], ok[it], dt) }
}

// ---------------------------------------------------------------- limpieza del recorrido

private fun quitarSaltos(p: Array<P>, v: BooleanArray, anchoHombros: Double, fps: Double) {
    val salto = SALTO_MAXIMO * anchoHombros * max(1.0, 30.0 / fps)
    val aceptarTras = max(2, redondear(0.2 * fps))
    var ultimo = -1; var rechazados = 0
    for (i in p.indices) {
        if (!v[i]) continue
        if (ultimo >= 0) {
            val pasos = i - ultimo
            if ((p[i] - p[ultimo]).norma() > salto * pasos && rechazados < aceptarTras) {
                v[i] = false; rechazados++; continue
            }
        }
        ultimo = i; rechazados = 0
    }
}

private fun direccion(p: Array<P>, v: BooleanArray, i: Int, hacia: Int, cantidad: Int): P? {
    val vecinos = mutableListOf<Int>()
    var j = i + hacia
    while (j in p.indices && vecinos.size < cantidad) { if (v[j]) vecinos.add(j); j += hacia }
    if (vecinos.isEmpty()) return null
    val lejano = vecinos.last()
    val vec = if (hacia < 0) p[i] - p[lejano] else p[lejano] - p[i]
    val n = vec.norma()
    return if (n > 1e-6) vec / n else null
}

private fun limitarAngulo(dir: P?, cuerda: P, maximoGrados: Double): P {
    if (dir == null) return cuerda
    val coseno = dir.punto(cuerda).coerceIn(-1.0, 1.0)
    if (coseno >= cos(Math.toRadians(maximoGrados))) return dir
    var lado = sign(cuerda.x * dir.y - cuerda.y * dir.x)
    if (lado == 0.0) lado = 1.0
    val ang = Math.toRadians(maximoGrados) * lado
    val c = cos(ang); val s = sin(ang)
    return P(c * cuerda.x - s * cuerda.y, s * cuerda.x + c * cuerda.y)
}

private fun rellenarHuecos(p: Array<P>, v: BooleanArray, fps: Double) {
    val maxHueco = max(1, redondear(SEGUNDOS_HUECO * fps))
    val cantidad = max(2, redondear(0.15 * fps))
    val indices = v.indices.filter { v[it] }
    for (n in 0 until indices.size - 1) {
        val a = indices[n]; val b = indices[n + 1]
        val largo = b - a
        if (largo < 2 || largo > maxHueco + 1) continue
        val p0 = p[a]; val p3 = p[b]
        val cuerda = p3 - p0
        val distancia = cuerda.norma()
        if (distancia < 1e-6) { for (i in a + 1 until b) { p[i] = p0; v[i] = true }; continue }
        val unitaria = cuerda / distancia
        val salida = limitarAngulo(direccion(p, v, a, -1, cantidad), unitaria, ANGULO_MAXIMO_HUECO)
        val llegada = limitarAngulo(direccion(p, v, b, +1, cantidad), unitaria, ANGULO_MAXIMO_HUECO)
        val brazo = distancia / 3.0
        val p1 = p0 + salida * brazo
        val p2 = p3 - llegada * brazo

        val ts = linspace(0.0, 1.0, 200)
        val curva = ts.map { t ->
            val u = 1 - t
            p0 * (u * u * u) + p1 * (3 * u * u * t) + p2 * (3 * u * t * t) + p3 * (t * t * t)
        }
        val acum = DoubleArray(200)
        for (i in 1 until 200) acum[i] = acum[i - 1] + (curva[i] - curva[i - 1]).norma()
        val cx = DoubleArray(200) { curva[it].x }; val cy = DoubleArray(200) { curva[it].y }
        val objetivos = linspace(0.0, acum.last(), largo + 1)
        for (k in 1 until largo) {
            p[a + k] = P(interp(objetivos[k], acum, cx), interp(objetivos[k], acum, cy))
            v[a + k] = true
        }
    }
}

private fun suavizarTramos(p: Array<P>, v: BooleanArray, fps: Double) {
    val ventana = max(3, redondear(SEGUNDOS_SUAVIZADO * fps) or 1)
    val mitad = ventana / 2
    val k = 1.0 / ventana
    var i = 0
    while (i < p.size) {
        if (!v[i]) { i++; continue }
        var j = i
        while (j < p.size && v[j]) j++
        val tramo = p.copyOfRange(i, j)
        if (tramo.size >= 3) {
            val relleno = Array(tramo.size + 2 * mitad) { tramo[(it - mitad).coerceIn(0, tramo.size - 1)] }
            for (t in tramo.indices) {
                var x = 0.0; var y = 0.0
                for (w in 0 until ventana) { x += relleno[t + ventana - 1 - w].x * k; y += relleno[t + ventana - 1 - w].y * k }
                p[i + t] = P(x, y)
            }
        }
        i = j
    }
}

private fun usarMano(p: Array<P>, v: BooleanArray, manos: Array<Mano>?, manoOk: BooleanArray?, fps: Double) {
    if (manoOk == null) return
    val n = manoOk.size
    if (n != p.size) return
    if (manos == null || manos.size != n) { manoOk.copyInto(v); return }

    val brazo = p.copyOf(); val brazoOk = v.copyOf()
    for (i in 0 until n) if (manoOk[i]) p[i] = manos[i][MANO_MUNECA]
    val nueva = manoOk.copyOf()

    val ambos = BooleanArray(n) { manoOk[it] && brazoOk[it] }
    val maxHueco = max(1, redondear(SEGUNDOS_HUECO * fps))
    val borde = max(1, redondear(BRAZO_BORDE * fps))
    val indicesAmbos = (0 until n).filter { ambos[it] }

    if (indicesAmbos.isNotEmpty()) {
        val dif = Array(n) { if (ambos[it]) p[it] - brazo[it] else P(0.0, 0.0) }
        for (i in 0 until n) {
            if (manoOk[i] || !brazoOk[i]) continue
            val a = indicesAmbos.lastOrNull { it < i }
            val b = indicesAmbos.firstOrNull { it > i }
            val correccion = when {
                a != null && b != null && b - a <= maxHueco + 1 -> {
                    val t = (i - a).toDouble() / (b - a)
                    dif[a] * (1 - t) + dif[b] * t
                }
                a != null && b == null && i - a <= borde -> dif[a]
                b != null && a == null && b - i <= borde -> dif[b]
                else -> continue
            }
            p[i] = brazo[i] + correccion
            nueva[i] = true
        }
    }
    nueva.copyInto(v)
}

private fun pincelPerezoso(p: Array<P>, v: BooleanArray, anchoHombros: Double) {
    val radio = RADIO_PINCEL * anchoHombros
    var pincel: P? = null
    for (i in p.indices) {
        if (!v[i]) { pincel = null; continue }
        val pc = pincel
        pincel = if (pc == null) p[i] else {
            val dif = p[i] - pc
            val dist = dif.norma()
            if (dist > radio) pc + dif * ((dist - radio) / dist) else pc
        }
        p[i] = pincel!!
    }
}

// limpia el recorrido de la muñeca, devuelve copias de puntos y validos
fun limpiarRecorrido(
    puntos: Array<P>, valida: BooleanArray, anchoHombros: Double, fps0: Double,
    manos: Array<Mano>? = null, manoOk: BooleanArray? = null
): Pair<Array<P>, BooleanArray> {
    val p = puntos.copyOf(); val v = valida.copyOf()
    val fps = max(fps0, 1.0)
    usarMano(p, v, manos, manoOk, fps)
    if (p.isEmpty() || v.none { it } || anchoHombros < 1e-6) return p to v
    quitarSaltos(p, v, anchoHombros, fps)
    rellenarHuecos(p, v, fps)
    pincelPerezoso(p, v, anchoHombros)
    suavizarTramos(p, v, fps)
    return p to v
}

fun recorridoAntebrazo(munecas: Array<P>, munecasOk: BooleanArray, codos: Array<P>, codosOk: BooleanArray,
                       anchoHombros: Double, fps: Double): Pair<Array<P>, BooleanArray> {
    val p = Array(munecas.size) { munecas[it] + (codos[it] - munecas[it]) * FRACCION_ANTEBRAZO }
    val v = BooleanArray(munecas.size) { munecasOk[it] && codosOk[it] }
    return limpiarRecorrido(p, v, anchoHombros, fps)
}

// ---------------------------------------------------------------- partes de la muestra (v13)

private fun medianaD(valores: DoubleArray): Double {
    val s = valores.sorted(); val n = s.size
    return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
}

private fun centroMano(m: Mano): P {
    var x = 0.0; var y = 0.0
    for (q in m) { x += q.x; y += q.y }
    return P(x / m.size, y / m.size)
}

// la seña empieza cuando la mano sube por encima de su codo; antes no se guarda nada
const val MOVIMIENTO_MINIMO_REPOSO = 0.35   // anchos de hombro: una mano abajo que se mueve menos esta en reposo
// si en una grabacion NINGUNA mano queda, se vuelve a armar sin quitar la de reposo
private val quitarReposo = booleanArrayOf(true)

private fun <T> sinQuedarVacio(hayAlgo: (T) -> Boolean, armar: () -> T): T {
    quitarReposo[0] = true
    val r = armar()
    if (hayAlgo(r)) return r
    quitarReposo[0] = false
    try { return armar() } finally { quitarReposo[0] = true }
}

fun indicesActivos(
    p: Array<P>, v: BooleanArray, centro: P, anchoHombros: Double,
    codos: Array<P>? = null, codosOk: BooleanArray? = null, quitarManoReposo: Boolean = quitarReposo[0]
): List<Int> {
    val alturaHombros = centro.y
    val linea = alturaHombros + UMBRAL_TORSO * anchoHombros
    val indices = p.indices.filter { v[it] && p[it].y < linea }
    if (indices.isEmpty()) return indices
    var tramo = (indices.first()..indices.last()).filter { v[it] }
    if (tramo.size < 2) return tramo
    val q = tramo.map { p[it] }
    fun esViaje(a: P, b: P) = abs(b.y - a.y) > 2.0 * abs(b.x - a.x) && max(a.y, b.y) > alturaHombros
    var inicio = 0
    while (inicio < q.size - 1 && q[inicio + 1].y < q[inicio].y && esViaje(q[inicio], q[inicio + 1])) inicio++
    var fin = q.size - 1
    while (fin > inicio && q[fin].y > q[fin - 1].y && esViaje(q[fin - 1], q[fin])) fin--
    tramo = tramo.subList(inicio, fin + 1)
    if (codos != null && codosOk != null && tramo.isNotEmpty()) {
        val primero = tramo.indexOfFirst { codosOk[it] && p[it].y < codos[it].y }
        if (primero >= 0) tramo = tramo.subList(primero, tramo.size)
        // v17: la mano nunca subio por encima de su codo y casi no se movio -> es la mano en reposo
        // (si se movio, cuenta aunque este abajo: ESTOMAGO, FALDA, PANTALON...)
        else if (tramo.any { codosOk[it] } && quitarManoReposo) {
            val mx = tramo.maxOf { p[it].x } - tramo.minOf { p[it].x }
            val my = tramo.maxOf { p[it].y } - tramo.minOf { p[it].y }
            if (max(mx, my) / max(anchoHombros, 1e-6) < MOVIMIENTO_MINIMO_REPOSO) return emptyList()
        }
    }
    return tramo
}

fun completarManosTapadas(manos: Array<Mano>, ok: BooleanArray, munecas: Array<P>, munecasOk: BooleanArray,
                          fps: Double): Pair<Array<Mano>, BooleanArray> {
    val salida = manos.copyOf()
    if (ok.none { it }) return salida to ok.copyOf()
    val limite = max(1, redondear(MANO_FANTASMA_SEGUNDOS * fps))
    val salidaOk = ok.copyOf()
    var ultimaForma: Mano? = null; var ultimoI = 0
    for (i in ok.indices) {
        if (ok[i]) {
            ultimaForma = Array(CANTIDAD_MANO) { manos[i][it] - manos[i][MANO_MUNECA] }; ultimoI = i
        } else if (ultimaForma != null && munecasOk[i] && i - ultimoI <= limite) {
            val f = ultimaForma
            salida[i] = Array(CANTIDAD_MANO) { munecas[i] + f[it] }
            salidaOk[i] = true
        }
    }
    return salida to salidaOk
}

// centro de la palma: muñeca y las 5 bases de los dedos (sin las puntas)
fun centroPalma(m: Mano): P {
    var x = 0.0; var y = 0.0
    for (j in PALMA_PUNTOS) { x += m[j].x; y += m[j].y }
    return P(x / PALMA_PUNTOS.size, y / PALMA_PUNTOS.size)
}

// por cada dedo: largo (punta a la palma), estirado (base->punta / suma de huesos) y la punta x/y, en tamaños de mano
private fun medidasDedos(m: Mano): Triple<DoubleArray, DoubleArray, Array<P>> {
    val tam = max((m[MANO_BASE_MEDIO] - m[MANO_MUNECA]).norma(), 1e-6)
    val palma = centroPalma(m)
    val largo = DoubleArray(5); val estirado = DoubleArray(5)
    val puntas = Array(5) { P(0.0, 0.0) }
    for (k in 0 until 5) {
        val base = DEDOS_BASES[k]; val punta = DEDOS_PUNTAS[k]
        var huesos = 0.0
        for (j in base until punta) huesos += (m[j + 1] - m[j]).norma()
        largo[k] = (m[punta] - palma).norma() / tam
        estirado[k] = (m[punta] - m[base]).norma() / max(huesos, 1e-6)
        puntas[k] = (m[punta] - palma) / tam
    }
    return Triple(largo, estirado, puntas)
}

// como np.interp: une los puntos (xp, fp) en linea recta; fuera de los extremos repite el extremo
private fun interpolar(x: Double, xp: DoubleArray, fp: DoubleArray): Double {
    if (x <= xp[0]) return fp[0]
    if (x >= xp[xp.size - 1]) return fp[fp.size - 1]
    var j = 0
    while (j < xp.size - 2 && xp[j + 1] <= x) j++
    val t = (x - xp[j]) / (xp[j + 1] - xp[j])
    return fp[j] + t * (fp[j + 1] - fp[j])
}

// v16: el brazo como curva a lo largo de la seña (PUNTOS_ANGULO puntos): angulo del codo / 180
// (hombro->codo con codo->muñeca), direccion del antebrazo x, y; + minimo, maximo y cuantas veces se dobla.
// El hombro sale del centro de hombros +- medio ancho.
fun angulosBrazo(b: Brazo, indices: List<Int>, lado: Int, centroHombros: P, anchoHombros: Double): Pair<FloatArray, Boolean> {
    val vacio = FloatArray(BRAZO_POR_MANO - 1)
    if (indices.isEmpty()) return vacio to false
    val ok = BooleanArray(b.munecas.size) { b.validas[it] && b.codosOk[it] }
    val todos = indices.filter { ok[it] }
    if (todos.isEmpty()) return vacio to false
    val hombro = P(centroHombros.x + LADO_HOMBRO[lado] * anchoHombros, centroHombros.y)

    // angulo y direccion en cada cuadro que se vio
    val ang = DoubleArray(todos.size); val dx = DoubleArray(todos.size); val dy = DoubleArray(todos.size)
    for ((k, i) in todos.withIndex()) {
        val arriba = hombro - b.codos[i]; val abajo = b.munecas[i] - b.codos[i]
        val la = max(arriba.norma(), 1e-6); val lb = max(abajo.norma(), 1e-6)
        val c = ((arriba.x * abajo.x + arriba.y * abajo.y) / (la * lb)).coerceIn(-1.0, 1.0)
        ang[k] = Math.toDegrees(kotlin.math.acos(c))
        dx[k] = abajo.x / lb; dy[k] = abajo.y / lb
    }

    // serie cuadro a cuadro del inicio al final (huecos en linea recta) y suavizada con 5 cuadros
    val i0 = indices.first(); val i1 = indices.last()
    val cuadros = DoubleArray(i1 - i0 + 1) { (i0 + it).toDouble() }
    val xp = DoubleArray(todos.size) { todos[it].toDouble() }
    fun serie(v: DoubleArray): DoubleArray {
        val s = DoubleArray(cuadros.size) { interpolar(cuadros[it], xp, v) }
        val n = s.size
        return DoubleArray(n) { i -> (-2..2).sumOf { k -> s[(i + k).coerceIn(0, n - 1)] } / 5.0 }
    }
    val posiciones = DoubleArray(PUNTOS_ANGULO) { j -> i0 + j * (i1 - i0).toDouble() / (PUNTOS_ANGULO - 1) }
    val sa = serie(ang); val sx = serie(dx); val sy = serie(dy)
    val curvaA = DoubleArray(PUNTOS_ANGULO) { interpolar(posiciones[it], cuadros, sa) }
    val curvaX = DoubleArray(PUNTOS_ANGULO) { interpolar(posiciones[it], cuadros, sx) }
    val curvaY = DoubleArray(PUNTOS_ANGULO) { interpolar(posiciones[it], cuadros, sy) }

    // cuantas veces se dobla (baja DOBLEZ_MINIMO grados; para contar otra vez tiene que volver a estirarse)
    var dobleces = 0; var bajando = false
    var alto = curvaA[0]; var bajo = curvaA[0]
    for (v in curvaA) {
        if (!bajando) {
            alto = max(alto, v)
            if (alto - v >= DOBLEZ_MINIMO) { bajando = true; bajo = v; dobleces++ }
        } else {
            bajo = min(bajo, v)
            if (v - bajo >= DOBLEZ_MINIMO) { bajando = false; alto = v }
        }
    }

    val salida = FloatArray(BRAZO_POR_MANO - 1)
    for (j in 0 until PUNTOS_ANGULO) {
        salida[j] = (curvaA[j] / 180.0).toFloat()
        salida[PUNTOS_ANGULO + j] = curvaX[j].toFloat()
        salida[2 * PUNTOS_ANGULO + j] = curvaY[j].toFloat()
    }
    salida[3 * PUNTOS_ANGULO] = (curvaA.min() / 180.0).toFloat()
    salida[3 * PUNTOS_ANGULO + 1] = (curvaA.max() / 180.0).toFloat()
    salida[3 * PUNTOS_ANGULO + 2] = dobleces.toFloat()
    return salida to true
}

// v14: los 5 dedos al inicio, mitad y final (largo y estirado) + las puntas en la mitad
fun formaDeMano(manos: Array<Mano>, ok: BooleanArray, indices: List<Int>): Pair<FloatArray, Boolean> {
    val vacio = FloatArray(DEDOS_POR_MANO)
    if (indices.isEmpty()) return vacio to false
    val todos = indices.filter { ok[it] }
    if (todos.isEmpty()) return vacio to false
    val n = indices.size
    val salida = FloatArray(DEDOS_POR_MANO)
    for (mo in 0 until MOMENTOS) {
        val desde = mo * n / MOMENTOS
        val tramo = indices.subList(desde, min(n, max(desde + 1, (mo + 1) * n / MOMENTOS)))
        val usar = tramo.filter { ok[it] }.ifEmpty { todos }
        val medidas = usar.map { medidasDedos(manos[it]) }
        for (k in 0 until 5) {
            salida[mo * 10 + k * 2] = medianaD(DoubleArray(medidas.size) { medidas[it].first[k] }).toFloat()
            salida[mo * 10 + k * 2 + 1] = medianaD(DoubleArray(medidas.size) { medidas[it].second[k] }).toFloat()
            if (mo == MOMENTOS / 2) {
                salida[MOMENTOS * 10 + k * 2] = medianaD(DoubleArray(medidas.size) { medidas[it].third[k].x }).toFloat()
                salida[MOMENTOS * 10 + k * 2 + 1] = medianaD(DoubleArray(medidas.size) { medidas[it].third[k].y }).toFloat()
            }
        }
    }
    return salida to true
}

// pone cada mano en su brazo en todos los cuadros
fun corregirManosGrabacion(brazos: Array<Brazo>): Pair<Array<Array<Mano>>, Array<BooleanArray>> {
    val mi = brazos[0].manos.copyOf(); val md = brazos[1].manos.copyOf()
    val oi = brazos[0].manosOk.copyOf(); val od = brazos[1].manosOk.copyOf()
    for (t in oi.indices) {
        if ((oi[t] || od[t]) && brazos[0].validas[t] && brazos[1].validas[t]) {
            val (a, b) = asignarManos(if (oi[t]) mi[t] else null, if (od[t]) md[t] else null,
                brazos[0].munecas[t], brazos[1].munecas[t])
            mi[t] = a ?: vacia(); oi[t] = a != null
            md[t] = b ?: vacia(); od[t] = b != null
        }
    }
    return arrayOf(mi, md) to arrayOf(oi, od)
}

// descarta los cuadros donde la deteccion de la mano es imposible (saltos, tamaño, mezcla con la cara)
fun filtrarMano(manos: Array<Mano>, manoOk: BooleanArray, anchoHombros: Double, fps: Double,
                caras: Array<Array<P>>? = null, carasOk: BooleanArray? = null): BooleanArray {
    val ok = manoOk.copyOf()
    if (manos.size != ok.size || ok.count { it } < 3 || anchoHombros <= 0) return ok
    val centros = Array(manos.size) { centroMano(manos[it]) }
    val tamanos = DoubleArray(manos.size) { (manos[it][MANO_BASE_MEDIO] - manos[it][MANO_MUNECA]).norma() }
    val normal = medianaD(DoubleArray(ok.count { it }).also { a -> var k = 0; for (i in ok.indices) if (ok[i]) a[k++] = tamanos[i] })
    if (normal <= 1e-6) return ok
    for (i in ok.indices) ok[i] = ok[i] && tamanos[i] > TAMANO_MANO_MIN * normal && tamanos[i] < TAMANO_MANO_MAX * normal

    var nariz: Array<P>? = null
    if (caras != null && carasOk != null && caras.size == ok.size && carasOk.any { it }) {
        val buenas = caras.indices.filter { carasOk[it] }
        val mx = medianaD(DoubleArray(buenas.size) { caras[buenas[it]][0].x })
        val my = medianaD(DoubleArray(buenas.size) { caras[buenas[it]][0].y })
        nariz = Array(caras.size) { if (carasOk[it]) caras[it][0] else P(mx, my) }
    }

    val salto = SALTO_MANO_CUADRO * anchoHombros * 30.0 / max(fps, 1.0)
    var ultimo = -1; var rechazos = 0
    for (i in ok.indices) {
        if (!ok[i]) continue
        if (ultimo < 0) { ultimo = i; continue }
        val lejos = (centros[i] - centros[ultimo]).norma() > salto * (i - ultimo)
        var caraMal = false
        if (nariz != null && (centros[i] - nariz[i]).norma() < RADIO_CARA * anchoHombros) {
            val r = tamanos[i] / max(tamanos[ultimo], 1e-6)
            caraMal = r > CAMBIO_TAMANO_CARA || r < 1 / CAMBIO_TAMANO_CARA
        }
        if ((lejos || caraMal) && rechazos < RECHAZOS_SEGUIDOS) { ok[i] = false; rechazos++; continue }
        rechazos = 0; ultimo = i
    }
    return ok
}

// lugares de la cara (y el pecho) durante la seña, en pixeles; null si no se vio la cara
internal fun referenciasCara(caras: Array<Array<P>>?, carasOk: BooleanArray?, indices: List<Int>,
                            centroHombros: P, anchoHombros: Double): Array<P>? {
    if (caras == null || carasOk == null || caras.isEmpty() || caras[0].size < CANTIDAD_CARA_REF) return null
    var usar = indices.filter { carasOk[it] }
    if (usar.isEmpty()) usar = caras.indices.filter { carasOk[it] }
    if (usar.isEmpty()) return null
    val c = Array(CANTIDAD_CARA_REF) { j ->
        P(medianaD(DoubleArray(usar.size) { caras[usar[it]][j].x }), medianaD(DoubleArray(usar.size) { caras[usar[it]][j].y }))
    }
    val pecho = centroHombros + P(0.0, 0.45 * anchoHombros)
    return arrayOf(c[0], (c[1] + c[2]) / 2.0, c[3], c[4], (c[5] + c[6]) / 2.0, (c[7] + c[8]) / 2.0, c[9], c[10], pecho)
}

// recorrido limpio del centro de la mano de un lado (lo que usan la muestra y el muñeco)
class LadoLimpio(
    val puntos: Array<P>, val valida: BooleanArray, val centro: Array<P>,
    val manoSuave: Array<Mano>, val okMano: BooleanArray, val indices: List<Int>, val fps: Double
)

internal fun limpiarLado(
    b: Brazo, manos: Array<Mano>, manosOk: BooleanArray, anchoHombros: Double, centroHombros: P,
    caras: Array<Array<P>>?, carasOk: BooleanArray?, usarCodos: Boolean
): LadoLimpio {
    val fps = b.munecas.size / SEGUNDOS_MUESTRA
    val okMano = filtrarMano(manos, manosOk, anchoHombros, fps, caras, carasOk)
    val manoSuave = suavizarManos(manos, okMano, fps)
    // con la mano: para completar la mano tapada y medir los dedos
    val (puntos, valida) = limpiarRecorrido(b.munecas, b.validas, anchoHombros, fps, manoSuave, okMano)
    // v17: el TRAZO sale siempre de la muñeca del esqueleto (punta del antebrazo): casi nunca se pierde,
    // asi no salta al pasar de la mano al brazo cuando la mano sale borrosa
    val (centro, validaTrazo) = limpiarRecorrido(b.munecas, b.validas, anchoHombros, fps)
    val indices = indicesActivos(centro, validaTrazo, centroHombros, anchoHombros,
        if (usarCodos) b.codos else null, if (usarCodos) b.codosOk else null)
    return LadoLimpio(puntos, valida, centro, manoSuave, okMano, indices, fps)
}

// para VER el trazo como lo recibe el modelo: centro de la mano (pixeles, suavizado) y cuadros de la seña
// si ninguna mano queda (seña hecha abajo), se repite sin quitar la mano en reposo
fun trazoFinalMano(brazos: Array<Brazo>, anchoHombros: Double, centroHombros: P,
                   caras: Array<Array<P>>? = null, carasOk: BooleanArray? = null): Array<Pair<Array<P>, List<Int>>?> =
    sinQuedarVacio({ r -> r.any { it != null } }) { trazoFinalManoUnaVez(brazos, anchoHombros, centroHombros, caras, carasOk) }

private fun trazoFinalManoUnaVez(brazos: Array<Brazo>, anchoHombros: Double, centroHombros: P,
                   caras: Array<Array<P>>?, carasOk: BooleanArray?): Array<Pair<Array<P>, List<Int>>?> {
    val salida = arrayOfNulls<Pair<Array<P>, List<Int>>>(2)
    if (!anchoHombros.isFinite() || anchoHombros < 1e-6) return salida
    val (manos, manosOk) = corregirManosGrabacion(brazos)
    for (lado in 0..1) {
        val l = limpiarLado(brazos[lado], manos[lado], manosOk[lado], anchoHombros, centroHombros, caras, carasOk, true)
        if (l.indices.size >= MIN_FRAMES_MANO) {
            // suavizar quita un punto a cada lado: se repiten los extremos para que el punto i sea el cuadro i
            val s = suavizar(l.centro)
            val alineado = if (s.size == l.centro.size - 2) arrayOf(s.first()) + s + arrayOf(s.last()) else s
            salida[lado] = alineado to l.indices
        }
    }
    return salida
}

// si ninguna mano queda (seña hecha abajo con poco movimiento), se arma otra vez sin quitar la de reposo
fun construirMuestraCompleta(
    brazos: Array<Brazo>, anchoHombros: Double, centroHombros: P, conCodos: Boolean = true,
    caras: Array<Array<P>>? = null, carasOk: BooleanArray? = null
): Muestra = sinQuedarVacio({ m: Muestra -> m.validos.any { it } }) {
    construirMuestraUnaVez(brazos, anchoHombros, centroHombros, conCodos, caras, carasOk)
}

private fun construirMuestraUnaVez(
    brazos: Array<Brazo>, anchoHombros: Double, centroHombros: P, conCodos: Boolean,
    caras: Array<Array<P>>?, carasOk: BooleanArray?
): Muestra {
    val trazos = Array(CANTIDAD_TRAZOS) { Array(PUNTOS_TRAZO) { P(0.0, 0.0) } }
    val validos = BooleanArray(CANTIDAD_TRAZOS); val extras = FloatArray(CANTIDAD_EXTRAS)
    val trazosB = Array(2) { Array(PUNTOS_TRAZO) { P(0.0, 0.0) } }; val validosB = BooleanArray(2)
    if (!anchoHombros.isFinite() || anchoHombros < 1e-6) return Muestra(trazos, validos, extras, trazosB, validosB)

    val (manos, manosOk) = corregirManosGrabacion(brazos)

    for (lado in 0..1) {
        val b = brazos[lado]
        val l = limpiarLado(b, manos[lado], manosOk[lado], anchoHombros, centroHombros, caras, carasOk, true)
        val indices = l.indices
        if (indices.size < MIN_FRAMES_MANO) continue
        val recorrido = indices.map { l.centro[it] }.toTypedArray()
        val media = P(recorrido.sumOf { it.x } / recorrido.size, recorrido.sumOf { it.y } / recorrido.size)

        // trazo de la palma
        val r = remuestrearPorLongitud(suavizar(recorrido), PUNTOS_TRAZO)
        val mr = P(r.sumOf { it.x } / r.size, r.sumOf { it.y } / r.size)
        trazos[lado] = Array(r.size) { (r[it] - mr) / anchoHombros }
        validos[lado] = true

        // la mano tapada se completa con su ultima forma (para medir los dedos)
        val (completa, okCompleto) = completarManosTapadas(l.manoSuave, l.okMano, l.puntos, l.valida, l.fps)

        // v15: ya no hay trazo del codo (dibujo 2); el brazo va como angulos en los extras

        // donde ocurre (respecto a los hombros)
        val rel = recorrido.map { (it - centroHombros) / anchoHombros }
        val o = lado * EXTRAS_POR_MANO
        extras[o] = 1f
        extras[o + 1] = rel.first().x.toFloat(); extras[o + 2] = rel.first().y.toFloat()
        extras[o + 3] = rel.last().x.toFloat(); extras[o + 4] = rel.last().y.toFloat()
        extras[o + 5] = (rel.sumOf { it.x } / rel.size).toFloat(); extras[o + 6] = (rel.sumOf { it.y } / rel.size).toFloat()
        extras[o + 7] = (rel.maxOf { it.x } - rel.minOf { it.x }).toFloat()
        extras[o + 8] = (rel.maxOf { it.y } - rel.minOf { it.y }).toFloat()

        val (forma, formaValida) = formaDeMano(completa, okCompleto, indices)
        extras[o + 9] = if (formaValida) 1f else 0f
        forma.copyInto(extras, o + 10)

        // donde esta la mano respecto a la cara y el pecho
        val refs = referenciasCara(caras, carasOk, indices, centroHombros, anchoHombros)
        if (refs != null) {
            extras[o + E_CARA_VALIDA] = 1f
            for ((j, ref) in refs.withIndex()) {
                val d = recorrido.map { (it - ref) / anchoHombros }
                extras[o + E_CARA_VALIDA + 1 + j * 3] = (d.sumOf { it.x } / d.size).toFloat()
                extras[o + E_CARA_VALIDA + 2 + j * 3] = (d.sumOf { it.y } / d.size).toFloat()
                extras[o + E_CARA_VALIDA + 3 + j * 3] = d.minOf { it.norma() }.toFloat()
            }
        }

        // el brazo: angulo del codo y direccion del antebrazo
        if (conCodos) {
            val (brazo, brazoValido) = angulosBrazo(b, indices, lado, centroHombros, anchoHombros)
            extras[o + E_BRAZO_VALIDO] = if (brazoValido) 1f else 0f
            brazo.copyInto(extras, o + E_BRAZO_VALIDO + 1)
        }
    }
    return Muestra(trazos, validos, extras, trazosB, validosB)
}

// ---------------------------------------------------------------- imagen del modelo

// imita cv2.line(..., grosor 2, LINE_AA): ajustado comparando con OpenCV
const val ALCANCE_LINEA = 2.3
const val PUNTA_LINEA = 0.45

// 2 trazos (codos) -> 2 canales; 12 trazos -> 4 canales: centro izq, centro der, dedos izq, dedos der
fun dibujarTrazos(trazos: Array<Array<P>>, validos: BooleanArray, tamano: Int = TAMANO_IMAGEN): FloatArray {
    val n = DEDOS_PUNTAS.size
    val grupos: List<List<Int>> = if (trazos.size == 2) listOf(listOf(0), listOf(1))
        else listOf(listOf(0), listOf(1), (2 until 2 + n).toList(), (2 + n until 2 + 2 * n).toList())
    val imagen = FloatArray(grupos.size * tamano * tamano)
    for ((canal, indices) in grupos.withIndex()) {
        val lienzo = DoubleArray(tamano * tamano)
        var hay = false
        for (indice in indices) {
            if (!validos[indice]) continue
            hay = true
            val p = trazos[indice].map {
                P(Math.rint((it.x / MEDIA_VENTANA * 0.5 + 0.5) * (tamano - 1)).coerceIn(0.0, tamano - 1.0),
                  Math.rint((it.y / MEDIA_VENTANA * 0.5 + 0.5) * (tamano - 1)).coerceIn(0.0, tamano - 1.0))
            }
            val cant = p.size
            for (k in 0 until cant - 1) {
                val intensidad = (90 + 165 * (k + 1) / (cant - 1).toDouble()).toInt().toDouble()
                val a = p[k]; val d = p[k + 1] - a
                val l2 = d.punto(d); val largo = Math.sqrt(l2)
                val m = ALCANCE_LINEA + 1.0
                val x0 = maxOf(0, Math.floor(minOf(a.x, p[k + 1].x) - m).toInt()); val x1 = minOf(tamano - 1, Math.ceil(maxOf(a.x, p[k + 1].x) + m).toInt())
                val y0 = maxOf(0, Math.floor(minOf(a.y, p[k + 1].y) - m).toInt()); val y1 = minOf(tamano - 1, Math.ceil(maxOf(a.y, p[k + 1].y) + m).toInt())
                for (y in y0..y1) for (x in x0..x1) {
                    val tt = if (l2 > 0) ((x - a.x) * d.x + (y - a.y) * d.y) / l2 else 0.0
                    val t = tt.coerceIn(0.0, 1.0)
                    val dist = hypot(x - (a.x + t * d.x), y - (a.y + t * d.y))
                    // pasado el extremo la linea se apaga mas rapido (como cv2.LINE_AA de grosor 2)
                    val fuera = if (l2 > 0) maxOf(0.0, -tt, tt - 1) * largo else dist
                    val cob = (ALCANCE_LINEA - dist - PUNTA_LINEA * fuera).coerceIn(0.0, 1.0)
                    if (cob <= 0.0) continue
                    val i = y * tamano + x
                    lienzo[i] = Math.rint(lienzo[i] * (1 - cob) + intensidad * cob)
                }
            }
        }
        if (hay) for (i in lienzo.indices) imagen[canal * tamano * tamano + i] = (lienzo[i] / 255.0).toFloat()
    }
    return imagen
}
