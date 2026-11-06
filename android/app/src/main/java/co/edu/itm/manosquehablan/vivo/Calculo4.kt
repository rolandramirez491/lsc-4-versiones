package co.edu.itm.manosquehablan.vivo

import kotlin.math.max

// Lo que ve cada una de las 4 versiones (igual que experimento_versiones.py en el PC).
// Sin nada de Android: se prueba en el PC contra Python (CalculoTest).

// una grabacion ya llevada a 99 cuadros y a la escala de 1280x720 (como las muestras del banco)
class GrabacionCruda(
    val munecas: Array<Array<P>>, val validas: Array<BooleanArray>,
    val manos: Array<Array<Mano>>, val manosOk: Array<BooleanArray>,
    val codos: Array<Array<P>>, val codosOk: Array<BooleanArray>,
    val caras: Array<Array<P>>, val carasOk: BooleanArray,
    val ancho: Double, val centro: P
) {
    val cuadros get() = munecas[0].size
    fun brazo(l: Int) = Brazo(munecas[l], validas[l], manos[l], manosOk[l], codos[l], codosOk[l])
}

const val ANCHO_REF = 1280.0
const val ALTO_REF = 720.0

// promedio de 5 cuadros (en los bordes se repite el extremo), como _suave() del PC
private fun suave(p: Array<P>, ventana: Int = 5): Array<P> {
    val n = p.size; val m = ventana / 2
    return Array(n) { i ->
        var x = 0.0; var y = 0.0
        for (k in -m..m) { val q = p[(i + k).coerceIn(0, n - 1)]; x += q.x; y += q.y }
        P(x / ventana, y / ventana)
    }
}

// ---------------------------------------------------------------- versiones 1 y 2: secuencias para la GRU

// versión 1: solo los 21 puntos de cada mano (0..1 de la imagen) + si se vio cada mano
fun secuenciaV1(g: GrabacionCruda): Array<FloatArray> = Array(g.cuadros) { t ->
    val f = FloatArray(21 * 2 * 2 + 2)
    var k = 0
    for (l in 0..1) for (q in g.manos[l][t]) { f[k++] = (q.x / ANCHO_REF).toFloat(); f[k++] = (q.y / ALTO_REF).toFloat() }
    f[k++] = if (g.manosOk[0][t]) 1f else 0f
    f[k] = if (g.manosOk[1][t]) 1f else 0f
    f
}

// versión 2: muñecas, codos, hombros y manos (estabilizados), medidos desde los hombros en anchos de hombro
fun secuenciaV2(g: GrabacionCruda): Array<FloatArray> {
    val fps = g.cuadros / SEGUNDOS_MUESTRA
    val manos = Array(2) { suavizarManos(g.manos[it], g.manosOk[it], fps) }
    val munecas = Array(2) { suave(g.munecas[it]) }
    val codos = Array(2) { suave(g.codos[it]) }
    val hombros = arrayOf(g.centro + P(0.5 * g.ancho, 0.0), g.centro - P(0.5 * g.ancho, 0.0))
    val a = max(g.ancho, 1e-6)
    return Array(g.cuadros) { t ->
        val puntos = listOf(munecas[0][t], munecas[1][t], codos[0][t], codos[1][t], hombros[0], hombros[1]) +
            manos[0][t].toList() + manos[1][t].toList()
        val f = FloatArray(puntos.size * 2 + 2)
        for ((k, q) in puntos.withIndex()) { f[2 * k] = ((q.x - g.centro.x) / a).toFloat(); f[2 * k + 1] = ((q.y - g.centro.y) / a).toFloat() }
        f[puntos.size * 2] = if (g.manosOk[0][t]) 1f else 0f
        f[puntos.size * 2 + 1] = if (g.manosOk[1][t]) 1f else 0f
        f
    }
}

// ---------------------------------------------------------------- versión 3: dibujo + lugar + dedos (= V14)

// trazo de la palma, dibujo del codo y extras (lugar, dedos en 3 momentos, cara). Sin ángulo del codo
// y SIN quitar la mano en reposo (eso llegó en la v17).
fun construirV3(g: GrabacionCruda): Muestra {
    val trazos = Array(CANTIDAD_TRAZOS) { Array(PUNTOS_TRAZO) { P(0.0, 0.0) } }
    val validos = BooleanArray(CANTIDAD_TRAZOS); val extras = FloatArray(CANTIDAD_EXTRAS)
    val trazosB = Array(2) { Array(PUNTOS_TRAZO) { P(0.0, 0.0) } }; val validosB = BooleanArray(2)
    val ancho = g.ancho; val centroH = g.centro
    if (!ancho.isFinite() || ancho < 1e-6) return Muestra(trazos, validos, extras, trazosB, validosB)
    val brazos = Array(2) { g.brazo(it) }
    val (manosC, okC) = corregirManosGrabacion(brazos)
    for (l in 0..1) {
        val brazo = g.munecas[l]; val bok = g.validas[l]
        val fps = brazo.size / SEGUNDOS_MUESTRA
        // la mano: sin detecciones imposibles y suavizada
        val mok = filtrarMano(manosC[l], okC[l], ancho, fps, g.caras, g.carasOk)
        val mano = suavizarManos(manosC[l], mok, fps)
        // el trazo: centro de la palma donde se ve la mano; donde no, la muñeca limpia (mano pegada al brazo)
        val (limpio, valida) = limpiarRecorrido(brazo, bok, ancho, fps, mano, mok)
        val centro = Array(brazo.size) { if (mok[it]) centroPalma(mano[it]) else limpio[it] }
        val indices = indicesActivos(centro, valida, centroH, ancho, g.codos[l], g.codosOk[l], quitarManoReposo = false)
        if (indices.size < MIN_FRAMES_MANO) continue

        val recorrido = indices.map { centro[it] }.toTypedArray()
        trazos[l] = centrado(remuestrearPorLongitud(suavizar(recorrido), PUNTOS_TRAZO), ancho)
        validos[l] = true

        // dibujo del codo (DIBUJO 2)
        val (pb, vb) = recorridoAntebrazo(brazo, bok, g.codos[l], g.codosOk[l], ancho, fps)
        val ib = indices.filter { vb[it] }
        if (ib.size >= MIN_FRAMES_MANO) {
            trazosB[l] = centrado(remuestrearPorLongitud(suavizar(ib.map { pb[it] }.toTypedArray()), PUNTOS_TRAZO), ancho)
            validosB[l] = true
        }

        // dónde ocurre (respecto a los hombros)
        val rel = recorrido.map { (it - centroH) / ancho }
        val o = l * EXTRAS_POR_MANO
        extras[o] = 1f
        extras[o + 1] = rel.first().x.toFloat(); extras[o + 2] = rel.first().y.toFloat()
        extras[o + 3] = rel.last().x.toFloat(); extras[o + 4] = rel.last().y.toFloat()
        extras[o + 5] = (rel.sumOf { it.x } / rel.size).toFloat(); extras[o + 6] = (rel.sumOf { it.y } / rel.size).toFloat()
        extras[o + 7] = (rel.maxOf { it.x } - rel.minOf { it.x }).toFloat()
        extras[o + 8] = (rel.maxOf { it.y } - rel.minOf { it.y }).toFloat()

        // forma de los dedos en 3 momentos (la mano tapada sigue a la muñeca limpia)
        val (completa, okCompleta) = completarManosTapadas(mano, mok, limpio, valida, fps)
        val (forma, formaValida) = formaDeMano(completa, okCompleta, indices)
        extras[o + 9] = if (formaValida) 1f else 0f
        forma.copyInto(extras, o + 10)

        // dónde está la mano respecto a la cara y el pecho
        val refs = referenciasCara(g.caras, g.carasOk, indices, centroH, ancho)
        if (refs != null) {
            extras[o + E_CARA_VALIDA] = 1f
            for ((j, ref) in refs.withIndex()) {
                val d = recorrido.map { (it - ref) / ancho }
                extras[o + E_CARA_VALIDA + 1 + j * 3] = (d.sumOf { it.x } / d.size).toFloat()
                extras[o + E_CARA_VALIDA + 2 + j * 3] = (d.sumOf { it.y } / d.size).toFloat()
                extras[o + E_CARA_VALIDA + 3 + j * 3] = d.minOf { it.norma() }.toFloat()
            }
        }
    }
    return Muestra(trazos, validos, extras, trazosB, validosB)
}

private fun centrado(t: Array<P>, ancho: Double): Array<P> {
    val m = P(t.sumOf { it.x } / t.size, t.sumOf { it.y } / t.size)
    return Array(t.size) { (t[it] - m) / ancho }
}

// las dos redes votan juntas: media geométrica pesada (w = peso de la versión 2)
fun fusionar(p1: FloatArray, p2: FloatArray, w: Double): FloatArray {
    val l = DoubleArray(p1.size) { w * kotlin.math.ln(p1[it].toDouble().coerceIn(1e-9, 1.0)) +
        (1 - w) * kotlin.math.ln(p2[it].toDouble().coerceIn(1e-9, 1.0)) }
    val m = l.max()
    val e = DoubleArray(l.size) { kotlin.math.exp(l[it] - m) }
    val s = e.sum()
    return FloatArray(e.size) { (e[it] / s).toFloat() }
}

// ---------------------------------------------------------------- de la grabacion en vivo a GrabacionCruda

// el registro ya normalizado a 99 cuadros; los puntos se pasan a la escala 1280x720 de las muestras
internal fun grabacionDeRegistro(r: Registro, ancho: Int, alto: Int): GrabacionCruda? {
    if (r.anchos.isEmpty()) return null
    val ex = ANCHO_REF / ancho; val ey = ALTO_REF / alto
    fun e(p: P) = P(p.x * ex, p.y * ey)
    val anchoH = mediana(r.anchos) * ex
    val centro = P(mediana(r.centros.map { it.x }) * ex, mediana(r.centros.map { it.y }) * ey)
    return GrabacionCruda(
        Array(2) { l -> r.munecas[l].map(::e).toTypedArray() }, Array(2) { l -> r.validas[l].toBooleanArray() },
        Array(2) { l -> r.manos[l].map { m -> Array(m.size) { e(m[it]) } }.toTypedArray() },
        Array(2) { l -> r.manosOk[l].toBooleanArray() },
        Array(2) { l -> r.codos[l].map(::e).toTypedArray() }, Array(2) { l -> r.codosOk[l].toBooleanArray() },
        r.caras.map { c -> Array(c.size) { e(c[it]) } }.toTypedArray(), r.carasOk.toBooleanArray(),
        anchoH, centro
    )
}
