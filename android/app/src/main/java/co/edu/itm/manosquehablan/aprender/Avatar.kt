package co.edu.itm.manosquehablan.aprender

import android.content.Context
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import co.edu.itm.manosquehablan.vivo.COLOR_DER
import co.edu.itm.manosquehablan.vivo.COLOR_IZQ
import co.edu.itm.manosquehablan.vivo.Mano
import co.edu.itm.manosquehablan.vivo.MANO_MUNECA
import co.edu.itm.manosquehablan.vivo.P
import co.edu.itm.manosquehablan.vivo.Senas
import co.edu.itm.manosquehablan.vivo.construirMuestraCompleta
import co.edu.itm.manosquehablan.vivo.envolvente
import co.edu.itm.manosquehablan.vivo.mediana
import co.edu.itm.manosquehablan.vivo.suavizarManos
import co.edu.itm.manosquehablan.vivo.textoAscii
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.ClipOp
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.ImageShader
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.TileMode
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.sqrt

// Muñeco de APRENDER que repite las señas grabadas.

const val SEGUNDOS_GRABACION_AVATAR = 3.4
const val FRAMES_TRANSICION = 8
const val PAUSA_ENTRE_SENAS = 0.15
const val SEGUNDOS_REVISAR = 2.0

// brazo del muñeco en anchos de hombro desde el centro de los hombros
class LadoAvatar(
    val muneca: Array<P>, val codo: Array<P>, val mano: Array<Mano>?,
    val trazo: Array<P>? = null, val trazoOk: BooleanArray? = null    // trazo final (como lo ve el modelo)
)

// sinTrazo = true para gestos que no son señas
class AvatarSena(
    val lados: Array<LadoAvatar?>, val T: Int, val fps: Double, val sinTrazo: Boolean = false,
    val caras: Array<Gestos>? = null,   // gestos de la cara por cuadro
    val giro: DoubleArray? = null,      // giro de la cabeza por cuadro (medido con la nariz)
    val clip: String? = null,           // video de la seña en assets (mismos cuadros que la muestra)
    val nariz: P? = null,               // nariz de la persona (la mano toca el mismo lugar de la cara del muñeco)
    val inclinacion: DoubleArray? = null // inclinación de la cabeza por cuadro, en grados (en vivo)
)

// ---------------------------------------------------------------- cargar

// forma de una mano real de cada lado, se usa cuando una seña no tiene mano
val manoPlantilla = arrayOfNulls<Mano>(2)

// mano de plantilla girada en la direccion del antebrazo
fun manoPorDefecto(lado: Int, muneca: P, codo: P): Mano? {
    val base = manoPlantilla[lado] ?: manoPlantilla[1 - lado]?.let { h -> Array(h.size) { P(-h[it].x, h[it].y) } } ?: return null
    val dir = muneca - codo
    if (dir.norma() < 1e-6) return Array(base.size) { muneca + base[it] }
    val giro = Math.atan2(dir.y, dir.x) - Math.atan2(base[9].y, base[9].x)
    val c = Math.cos(giro); val s = Math.sin(giro)
    return Array(base.size) { muneca + P(c * base[it].x - s * base[it].y, s * base[it].x + c * base[it].y) }
}

// logo del ITM (assets/logo_itm.png), se carga junto con las señas
var logoItm: ImageBitmap? = null

fun cargarLogoItm(contexto: Context) {
    if (logoItm == null) logoItm = runCatching {
        contexto.assets.open("logo_itm.png").use { android.graphics.BitmapFactory.decodeStream(it).asImageBitmap() }
    }.getOrNull()
}

// muñecos y videos de APRENDER dentro del APK (cel_preparar_aprender.py):
// assets/aprender/<SENA>/avatar.json y clip.mp4; se leen solo cuando se necesitan
class AvatarPc(private val contexto: Context) {
    private val carpetas: Set<String> = (contexto.assets.list("aprender") ?: emptyArray()).toSet()
    private val cache = java.util.concurrent.ConcurrentHashMap<String, Pair<AvatarSena?, Unit>>()

    init { cargarLogoItm(contexto) }

    val cantidad: Int get() = carpetas.size

    operator fun get(nombre: String): AvatarSena? = cache.getOrPut(nombre) {
        val c = Senas.nombreCarpeta(nombre)
        (if (c in carpetas) leer(c) else null) to Unit
    }.first

    // una seña cualquiera (para la pose de reposo)
    fun cualquiera(): AvatarSena? = carpetas.firstOrNull()?.let { leer(it) }

    // GRABAR: cuantas muestras del PC tiene la seña (la tipica + aprender/<SENA>/muestras/NN)
    fun cantidadPc(nombre: String): Int {
        val c = Senas.nombreCarpeta(nombre)
        if (c !in carpetas) return 0
        return 1 + (contexto.assets.list("aprender/$c/muestras")?.size ?: 0)
    }

    // GRABAR: la muestra k del PC (0 = la tipica)
    fun muestraPc(nombre: String, k: Int): AvatarSena? =
        if (k == 0) get(nombre) else leer("${Senas.nombreCarpeta(nombre)}/muestras/%02d".format(k))

    private fun leer(c: String): AvatarSena? = try {
        val o = JSONObject(contexto.assets.open("aprender/$c/avatar.json").bufferedReader().readText())
        fun pts(a: org.json.JSONArray) = Array(a.length()) { val q = a.getJSONArray(it); P(q.getDouble(0), q.getDouble(1)) }
        val l = o.getJSONObject("lados")
        val lados = Array(2) { i ->
            val d = l.optJSONObject(if (i == 0) "izq" else "der") ?: return@Array null
            LadoAvatar(
                pts(d.getJSONArray("muneca")), pts(d.getJSONArray("codo")),
                d.optJSONArray("mano")?.let { a -> Array(a.length()) { pts(a.getJSONArray(it)) } },
                d.optJSONArray("trazo")?.let(::pts),
                d.optJSONArray("trazo_ok")?.let { a -> BooleanArray(a.length()) { a.getBoolean(it) } }
            )
        }
        for (k in 0..1) if (manoPlantilla[k] == null) lados[k]?.mano?.firstOrNull()?.let { h ->
            manoPlantilla[k] = Array(h.size) { h[it] - h[MANO_MUNECA] }
        }
        val giro = o.optJSONArray("giro")?.let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
        val nariz = o.optJSONArray("nariz")?.let { P(it.getDouble(0), it.getDouble(1)) }
        AvatarSena(lados, o.getInt("T"), o.getDouble("fps"), giro = giro, clip = if (o.optBoolean("clip")) "aprender/$c/clip.mp4" else null, nariz = nariz)
    } catch (_: Exception) { null }
}

private fun suavizarSerie(p: Array<P>, ventana: Int = 5): Array<P> {
    if (p.size < 3) return p
    val mitad = ventana / 2
    return Array(p.size) { i ->
        var s = P(0.0, 0.0)
        for (k in -mitad..mitad) s += p[(i + k).coerceIn(0, p.size - 1)]
        s / ventana.toDouble()
    }
}

private fun rellenar(p: Array<P>, ok: BooleanArray): Array<P>? {
    val primero = ok.indexOfFirst { it }
    if (primero < 0) return null
    var ultimo = p[primero]
    return Array(p.size) { i -> if (ok[i]) { ultimo = p[i]; p[i] } else ultimo }
}

private fun manoPegada(mano0: Array<Mano>, ok: BooleanArray, munecaBrazo: Array<P>, fps: Double): Array<Mano>? {
    val mano = suavizarManos(mano0, ok, fps)
    val primero = ok.indexOfFirst { it }
    if (primero < 0) return null
    val salida = mano.copyOf()
    var forma: Mano? = null; var desfase = P(0.0, 0.0)
    for (i in (primero until mano.size) + (primero - 1 downTo 0)) {
        if (ok[i]) {
            forma = Array(mano[i].size) { mano[i][it] - mano[i][MANO_MUNECA] }
            desfase = mano[i][MANO_MUNECA] - munecaBrazo[i]
        } else {
            val f = forma!!
            salida[i] = Array(f.size) { munecaBrazo[i] + desfase + f[it] }
        }
    }
    return salida
}

// carga una seña desde las muestras del celular usando la grabacion mas tipica
fun cargarAvatarCelular(senas: Senas, s: Senas.Sena): AvatarSena? {
    val muestras = senas.muestrasCelular(s).map { senas.leer(it) }
    if (muestras.isEmpty()) return null
    val vectores = muestras.map { (brazos, ancho, centro) ->
        val m = construirMuestraCompleta(brazos, ancho, centro)
        m.trazos.flatMap { t -> t.flatMap { listOf(it.x, it.y) } } + m.extras.map { it.toDouble() }
    }
    val media = DoubleArray(vectores[0].size) { k -> vectores.sumOf { it[k] } / vectores.size }
    val tipica = vectores.indices.minBy { i -> sqrt(vectores[i].indices.sumOf { k -> (vectores[i][k] - media[k]).let { it * it } }) }
    val (brazos, ancho, centro) = muestras[tipica]
    return avatarDeMuestra(brazos, ancho, centro)
}

// una muestra grabada en el celular (GRABAR: el deslizador elige cual)
fun avatarDeArchivo(senas: Senas, archivo: java.io.File): AvatarSena? = try {
    val (brazos, ancho, centro) = senas.leer(archivo)
    avatarDeMuestra(brazos, ancho, centro)
} catch (_: Exception) { null }

fun avatarDeMuestra(brazos: Array<co.edu.itm.manosquehablan.vivo.Brazo>, ancho: Double, centro: P): AvatarSena {
    val T = brazos[0].munecas.size
    val fps = T / SEGUNDOS_GRABACION_AVATAR
    fun norm(p: P) = (p - centro) / ancho
    val lados = Array(2) { l ->
        val b = brazos[l]
        val m0 = rellenar(b.munecas, b.validas) ?: return@Array null
        val c0 = rellenar(b.codos, b.codosOk) ?: return@Array null
        val m = suavizarSerie(m0); val c = suavizarSerie(c0)
        val mano = manoPegada(b.manos, b.manosOk, m, fps)
        LadoAvatar(m.map(::norm).toTypedArray(), c.map(::norm).toTypedArray(),
            mano?.map { h -> h.map(::norm).toTypedArray() }?.toTypedArray())
    }
    return AvatarSena(lados, T, fps)
}

// ---------------------------------------------------------------- frases

// pasa a minusculas sin tildes, la ñ queda como n#
fun normalizar(palabra: String): String =
    textoAscii(palabra.replace("ñ", "n#").replace("Ñ", "n#")).lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")

fun diccionario(senas: List<Senas.Sena>): Map<List<String>, Senas.Sena> =
    senas.associateBy { normalizar(it.nombre).split(" ") }

class Trozo(val desde: Int, val hasta: Int, val sena: Senas.Sena?)

fun analizarTexto(palabras: List<String>, dic: Map<List<String>, Senas.Sena>): List<Trozo> {
    val claves = palabras.map(::normalizar)
    val largoMax = dic.keys.maxOfOrNull { it.size } ?: 1
    val resultado = mutableListOf<Trozo>()
    var i = 0
    while (i < claves.size) {
        var encontrado = false
        for (largo in minOf(largoMax, claves.size - i) downTo 1) {
            val s = dic[claves.subList(i, i + largo)]
            if (s != null) { resultado += Trozo(i, i + largo, s); i += largo; encontrado = true; break }
        }
        if (!encontrado) { resultado += Trozo(i, i + 1, null); i++ }
    }
    return resultado
}

fun esPrefijo(claves: List<String>, dic: Map<List<String>, Senas.Sena>) =
    dic.keys.any { it.size > claves.size && it.subList(0, claves.size) == claves }

// ---------------------------------------------------------------- reproductor

private fun mezclarLados(a: AvatarSena, b: AvatarSena, t: Double): AvatarSena {
    val lados = Array(2) { l ->
        val da = a.lados[l] ?: return@Array null
        val db = b.lados[l] ?: return@Array null
        fun mezcla(x: P, y: P) = x * (1 - t) + y * t
        val mano = when {
            da.mano == null && db.mano == null -> null
            da.mano == null -> arrayOf(db.mano!![0])
            db.mano == null -> arrayOf(da.mano.last())
            else -> arrayOf(Array(da.mano.last().size) { mezcla(da.mano.last()[it], db.mano[0][it]) })
        }
        LadoAvatar(arrayOf(mezcla(da.muneca.last(), db.muneca[0])), arrayOf(mezcla(da.codo.last(), db.codo[0])), mano)
    }
    return AvatarSena(lados, 1, 30.0)
}

// seña para reproducir: texto que se muestra y la palabra acumulada si se esta deletreando
class Entrada(val texto: String, val acumulado: String?, val sena: AvatarSena)

class Paso(val sena: AvatarSena?, val cuadro: Int, val texto: String, val acumulado: String?)

// reproduce varias señas seguidas con transicion entre ellas
class Reproductor {
    private var lista = listOf<Entrada>()
    private var actual = -1
    private var frame = 0.0
    private var transicion = 0
    private var espera = 0.0

    fun cargar(nueva: List<Entrada>) {
        lista = nueva
        actual = if (lista.isNotEmpty()) 0 else -1
        frame = 0.0; transicion = 0; espera = 0.0
    }

    val activo get() = actual in lista.indices

    // avanza un cuadro
    fun paso(velocidad: Double): Paso {
        if (!activo) return Paso(null, 0, "", null)
        val e = lista[actual]
        if (transicion > 0) {
            val anterior = lista[actual - 1].sena
            val t = 1.0 - transicion.toDouble() / FRAMES_TRANSICION
            transicion--
            return Paso(mezclarLados(anterior, e.sena, t), 0, e.texto, e.acumulado)
        }
        val cuadro = minOf(frame.toInt(), e.sena.T - 1)
        frame += velocidad * e.sena.fps / 30.0
        if (frame >= e.sena.T) {
            espera += velocidad / 30.0
            if (espera >= PAUSA_ENTRE_SENAS) {
                actual++; frame = 0.0; espera = 0.0
                if (activo) transicion = FRAMES_TRANSICION
            }
        }
        return Paso(e.sena, cuadro, e.texto, e.acumulado)
    }
}

// ---------------------------------------------------------------- rascarse

// probabilidad de rascarse
const val PROBABILIDAD_RASCARSE = 0.10

// la mano derecha sube a rascarse un ojo o la cabeza y vuelve
fun gestoRascarse(base: AvatarSena, rng: kotlin.random.Random): AvatarSena? {
    // mano derecha
    val lado = 1
    val d = base.lados[lado] ?: return null
    val signo = if (lado == 0) 1.0 else -1.0
    val ojo = rng.nextBoolean()
    // posicion del ojo o de la cabeza
    val objetivo = if (ojo) P(signo * 0.09, -0.80) else P(signo * 0.16, -1.08)

    val m0 = d.muneca.last(); val c0 = d.codo.last()
    val forma0 = d.mano?.last()
    val hombro = P(signo * 0.5, 0.0)
    // la punta del indice va al objetivo
    val munecaMeta = objetivo + P(signo * 0.10, 0.22)
    val codoMeta = hombro + (munecaMeta - hombro) * 0.5 + P(signo * 0.38, 0.30)

    val subir = 18; val rascar = 30; val bajar = 18          // cuadros de cada parte
    val total = subir + rascar + bajar
    val munecas = ArrayList<P>(); val codos = ArrayList<P>(); val manos = ArrayList<Mano>()
    for (f in 0 until total) {
        val (t, temblor) = when {
            f < subir -> suave(f / subir.toFloat()).toDouble() to 0.0
            f < subir + rascar -> 1.0 to 1.0
            else -> (1 - suave((f - subir - rascar) / bajar.toFloat())).toDouble() to 0.0
        }
        // vaiven de la mano
        val vaiven = temblor * kotlin.math.sin((f - subir) * 2 * Math.PI / 7.5) * 0.035
        val desplaza = if (ojo) P(vaiven, vaiven * 0.3) else P(vaiven * 0.6, -kotlin.math.abs(vaiven))
        val m = m0 + (munecaMeta - m0) * t + desplaza
        munecas += m
        codos += c0 + (codoMeta - c0) * t
        if (forma0 != null) {
            // se gira la mano para que apunte a la cara
            val rel = Array(forma0.size) { forma0[it] - forma0[MANO_MUNECA] }
            val actual = rel[8]; val deseada = objetivo - munecaMeta
            val giro = (Math.atan2(deseada.y, deseada.x) - Math.atan2(actual.y, actual.x)).let {
                var a = it; while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; a
            } * t
            val escalaMano = if (actual.norma() > 1e-6) (0.9 * deseada.norma() / actual.norma()).coerceIn(0.6, 1.4) else 1.0
            val k = 1 + (escalaMano - 1) * t
            manos += Array(rel.size) { m + girarP(rel[it], giro) * k }
        }
    }
    val nuevos = Array(2) { l ->
        val o = base.lados[l] ?: return@Array null
        if (l == lado) LadoAvatar(munecas.toTypedArray(), codos.toTypedArray(), if (forma0 != null) manos.toTypedArray() else null)
        else LadoAvatar(Array(total) { o.muneca.last() }, Array(total) { o.codo.last() }, o.mano?.let { h -> Array(total) { h.last() } })
    }
    return AvatarSena(nuevos, total, 30.0, sinTrazo = true)
}

private fun girarP(v: P, a: Double): P {
    val c = Math.cos(a); val s = Math.sin(a)
    return P(c * v.x - s * v.y, s * v.x + c * v.y)
}

// ---------------------------------------------------------------- deletreo

// palabra toda en mayusculas = se deletrea
fun esMayusculaSostenida(palabra: String): Boolean {
    val letras = palabra.filter { it.isLetter() }
    return letras.isNotEmpty() && letras.all { it.isUpperCase() }
}

// separa la palabra en letras (ll y rr juntas), devuelve (letra, clave)
fun letrasDe(palabra: String): List<Pair<String, String>> {
    val p = palabra.filter { it.isLetter() }
    val salida = mutableListOf<Pair<String, String>>()
    var i = 0
    while (i < p.length) {
        val doble = if (i + 1 < p.length) p.substring(i, i + 2).uppercase() else ""
        if (doble == "LL" || doble == "RR") { salida += doble to normalizar(doble); i += 2; continue }
        val letra = p[i].toString()
        salida += letra.uppercase() to normalizar(letra)
        i++
    }
    return salida
}

// ---------------------------------------------------------------- dibujo
// Muñeco realista (igual que munieco.py del PC): cuerpo con volumen, cabeza detallada, pelo de mechones
// que se mece, respiración por la nariz, nuez que traga, giro leve del cuerpo, brazos 3D, manos con
// uñas, vellos, barba de pocos días y logo de manos en la camisa.

val FONDO = Color(26, 28, 32)
val FONDO_LISTA = Color(38, 40, 45)

// colores de cada personaje (tonos de persona normal)
class Apariencia(val piel: Color, val pielSombra: Color, val camisa: Color, val pelo: Color, val mujer: Boolean)

val HOMBRE = Apariencia(Color(214, 170, 138), Color(178, 132, 100), Color(62, 80, 104), Color(52, 40, 34), false)
val MUJER = Apariencia(Color(226, 186, 160), Color(194, 148, 120), Color(132, 58, 72), Color(122, 84, 56), true)

private val CONTORNO = Color(92, 76, 70)
private val COLOR_CODO = Color(255, 215, 0)   // dorado

// ---------------------------------------------------------------- utilidades de color y geometría

private fun mezcla(a: Color, b: Color, k: Float): Color {
    val q = k.coerceIn(0f, 1f)
    return Color(a.red + (b.red - a.red) * q, a.green + (b.green - a.green) * q, a.blue + (b.blue - a.blue) * q, 1f)
}

private fun escalar(c: Color, k: Float) =
    Color((c.red * k).coerceIn(0f, 1f), (c.green * k).coerceIn(0f, 1f), (c.blue * k).coerceIn(0f, 1f), 1f)

private val BLANCO_PIEL = Color(255, 250, 245)
private fun Apariencia.contornoPiel() = mezcla(pielSombra, Color(12, 14, 20), 0.38f)

private fun ahora() = System.nanoTime() / 1e9

private fun normal(d: Offset): Offset {
    val l = d.getDistance()
    return if (l < 1e-6f) Offset(0f, -1f) else Offset(-d.y / l, d.x / l)
}

private fun bez2(p0: Offset, c: Offset, p1: Offset, n: Int): List<Offset> = List(n) { i ->
    val t = i / (n - 1f); val u = 1 - t
    p0 * (u * u) + c * (2 * u * t) + p1 * (t * t)
}

private fun bez3(p0: Offset, c1: Offset, c2: Offset, p1: Offset, n: Int): List<Offset> = List(n) { i ->
    val t = i / (n - 1f); val u = 1 - t
    p0 * (u * u * u) + c1 * (3 * u * u * t) + c2 * (3 * u * t * t) + p1 * (t * t * t)
}

private fun poligono(p: List<Offset>, cerrar: Boolean = true) = Path().apply {
    if (p.isEmpty()) return@apply
    moveTo(p[0].x, p[0].y); for (i in 1 until p.size) lineTo(p[i].x, p[i].y); if (cerrar) close()
}

private fun dentroDe(p: Offset, poli: List<Offset>): Boolean {
    var dentro = false
    var j = poli.size - 1
    for (i in poli.indices) {
        val a = poli[i]; val b = poli[j]
        if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y + 1e-9f) + a.x) dentro = !dentro
        j = i
    }
    return dentro
}

// mancha suave (sombra, rubor o brillo) con bordes difuminados; rx, ry = radios
private fun DrawScope.mancha(c: Offset, rx: Float, ry: Float, color: Color, alfa: Float) {
    if (alfa <= 0.01f || rx < 0.5f || ry < 0.5f) return
    scale(1f, ry / rx, pivot = c) {
        drawCircle(
            Brush.radialGradient(
                0f to color.copy(alpha = alfa), 0.5f to color.copy(alpha = alfa * 0.7f), 1f to color.copy(alpha = 0f),
                center = c, radius = rx
            ), rx, c
        )
    }
}

// relleno con volumen: luz arriba y al centro, sombra abajo, a los lados y en el borde
private fun DrawScope.rellenoSombreado(
    path: Path, color: Color, luz: Float = 0.10f, sombra: Float = 0.20f, borde: Float = 0.22f, centro: Float = 0.45f
) {
    val b = path.getBounds()
    if (b.width < 1f || b.height < 1f) return
    drawPath(path, Brush.verticalGradient(0f to escalar(color, 1 + luz), 1f to escalar(color, 1 - sombra * 0.6f), startY = b.top, endY = b.bottom))
    val cc = centro.coerceIn(0.15f, 0.85f)
    drawPath(path, Brush.horizontalGradient(
        0f to Color.Black.copy(alpha = sombra * 0.9f), (cc * 0.55f) to Color.Transparent,
        (cc + (1 - cc) * 0.45f) to Color.Transparent, 1f to Color.Black.copy(alpha = sombra * 0.9f),
        startX = b.left, endX = b.right
    ))
    if (borde > 0f) {
        val w = maxOf(3f, 0.10f * minOf(b.width, b.height))
        clipPath(path) {
            drawPath(path, Color.Black.copy(alpha = borde * 0.45f), style = Stroke(w))
            drawPath(path, Color.Black.copy(alpha = borde * 0.35f), style = Stroke(w * 0.45f))
        }
    }
}

// ---------------------------------------------------------------- tiempo: respiración, tragar, giro

const val PERIODO_RESPIRACION = 4.2

// 0 = aire afuera, 1 = pulmones llenos (inhala un poco más rápido de lo que exhala)
fun respiracion(t: Double = ahora()): Float {
    val f = ((t % PERIODO_RESPIRACION) / PERIODO_RESPIRACION).toFloat()
    return if (f < 0.42f) suave(f / 0.42f) else 1 - suave((f - 0.42f) / 0.58f)
}

// de vez en cuando traga saliva: la nuez sube y baja (0 = quieta, 1 = arriba)
fun trago(t: Double = ahora()): Float {
    val ventana = (t / 14.0).toLong()
    val rng = kotlin.random.Random(ventana * 7919 + 13)
    if (rng.nextDouble() < 0.25) return 0f
    val inicio = ventana * 14.0 + 1.0 + rng.nextDouble() * 11.0
    val u = (t - inicio) / 0.85
    if (u < 0 || u > 1) return 0f
    return Math.pow(sin(Math.PI * u), 1.5).toFloat()
}

// de vez en cuando gira un poquito el cuerpo y vuelve (radianes)
fun giroCuerpo(t: Double = ahora()): Float {
    val ventana = (t / 16.0).toLong()
    val rng = kotlin.random.Random(ventana * 104729 + 7)
    if (rng.nextDouble() < 0.35) return 0f
    val inicio = ventana * 16.0 + 1.0 + rng.nextDouble() * 8.0
    val amp = (if (rng.nextBoolean()) 1f else -1f) * (0.14f + rng.nextFloat() * 0.10f)
    val u = (t - inicio).toFloat()
    return when {
        u < 0f || u > 5f -> 0f
        u < 1.4f -> amp * suave(u / 1.4f)
        u < 3.6f -> amp
        else -> amp * (1 - suave((u - 3.6f) / 1.4f))
    }
}

// ---------------------------------------------------------------- texturas (poros y barba)

private val texturas = HashMap<String, ImageBitmap>()

private fun textura(nombre: String, tam: Int, semilla: Long, generar: (java.util.Random) -> Int): ImageBitmap =
    texturas.getOrPut(nombre) {
        val rng = java.util.Random(semilla)
        val bmp = android.graphics.Bitmap.createBitmap(tam, tam, android.graphics.Bitmap.Config.ARGB_8888)
        val px = IntArray(tam * tam) { generar(rng) }
        bmp.setPixels(px, 0, tam, 0, 0, tam, tam)
        bmp.asImageBitmap()
    }

// poros: puntitos un poco más oscuros y leves cambios de tono
private fun texturaPoros() = textura("poros", 96, 11) { rng ->
    val v = rng.nextGaussian()
    val a = if (rng.nextDouble() < 0.02) 55 else (kotlin.math.abs(v) * 12).toInt().coerceAtMost(40)
    val c = if (v > 0 && a < 55) 255 else 40
    (a shl 24) or (c shl 16) or (c shl 8) or c
}

// barba de pocos días: puntitos oscuros
private fun texturaBarba() = textura("barba", 64, 23) { rng ->
    if (rng.nextDouble() < 0.30) ((150 + rng.nextInt(80)) shl 24) or (0x302622) else 0
}

private fun DrawScope.texturaEn(path: Path, img: ImageBitmap, alfa: Float) {
    val b = path.getBounds()
    clipPath(path) {
        translate(b.left, b.top) {
            drawRect(ShaderBrush(ImageShader(img, TileMode.Repeated, TileMode.Repeated)), size = Size(b.width, b.height), alpha = alfa)
        }
    }
}

// ---------------------------------------------------------------- pelo de mechones

private class Mechones(n: Int) {
    val ctrl = FloatArray(n * 8)      // 4 puntos de control (x, y) por mechón, en radios de cabeza
    val tono = FloatArray(n)          // -1 oscuro .. 1 claro
    val grosor = IntArray(n)
    val fase = FloatArray(n); val fase2 = FloatArray(n); val vel = FloatArray(n)
    var giroPrevio = 0f; var tPrevio = 0.0; var inercia = 0f
    val n = n
}

private val cacheMechones = HashMap<String, Mechones>()

private fun mechones(mujer: Boolean, parte: String): Mechones = cacheMechones.getOrPut("$mujer$parte") {
    val rng = java.util.Random(if (mujer) 7L else 3L)
    fun u(a: Double, b: Double) = a + (b - a) * rng.nextDouble()
    fun g(m: Double, s: Double) = m + s * rng.nextGaussian()
    val A = 0.80
    val lista = ArrayList<DoubleArray>(); val tonos = ArrayList<Double>(); val gros = ArrayList<Int>()
    if (!mujer) {
        // pelo corto peinado hacia atrás, pegado a la cabeza
        fun dentro(x: Double, y: Double, h: Double): Pair<Double, Double> {
            val e = (x / (A * 1.03)).let { it * it } + ((y + 0.12) / 1.0).let { it * it }
            if (e > h) { val k = sqrt(h / e); return x * k to (y + 0.12) * k - 0.12 }
            return x to y
        }
        repeat(480) {
            val uu = u(-1.0, 1.0)
            val ang = Math.PI * (1.5 + 0.48 * uu)
            val linea = -0.60 + 0.30 * uu * uu
            val t0 = Math.pow(rng.nextDouble(), 0.6)
            val bordeY = -0.12 + sin(ang)
            val x0 = A * 1.02 * cos(ang) * (1 - 0.15 * (1 - t0)) + g(0.0, 0.015)
            val y0 = linea + (bordeY - linea) * (1 - t0) + g(0.0, 0.015)
            val largo = u(0.10, 0.22)
            val dx = cos(ang) * 0.25 + g(0.0, 0.06)
            var (x1, y1) = (x0 + dx * largo) to (y0 - largo * (if (kotlin.math.abs(uu) < 0.8) 0.9 else 0.3) + g(0.0, 0.02))
            dentro(x1, y1, 1.0 + u(0.0, 0.03)).let { x1 = it.first; y1 = it.second }
            val c1 = dentro(x0 + dx * largo * 0.2, y0 - largo * 0.45, 1.0)
            val c2 = dentro(x1 - dx * 0.02, y1 + largo * 0.15, 1.0)
            lista += doubleArrayOf(x0, y0, c1.first, c1.second, c2.first, c2.second, x1, y1)
            tonos += g(-0.05, 0.5); gros += if (rng.nextDouble() < 0.75) 1 else 2
        }
        for (s in intArrayOf(-1, 1)) repeat(12) {   // patillas
            val x = s * A * (0.98 + u(-0.03, 0.02)); val y0 = -0.30 + u(-0.05, 0.05)
            lista += doubleArrayOf(x, y0, x + s * 0.01, y0 + 0.06, x, y0 + 0.12, x - s * 0.01, -0.10 + u(-0.04, 0.04))
            tonos += g(-0.3, 0.3); gros += 1
        }
    } else if (parte == "arriba") {
        // raya a un lado; los mechones siguen la cabeza y caen por los lados
        fun E(t: Double, k: Double = 1.0) = doubleArrayOf(A * 1.06 * cos(t) * k, -0.12 + sin(t) * k)
        val tp = 2 * Math.PI - Math.acos(0.17)
        repeat(460) {
            val s = if (rng.nextDouble() < 0.55) -1 else 1
            val t0 = tp + g(0.0, 0.03)
            val ts = if (s < 0) Math.PI + 0.08 else 2 * Math.PI - 0.08
            val p0 = E(t0, 1.0 + u(0.0, 0.02))
            val p1: DoubleArray; val p2: DoubleArray; val p3: DoubleArray
            if (rng.nextDouble() < 0.6) {
                p1 = E(t0 + (ts - t0) * u(0.45, 0.7), 1.03 + u(0.0, 0.03))
                p2 = doubleArrayOf(s * A * u(1.06, 1.13), u(-0.25, 0.05))
                p3 = doubleArrayOf(s * A * u(0.98, 1.16), u(0.3, 1.5))
            } else {
                val uu = u(0.15, 1.0)
                val x3 = if (s < 0) s * A * uu else 0.17 * A + (A - 0.17 * A) * uu
                val y3 = -0.78 + 0.62 * Math.pow(kotlin.math.abs(x3 / A - 0.17), 1.4) + g(0.0, 0.02)
                p1 = E(t0 + (ts - t0) * uu * 0.6, 1.02 + u(0.0, 0.02))
                p2 = doubleArrayOf(x3 * 1.04, y3 - 0.25)
                p3 = doubleArrayOf(x3, y3)
            }
            lista += doubleArrayOf(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
            tonos += g(0.05, 0.5); gros += if (rng.nextDouble() < 0.85) 1 else 2
        }
    } else {
        // melena de atrás
        repeat(260) {
            val s = if (rng.nextBoolean()) 1 else -1
            lista += doubleArrayOf(s * A * u(0.5, 1.05), u(-0.9, -0.2), s * A * 1.30, 0.2, s * A * 1.25, 1.2, s * A * u(0.95, 1.35), u(1.9, 2.6))
            tonos += g(-0.15, 0.45); gros += intArrayOf(1, 2, 2, 3)[rng.nextInt(4)]
        }
    }
    val m = Mechones(lista.size)
    val rmov = java.util.Random(99L + (if (mujer) 10 else 0) + parte.length)
    for (i in lista.indices) {
        for (k in 0 until 8) m.ctrl[i * 8 + k] = lista[i][k].toFloat()
        m.tono[i] = tonos[i].coerceIn(-1.0, 1.0).toFloat(); m.grosor[i] = gros[i]
        m.fase[i] = (rmov.nextDouble() * 2 * Math.PI).toFloat(); m.fase2[i] = (rmov.nextDouble() * 2 * Math.PI).toFloat()
        m.vel[i] = (0.18 + rmov.nextDouble() * 0.27).toFloat()
    }
    m.tPrevio = ahora()
    m
}

// gira el pelo sobre la cabeza como si estuviera en una esfera
private fun girarPelo(x: Float, fi: Float, R: Float = 0.86f): Float {
    if (kotlin.math.abs(fi) < 1e-4f) return x
    val u = (x / R).coerceIn(-1f, 1f)
    val th = kotlin.math.asin(u)
    return R * sin((th + fi).coerceIn((-Math.PI / 2).toFloat(), (Math.PI / 2).toFloat())) + (x - R * u)
}

private fun DrawScope.dibujarMechones(cx: Float, cy: Float, r: Float, ap: Apariencia, parte: String, giro: Float) {
    val m = mechones(ap.mujer, parte)
    val t = ahora()
    // movimiento: cada mechón se mece con su ritmo; al girar la cabeza el pelo se queda un instante atrás
    val dt = maxOf(1e-3, t - m.tPrevio)
    val v = ((giro - m.giroPrevio) / dt).toFloat()
    m.inercia = m.inercia * 0.88f + v * 0.12f
    m.giroPrevio = giro; m.tPrevio = t
    val kIn = when { parte == "atras" -> 0.10f; ap.mujer -> 0.06f; else -> 0.015f }
    val oscuro = mezcla(ap.pelo, Color(5, 6, 8), 0.55f)
    val claro = mezcla(ap.pelo, Color(255, 236, 200), if (ap.mujer) 0.22f else 0.16f)
    val niveles = 9
    val cintas = Array(niveles) { Path() }
    val hebras = Array(niveles) { Path() }
    val pesos = floatArrayOf(0f, 0.25f, 0.65f, 1f)
    val ancho = r * (if (ap.mujer) 0.030f else 0.024f)
    val muestras = 10
    val px = FloatArray(muestras); val py = FloatArray(muestras)
    val otroX = FloatArray(muestras); val otroY = FloatArray(muestras)
    for (i in 0 until m.n) {
        val o = i * 8
        val largo = kotlin.math.abs(m.ctrl[o + 7] - m.ctrl[o + 1]) + kotlin.math.abs(m.ctrl[o + 6] - m.ctrl[o])
        val amp = when {
            parte == "atras" -> 0.035f * (largo / 2f).coerceIn(0.3f, 1.2f)
            ap.mujer -> 0.018f * (largo / 1.2f).coerceIn(0.2f, 1.2f)
            else -> 0.006f
        }
        val w = (2 * Math.PI * m.vel[i]).toFloat()
        val tt = t.toFloat()
        val mece = amp * (sin(w * tt + m.fase[i]) + 0.45f * sin(1.9f * w * tt + m.fase2[i])) - kIn * m.inercia
        val sube = 0.25f * amp * sin(w * tt * 0.7f + m.fase2[i])
        val cxs = FloatArray(4); val cys = FloatArray(4)
        for (k in 0 until 4) {
            var x = m.ctrl[o + k * 2] + mece * pesos[k]
            val y = m.ctrl[o + k * 2 + 1] + sube * pesos[k]
            x = girarPelo(x, giro)
            cxs[k] = cx + x * r; cys[k] = cy + y * r
        }
        for (s in 0 until muestras) {
            val q = s / (muestras - 1f); val u = 1 - q
            px[s] = u * u * u * cxs[0] + 3 * u * u * q * cxs[1] + 3 * u * q * q * cxs[2] + q * q * q * cxs[3]
            py[s] = u * u * u * cys[0] + 3 * u * u * q * cys[1] + 3 * u * q * q * cys[2] + q * q * q * cys[3]
        }
        val nivel = (((m.tono[i] + 1f) / 2f) * (niveles - 1)).roundToInt().coerceIn(0, niveles - 1)
        // cinta que se afina hacia la punta
        val cinta = cintas[nivel]
        for (s in 0 until muestras) {
            val a = maxOf(0, s - 1); val b = minOf(muestras - 1, s + 1)
            val nn = normal(Offset(px[b] - px[a], py[b] - py[a]))
            val q = s / (muestras - 1f)
            val perfil = Math.pow((1 - q).toDouble(), 0.6).toFloat() * (0.6f + 0.4f * sin(Math.PI.toFloat() * minOf(1f, q * 3)))
            val wv = ancho * (0.6f + 0.4f * m.grosor[i]) * perfil
            if (s == 0) cinta.moveTo(px[s] + nn.x * wv, py[s] + nn.y * wv) else cinta.lineTo(px[s] + nn.x * wv, py[s] + nn.y * wv)
            otroX[s] = px[s] - nn.x * wv; otroY[s] = py[s] - nn.y * wv
        }
        for (s in muestras - 1 downTo 0) cinta.lineTo(otroX[s], otroY[s])
        cinta.close()
        val hebra = hebras[nivel]
        hebra.moveTo(px[0], py[0]); for (s in 1 until muestras) hebra.lineTo(px[s], py[s])
    }
    fun colorDe(n: Int, k: Float): Color {
        val tn = (n / (niveles - 1f)) * 2 - 1
        return if (tn > 0) mezcla(ap.pelo, claro, tn * k) else mezcla(ap.pelo, oscuro, -tn * k)
    }
    for (n in 0 until niveles) drawPath(cintas[n], colorDe(n, 0.8f))
    val gh = maxOf(0.8f, r / 90f)
    for (n in 0 until niveles) drawPath(hebras[n], if (n < niveles / 2) colorDe(0, 0.6f) else colorDe(niveles - 1, 1f),
        style = Stroke(gh, cap = StrokeCap.Round, join = StrokeJoin.Round), alpha = 0.7f)
    // brillo en banda: la luz se refleja en una franja que rodea la cabeza
    val banda = Path().apply {
        fillType = PathFillType.EvenOdd
        addOval(Rect(cx - r * 0.86f, cy - r * 0.15f - r * 0.84f, cx + r * 0.86f, cy - r * 0.15f + r * 0.84f))
        addOval(Rect(cx - r * 0.72f, cy - r * 0.15f - r * 0.70f, cx + r * 0.72f, cy - r * 0.15f + r * 0.70f))
    }
    clipPath(banda) {
        val luz = mezcla(ap.pelo, Color(255, 240, 215), 0.55f)
        for (n in 0 until niveles) drawPath(cintas[n], luz, alpha = 0.28f)
    }
}

private fun DrawScope.peloLargo(c: Offset, r: Float, ap: Apariencia, giro: Float) {
    val A = r * 0.80f
    val izq = bez3(Offset(c.x - A * 0.9f, c.y - r * 0.7f), Offset(c.x - A * 1.40f, c.y + r * 0.2f), Offset(c.x - A * 1.30f, c.y + r * 1.6f), Offset(c.x - A * 1.15f, c.y + r * 2.45f), 16)
    val der = bez3(Offset(c.x + A * 0.9f, c.y - r * 0.7f), Offset(c.x + A * 1.40f, c.y + r * 0.2f), Offset(c.x + A * 1.30f, c.y + r * 1.6f), Offset(c.x + A * 1.15f, c.y + r * 2.45f), 16).reversed()
    val fi = giro * 0.6f
    val poli = (izq + der).map { Offset(c.x + girarPelo((it.x - c.x) / r, fi) * r, it.y) }
    rellenoSombreado(poligono(poli), mezcla(ap.pelo, Color.Black, 0.35f), luz = 0.1f, sombra = 0.3f, borde = 0.3f)
    dibujarMechones(c.x, c.y, r, ap, "atras", fi)
}

// ---------------------------------------------------------------- cabeza

private fun DrawScope.boca(bx: Float, by: Float, r: Float, sonrisa: Float, labios: Color, labioOsc: Color) {
    val w = r * (0.18f + 0.03f * sonrisa)
    val h = r * (0.11f + 0.08f * sonrisa)
    val sube = r * 0.05f * sonrisa
    val izq = Offset(bx - w, by - sube); val der = Offset(bx + w, by - sube)
    val arriba = bez2(izq, Offset(bx, by - h * 0.25f), der, 20)
    val abajo = bez2(der, Offset(bx, by + h * 1.6f), izq, 20)
    val bocaP = poligono(arriba + abajo.drop(1))
    fun bordeSup(x: Float): Float { val u = (x - (bx - w)) / (2 * w); return (1 - u) * (1 - u) * izq.y + 2 * (1 - u) * u * (by - h * 0.25f) + u * u * der.y }
    fun bordeInf(x: Float): Float { val u = (x - (bx - w)) / (2 * w); return (1 - u) * (1 - u) * izq.y + 2 * (1 - u) * u * (by + h * 1.6f) + u * u * der.y }
    clipPath(bocaP) {
        // fondo: garganta oscura, paladar rojizo, mejillas por dentro
        drawPath(bocaP, Color(58, 20, 26))
        mancha(Offset(bx, by + h * 0.35f), w * 0.75f, h * 0.55f, Color(104, 38, 48), 1f)
        mancha(Offset(bx, by + h * 0.55f), w * 0.28f, h * 0.30f, Color(30, 8, 12), 0.9f)
        for (s in intArrayOf(-1, 1)) mancha(Offset(bx + s * w * 0.86f, by + h * 0.4f), w * 0.2f, h * 0.7f, Color(140, 62, 72), 0.9f)
        // lengua con surco, papilas y brillo
        val lx = bx; val ly = by + h * 0.80f
        val lengua = Color(196, 92, 104)
        drawOval(mezcla(lengua, Color(90, 30, 40), 0.35f), Offset(lx - w * 0.70f, ly - h * 0.62f), Size(w * 1.4f, h * 1.24f))
        drawOval(lengua, Offset(lx - w * 0.64f, ly - h * 0.54f + 1f), Size(w * 1.28f, h * 1.08f))
        drawLine(mezcla(lengua, Color(90, 30, 40), 0.5f), Offset(lx, ly - h * 0.45f), Offset(lx, ly + h * 0.15f), maxOf(0.8f, r / 90f))
        mancha(Offset(lx - w * 0.22f, ly - h * 0.28f), w * 0.16f, h * 0.09f, Color(228, 150, 160), 0.9f)
        // dientes de abajo
        val inferiores = floatArrayOf(-1f, -0.78f, -0.56f, -0.36f, -0.17f, 0f, 0.17f, 0.36f, 0.56f, 0.78f, 1f)
        for (k in 0 until inferiores.size - 1) {
            val t0 = inferiores[k]; val t1 = inferiores[k + 1]
            val xa = bx + t0 * w * 0.70f; val xb = bx + t1 * w * 0.70f
            val lejos = maxOf(kotlin.math.abs(t0), kotlin.math.abs(t1))
            val largo = h * 0.30f * (1 - 0.35f * Math.pow(lejos.toDouble(), 1.5).toFloat())
            val ba = bordeInf(xa) - h * 0.10f; val bb = bordeInf(xb) - h * 0.10f
            drawPath(poligono(listOf(Offset(xa + 0.6f, ba + h * 0.2f), Offset(xa + 0.6f, ba - largo * 0.9f), Offset((xa + xb) / 2, (ba + bb) / 2 - largo),
                Offset(xb - 0.6f, bb - largo * 0.9f), Offset(xb - 0.6f, bb + h * 0.2f))), mezcla(Color(240, 236, 226), Color(190, 180, 166), 0.20f + 0.45f * lejos))
            drawLine(Color(170, 155, 146), Offset(xa, ba - largo * 0.85f), Offset(xa, ba + 1f), maxOf(0.6f, r / 120f))
        }
        drawPath(poligono(abajo + abajo.reversed().map { it - Offset(0f, h * 0.14f) }), Color(206, 104, 116))   // encía de abajo
        drawPath(poligono(arriba + arriba.reversed().map { it + Offset(0f, h * 0.20f) }), Color(214, 112, 122))  // encía de arriba
        // dientes de arriba: los del centro más grandes, los de los lados más chicos y en sombra
        val dientes = floatArrayOf(-1f, -0.80f, -0.60f, -0.36f, 0f, 0.36f, 0.60f, 0.80f, 1f)
        for (k in 0 until dientes.size - 1) {
            val t0 = dientes[k]; val t1 = dientes[k + 1]
            val xa = bx + t0 * w * 0.78f; val xb = bx + t1 * w * 0.78f
            val lejos = maxOf(kotlin.math.abs(t0), kotlin.math.abs(t1))
            val largo = h * 0.36f * (1 - 0.40f * Math.pow(lejos.toDouble(), 1.5).toFloat())
            val ra = bordeSup(xa) + h * 0.10f; val rb = bordeSup(xb) + h * 0.10f
            val tono = mezcla(Color(246, 243, 234), Color(200, 190, 175), 0.12f + 0.45f * lejos)
            drawPath(poligono(listOf(Offset(xa + 0.8f, ra - h * 0.06f), Offset(xb - 0.8f, rb - h * 0.06f), Offset(xb - 0.8f, rb + largo * 0.9f),
                Offset((xa + xb) / 2, (ra + rb) / 2 + largo), Offset(xa + 0.8f, ra + largo * 0.9f))), tono)
            drawLine(Color(185, 172, 162), Offset(xa, ra), Offset(xa, ra + largo * 0.85f), maxOf(0.6f, r / 120f))
            drawLine(Color.White.copy(alpha = 0.8f), Offset((xa + xb) / 2 - (xb - xa) * 0.12f, ra + largo * 0.25f), Offset((xa + xb) / 2 - (xb - xa) * 0.12f, ra + largo * 0.6f), maxOf(0.6f, r / 120f))
        }
        drawPath(poligono(arriba + arriba.reversed().map { it + Offset(0f, h * 0.16f) }), Color(90, 50, 55), alpha = 0.3f)   // sombra del labio
    }
    // labios con volumen
    val g = maxOf(1f, r / 30f)
    drawPath(poligono(arriba + bez2(der + Offset(0f, -h * 0.05f), Offset(bx, by - h * 0.55f), izq + Offset(0f, -h * 0.05f), 20)), labioOsc)
    drawPath(poligono(abajo + bez2(izq, Offset(bx, by + h * 2.15f), der, 20)), labios)
    drawPath(poligono(arriba, false), mezcla(labioOsc, CONTORNO, 0.4f), style = Stroke(g, cap = StrokeCap.Round))
    for (p in listOf(izq, der)) drawCircle(mezcla(labioOsc, CONTORNO, 0.5f), maxOf(1f, r / 40f), p)
}

// arrugas sutiles con los gestos y rasgos que siempre están
private fun DrawScope.arrugas(cx: Float, cy: Float, r: Float, ap: Apariencia, sonrisa: Float, cejas: FloatArray, bostezo: Float, gx: Float) {
    val color = mezcla(ap.piel, ap.pielSombra, 0.85f)
    val g = maxOf(0.8f, r / 90f)
    fun curva(p0: Offset, c: Offset, p1: Offset, fuerza: Float) {
        if (fuerza <= 0.05f) return
        val f = fuerza.coerceAtMost(1f)
        drawPath(poligono(bez2(p0, c, p1, 10), false), mezcla(ap.piel, color, f), style = Stroke(g, cap = StrokeCap.Round))
        val l = Offset(0f, -1.5f)
        drawPath(poligono(bez2(p0 + l, c + l, p1 + l, 10), false), mezcla(ap.piel, BLANCO_PIEL, 0.18f * f), style = Stroke(g, cap = StrokeCap.Round))
    }
    val s = maxOf(0f, sonrisa)
    val x = cx + gx
    for (lado in intArrayOf(-1, 1)) curva(Offset(x + lado * r * 0.035f, cy + r * 0.28f), Offset(x + lado * r * 0.045f, cy + r * 0.34f), Offset(x + lado * r * 0.05f, cy + r * 0.40f), 0.30f)
    curva(Offset(x - r * 0.12f, cy + r * 0.70f), Offset(x, cy + r * (0.66f + 0.02f * s)), Offset(x + r * 0.12f, cy + r * 0.70f), 0.35f)
    mancha(Offset(x, cy + r * 0.83f), r * 0.10f, r * 0.05f, ap.pielSombra, 0.3f)
    for (lado in intArrayOf(-1, 1)) {
        val ox = x + lado * r * 0.34f
        curva(Offset(ox - r * 0.10f, cy + r * 0.10f), Offset(ox, cy + r * 0.15f), Offset(ox + r * 0.11f, cy + r * 0.09f), 0.22f)
        mancha(Offset(x + lado * r * 0.62f, cy - r * 0.30f), r * 0.14f, r * 0.24f, ap.pielSombra, 0.3f)
        mancha(Offset(x + lado * r * 0.50f, cy + r * (0.42f - 0.05f * s)), r * 0.14f, r * 0.2f, ap.pielSombra, 0.2f)
        curva(Offset(x + lado * r * 0.15f, cy + r * 0.22f), Offset(x + lado * r * 0.30f, cy + r * 0.40f), Offset(x + lado * r * (0.30f + 0.04f * s), cy + r * (0.50f - 0.04f * s)), 0.08f + 0.55f * s)
        val ox2 = x + lado * r * 0.52f; val oy2 = cy - r * 0.07f
        for (ang in floatArrayOf(-25f, 0f, 25f)) {
            val a = Math.toRadians(ang.toDouble())
            val p1 = Offset(ox2 + lado * r * 0.12f * cos(a).toFloat(), oy2 + r * 0.12f * sin(a).toFloat())
            curva(Offset(ox2, oy2), (Offset(ox2, oy2) + p1) / 2f + Offset(0f, r * 0.01f), p1, 0.75f * s - 0.15f)
        }
        curva(Offset(ox - r * 0.12f, cy + r * 0.06f), Offset(ox, cy + r * (0.11f - 0.02f * s)), Offset(ox + r * 0.12f, cy + r * 0.06f), 0.05f + 0.55f * s)
    }
    val subeC = maxOf(0f, maxOf(cejas[0], cejas[1]))
    for (k in 0 until 3) {
        val y = cy - r * (0.36f + 0.07f * k)
        curva(Offset(x - r * 0.30f, y + r * 0.02f), Offset(x, y - r * 0.03f), Offset(x + r * 0.30f, y + r * 0.02f), 0.8f * subeC - 0.15f * k)
    }
    val frunce = maxOf(0f, -minOf(cejas[0], cejas[1]))
    for (lado in intArrayOf(-1, 1)) curva(Offset(x + lado * r * 0.05f, cy - r * 0.28f), Offset(x + lado * r * 0.06f, cy - r * 0.20f), Offset(x + lado * r * 0.04f, cy - r * 0.13f), 1.2f * frunce)
    curva(Offset(x - r * 0.14f, cy + r * 0.86f), Offset(x, cy + r * 0.82f), Offset(x + r * 0.14f, cy + r * 0.86f), 0.9f * bostezo)
}

// cabeza del muñeco, parpadeo y sonrisa van de 0 a 1
private fun DrawScope.cabeza(
    c0: Offset, r: Float, ap: Apariencia, parpadeo: Float, sonrisa: Float, cejas: FloatArray = floatArrayOf(0f, 0f),
    giro: Float = 0f, besando: Boolean = false, gestosCara: Gestos? = null, guino: Boolean = false,
    conCuello: Boolean = true, aire: Float = respiracion()
) {
    val mx = (gestosCara?.miradaX ?: 0f) * r * 0.06f
    val my = (gestosCara?.miradaY ?: 0f) * r * 0.035f
    val bizco = gestosCara?.bizco ?: 0f
    val bostezo = gestosCara?.bostezo ?: 0f
    val A = r * 0.80f
    val colorCejas = if (ap.mujer) mezcla(ap.pelo, CONTORNO, 0.45f) else Color(52, 42, 38)
    val pielOsc = ap.pielSombra
    if (conCuello) {
        if (ap.mujer) peloLargo(c0, r, ap, giro)
        val cw = r * (if (ap.mujer) 0.24f else 0.30f)
        drawRect(pielOsc, Offset(c0.x - cw, c0.y + r * 0.6f), Size(cw * 2, r * 1.4f))
    }
    // al girar se corre toda la cabeza; las orejas van al lado contrario
    val giroPx = giro * r * 0.45f
    val fiPelo = giro * 0.55f
    val cx = c0.x + giroPx * 0.35f; val cy = c0.y
    for (lado in intArrayOf(-1, 1)) {
        val ox = cx + lado * A * 0.98f - giroPx * (if (lado * giroPx < 0) 0.25f else 0.6f)
        drawOval(ap.contornoPiel(), Offset(ox - r * 0.13f, cy - r * 0.22f), Size(r * 0.26f, r * 0.46f))
        drawOval(ap.piel, Offset(ox - r * 0.11f, cy - r * 0.20f), Size(r * 0.22f, r * 0.42f))
        drawOval(pielOsc, Offset(ox - r * 0.06f, cy - r * 0.13f), Size(r * 0.12f, r * 0.28f), style = Stroke(maxOf(1f, r / 30f)))
    }
    // forma de la cara: frente redonda, mandíbula y barbilla
    val arriba = List(24) { i -> val t = Math.PI + Math.PI * i / 23.0; Offset(cx + A * cos(t).toFloat(), cy - r * 0.05f + r * 0.95f * sin(t).toFloat()) }
    val der = bez3(Offset(cx + A, cy - r * 0.05f), Offset(cx + A, cy + r * 0.55f), Offset(cx + A * 0.55f, cy + r * 0.98f), Offset(cx, cy + r), 14)
    val izq = bez3(Offset(cx, cy + r), Offset(cx - A * 0.55f, cy + r * 0.98f), Offset(cx - A, cy + r * 0.55f), Offset(cx - A, cy - r * 0.05f), 14)
    val caraP = poligono(arriba + der.drop(1) + izq.drop(1))
    drawPath(caraP, ap.contornoPiel(), style = Stroke(maxOf(2f, r / 24f) + 1f, join = StrokeJoin.Round))
    rellenoSombreado(caraP, ap.piel, luz = 0.06f, sombra = 0.10f, borde = 0.12f)
    texturaEn(caraP, texturaPoros(), 0.55f)          // piel con poros (no perfecta)
    val gxm = giro * r * 0.45f * 0.75f
    if (!ap.mujer) {
        // barba de pocos días en la mandíbula, la barbilla y el bigote (sin los labios)
        val zona = Path().apply {
            addArc(Rect(cx + gxm * 0.6f - A * 0.98f, cy + r * 0.40f - r * 0.62f, cx + gxm * 0.6f + A * 0.98f, cy + r * 0.40f + r * 0.62f), 0f, 180f)
            close()
            addOval(Rect(cx + gxm - r * 0.25f, cy + r * 0.37f - r * 0.075f, cx + gxm + r * 0.25f, cy + r * 0.37f + r * 0.075f))
        }
        val labiosZona = Path().apply { addOval(Rect(cx + gxm - r * 0.22f, cy + r * 0.50f - r * 0.065f, cx + gxm + r * 0.22f, cy + r * 0.50f + r * 0.065f)) }
        clipPath(caraP) { clipPath(labiosZona, clipOp = ClipOp.Difference) { texturaEn(zona, texturaBarba(), 0.40f) } }
    }
    // pelo: sombra en la frente, base y mechones
    if (!ap.mujer) {
        mancha(Offset(cx, cy - r * 0.55f), A * 0.9f, r * 0.22f, mezcla(ap.piel, ap.pelo, 0.4f), 0.35f)
        val base = List(24) { i -> val t = Math.PI * 1.06 + Math.PI * 0.88 * i / 23.0; Offset(cx + A * cos(t).toFloat(), cy - r * 0.10f + r * 0.98f * sin(t).toFloat()) } +
            bez2(Offset(cx + A * 0.92f, cy - r * 0.30f), Offset(cx, cy - r * 0.78f), Offset(cx - A * 0.92f, cy - r * 0.30f), 12)
        rellenoSombreado(poligono(base), mezcla(ap.pelo, Color.Black, 0.25f), luz = 0.1f, sombra = 0.2f, borde = 0.2f)
    } else {
        mancha(Offset(cx - A * 0.2f, cy - r * 0.62f), A * 0.8f, r * 0.2f, mezcla(ap.piel, ap.pelo, 0.4f), 0.30f)
        val base = List(28) { i -> val t = Math.PI * 0.98 + Math.PI * 1.04 * i / 27.0; Offset(cx + A * 1.06f * cos(t).toFloat(), cy - r * 0.12f + r * 0.96f * sin(t).toFloat()) } +
            bez3(Offset(cx + A * 1.04f, cy - r * 0.1f), Offset(cx + A * 0.7f, cy - r * 0.6f), Offset(cx + A * 0.35f, cy - r * 0.68f), Offset(cx + A * 0.18f, cy - r * 0.78f), 10) +
            bez3(Offset(cx + A * 0.18f, cy - r * 0.78f), Offset(cx - A * 0.3f, cy - r * 0.55f), Offset(cx - A * 0.85f, cy - r * 0.45f), Offset(cx - A * 1.04f, cy - r * 0.1f), 12)
        rellenoSombreado(poligono(base), mezcla(ap.pelo, Color.Black, 0.25f), luz = 0.1f, sombra = 0.2f, borde = 0.2f)
    }
    dibujarMechones(cx, cy, r, ap, "arriba", fiPelo)
    // cuencas de los ojos, sombra de barba, mejillas
    for (lado in intArrayOf(-1, 1)) mancha(Offset(cx + giro * r * 0.34f + lado * r * 0.34f, cy - r * 0.10f), r * 0.20f, r * 0.12f, pielOsc, 0.45f)
    if (!ap.mujer) mancha(Offset(cx, cy + r * 0.40f), r * 0.26f, r * 0.08f, Color(120, 100, 92), 0.10f)
    for (lado in intArrayOf(-1, 1)) {
        val aleja = 1 - 0.35f * maxOf(0f, -lado * giro)
        mancha(Offset(cx + gxm + lado * r * 0.45f * (0.85f + 0.15f * aleja), cy + r * 0.28f), r * 0.16f * aleja, r * 0.09f,
            Color(230, 120, 120), if (ap.mujer) 0.35f else 0.15f)
    }
    val gx = giro * r * 0.45f * 0.75f
    arrugas(cx, cy, r, ap, sonrisa, cejas, bostezo, gx)
    // cejas y ojos
    val g = maxOf(2f, r / 24f)
    val irisColor = if (ap.mujer) Color(70, 120, 120) else Color(95, 65, 45)
    for ((k, lado) in intArrayOf(-1, 1).withIndex()) {
        val x = cx + gx + lado * r * 0.34f * (1 - 0.25f * kotlin.math.abs(giro))
        val sube = sonrisa * r * 0.03f + cejas[k] * r * 0.13f
        val ceja = bez2(Offset(x - lado * r * 0.17f, cy - r * 0.22f - sube), Offset(x + lado * r * 0.02f, cy - r * 0.31f - sube - cejas[k] * r * 0.03f),
            Offset(x + lado * r * 0.19f, cy - r * 0.24f - sube), 8)
        for (j in 0 until ceja.size - 1) drawLine(colorCejas, ceja[j], ceja[j + 1], maxOf(1f, g * (1.3f - j / ceja.size.toFloat() * 0.7f)), cap = StrokeCap.Round)
        val cierre = gestosCara?.ojos?.get(k) ?: maxOf(parpadeo, 0.75f * bostezo)
        val alto = r * 0.15f * (1 - if (guino && k == 0) 1f else cierre)
        val ey = cy - r * 0.06f
        if (alto > r * 0.02f) {
            val ojo = poligono(bez2(Offset(x - r * 0.15f, ey), Offset(x, ey - alto * 1.2f), Offset(x + r * 0.15f, ey), 10) +
                bez2(Offset(x + r * 0.15f, ey), Offset(x, ey + alto * 0.9f), Offset(x - r * 0.15f, ey), 10).drop(1))
            drawPath(ojo, Color(246, 244, 240))
            val pupila = Offset(x + mx - lado * bizco * r * 0.075f, ey + my)
            clipPath(ojo) {
                drawCircle(irisColor, maxOf(2f, r * 0.075f), pupila)
                drawCircle(Color(12, 12, 15), maxOf(1f, r * 0.035f), pupila)
                drawCircle(Color.White, maxOf(1f, r * 0.016f), pupila + Offset(r * 0.025f, -r * 0.025f))
            }
            val parpado = bez2(Offset(x - r * 0.16f, ey + r * 0.005f), Offset(x, ey - alto * 1.25f), Offset(x + r * 0.16f, ey + r * 0.005f), 10)
            drawPath(poligono(parpado, false), Color(45, 35, 32), style = Stroke(maxOf(1f, r / 30f) + 1f, cap = StrokeCap.Round))
            if (ap.mujer) for (tq in floatArrayOf(0.2f, 0.5f, 0.8f)) {
                val p0 = parpado[(tq * (parpado.size - 1)).toInt()]
                drawLine(Color(45, 35, 32), p0, p0 + Offset(lado * r * 0.05f * (tq - 0.2f), -r * 0.05f), maxOf(0.8f, r / 80f))
            }
            drawPath(poligono(bez2(Offset(x - r * 0.12f, ey + alto * 0.95f), Offset(x, ey + alto * 1.15f), Offset(x + r * 0.12f, ey + alto * 0.95f), 8), false),
                mezcla(ap.piel, CONTORNO, 0.25f), style = Stroke(maxOf(0.8f, r / 80f)))
        } else {
            drawPath(poligono(bez2(Offset(x - r * 0.15f, ey), Offset(x, ey + r * 0.04f), Offset(x + r * 0.15f, ey), 8), false),
                Color(45, 35, 32), style = Stroke(maxOf(2f, r / 26f), cap = StrokeCap.Round))
        }
    }
    // nariz que respira: al inhalar las aletas se abren y las fosas crecen; al exhalar se recogen
    val cn = Offset(cx + gx * 1.1f, cy)
    val abre = 1f + 0.10f * aire
    val fosa = 0.75f + 0.55f * aire
    drawPath(poligono(bez2(Offset(cn.x - r * 0.03f, cn.y - r * 0.12f), Offset(cn.x - r * 0.10f, cn.y + r * 0.12f), Offset(cn.x - r * 0.07f, cn.y + r * 0.24f), 8), false),
        mezcla(ap.piel, pielOsc, 0.8f), style = Stroke(maxOf(1f, r / 26f), cap = StrokeCap.Round))
    for (lado in intArrayOf(-1, 1)) {
        val ac = Offset(cn.x + lado * r * 0.10f * abre, cn.y + r * 0.205f)
        val rx = r * 0.055f * abre; val ry = r * 0.05f
        drawArc(mezcla(ap.piel, pielOsc, 0.7f), if (lado < 0) 100f else -80f, 160f, false, Offset(ac.x - rx, ac.y - ry), Size(rx * 2, ry * 2),
            style = Stroke(maxOf(1f, r / 34f), cap = StrokeCap.Round))
    }
    mancha(Offset(cn.x, cn.y + r * 0.19f), r * 0.06f, r * 0.035f, BLANCO_PIEL, 0.5f)
    for (lado in intArrayOf(-1, 1)) {
        val an = r * 0.05f * fosa; val al = r * 0.028f * fosa
        drawOval(mezcla(pielOsc, CONTORNO, 0.55f), Offset(cn.x + lado * r * 0.068f * abre - an / 2, cn.y + r * 0.235f - al / 2), Size(an, al))
    }
    // boca
    val labios = if (ap.mujer) Color(196, 82, 98) else Color(176, 112, 104)
    val labioOsc = mezcla(labios, CONTORNO, 0.35f)
    if (besando) {
        val bx = cx + giro * r * 0.62f
        drawOval(labios, Offset(bx - r * 0.10f, cy + r * 0.42f), Size(r * 0.20f, r * 0.17f))
        drawOval(Color(110, 30, 40), Offset(bx - r * 0.035f, cy + r * 0.475f), Size(r * 0.07f, r * 0.06f))
        return
    }
    val bx = cx + gx; val by = cy + r * 0.50f
    if (sonrisa > 0.3f) { boca(bx, by, r, sonrisa, labios, labioOsc); return }
    val abierta = maxOf(bostezo, gestosCara?.boca ?: 0f)
    if (abierta > 0.05f) {
        val an = r * (0.22f + 0.10f * abierta); val al = r * (0.08f + 0.36f * abierta)
        drawOval(labios, Offset(bx - an / 2 - r * 0.02f, by - al / 2 - r * 0.02f), Size(an + r * 0.04f, al + r * 0.04f))
        drawOval(Color(90, 25, 35), Offset(bx - an / 2, by - al / 2), Size(an, al))
        return
    }
    // labios cerrados: el de arriba con arco de cupido, el de abajo más lleno y con brillo
    val ancho = r * (0.40f + 0.18f * sonrisa)
    val w2 = ancho / 2 * (1 + 0.15f * sonrisa)
    val curva = sonrisa * r * 0.05f
    val sup = bez2(Offset(bx - w2, by - curva), Offset(bx - w2 * 0.5f, by - r * 0.07f), Offset(bx, by - r * 0.035f), 8) +
        bez2(Offset(bx, by - r * 0.035f), Offset(bx + w2 * 0.5f, by - r * 0.07f), Offset(bx + w2, by - curva), 8).drop(1) +
        bez2(Offset(bx + w2, by - curva), Offset(bx, by + r * 0.01f + curva), Offset(bx - w2, by - curva), 10).drop(1)
    val inf = bez2(Offset(bx - w2 * 0.92f, by - curva * 0.8f), Offset(bx, by + r * 0.01f + curva), Offset(bx + w2 * 0.92f, by - curva * 0.8f), 10) +
        bez2(Offset(bx + w2 * 0.92f, by - curva * 0.8f), Offset(bx, by + r * (if (ap.mujer) 0.13f else 0.10f)), Offset(bx - w2 * 0.92f, by - curva * 0.8f), 10).drop(1)
    drawPath(poligono(inf), mezcla(labios, Color.White, 0.08f))
    drawPath(poligono(sup), labioOsc)
    drawPath(poligono(bez2(Offset(bx - w2, by - curva), Offset(bx, by + r * 0.01f + curva), Offset(bx + w2, by - curva), 10), false),
        mezcla(labioOsc, CONTORNO, 0.5f), style = Stroke(maxOf(0.8f, r / 80f)))
    mancha(Offset(bx, by + r * 0.05f), w2 * 0.25f, r * 0.015f, BLANCO_PIEL, 0.45f)
}

// gestos de la cara, la mirada va de -1 a 1 y lo demas de 0 a 1
class Gestos(
    val parpadeo: Float, val sonrisa: Float, val cejas: FloatArray,
    val miradaX: Float = 0f, val miradaY: Float = 0f, val bizco: Float = 0f, val lengua: Float = 0f,
    val bostezo: Float = 0f,
    val ojos: FloatArray? = null,   // en vivo: cuánto está cerrado cada ojo (0 abierto, 1 cerrado)
    val boca: Float = 0f            // en vivo: cuánto está abierta la boca (0 a 1)
) {
    // mezcla el parpadeo con la cara de un gesto
    fun con(movimiento: Gestos?): Gestos = if (movimiento == null) this else Gestos(
        maxOf(parpadeo, movimiento.parpadeo), movimiento.sonrisa, movimiento.cejas,
        movimiento.miradaX, movimiento.miradaY, movimiento.bizco, 0f, movimiento.bostezo
    )
}

// 0..1 dentro de la ventana [desde, hasta) con subida y bajada de 300 ms
private fun ventana(fase: Long, desde: Long, hasta: Long): Float = when {
    fase < desde || fase >= hasta -> 0f
    fase < desde + 300 -> suave((fase - desde) / 300f)
    fase >= hasta - 300 -> suave((hasta - fase) / 300f)
    else -> 1f
}

private fun suave(t: Float) = t.coerceIn(0f, 1f).let { it * it * (3 - 2 * it) }

// gestos de la cara segun el tiempo: parpadeo, sonrisa, cejas, mirada y biscos (ya no saca la lengua)
fun gestos(ms: Long): Gestos {
    val (parpadeo, sonrisa) = gestosBase(ms)
    val ciclo = ms / 9000
    val fase = ms % 9000
    val cual = ((ciclo * 7919 + 3) % 5).toInt()  // que ceja sube
    val nivel = when {
        fase in 2500 until 2800 -> suave((fase - 2500) / 300f)
        fase in 2800 until 3900 -> 1f
        fase in 3900 until 4200 -> 1f - suave((fase - 3900) / 300f)
        else -> 0f
    }
    val cejas = when (cual) {
        0, 1 -> floatArrayOf(nivel, 0f)
        2 -> floatArrayOf(0f, nivel)
        3 -> floatArrayOf(nivel, nivel)
        else -> floatArrayOf(0f, 0f)
    }
    val cicloM = ms / 6500
    val (dirX, dirY) = when (((cicloM * 31 + 7) % 6).toInt()) {
        0 -> -1f to 0f; 1 -> 1f to 0f; 2 -> 0f to 1f; 3 -> -0.8f to 0.7f; 4 -> 0.8f to -0.8f; else -> 0f to 0f
    }
    val mira = ventana(ms % 6500, 1200, 3300)
    val bizco = if ((ms / 17000) % 3 == 1L) ventana(ms % 17000, 500, 1900) else 0f
    return Gestos(parpadeo, sonrisa, cejas, dirX * mira * (1 - bizco), dirY * mira * (1 - bizco), bizco, 0f)
}

private fun gestosBase(ms: Long): Pair<Float, Float> {
    val fase = ms % 2000
    val parpadeo = when {
        fase < 70 -> fase / 70f
        fase < 140 -> 1f - (fase - 70) / 70f
        else -> 0f
    }
    val ciclo = ms % 11000                    // sonrisa cada 11 s
    val sonrisa = when {
        ciclo in 7000 until 7300 -> (ciclo - 7000) / 300f
        ciclo in 7300 until 8800 -> 1f
        ciclo in 8800 until 9100 -> 1f - (ciclo - 8800) / 300f
        else -> 0f
    }
    return parpadeo to sonrisa
}

// estiramiento horizontal del muñeco
const val ANCHO_MUNIECO = 1.35f

fun escalaMunieco(area: Rect) = minOf(area.width / (3.6f * ANCHO_MUNIECO), area.height / 3.6f)
// muñeco que imita a la persona EN VIVO (TRADUCIR): brazos, manos, giro de la cabeza y el cuerpo.
// Devuelve la pose de un cuadro y cuánto se corre el cuerpo (dx), o null si no se ve el cuerpo.
fun avatarEnVivo(e: co.edu.itm.manosquehablan.vivo.Escena): Triple<AvatarSena, Float, Gestos>? {
    val (centro, ancho) = e.hombros ?: return null
    if (ancho < 1e-6) return null
    fun n(p: P) = (p - centro) / ancho
    val lados = Array(2) { l ->
        val ab = e.antebrazos.getOrNull(l) ?: return@Array null
        val (muneca, codo) = ab
        val mano = e.manos.getOrNull(l)
        LadoAvatar(arrayOf(n(muneca)), arrayOf(n(codo)), mano?.let { h -> arrayOf(Array(h.size) { n(h[it]) }) })
    }
    // giro de la cabeza: donde queda la nariz entre los dos lados de la cara (0.5 = de frente)
    var giro: DoubleArray? = null
    var incl: DoubleArray? = null
    var ojos: FloatArray? = null
    var boca = 0f
    e.caraSuave?.let { c ->
        if (c.size > 454) {
            val w = c[454].x - c[234].x
            // en signo contrario: el muñeco es un espejo y esta medida sale de la cámara sin espejo
            if (kotlin.math.abs(w) > 1e-3) giro = doubleArrayOf((-((c[1].x - c[234].x) / w - 0.5) * 4.5).coerceIn(-1.4, 1.4))
            // ojos: qué tan abiertos (párpados / ancho del ojo). Espejo: tu ojo derecho es el de la derecha del muñeco
            fun cierre(arriba: Int, abajo: Int, a: Int, b: Int): Float {
                val ancho = (c[a] - c[b]).norma()
                if (ancho < 1e-3) return 0f
                return ((0.22 - (c[arriba] - c[abajo]).norma() / ancho) / 0.14).coerceIn(0.0, 1.0).toFloat()
            }
            ojos = floatArrayOf(cierre(386, 374, 362, 263), cierre(159, 145, 33, 133))
            val ojosAncho = (c[33] - c[263]).norma()
            if (ojosAncho > 1e-3) boca = (((c[13] - c[14]).norma() / ojosAncho - 0.05) / 0.25).coerceIn(0.0, 1.0).toFloat()
            // inclinación: ángulo de la línea entre los ojos (el muñeco está en espejo)
            val ang = Math.toDegrees(Math.atan2(c[263].y - c[33].y, c[263].x - c[33].x))
            incl = doubleArrayOf((-ang).coerceIn(-20.0, 20.0))
        }
    }
    // el cuerpo se corre como el de la persona (en espejo)
    val dx = ((e.ancho / 2.0 - centro.x) / ancho * 0.5).coerceIn(-1.0, 1.0).toFloat()
    val gestos = Gestos(0f, 0f, floatArrayOf(0f, 0f), ojos = ojos, boca = boca)
    return Triple(AvatarSena(lados, 1, 30.0, sinTrazo = true, giro = giro, inclinacion = incl), dx, gestos)
}

// area para que el muñeco ocupe TODO el alto (de la cabeza a los pies), centrado en cx
// (es más ancha que el muñeco: los brazos pueden salirse un poco del lienzo)
fun areaPorAlto(cx: Float, arriba: Float, alto: Float) =
    Rect(cx - alto * ANCHO_MUNIECO / 2f, arriba - alto * 0.06f, cx + alto * ANCHO_MUNIECO / 2f, arriba + alto * 0.94f)
const val CENTRO_X_BRAZOS = 0.9f   // manos frente a la cara: menos estiradas (como en el PC)
private val COLOR_JEAN = Color(52, 70, 104)

// cabeza asomandose por la derecha, visible de 0 a 1
fun DrawScope.dibujarAsomo(area: Rect, apariencia: Apariencia, visible: Float, guino: Boolean) {
    if (visible <= 0f) return
    val escala = escalaMunieco(area)
    val r = escala * 0.36f
    val c = Offset(area.right + r * 1.3f - visible * r * 2.3f, area.top + (1.25f - 0.78f) * escala)
    rotate(-22f, c) {
        cabeza(c, r, apariencia, 0f, if (guino) 1f else 0.4f, floatArrayOf(if (guino) 0.6f else 0f, 0f), giro = -0.6f, guino = guino)
    }
}

// corazon del beso
fun DrawScope.dibujarCorazon(centro: Offset, tam: Float, alfa: Float) {
    val c = Color(235, 60, 110).copy(alpha = alfa)
    val p = Path().apply {
        moveTo(centro.x, centro.y + tam * 0.35f)
        cubicTo(centro.x - tam * 0.9f, centro.y - tam * 0.2f, centro.x - tam * 0.35f, centro.y - tam * 0.8f, centro.x, centro.y - tam * 0.3f)
        cubicTo(centro.x + tam * 0.35f, centro.y - tam * 0.8f, centro.x + tam * 0.9f, centro.y - tam * 0.2f, centro.x, centro.y + tam * 0.35f)
        close()
    }
    drawPath(p, c)
}

// ---------------------------------------------------------------- brazos y manos 3D

private val LUZ = Offset(-0.6f, -0.8f)

// tramo plano (para el contorno)
private fun DrawScope.tramoPlano(a: Offset, b: Offset, ra: Float, rb: Float, color: Color) {
    val n = normal(b - a)
    drawPath(poligono(listOf(a + n * ra, b + n * rb, b - n * rb, a - n * ra)), color)
    drawCircle(color, ra, a); drawCircle(color, rb, b)
}

// tramo con volumen: luz de arriba-izquierda, brillo y lado en sombra (como un cilindro)
private fun DrawScope.tramo3d(a: Offset, b: Offset, ra: Float, rb: Float, color: Color, brillo: Float = 0.10f) {
    var n = normal(b - a)
    if (n.x * LUZ.x + n.y * LUZ.y < 0) n = -n
    val m = (a + b) / 2f
    val rm = (ra + rb) / 2f
    val pincel = Brush.linearGradient(
        0f to escalar(color, 0.60f), 0.30f to escalar(color, 0.86f), 0.62f to escalar(color, 1.08f),
        0.74f to mezcla(escalar(color, 1.1f), Color.White, brillo), 0.90f to escalar(color, 1.0f), 1f to escalar(color, 0.80f),
        start = m - n * rm, end = m + n * rm
    )
    drawPath(poligono(listOf(a + n * ra, b + n * rb, b - n * rb, a - n * ra)), pincel)
}

// unión redonda entre tramos (codo, muñeca, nudillos)
private fun DrawScope.union3d(p: Offset, rr: Float, color: Color) {
    drawCircle(Brush.radialGradient(0f to escalar(color, 1.08f), 0.55f to escalar(color, 0.95f), 1f to escalar(color, 0.66f),
        center = p + LUZ * (rr * 0.3f), radius = rr * 1.3f), rr, p)
}

private fun DrawScope.tubo(pts: List<Offset>, rad: List<Float>, color: Color, brillo: Float = 0.10f) {
    for (k in 1 until pts.size - 1) union3d(pts[k], rad[k], color)
    union3d(pts[0], rad[0], color); union3d(pts.last(), rad.last(), color)
    for (k in 0 until pts.size - 1) tramo3d(pts[k], pts[k + 1], rad[k], rad[k + 1], color, brillo)
}

private fun cadenaBrazo(hombro: Offset, codo: Offset, muneca: Offset, rb: Float, rc: Float, rm: Float): Pair<List<Offset>, List<Float>> {
    val B = hombro + (codo - hombro) * 0.42f
    val F = codo + (muneca - codo) * 0.33f
    return listOf(hombro, B, codo, F, muneca) to listOf(rb, rb * 1.07f, rc * 1.10f, rc * 1.04f, rm)
}

// vellos cortos que siguen el brazo (siempre los mismos)
private val cacheVellos = HashMap<Long, FloatArray>()

private fun DrawScope.vellos(A: Offset, B: Offset, ra: Float, rb: Float, desde: Float, hasta: Float, cantidad: Int, semilla: Long,
                             color: Color, largo0: Float = 0.18f, largo1: Float = 0.34f) {
    val d = cacheVellos.getOrPut(semilla * 1000 + cantidad) {
        val rng = java.util.Random(semilla)
        FloatArray(cantidad * 4) { i -> when (i % 4) { 0 -> rng.nextFloat(); 1 -> rng.nextFloat() * 1.76f - 0.88f; 2 -> (rng.nextGaussian() * 0.35).toFloat(); else -> rng.nextFloat() } }
    }
    val eje = B - A
    val l = eje.getDistance()
    if (l < 1f) return
    val u = eje / l
    val n = Offset(-u.y, u.x)
    val path = Path()
    for (i in 0 until cantidad) {
        val t = desde + (hasta - desde) * d[i * 4]
        val rad = ra + (rb - ra) * t
        val p0 = A + eje * t + n * (d[i * 4 + 1] * rad)
        val ang = d[i * 4 + 2]
        val dir = u * cos(ang) + n * (sin(ang) * 0.6f)
        val lp = (largo0 + (largo1 - largo0) * d[i * 4 + 3]) * rad
        val p1 = p0 + dir * lp
        val pm = (p0 + p1) / 2f + n * (lp * 0.12f)
        path.moveTo(p0.x, p0.y); path.quadraticTo(pm.x, pm.y, p1.x, p1.y)
    }
    drawPath(path, color, style = Stroke(maxOf(0.7f, l / 220f), cap = StrokeCap.Round))
}

private val DEDOS = listOf(intArrayOf(1, 2, 3, 4), intArrayOf(5, 6, 7, 8), intArrayOf(9, 10, 11, 12), intArrayOf(13, 14, 15, 16), intArrayOf(17, 18, 19, 20))
private val GROSOR_DEDO = floatArrayOf(1.18f, 1.0f, 1.02f, 0.95f, 0.80f)

// brazo de arriba: hombro, bíceps y codo con volumen, y la manga corta encima
private fun DrawScope.brazoArriba(hombro: Offset, codo: Offset, muneca: Offset, rb: Float, rc: Float, rm: Float, ap: Apariencia) {
    val (pts, rad) = cadenaBrazo(hombro, codo, muneca, rb, rc, rm)
    val linea = ap.contornoPiel()
    for (k in 0 until 2) tramoPlano(pts[k], pts[k + 1], rad[k] + 1.5f, rad[k + 1] + 1.5f, linea)
    tubo(pts.subList(0, 3), rad.subList(0, 3), ap.piel)
    val fin = hombro + (codo - hombro) * 0.40f
    tramoPlano(hombro, fin, rb * 1.22f + 1.5f, rb * 1.15f + 1.5f, CONTORNO)
    tubo(listOf(hombro, fin), listOf(rb * 1.22f, rb * 1.15f), ap.camisa, 0.06f)
}

// codo, antebrazo, muñeca y mano realista (siempre encima de todo)
private fun DrawScope.manoYAntebrazo(p: List<Offset>?, colorLado: Color, puntos: Boolean, ap: Apariencia,
                                     hombro: Offset, c: Offset, m: Offset, rb: Float, rc: Float, rm: Float) {
    val piel = ap.piel; val osc = ap.pielSombra; val linea = ap.contornoPiel()
    val (ptsB, radB) = cadenaBrazo(hombro, c, m, rb, rc, rm)
    val X = (ptsB[1] + ptsB[2]) / 2f
    val rX = (radB[1] + radB[2]) / 2f
    if (p == null) {
        val cad = listOf(X) + ptsB.subList(2, 5); val rad = listOf(rX) + radB.subList(2, 5)
        for (k in 0 until cad.size - 1) tramoPlano(cad[k], cad[k + 1], rad[k] + 1.5f, rad[k + 1] + 1.5f, linea)
        tubo(cad, rad, piel)
        vellos(hombro, c, rb, rc * 1.1f, 0.45f, 0.92f, if (ap.mujer) 14 else 45, 31L + (if (ap.mujer) 1 else 0), mezcla(piel, Color(70, 50, 40), if (ap.mujer) 0.12f else 0.30f), 0.15f, 0.28f)
        return
    }
    val tamano = maxOf((p[9] - p[0]).getDistance(), 4f)
    val base = tamano * 0.118f
    val borde = maxOf(1.2f, tamano * 0.022f)
    val eje = (p[9] - p[0]) / tamano
    val perp = Offset(-eje.y, eje.x)
    val anchoMuneca = tamano * 0.30f
    val palma = envolvente(listOf(p[0] + perp * anchoMuneca, p[0] - perp * anchoMuneca, p[1], p[2], p[5], p[9], p[13], p[17]))
    val palmaP = poligono(palma)
    val cad = listOf(X) + ptsB.subList(2, 5) + listOf(p[0])
    val rad = listOf(rX) + radB.subList(2, 5) + listOf(anchoMuneca * 0.92f)
    // radios de cada falange
    val dedosRad = DEDOS.mapIndexed { i, dedo ->
        val k = GROSOR_DEDO[i]
        if (dedo[0] == 1) listOf(1.25f, 1.05f, 0.9f, 0.78f).map { base * k * it } else listOf(1.0f, 0.92f, 0.84f, 0.74f).map { base * k * it }
    }
    // 1) contorno de piel de todo junto
    for (k in 0 until cad.size - 1) tramoPlano(cad[k], cad[k + 1], rad[k] + borde, rad[k + 1] + borde, linea)
    for ((i, dedo) in DEDOS.withIndex()) for (j in 0 until 3) tramoPlano(p[dedo[j]], p[dedo[j + 1]], dedosRad[i][j] + borde, dedosRad[i][j + 1] + borde, linea)
    drawPath(palmaP, linea); drawPath(palmaP, linea, style = Stroke(borde * 2 + 1, join = StrokeJoin.Round))
    // 2) brazo 3D, vellos, palma, piel entre dedos, base del pulgar y dedos 3D
    tubo(cad, rad, piel)
    vellos(hombro, c, rb, rc * 1.1f, 0.45f, 0.92f, if (ap.mujer) 14 else 45, 31L + (if (ap.mujer) 1 else 0), mezcla(piel, Color(70, 50, 40), if (ap.mujer) 0.12f else 0.30f), 0.15f, 0.28f)
    vellos(c, m, rc * 1.05f, rm, 0.08f, 0.86f, if (ap.mujer) 28 else 90, 17L + (if (ap.mujer) 1 else 0), mezcla(piel, Color(70, 50, 40), if (ap.mujer) 0.15f else 0.35f))
    rellenoSombreado(palmaP, piel, luz = 0.08f, sombra = 0.14f, borde = 0.18f)
    for ((k1, k2) in listOf(5 to 9, 9 to 13, 13 to 17)) {
        drawPath(poligono(listOf(p[k1], p[k1] + (p[k1 + 1] - p[k1]) * 0.25f, p[k2] + (p[k2 + 1] - p[k2]) * 0.25f, p[k2])), piel)
    }
    val tenar = listOf(p[0], p[1], p[2], p[0] + (p[5] - p[0]) * 0.55f)
    drawPath(poligono(envolvente(tenar)), piel)
    mancha((p[1] + p[2]) / 2f, base * 1.2f, base * 0.9f, BLANCO_PIEL, 0.25f)
    for ((i, dedo) in DEDOS.withIndex()) tubo(dedo.map { p[it] }, dedosRad[i], piel, 0.12f)
    // codo: hueso por fuera y pliegue por dentro cuando se dobla
    val uu = (c - hombro) / maxOf(1e-3f, (c - hombro).getDistance())
    val vv = (m - c) / maxOf(1e-3f, (m - c).getDistance())
    val dobla = 1 - ((uu.x * vv.x + uu.y * vv.y + 1) / 2).coerceIn(0f, 1f)
    val dentro = vv - uu
    if (dentro.getDistance() > 1e-3f) {
        val dn = dentro / dentro.getDistance()
        val hueso = c - dn * (rc * 0.55f)
        mancha(hueso, rc * 0.6f, rc * 0.6f, BLANCO_PIEL, 0.35f + 0.3f * dobla)
        mancha(hueso - dn * (rc * 0.35f), rc * 0.5f, rc * 0.5f, osc, 0.25f + 0.3f * dobla)
        if (dobla > 0.12f) {
            val pl = c + dn * (rc * 0.45f)
            val nn = Offset(-dn.y, dn.x)
            drawPath(poligono(bez2(pl - nn * (rc * 0.55f), pl + dn * (rc * 0.18f), pl + nn * (rc * 0.55f), 8), false),
                mezcla(piel, linea, minOf(1f, dobla * 1.4f)), style = Stroke(maxOf(0.8f, rc / 12f), cap = StrokeCap.Round))
        }
    }
    // yemas y nudillos más rosados, venas finas, tendones, pliegues y uñas
    val rosado = mezcla(piel, Color(230, 140, 140), 0.5f)
    for ((i, dedo) in DEDOS.withIndex()) mancha(p[dedo[3]], base * GROSOR_DEDO[i] * 0.8f, base * GROSOR_DEDO[i] * 0.8f, rosado, 0.3f)
    for (k in intArrayOf(6, 10, 14, 18)) mancha(p[k], base * 0.65f, base * 0.65f, rosado, 0.22f)
    val vena = mezcla(piel, Color(120, 140, 175), 0.18f)
    for (k in intArrayOf(9, 13)) drawPath(poligono(bez2(p[0] + (p[k] - p[0]) * 0.15f + perp * (base * 0.4f), p[0] + (p[k] - p[0]) * 0.45f - perp * (base * 0.3f),
        p[0] + (p[k] - p[0]) * 0.70f, 8), false), vena, style = Stroke(maxOf(0.8f, base / 6f), cap = StrokeCap.Round))
    for (k in intArrayOf(5, 9, 13, 17)) drawLine(mezcla(piel, osc, 0.30f), p[0] + (p[k] - p[0]) * 0.25f, p[0] + (p[k] - p[0]) * 0.85f, maxOf(0.7f, base / 8f))
    for ((i, dedo) in DEDOS.withIndex()) for (j in intArrayOf(1, 2)) {
        val q = p[dedo[j]]
        var d = p[dedo[j + 1]] - p[dedo[j - 1]]
        d = d / maxOf(1e-3f, d.getDistance())
        val n = Offset(-d.y, d.x)
        val w = base * GROSOR_DEDO[i] * 0.55f
        for (off in floatArrayOf(-0.12f, 0.12f)) drawLine(mezcla(piel, linea, 0.45f), q + d * (base * off) - n * w, q + d * (base * off) + n * w, maxOf(0.6f, base / 10f))
    }
    for ((i, dedo) in DEDOS.withIndex()) {
        val a = p[dedo[2]]; val b = p[dedo[3]]
        val d = b - a
        val l = d.getDistance()
        if (l < 2f) continue
        val k = GROSOR_DEDO[i]
        val ang = Math.toDegrees(kotlin.math.atan2(d.y, d.x).toDouble()).toFloat()
        val centro = b - d / l * (base * k * 0.42f)
        val ex = base * k * 0.70f; val ey = base * k * 0.58f
        rotate(ang, centro) {
            drawOval(mezcla(piel, linea, 0.85f), Offset(centro.x - ex - 1, centro.y - ey - 1), Size(ex * 2 + 2, ey * 2 + 2))
            drawOval(if (ap.mujer) Color(238, 178, 178) else Color(232, 186, 176), Offset(centro.x - ex, centro.y - ey), Size(ex * 2, ey * 2))
            drawOval(Color(250, 248, 248), Offset(centro.x - ex * 0.65f, centro.y - ey * 0.6f), Size(ex, ey * 0.5f))           // brillo
            drawOval(Color(245, 222, 214), Offset(centro.x - ex * 0.8f, centro.y - ey * 0.7f), Size(ex * 0.5f, ey * 1.4f))      // lúnula
            drawOval(mezcla(piel, Color(250, 245, 240), 0.6f), Offset(centro.x + ex * 0.45f, centro.y - ey * 0.9f), Size(ex * 0.5f, ey * 1.8f))  // borde
        }
    }
    if (puntos) p.forEach { drawCircle(colorLado, 2.5f, it) }
}

// trazo que, donde se cruza consigo mismo, oscurece el pedacito de atrás y aclara el de adelante
private fun DrawScope.trazoConCruces(pts: List<Offset>, color: Color, grosor: Float = 4f) {
    if (pts.size < 2) return
    val linea = poligono(pts, false)
    drawPath(linea, Color.White, style = Stroke(grosor + 4f, cap = StrokeCap.Round, join = StrokeJoin.Round))   // borde blanco
    drawPath(linea, color, style = Stroke(grosor, cap = StrokeCap.Round, join = StrokeJoin.Round))
    if (pts.size < 4) return
    val oscuro = escalar(color, 0.45f)
    val brillante = mezcla(color, Color.White, 0.8f)
    val radio = grosor / 2
    val hechos = ArrayList<Offset>()
    for (i in 2 until pts.size - 1) {
        val a0 = pts[i]; val s = pts[i + 1] - a0
        for (j in 0 until i - 1) {
            if (i - j < 4) continue
            val q0 = pts[j]; val rr = pts[j + 1] - q0
            val den = rr.x * s.y - rr.y * s.x
            val largo = rr.getDistance() * s.getDistance() + 1e-9f
            if (kotlin.math.abs(den) / largo <= 0.25f) continue
            val w = a0 - q0
            val t = (w.x * s.y - w.y * s.x) / den
            val u = (w.x * rr.y - w.y * rr.x) / den
            if (t < 0 || t > 1 || u < 0 || u > 1) continue
            val pc = q0 + rr * t
            if (hechos.any { (it - pc).getDistance() < grosor * 2 }) continue
            hechos += pc
            val dv = rr / maxOf(1e-3f, rr.getDistance()) * radio
            val dn = s / maxOf(1e-3f, s.getDistance()) * radio
            drawLine(oscuro, pc - dv, pc + dv, maxOf(1f, grosor / 2), cap = StrokeCap.Round)       // por detrás
            drawLine(brillante, pc - dn, pc + dn, maxOf(1f, grosor / 2), cap = StrokeCap.Round)    // por delante
        }
    }
}

// logo del pecho: dos manos abiertas haciendo señas dentro de un círculo
private fun DrawScope.logoLsc(c: Offset, tam: Float, color: Color) {
    val g = maxOf(1f, tam / 9f)
    drawCircle(color, tam, c, style = Stroke(g))
    for ((lado, ang) in listOf(-1 to -0.35f, 1 to 0.30f)) {
        val bx = c.x + lado * tam * 0.30f
        val by = c.y + tam * 0.15f * (if (lado < 0) 1f else -0.2f)
        val ca = cos(ang); val sa = sin(ang)
        fun R(x: Float, y: Float) = Offset(bx + x * ca - y * sa, by + x * sa + y * ca)
        drawPath(poligono(listOf(R(-0.17f * tam, -0.05f * tam), R(0.17f * tam, -0.05f * tam), R(0.15f * tam, 0.25f * tam), R(-0.15f * tam, 0.25f * tam))), color)
        val largos = floatArrayOf(0.30f, 0.36f, 0.34f, 0.27f)
        for ((k, dx) in floatArrayOf(-0.12f, -0.04f, 0.04f, 0.12f).withIndex())
            drawLine(color, R(dx * tam, -0.05f * tam), R(dx * tam * 1.15f, -0.05f * tam - largos[k] * tam), maxOf(1f, tam / 10f), cap = StrokeCap.Round)
        drawLine(color, R(-lado * 0.16f * tam, 0.10f * tam), R(-lado * 0.30f * tam, -0.05f * tam), maxOf(1f, tam / 10f), cap = StrokeCap.Round)
        val ac = R(lado * 0.32f * tam, -0.20f * tam)
        drawArc(color, if (lado > 0) -60f + Math.toDegrees(ang.toDouble()).toFloat() else 120f + Math.toDegrees(ang.toDouble()).toFloat(), 120f, false,
            Offset(ac.x - tam * 0.12f, ac.y - tam * 0.20f), Size(tam * 0.24f, tam * 0.40f), style = Stroke(maxOf(1f, tam / 12f), cap = StrokeCap.Round))
    }
}

// ---------------------------------------------------------------- muñeco completo

// dibuja el muñeco en espejo dentro de area
fun DrawScope.dibujarMunieco(
    sena: AvatarSena, frame: Int, area: Rect, puntos: Boolean, cara: Gestos? = null,
    apariencia: Apariencia = HOMBRE, dx: Float = 0f, sonriendo: Boolean = false,
    beso: Float = 0f,           // inclinacion de la cabeza para el beso
    mostrarAngulo: Boolean = false,  // pantalla LSC: arco y numero del angulo del codo
    version: Int = 0            // LSC 4 versiones: solo lo que ve esa version (0 = todo, como antes)
) {
    val ap = apariencia
    val parpadeo = cara?.parpadeo ?: 0f
    val sonrisa = if (sonriendo) 1f else cara?.sonrisa ?: 0f
    val escala = escalaMunieco(area)
    val escalaX = escala * ANCHO_MUNIECO
    val origen = Offset(area.center.x + dx * escalaX, area.top + 1.25f * escala)
    fun a(x: Double, y: Double) = Offset(origen.x - x.toFloat() * escalaX, origen.y + y.toFloat() * escala)
    fun a(p: P) = a(p.x, p.y)

    val ancho = if (ap.mujer) 0.84f else 1f
    val yaw = giroCuerpo()
    val cyw = cos(yaw); val syw = sin(yaw)
    // el pecho está más adelante que los costados: al girar se corre más que los bordes
    fun aw(x: Float, y: Float): Offset {
        val z = 0.38f * maxOf(0f, 1 - (x / 0.62f) * (x / 0.62f)) * (if (y < 1.8f) 1f else 0f)
        return a(((x * cyw + z * syw) * ancho).toDouble(), y.toDouble())
    }
    // como en el PC: hombros caidos, brazos gruesos como los de una persona
    val hi = aw(0.47f, 0.12f); val hd = aw(-0.47f, 0.12f)
    val grosorBrazo = escala * 0.21f * (if (ap.mujer) 0.85f else 1f)
    val grosorAntebrazo = escala * 0.165f * (if (ap.mujer) 0.88f else 1f)
    val r = escala * 0.36f
    // codos y muñecas: menos estirados cerca del centro (manos juntas frente a la cara quedan juntas)
    fun ab(p: P): Offset {
        val u = kotlin.math.abs(p.x).toFloat()
        val k = CENTRO_X_BRAZOS + (ANCHO_MUNIECO - CENTRO_X_BRAZOS) * minOf(1f, u / 0.5f)
        return Offset(origen.x - p.x.toFloat() * k * escala, origen.y + p.y.toFloat() * escala)
    }
    // cerca de la cara, la mano va de la nariz de la persona a la nariz del muñeco
    fun haciaCara(p: P, peso: Double = 1.0): P {
        val n = sena.nariz ?: return p
        val objetivo = P(0.0, -0.66 + 0.04)
        val w = (1.0 - (p - n).norma() / 0.9).coerceIn(0.0, 1.0) * peso
        return p + (objetivo - n) * w
    }
    val centroCabeza = a(0.0, -0.66) + Offset(beso * 0.145f * escalaX - syw * 0.22f * escalaX, kotlin.math.abs(beso) * 0.04f * escala)
    val aire = respiracion()
    val sube = aire * escala * 0.012f
    val hx = centroCabeza.x; val hy = centroCabeza.y
    // si la seña trae el giro medido (nariz entre las orejas) la cabeza gira como la persona
    val giroMedido = sena.giro?.let { g -> g[frame.coerceIn(0, g.size - 1)].toFloat() }
    val giroCabeza = if (giroMedido != null) beso + giroMedido else beso - yaw * 2.2f
    if (ap.mujer) peloLargo(centroCabeza, r, ap, giroCabeza)

    // silueta: axila -> hombro -> trapecio -> cuello, una sola curva como un cuerpo real
    val nw = escala * (if (ap.mujer) 0.095f else 0.12f)
    val yBaseCuello = a(0.0, -0.26).y
    fun lado(s: Float): Triple<List<Offset>, List<Offset>, List<Offset>> {
        fun X(x: Float, y: Float) = aw(s * x, y)
        val cuelloX = hx + (if (s > 0) -nw else nw)
        val delt = bez3(X(0.44f, 0.36f), X(0.55f + aire * 0.006f, 0.27f), X(0.56f + aire * 0.006f, 0.09f), X(0.48f, 0.01f), 12)
            .mapIndexed { i, q -> if (i > 0) q - Offset(0f, sube) else q }
        val trap = bez3(X(0.48f, 0.01f), X(0.38f, -0.07f), X(0.27f, -0.12f), X(0.20f, -0.135f), 10).map { it - Offset(0f, sube) }
        val cuello = bez3(X(0.20f, -0.135f) - Offset(0f, sube), Offset(cuelloX, X(0f, -0.15f).y), Offset(cuelloX, yBaseCuello), Offset(cuelloX, hy + r * 0.85f), 10)
        return Triple(delt, trap, cuello)
    }
    val (di, ti, ci) = lado(1f)
    val (dd, td, cd) = lado(-1f)
    val (ladoD, ladoI) = if (ap.mujer) {
        bez3(aw(-0.45f, 0.34f), aw(-0.38f, 0.75f), aw(-0.24f, 1.0f), aw(-0.34f, 1.7f), 14) to
            bez3(aw(0.34f, 1.7f), aw(0.24f, 1.0f), aw(0.38f, 0.75f), aw(0.45f, 0.34f), 14)
    } else {
        bez2(aw(-0.45f, 0.34f), aw(-0.35f, 1.0f), aw(-0.31f, 1.7f), 10) to bez2(aw(0.31f, 1.7f), aw(0.35f, 1.0f), aw(0.45f, 0.34f), 10)
    }
    // piernas (pantalon), detras de la camisa
    for (sl in floatArrayOf(1f, -1f)) {
        val t = ahora()
        val vaiven = (0.022 * sin(t * 2 * Math.PI / 7.0) + 0.010 * sin(t * 2 * Math.PI / 3.1 + 1.3)).toFloat()
        val w = kotlin.math.abs(cyw) + 0.45f * sqrt(maxOf(0f, 1 - cyw * cyw))
        val poli = listOf(
            a((sl * 0.31f * w + vaiven).toDouble(), 1.50), a((sl * 0.012f * w + vaiven).toDouble(), 1.50),
            a((sl * 0.045f * w + 0.25f * vaiven).toDouble(), 2.5), a((sl * 0.275f * w + 0.25f * vaiven).toDouble(), 2.5)
        )
        val pp = poligono(poli)
        rellenoSombreado(pp, COLOR_JEAN, luz = 0.10f, sombra = 0.25f, borde = 0.28f, centro = 0.5f)
        drawPath(pp, CONTORNO, style = Stroke(2f))
        drawLine(mezcla(COLOR_JEAN, Color(90, 170, 200), 0.35f), poli[0] * 0.8f + poli[1] * 0.2f, poli[3] * 0.82f + poli[2] * 0.18f, 1f)
    }
    val silueta = di + ti.drop(1) + ci.drop(1) + cd.reversed() + td.reversed().drop(1) + dd.reversed().drop(1) + ladoD.drop(1) + ladoI.dropLast(1)
    val siluetaP = poligono(silueta)

    // datos de cada brazo
    class Brazo(val l: Int, val hombro: Offset, val c: Offset, val m: Offset, val mano: List<Offset>?)
    val brazos = ArrayList<Brazo>()
    for ((l, hombro) in listOf(0 to hi, 1 to hd)) {
        val d = sena.lados[l] ?: continue
        val f = frame.coerceIn(0, d.muneca.size - 1)
        val pm = d.mano?.let { it[minOf(f, it.size - 1)] } ?: manoPorDefecto(l, d.muneca[f], d.codo[f])
        val mw = ab(haciaCara(d.muneca[f]))
        val cw = ab(haciaCara(d.codo[f], 0.4))
        // la mano siempre pegada a la muñeca del antebrazo (no se puede separar del brazo)
        val manoPx = pm?.map { a(it) }?.let { h -> val corr = mw - h[0]; h.map { it + corr } }
        // el hombro acompaña al brazo: si el codo sube, el hombro sube un poquito
        val subeH = ((hombro.y - cw.y) / maxOf(1f, escala)).coerceIn(0f, 1f)
        brazos += Brazo(l, hombro - Offset(0f, 0.07f * escala * subeH), cw, mw, manoPx)
    }
    val rb = grosorBrazo * 0.50f; val rc = grosorAntebrazo * 0.60f; val rm = grosorAntebrazo * 0.40f
    // 1) brazos de arriba que salen de detrás del hombro (codo al lado del cuerpo)
    for (b in brazos) if (!dentroDe(b.c, silueta)) brazoArriba(b.hombro, b.c, b.m, rb, rc, rm, ap)

    // 2) piel del cuello con volumen, sombra de la barbilla y anatomía del cuello
    rellenoSombreado(siluetaP, ap.piel, luz = 0.04f, sombra = 0.12f, borde = 0.18f)
    mancha(Offset(hx, hy + r * 0.92f), nw * 1.2f, r * 0.3f, ap.pielSombra, 0.85f)
    val luzC = mezcla(ap.piel, BLANCO_PIEL, 0.20f)
    val yHueco = yBaseCuello + r * 0.22f
    for (s in floatArrayOf(1f, -1f)) {
        val cx_ = hx + s * nw * 0.55f
        val musculo = bez2(Offset(cx_ + s * nw * 0.35f, hy + r * 0.80f), Offset(cx_, hy + r * 1.3f), Offset(hx + s * nw * 0.18f, yHueco), 12)
        drawPath(poligono(musculo.map { it + Offset(s * 2f, 0f) }, false), ap.pielSombra, alpha = 0.3f, style = Stroke(maxOf(2f, r * 0.07f), cap = StrokeCap.Round))
        drawPath(poligono(musculo, false), mezcla(ap.piel, ap.pielSombra, 0.4f), style = Stroke(maxOf(0.8f, r / 90f)))
        drawPath(poligono(musculo.map { it - Offset(s * 2f, 0f) }, false), luzC, style = Stroke(maxOf(0.8f, r / 90f)))
        drawPath(poligono(bez2(Offset(hx + s * nw * 0.95f, hy + r * 1.2f), Offset(hx + s * nw * 1.05f, yBaseCuello), Offset(hx + s * nw * 1.6f, yBaseCuello + r * 0.20f), 10), false),
            mezcla(ap.piel, ap.pielSombra, 0.45f), style = Stroke(maxOf(0.8f, r / 90f)))
    }
    mancha(Offset(hx, yHueco), nw * 0.3f, r * 0.08f, ap.pielSombra, 0.7f)
    for ((k, yy) in floatArrayOf(0.98f, 1.16f).withIndex()) {
        drawPath(poligono(bez2(Offset(hx - nw * 0.75f, hy + r * (yy + 0.03f)), Offset(hx, hy + r * (yy - 0.02f)), Offset(hx + nw * 0.75f, hy + r * (yy + 0.03f)), 10), false),
            mezcla(ap.piel, ap.pielSombra, 0.45f - 0.12f * k), style = Stroke(maxOf(0.8f, r / 90f)))
    }
    // nuez: sube y baja al tragar saliva (en la mujer apenas se nota)
    val ny = hy + r * 1.30f - trago() * r * 0.13f
    val tam = if (ap.mujer) 0.55f else 1f
    mancha(Offset(hx, ny + r * 0.08f * tam), nw * 0.26f * tam, r * 0.07f * tam, ap.pielSombra, 0.65f * tam)
    mancha(Offset(hx, ny - r * 0.01f), nw * 0.17f * tam, r * 0.055f * tam, BLANCO_PIEL, 0.35f * tam)
    if (!ap.mujer) {
        drawArc(luzC, 200f, 140f, false, Offset(hx - nw * 0.17f, ny - r * 0.075f), Size(nw * 0.34f, r * 0.15f), style = Stroke(maxOf(0.8f, r / 90f)))
        drawArc(mezcla(ap.piel, ap.pielSombra, 0.6f), 20f, 140f, false, Offset(hx - nw * 0.19f, ny + r * 0.02f - r * 0.075f), Size(nw * 0.38f, r * 0.15f), style = Stroke(maxOf(0.8f, r / 90f)))
    }
    // 3) camisa con escote, clavículas en la mujer, ribete, logo y pliegues
    val prof = if (ap.mujer) 0.16f else -0.02f
    val escote = bez3(aw(0.20f, -0.135f), aw(0.13f, prof), aw(-0.13f, prof), aw(-0.20f, -0.135f), 16).map { it - Offset(0f, sube) }
    if (ap.mujer) for (s in floatArrayOf(1f, -1f))
        drawPath(poligono(bez2(aw(s * 0.03f, -0.06f), aw(s * 0.12f, -0.11f), aw(s * 0.22f, -0.09f), 8), false),
            mezcla(ap.piel, ap.pielSombra, 0.7f), style = Stroke(maxOf(0.8f, r / 90f)))
    val camisa = poligono(di + ti.drop(1) + escote.drop(1) + td.reversed().drop(1) + dd.reversed().drop(1) + ladoD.drop(1) + ladoI.dropLast(1))
    rellenoSombreado(camisa, ap.camisa, luz = 0.12f, sombra = 0.22f, borde = 0.25f, centro = 0.45f - yaw * 0.9f)
    drawPath(poligono(escote, false), mezcla(ap.camisa, CONTORNO, 0.35f), style = Stroke(maxOf(2f, r / 14f), cap = StrokeCap.Round))
    drawPath(poligono(escote.map { it + Offset(0f, maxOf(2f, r / 14f)) }, false), mezcla(ap.camisa, Color.White, 0.12f), style = Stroke(1f))
    drawPath(siluetaP, CONTORNO, style = Stroke(2f, join = StrokeJoin.Round))
    val colorLogo = mezcla(ap.camisa, Color(240, 245, 245), 0.70f)
    logoLsc(aw(-0.24f, 0.40f) - Offset(0f, sube), escala * 0.17f, colorLogo)
    // logo del ITM al otro lado del pecho (imagen con fondo transparente)
    logoItm?.let { img ->
        val anchoL = escala * 0.40f
        val altoL = anchoL * img.height / img.width
        val c = aw(0.25f, 0.40f) - Offset(0f, sube)
        drawImage(img, dstOffset = androidx.compose.ui.unit.IntOffset((c.x - anchoL / 2).toInt(), (c.y - altoL / 2).toInt()),
            dstSize = androidx.compose.ui.unit.IntSize(anchoL.toInt().coerceAtLeast(1), altoL.toInt().coerceAtLeast(1)))
    }
    val pliegue = mezcla(ap.camisa, CONTORNO, 0.22f)
    for (s in floatArrayOf(1f, -1f))
        drawPath(poligono(bez2(aw(s * 0.36f, 0.40f), aw(s * 0.25f, 0.62f), aw(s * 0.27f, 0.88f), 10), false), pliegue, style = Stroke(1.2f))

    // 4) brazos de arriba que van por delante (codo delante del pecho)
    for (b in brazos) if (dentroDe(b.c, silueta)) brazoArriba(b.hombro, b.c, b.m, rb, rc, rm, ap)

    // 5) cabeza
    // inclinación de la cabeza (grados): gira alrededor del cuello, como la de la persona
    val incl = sena.inclinacion?.let { g -> g[frame.coerceIn(0, g.size - 1)].toFloat().coerceIn(-20f, 20f) } ?: 0f
    rotate(incl, pivot = Offset(hx, hy + r * 0.95f)) {
        cabeza(centroCabeza, r, ap, parpadeo, sonrisa, cara?.cejas ?: floatArrayOf(0f, 0f), giro = giroCabeza,
            besando = kotlin.math.abs(beso) > 0.6f, gestosCara = cara, conCuello = false, aire = aire)
    }

    // 6) codos, antebrazos, muñecas y manos: siempre encima de todo
    for (b in brazos) manoYAntebrazo(b.mano, if (b.l == 0) COLOR_IZQ else COLOR_DER, puntos, ap, b.hombro, b.c, b.m, rb, rc, rm)

    // angulo del codo: el mismo calculo que usa el modelo (hombro = centro +- medio ancho, codo y muñeca)
    if (mostrarAngulo) for (b in brazos) {
        val d = sena.lados[b.l] ?: continue
        val f = frame.coerceIn(0, minOf(d.muneca.size, d.codo.size) - 1)
        val hombroModelo = P(co.edu.itm.manosquehablan.vivo.LADO_HOMBRO[b.l], 0.0)
        val arriba = hombroModelo - d.codo[f]; val abajo = d.muneca[f] - d.codo[f]
        val la = arriba.norma(); val lb = abajo.norma()
        if (la < 1e-6 || lb < 1e-6) continue
        val grados = Math.toDegrees(kotlin.math.acos(((arriba.x * abajo.x + arriba.y * abajo.y) / (la * lb)).coerceIn(-1.0, 1.0)))
        // arco en el codo del muñeco, entre el brazo y el antebrazo
        val radio = rc * 2.4f
        val a1 = Math.toDegrees(kotlin.math.atan2((b.hombro - b.c).y.toDouble(), (b.hombro - b.c).x.toDouble())).toFloat()
        val a2 = Math.toDegrees(kotlin.math.atan2((b.m - b.c).y.toDouble(), (b.m - b.c).x.toDouble())).toFloat()
        var barrido = a2 - a1
        while (barrido > 180f) barrido -= 360f
        while (barrido < -180f) barrido += 360f
        val amarillo = Color(255, 215, 0)
        drawArc(amarillo, a1, barrido, false, topLeft = b.c - Offset(radio, radio),
            size = androidx.compose.ui.geometry.Size(radio * 2, radio * 2), style = Stroke(maxOf(3f, rc * 0.25f)))
        // numero junto al codo, hacia afuera del cuerpo
        val pincel = android.graphics.Paint().apply {
            color = android.graphics.Color.rgb(255, 215, 0); textSize = maxOf(28f, escala * 0.16f)
            typeface = android.graphics.Typeface.DEFAULT_BOLD; isAntiAlias = true
            setShadowLayer(4f, 0f, 0f, android.graphics.Color.BLACK)
        }
        val afuera = if (b.c.x < origen.x) -1f else 1f
        val texto = "%.0f°".format(grados)
        val ancho = pincel.measureText(texto)
        drawContext.canvas.nativeCanvas.drawText(texto,
            b.c.x + afuera * (radio + 6f) - (if (afuera < 0) ancho else 0f), b.c.y + pincel.textSize * 0.35f, pincel)
    }

    // LSC 4 versiones: version 2 y 4 ven el brazo (hombro, codo y muñeca); 3 y 4 ven el lugar en la cara y el pecho
    if (version == 2 || version == 4) for (b in brazos) {
        val color = Color(255, 255, 255)
        drawLine(color, b.hombro, b.c, maxOf(3f, escala * 0.02f)); drawLine(color, b.c, b.m, maxOf(3f, escala * 0.02f))
        for (q in listOf(b.hombro, b.c, b.m)) { drawCircle(Color(0, 200, 255), maxOf(6f, escala * 0.035f), q); drawCircle(Color.White, maxOf(6f, escala * 0.035f), q, style = Stroke(2f)) }
    }
    if (version >= 3) {
        // nariz, boca, mentón, frente, ojos, orejas y pecho (los lugares que mide la versión 3)
        val lugares = listOf(Offset(0f, 0.15f), Offset(0f, 0.5f), Offset(0f, 0.92f), Offset(0f, -0.62f),
            Offset(-0.38f, -0.12f), Offset(0.38f, -0.12f), Offset(-1.0f, 0.05f), Offset(1.0f, 0.05f))
        for (q in lugares) drawCircle(Color(255, 215, 0), maxOf(5f, r * 0.07f), centroCabeza + Offset(q.x * r, q.y * r))
        drawCircle(Color(255, 215, 0), maxOf(5f, r * 0.07f), a(0.0, 0.45))
    }

    // 7) trazos del codo (dorado) y de la mano, encima del muñeco
    //    el de la mano es el FINAL (como lo recibe el modelo, sin cortes) si la seña lo trae
    //    (LSC 4 versiones: solo la 3 y la 4 ven el dibujo de la seña)
    if (!sena.sinTrazo && (version == 0 || version >= 3)) for ((l, hombro) in listOf(0 to hi, 1 to hd)) {
        val d = sena.lados[l] ?: continue
        val f = frame.coerceIn(0, d.muneca.size - 1)
        if (f > 0) {
            trazoConCruces((0..f).map { ab(haciaCara(d.codo[it], 0.4)) }, COLOR_CODO, maxOf(3f, grosorAntebrazo * 0.2f))
            val color = if (l == 0) COLOR_IZQ else COLOR_DER
            val t = d.trazo; val ok = d.trazoOk
            if (t != null && ok != null) {
                var i = 0
                val fin = minOf(f, t.size - 1, ok.size - 1)
                while (i <= fin) {
                    if (!ok[i]) { i++; continue }
                    var j = i
                    while (j + 1 <= fin && ok[j + 1]) j++
                    if (j > i) trazoConCruces((i..j).map { ab(haciaCara(t[it])) }, color)
                    i = j + 1
                }
            } else trazoConCruces((0..f).map { ab(haciaCara(d.muneca[it])) }, color)
            // MOSTRAR TRAZOS: las 5 puntas de los dedos, muy delgadas
            val mano = d.mano
            if (mostrarDedos && mano != null) for ((k, punta) in intArrayOf(4, 8, 12, 16, 20).withIndex()) {
                val pts = (0..minOf(f, mano.size - 1)).map { ab(haciaCara(d.muneca[it])) + a(mano[it][punta]) - a(mano[it][MANO_MUNECA]) }
                if (pts.size > 1) drawPath(poligono(pts, false), COLORES_DEDOS[k], style = Stroke(1f))
            }
        }
    }
}

// MOSTRAR TRAZOS (APRENDER): pulgar, índice, medio, anular, meñique
var mostrarDedos = false
val COLORES_DEDOS = listOf(Color(238, 130, 238), Color(255, 0, 0), Color(0, 110, 0), Color(255, 140, 0), Color(0, 0, 139))
