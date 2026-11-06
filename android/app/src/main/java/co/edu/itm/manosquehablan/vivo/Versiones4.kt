package co.edu.itm.manosquehablan.vivo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer

// Las 4 versiones en el celular (igual que codigo/versiones.py en el PC):
//   1 solo manos (GRU) · 2 manos + cuerpo (GRU) · 3 dibujo + lugar + dedos (cascada) · 4 fusión 2 + 3
object InfoVersiones {
    val titulos = mapOf(
        1 to "Versión 1: solo manos",
        2 to "Versión 2: manos + cuerpo",
        3 to "Versión 3: dibujo + lugar + dedos",
        4 to "Versión 4: fusión 2 + 3",
    )
    val detalles = mapOf(
        1 to "21 puntos de cada mano (Hand Landmarker). Es la línea base del estado del arte.",
        2 to "Se agregan las muñecas y los codos, y todo se mide desde los hombros.",
        3 to "La seña como dibujo, dónde ocurre (cuerpo y cara) y la forma de los dedos en 3 momentos.",
        4 to "La versión 2 y la 3 votan juntas. Es la mejor.",
    )
    // acierto con personas no vistas (experimento: 90 señas, 5 personas por seña, semilla 42)
    val acierto = mapOf(1 to 55.3f, 2 to 59.2f, 3 to 57.5f, 4 to 62.8f)
}

class Versiones4(contexto: Context) {
    private val entorno = OrtEnvironment.getEnvironment()
    private fun sesion(c: Context, nombre: String): OrtSession =
        entorno.createSession(c.assets.open(nombre).use { it.readBytes() }, OrtSession.SessionOptions())

    private val v1 = sesion(contexto, "v1_manos.onnx")
    private val v2 = sesion(contexto, "v2_cuerpo.onnx")
    private val coord = sesion(contexto, "v3_coord.onnx")
    private val dibujo = sesion(contexto, "v3_dibujo.onnx")
    private val codo = sesion(contexto, "v3_codo.onnx")
    private val config = JSONObject(contexto.assets.open("cuatro_versiones.json").bufferedReader().readText())
    private fun numeros(o: JSONObject, k: String) = o.getJSONArray(k).let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }

    val nombres: List<String> = config.getJSONArray("nombres").let { a -> List(a.length()) { a.getString(it) } }
    private val pesoFusion = config.getDouble("peso_fusion")
    private val k = config.getInt("candidatas_k")
    private val media1 = numeros(config.getJSONObject("v1"), "media"); private val desv1 = numeros(config.getJSONObject("v1"), "desv")
    private val media2 = numeros(config.getJSONObject("v2"), "media"); private val desv2 = numeros(config.getJSONObject("v2"), "desv")
    private val c3 = config.getJSONObject("v3")
    private val mediaImg = numeros(c3, "media_imagen"); private val desvImg = numeros(c3, "desv_imagen")
    private val mediaExt = numeros(c3, "media_extras"); private val desvExt = numeros(c3, "desv_extras")
    private val mediaImg2 = numeros(c3, "media_imagen2"); private val desvImg2 = numeros(c3, "desv_imagen2")

    private fun correr(s: OrtSession, datos: FloatArray, forma: LongArray): FloatArray =
        OnnxTensor.createTensor(entorno, FloatBuffer.wrap(datos), forma).use { t ->
            s.run(mapOf("entrada" to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                softmax((r[0].value as Array<FloatArray>)[0])
            }
        }

    private fun softmax(x: FloatArray): FloatArray {
        val m = x.max(); val e = DoubleArray(x.size) { kotlin.math.exp((x[it] - m).toDouble()) }; val s = e.sum()
        return FloatArray(x.size) { (e[it] / s).toFloat() }
    }

    private fun gru(s: OrtSession, seq: Array<FloatArray>, media: FloatArray, desv: FloatArray): FloatArray {
        val f = media.size
        val datos = FloatArray(seq.size * f)
        for (t in seq.indices) for (j in 0 until f) datos[t * f + j] = (seq[t][j] - media[j]) / desv[j]
        return correr(s, datos, longArrayOf(1, seq.size.toLong(), f.toLong()))
    }

    private fun imagenNormalizada(img: FloatArray, media: FloatArray, desv: FloatArray): FloatArray {
        val area = TAMANO_IMAGEN * TAMANO_IMAGEN
        return FloatArray(img.size) { (img[it] - media[it / area]) / desv[it / area] }
    }

    private fun version3(g: GrabacionCruda): Pair<FloatArray, Map<String, FloatArray>> {
        val m = construirV3(g)
        val img = imagenNormalizada(dibujarTrazos(m.trazos, m.validos), mediaImg, desvImg)
        val img2 = imagenNormalizada(dibujarTrazos(m.trazosBrazo, m.validosBrazo), mediaImg2, desvImg2)
        val ext = FloatArray(m.extras.size) { (m.extras[it] - mediaExt[it]) / desvExt[it] }
        val forma = longArrayOf(1, 2, TAMANO_IMAGEN.toLong(), TAMANO_IMAGEN.toLong())
        val pc = correr(coord, ext, longArrayOf(1, ext.size.toLong()))
        val pd = correr(dibujo, img, forma)
        val p2 = correr(codo, img2, forma)
        val (final, _) = combinarCascada(pc, pd, p2, k, 0f)
        return final to linkedMapOf("Coordenadas (lugar y dedos)" to pc, "Dibujo de la mano" to pd, "Dibujo del codo" to p2)
    }

    // probabilidad final y lo que opina cada parte de la versión
    fun probabilidades(g: GrabacionCruda, version: Int): Pair<FloatArray, Map<String, FloatArray>> = when (version) {
        1 -> gru(v1, secuenciaV1(g), media1, desv1).let { it to mapOf("Manos (GRU)" to it) }
        2 -> gru(v2, secuenciaV2(g), media2, desv2).let { it to mapOf("Manos + cuerpo (GRU)" to it) }
        3 -> version3(g)
        else -> {
            val p2 = gru(v2, secuenciaV2(g), media2, desv2)
            val p3 = version3(g).first
            fusionar(p2, p3, pesoFusion) to linkedMapOf("Versión 2" to p2, "Versión 3" to p3)
        }
    }

    // de la grabación en vivo a las 4 señas más parecidas (top en 0..1; partes en el mismo orden que top)
    @Synchronized
    internal fun reconocer(r: Registro, ancho: Int, alto: Int, version: Int, objetivo: String? = null, top: Int = 4): Resultado {
        val g = grabacionDeRegistro(r, ancho, alto) ?: return Resultado(error = "NO SE VE EL CUERPO", version = version)
        if ((0..1).none { l -> g.manosOk[l].any { it } || g.validas[l].any { it } }) return Resultado(error = "SUBE LAS MANOS", version = version)
        val (p, partes) = probabilidades(g, version)
        val todos = p.indices.sortedByDescending { p[it] }
        val orden = todos.take(top)
        val o = objetivo?.let { n -> nombres.indexOfFirst { textoAscii(it) == textoAscii(n) } } ?: -1
        return Resultado(
            top = orden.map { nombres[it] to p[it] },
            partes = partes.mapValues { (_, v) -> FloatArray(orden.size) { v[orden[it]] } },
            version = version,
            objetivoPuesto = if (o >= 0) todos.indexOf(o) + 1 else 0,
            objetivoProb = if (o >= 0) p[o] else 0f,
            objetivoPartes = if (o >= 0) partes.mapValues { (_, v) -> v[o] } else emptyMap()
        )
    }
}
