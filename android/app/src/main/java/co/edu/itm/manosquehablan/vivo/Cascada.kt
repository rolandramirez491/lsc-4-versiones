package co.edu.itm.manosquehablan.vivo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer
import java.text.Normalizer

// quita tildes y signos y pasa a mayusculas
fun textoAscii(texto: String): String =
    Normalizer.normalize(texto, Normalizer.Form.NFD)
        .filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
        .replace("¿", "").replace("¡", "")
        .filter { it.code < 128 }
        .uppercase()

class Candidata(val nombre: String, val pCoord: Float, val pDibujo: Float, val pBrazo: Float?)

class Resultado(
    val error: String? = null,
    val top: List<Pair<String, Float>> = emptyList(),
    val candidatas: List<Candidata> = emptyList(),
    val parecidas: List<String> = emptyList(),     // PRACTICAR: hasta 7 señas parecidas
    // 4 versiones: lo que opina cada parte de la versión, en las señas de top (nombre de la parte -> %)
    val partes: Map<String, FloatArray> = emptyMap(),
    val version: Int = 0,
    // la seña que se intentaba: en qué puesto quedó (1 = acertó), con qué probabilidad y lo que opinó cada parte
    val objetivoPuesto: Int = 0,
    val objetivoProb: Float = 0f,
    val objetivoPartes: Map<String, Float> = emptyMap()
)

// Cascada v13 con el modelo del PC (ONNX): coordenadas + dibujo 1 (4 canales) + dibujo 2 (codo, 2 canales).
class Cascada(contexto: Context) {
    private val entorno = OrtEnvironment.getEnvironment()
    private fun sesion(contexto: Context, nombre: String): OrtSession? {
        val bytes = try { contexto.assets.open(nombre).use { it.readBytes() } } catch (_: java.io.IOException) { return null }
        return entorno.createSession(bytes, OrtSession.SessionOptions())
    }

    private val coord = sesion(contexto, "coordenadas.onnx")!!
    private val dibujo = sesion(contexto, "dibujo.onnx")!!
    private val brazo = sesion(contexto, "dibujo_brazo.onnx")
    // v15: ANGULO CODO aparte (solo ve los numeros del brazo); coordenadas ve todo lo demas
    private val angulo = sesion(contexto, "angulo_codo.onnx")

    private val config = JSONObject(contexto.assets.open("config.json").bufferedReader().readText())
    private fun numeros(clave: String): FloatArray? =
        config.optJSONArray(clave)?.let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }

    private val mediaImagen = numeros("media_imagen")!!
    private val desvImagen = numeros("desv_imagen")!!
    private val mediaExtras = numeros("media_extras")!!
    private val desvExtras = numeros("desv_extras")!!
    private val mediaImagenBrazo = numeros("media_imagen_brazo")
    private val desvImagenBrazo = numeros("desv_imagen_brazo")
    private val mascaraAngulo = numeros("mascara_angulo")
    private val mascaraCoord = numeros("mascara_coord")
    val candidatasK = config.getInt("candidatas_k")
    val probabilidadMinima = config.getDouble("probabilidad_minima").toFloat()
    val nombres: List<String> = config.getJSONArray("etiquetas").let { a ->
        val porClase = sortedMapOf<Int, String>()
        for (i in 0 until a.length()) { val e = a.getJSONObject(i); porClase[e.getInt("clase_modelo")] = e.getString("nombre") }
        porClase.values.map(::textoAscii)
    }

    private fun correr(s: OrtSession, datos: FloatArray, forma: LongArray): FloatArray =
        OnnxTensor.createTensor(entorno, FloatBuffer.wrap(datos), forma).use { t ->
            s.run(mapOf("entrada" to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                (r[0].value as Array<FloatArray>)[0]
            }
        }

    private fun imagen(s: OrtSession, img: FloatArray, media: FloatArray, desv: FloatArray): FloatArray {
        val area = TAMANO_IMAGEN * TAMANO_IMAGEN
        val canales = img.size / area
        val x = FloatArray(img.size) { (img[it] - media[it / area]) / desv[it / area] }
        return correr(s, x, longArrayOf(1, canales.toLong(), TAMANO_IMAGEN.toLong(), TAMANO_IMAGEN.toLong()))
    }

    @Synchronized
    fun analizar(m: Muestra, k: Int = candidatasK): Resultado {
        if (m.validos.none { it }) return Resultado(error = "SUBE LAS MANOS")
        val ext = FloatArray(m.extras.size) { (m.extras[it] - mediaExtras[it]) / desvExtras[it] }
        val extCoord = mascaraCoord?.let { mc -> FloatArray(ext.size) { ext[it] * mc[it] } } ?: ext
        val pC = correr(coord, extCoord, longArrayOf(1, ext.size.toLong()))
        val pD = imagen(dibujo, dibujarTrazos(m.trazos, m.validos), mediaImagen, desvImagen)
        // el ANGULO CODO va en el lugar que tenia el dibujo 2
        val pB = if (angulo != null && mascaraAngulo != null) {
            val ma = mascaraAngulo
            correr(angulo, FloatArray(ext.size) { ext[it] * ma[it] }, longArrayOf(1, ext.size.toLong()))
        } else if (brazo != null && mediaImagenBrazo != null && desvImagenBrazo != null && m.validosBrazo.any { it })
            imagen(brazo, dibujarTrazos(m.trazosBrazo, m.validosBrazo), mediaImagenBrazo, desvImagenBrazo) else null
        val (puntaje, candidatas) = combinarCascada(pC, pD, pB, k, probabilidadMinima)
        val listaCand = candidatas.indices.filter { candidatas[it] }.sortedByDescending { pC[it] }
            .map { Candidata(nombres[it], pC[it], pD[it], pB?.get(it)) }
        val top = puntaje.indices.sortedByDescending { puntaje[it] }.take(3).map { nombres[it] to puntaje[it] }
        return Resultado(top = top, candidatas = listaCand)
    }
}

// coordenadas escogen hasta k candidatas y los dibujos deciden entre ellas
fun combinarCascada(pC: FloatArray, pD: FloatArray, pB: FloatArray?, k: Int, minimo: Float): Pair<FloatArray, BooleanArray> {
    val orden = pC.indices.sortedByDescending { pC[it] }
    val candidatas = BooleanArray(pC.size)
    for (pos in 0 until minOf(k, pC.size)) {
        val c = orden[pos]
        if (pos == 0 || pC[c] >= minimo) candidatas[c] = true
    }
    val puntaje = FloatArray(pC.size) { i -> if (candidatas[i]) pC[i] * pD[i] * (pB?.get(i) ?: 1f) else 0f }
    val suma = maxOf(puntaje.sum(), 1e-12f)
    for (i in puntaje.indices) puntaje[i] /= suma
    return puntaje to candidatas
}
