package co.edu.itm.manosquehablan.vivo

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.max

// Compara lo que calcula el celular (Kotlin) con lo que calcula el PC (Python) en 8 muestras del banco.
// Los casos los genera  codigo/exportar_celular.py  ->  src/test/resources/casos.json
class CalculoTest {

    private fun numeros(a: Any?): List<Double> = when (a) {
        is JSONArray -> (0 until a.length()).flatMap { numeros(a.get(it)) }
        is Number -> listOf(a.toDouble())
        is Boolean -> listOf(if (a) 1.0 else 0.0)
        else -> emptyList()
    }

    private fun puntos(a: JSONArray) = Array(a.length()) { val q = a.getJSONArray(it); P(q.getDouble(0), q.getDouble(1)) }
    private fun bools(a: JSONArray) = BooleanArray(a.length()) { a.getBoolean(it) }

    private fun grabacion(m: JSONObject): GrabacionCruda {
        val anchos = numeros(m.getJSONArray("ancho_hombros")).sorted()
        val centros = puntos(m.getJSONArray("centro_hombros"))
        fun med(v: List<Double>): Double { val s = v.sorted(); val n = s.size; return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2 }
        fun manos(k: String) = m.getJSONArray(k).let { a -> Array(a.length()) { puntos(a.getJSONArray(it)) } }
        return GrabacionCruda(
            arrayOf(puntos(m.getJSONArray("muneca_izq")), puntos(m.getJSONArray("muneca_der"))),
            arrayOf(bools(m.getJSONArray("valida_izq")), bools(m.getJSONArray("valida_der"))),
            arrayOf(manos("mano_izq"), manos("mano_der")),
            arrayOf(bools(m.getJSONArray("mano_izq_ok")), bools(m.getJSONArray("mano_der_ok"))),
            arrayOf(puntos(m.getJSONArray("codo_izq")), puntos(m.getJSONArray("codo_der"))),
            arrayOf(bools(m.getJSONArray("codo_izq_ok")), bools(m.getJSONArray("codo_der_ok"))),
            manos("cara"), bools(m.getJSONArray("cara_ok")),
            med(anchos), P(med(centros.map { it.x }), med(centros.map { it.y }))
        )
    }

    private fun difMax(a: List<Double>, b: List<Double>): Double {
        assertTrue("tamaños distintos ${a.size} vs ${b.size}", a.size == b.size)
        return a.indices.maxOfOrNull { abs(a[it] - b[it]) } ?: 0.0
    }

    @Test
    fun igualQuePython() {
        val texto = javaClass.classLoader!!.getResource("casos.json").readText()
        val casos = JSONArray(texto)
        var peorSeq = 0.0; var peorTrazo = 0.0; var peorExtra = 0.0; var peorImagen = 0.0
        for (c in 0 until casos.length()) {
            val caso = casos.getJSONObject(c)
            val g = grabacion(caso.getJSONObject("muestra"))
            val s1 = difMax(secuenciaV1(g).flatMap { it.map(Float::toDouble) }, numeros(caso.get("seq_v1")))
            val s2 = difMax(secuenciaV2(g).flatMap { it.map(Float::toDouble) }, numeros(caso.get("seq_v2")))
            val m = construirV3(g)
            val validos = m.validos.toList() == numeros(caso.get("validos")).map { it > 0.5 } &&
                m.validosBrazo.toList() == numeros(caso.get("validos_codo")).map { it > 0.5 }
            val t = max(difMax(m.trazos.flatMap { tr -> tr.flatMap { listOf(it.x, it.y) } }, numeros(caso.get("trazos"))),
                difMax(m.trazosBrazo.flatMap { tr -> tr.flatMap { listOf(it.x, it.y) } }, numeros(caso.get("trazos_codo"))))
            val e = difMax(m.extras.map(Float::toDouble), numeros(caso.get("extras")))
            val im = max(difMax(dibujarTrazos(m.trazos, m.validos).map(Float::toDouble), numeros(caso.get("imagen"))),
                difMax(dibujarTrazos(m.trazosBrazo, m.validosBrazo).map(Float::toDouble), numeros(caso.get("imagen_codo"))))
            val ik = dibujarTrazos(m.trazos, m.validos).map(Float::toDouble); val ip = numeros(caso.get("imagen"))
            val media = ik.indices.sumOf { abs(ik[it] - ip[it]) } / ik.size
            val distintos = ik.indices.count { abs(ik[it] - ip[it]) > 0.1 }
            val encendidos = ip.count { it > 0.1 }
            println("%-16s seqV1 %.1e  seqV2 %.1e  trazos %.1e  extras %.1e  imagen max %.3f media %.4f (%d de %d px encendidos difieren >0.1)  validos %s"
                .format(caso.getString("nombre"), s1, s2, t, e, im, media, distintos, encendidos, if (validos) "OK" else "DISTINTOS"))
            assertTrue("validos distintos en ${caso.getString("nombre")}", validos)
            peorSeq = max(peorSeq, max(s1, s2)); peorTrazo = max(peorTrazo, t); peorExtra = max(peorExtra, e)
            peorImagen = max(peorImagen, im)
        }
        println("PEOR: secuencias %.1e  trazos %.1e  extras %.1e  imagen %.3f".format(peorSeq, peorTrazo, peorExtra, peorImagen))
        assertTrue("secuencias distintas", peorSeq < 1e-4)
        assertTrue("trazos distintos", peorTrazo < 1e-4)
        assertTrue("extras distintos", peorExtra < 1e-3)
    }
}
