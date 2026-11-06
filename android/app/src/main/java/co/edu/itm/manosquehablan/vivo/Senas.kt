package co.edu.itm.manosquehablan.vivo

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// Maneja la lista de señas y las muestras guardadas en el celular.
// senas.json trae las señas base, mis_senas_celular.txt las que se crean en la app
// y cada muestra queda en datos_propios/NN_NOMBRE/muestra_*.json
class Senas(private val contexto: Context) {
    class Sena(val indice: Int, val nombre: String, val muestrasPc: Int)

    val raiz: File = carpetaDatos(contexto)
    private val carpetaDatos = File(raiz, "datos_propios")
    private val archivoNuevas = File(raiz, "mis_senas_celular.txt")
    private val base: List<Sena> = JSONArray(contexto.assets.open("senas.json").bufferedReader().readText()).let { a ->
        List(a.length()) { val o = a.getJSONObject(it); Sena(o.getInt("indice"), o.getString("nombre"), o.getInt("muestras")) }
    }

    private val archivoEliminadas = File(raiz, "senas_eliminadas.txt")
    private fun eliminadas() = if (archivoEliminadas.exists()) archivoEliminadas.readLines().map { it.trim() }.filter { it.isNotEmpty() }.toSet() else emptySet()

    fun lista(): List<Sena> {
        val nuevas = if (archivoNuevas.exists())
            archivoNuevas.readLines().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") } else emptyList()
        val fuera = eliminadas()
        // los indices se mantienen aunque se borre una seña
        return (base + nuevas.mapIndexed { i, n -> Sena(base.size + i, n, 0) }).filter { nombreCarpeta(it.nombre) !in fuera }
    }

    // quita la seña de la lista y mueve sus muestras a datos_propios_borradas
    fun eliminar(s: Sena) {
        borrarTodas(s)
        carpeta(s).delete()
        archivoEliminadas.appendText(nombreCarpeta(s.nombre) + "\n")
    }

    // agrega una seña nueva al final, devuelve el error o null si salio bien
    fun agregar(nombre0: String): String? {
        val nombre = nombre0.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
        if (nombre.isEmpty()) return "ESCRIBE UN NOMBRE"
        if (nombreCarpeta(nombre).trim('_').isEmpty()) return "NOMBRE NO VALIDO"
        lista().firstOrNull { nombreCarpeta(it.nombre) == nombreCarpeta(nombre) }?.let { return "YA EXISTE: ${textoAscii(it.nombre)}" }
        // si estaba borrada se vuelve a activar con el mismo indice
        val clave = nombreCarpeta(nombre)
        if (clave in eliminadas()) {
            archivoEliminadas.writeText(eliminadas().filter { it != clave }.joinToString("") { it + "\n" })
            return null
        }
        archivoNuevas.appendText(nombre + "\n")
        return null
    }

    fun carpeta(s: Sena) = File(carpetaDatos, "%02d_%s".format(s.indice, nombreCarpeta(s.nombre)))
    fun muestrasCelular(s: Sena): List<File> =
        carpeta(s).listFiles { f -> f.name.startsWith("muestra_") && f.extension == "json" }?.sorted() ?: emptyList()
    // si ya estan las muestras base en el celular, no se suman las del PC al contar
    val pcImportado get() = File(raiz, "pc_importado.txt").exists()
    fun contar(s: Sena) = s.muestrasPc + muestrasCelular(s).size

    // mueve UNA muestra a datos_propios_borradas/fecha (por si te arrepientes)
    fun borrarUna(s: Sena, archivo: File): Boolean {
        val destino = File(raiz, "datos_propios_borradas/${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}/${carpeta(s).name}")
        destino.mkdirs()
        return archivo.renameTo(File(destino, archivo.name))
    }

    // guarda la grabacion en json, devuelve el archivo (o null) y el mensaje
    internal fun guardar(r: Registro, s: Sena): Pair<File?, String> {
        if (r.anchos.isEmpty()) return null to "NO SE VIO EL CUERPO - NO GUARDADA"
        val m = construirMuestraCompleta(r.brazos(), mediana(r.anchos), P(mediana(r.centros.map { it.x }), mediana(r.centros.map { it.y })),
            true, r.caras.toTypedArray(), r.carasOk.toBooleanArray())
        if (m.validos.none { it }) return null to "SUBE LAS MANOS - NO GUARDADA"

        fun pts(l: List<P>) = JSONArray().apply { l.forEach { put(JSONArray().put(it.x).put(it.y)) } }
        val o = JSONObject().put("clase", s.indice).put("nombre", s.nombre)
        for ((lado, sufijo) in listOf(0 to "izq", 1 to "der")) {
            o.put("muneca_$sufijo", pts(r.munecas[lado]))
            o.put("valida_$sufijo", JSONArray(r.validas[lado]))
            o.put("mano_$sufijo", JSONArray().apply { r.manos[lado].forEach { put(pts(it.toList())) } })
            o.put("mano_${sufijo}_ok", JSONArray(r.manosOk[lado]))
            o.put("codo_$sufijo", pts(r.codos[lado]))
            o.put("codo_${sufijo}_ok", JSONArray(r.codosOk[lado]))
        }
        o.put("ancho_hombros", JSONArray(r.anchos))
        o.put("centro_hombros", pts(r.centros))
        o.put("cara", JSONArray().apply { r.caras.forEach { put(pts(it.toList())) } })
        o.put("cara_ok", JSONArray(r.carasOk))

        val c = carpeta(s).apply { mkdirs() }
        val archivo = File(c, "muestra_${SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())}.json")
        archivo.writeText(o.toString())
        return archivo to "GUARDADA"
    }

    // mueve las muestras de la seña a datos_propios_borradas/fecha
    fun borrarTodas(s: Sena): Int {
        val archivos = muestrasCelular(s)
        val destino = File(raiz, "datos_propios_borradas/${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}/${carpeta(s).name}")
        if (archivos.isNotEmpty()) destino.mkdirs()
        archivos.forEach { it.renameTo(File(destino, it.name)) }
        return archivos.size
    }

    // lee una muestra json -> brazos, ancho de hombros y centro
    fun leer(archivo: File): Triple<Array<Brazo>, Double, P> {
        val o = JSONObject(archivo.readText())
        fun pts(a: JSONArray) = Array(a.length()) { val q = a.getJSONArray(it); P(q.getDouble(0), q.getDouble(1)) }
        fun bools(a: JSONArray) = BooleanArray(a.length()) { a.getBoolean(it) }
        val brazos = Array(2) { lado ->
            val s = if (lado == 0) "izq" else "der"
            Brazo(
                pts(o.getJSONArray("muneca_$s")), bools(o.getJSONArray("valida_$s")),
                o.getJSONArray("mano_$s").let { a -> Array(a.length()) { pts(a.getJSONArray(it)) } },
                bools(o.getJSONArray("mano_${s}_ok")),
                pts(o.getJSONArray("codo_$s")), bools(o.getJSONArray("codo_${s}_ok"))
            )
        }
        val anchos = o.getJSONArray("ancho_hombros").let { a -> List(a.length()) { a.getDouble(it) } }
        val centros = pts(o.getJSONArray("centro_hombros"))
        return Triple(brazos, mediana(anchos), P(mediana(centros.map { it.x }), mediana(centros.map { it.y })))
    }

    // puntos de la cara de una muestra (las viejas no los tienen)
    fun leerCaras(archivo: File): Pair<Array<Array<P>>, BooleanArray>? {
        val o = JSONObject(archivo.readText())
        if (!o.has("cara") || !o.has("cara_ok")) return null
        val c = o.getJSONArray("cara"); val k = o.getJSONArray("cara_ok")
        return Array(c.length()) { i -> val f = c.getJSONArray(i); Array(f.length()) { j -> val q = f.getJSONArray(j); P(q.getDouble(0), q.getDouble(1)) } } to
            BooleanArray(k.length()) { k.getBoolean(it) }
    }

    companion object {
        fun nombreCarpeta(nombre: String) = textoAscii(nombre).replace(" ", "_").replace("?", "").replace("!", "")
    }
}

// carpeta interna de la app, si habia algo en la externa se pasa aqui la primera vez
fun carpetaDatos(contexto: Context): File {
    val interna = contexto.filesDir
    val externa = contexto.getExternalFilesDir(null)
    val marca = File(interna, ".migrado")
    if (!marca.exists() && externa != null && externa.exists()) {
        // "aprender" (muñeco y videos que manda el PC) se queda afuera
        externa.listFiles()?.filter { it.name != "aprender" }?.forEach { it.copyRecursively(File(interna, it.name), overwrite = false) }
        externa.listFiles()?.filter { it.name != "aprender" }?.forEach { it.deleteRecursively() }
    }
    marca.writeText("ok")
    return interna
}

// muestras que vienen en el apk (assets/muestras_base.zip)
// se descomprimen la primera vez que se abre la app, las que ya existen no se sobreescriben
object MuestrasBase {
    private const val ARCHIVO = "muestras_base.zip"

    private fun version(contexto: Context): String? = try {
        contexto.assets.openFd(ARCHIVO).use { "${it.length}" }
    } catch (_: Exception) {
        try { contexto.assets.open(ARCHIVO).use { "${it.available()}" } } catch (_: Exception) { null }
    }

    private fun marca(contexto: Context) = File(carpetaDatos(contexto), ".muestras_base")

    fun faltaPreparar(contexto: Context): Boolean {
        val v = version(contexto) ?: return false
        val m = marca(contexto)
        return !m.exists() || m.readText() != v
    }

    // descomprime el zip, avance(hechas, total) para la barra
    fun preparar(contexto: Context, avance: (Int, Int) -> Unit) {
        val v = version(contexto) ?: return
        val raiz = carpetaDatos(contexto)
        val total = contexto.assets.open(ARCHIVO).use { e -> java.util.zip.ZipInputStream(e).use { z -> generateSequence { z.nextEntry }.count() } }
        var hechas = 0
        // no copiar las señas que fueron borradas
        val eliminadas = File(raiz, "senas_eliminadas.txt").let { f -> if (f.exists()) f.readLines().map { it.trim() }.filter { it.isNotEmpty() }.toSet() else emptySet() }
        contexto.assets.open(ARCHIVO).use { e ->
            java.util.zip.ZipInputStream(e.buffered()).use { z ->
                while (true) {
                    val entrada = z.nextEntry ?: break
                    val carpetaSena = entrada.name.split("/").getOrNull(1) ?: ""
                    if (!entrada.isDirectory && carpetaSena.substringAfter("_") !in eliminadas) {
                        val destino = File(raiz, entrada.name)
                        if (destino.canonicalPath.startsWith(raiz.canonicalPath) && !destino.exists()) {
                            destino.parentFile?.mkdirs()
                            destino.outputStream().use { z.copyTo(it) }
                        }
                    }
                    hechas++
                    if (hechas % 25 == 0 || hechas == total) avance(hechas, total)
                }
            }
        }
        File(raiz, "pc_importado.txt").writeText("Las grabaciones del PC vienen dentro del APK.\n")
        marca(contexto).writeText(v)
    }
}

internal fun mediana(v: List<Double>): Double {
    val s = v.sorted(); val n = s.size
    return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
}
