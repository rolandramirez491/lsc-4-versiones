package co.edu.itm.manosquehablan.vivo

import android.content.Context
import org.json.JSONObject
import kotlin.math.max

// Bucle del traductor: abrir la boca, grabar 3.4 s, analizar y agregar a la frase.

const val SEGUNDOS_GRABACION = 3.4
const val DURACION_DESTELLO = 0.45
const val UMBRAL_CONFIANZA = 0.60
const val VISIBILIDAD_MINIMA = 0.5
const val FRAMES_BOCA_ABIERTA = 3

enum class Estado { PREPARACION, GRABANDO, ANALISIS, RESULTADO }
// PRACTICAR (en APRENDER): reconoce igual que TRADUCIR pero no arma frase y busca las señas parecidas
enum class Modo { TRADUCIR, GRABAR, PRACTICAR }

const val CANDIDATAS_PRACTICA = 8      // la que gana + 7 parecidas

const val META_POR_SENA = 15

// tramo del trazo para dibujar
class Tramo(val puntos: List<P>, val color: Int, val grosor: Float)   // color: 0 izq, 1 der, 2 antebrazo

// circulo al final del trazo: lleno si se ve la mano, aro si se perdio hace poco
class Punta(val punto: P, val lado: Int, val lleno: Boolean)

// todo lo que se dibuja en un cuadro
class Escena(
    val ancho: Int,
    val alto: Int,
    val manos: Array<Mano?>,
    val fantasma: BooleanArray,
    val antebrazos: Array<Pair<P, P>?>,
    val giros: DoubleArray,
    val cara: List<P>?,          // puntos de la cara con los extras
    val caraSuave: Array<P>?,    // los 478 puntos, para el destello
    val nivelCara: Double,
    val alturaTorso: Double?,
    val estado: Estado,
    val hayCuerpo: Boolean,
    val restante: Double,
    val resultado: Resultado?,
    val frase: List<String>,
    val tramos: List<Tramo>,
    val puntas: List<Punta>,
    val destello: Double?,       // progreso 0..1 o null
    val modo: Modo = Modo.TRADUCIR,
    val sena: String = "",       // seña elegida en grabar
    val cantidad: Int = 0,       // muestras que tiene
    val mensaje: String? = null, // guardada, errores, etc
    val hayCara: Boolean = true,
    val hombros: Pair<P, Double>? = null,                    // centro y ancho de los hombros (muñeco en vivo)
    val repetir: Triple<Array<Brazo>, Double, P>? = null,    // ultima seña reconocida (el muñeco la repite)
    val repetirId: Int = 0
)

// datos de un cuadro (leer_frame)
internal class Cuadro(
    val munecas: Array<P>, val validas: BooleanArray,
    val manos: Array<Mano>, val manosOk: BooleanArray,
    val codos: Array<P>, val codosOk: BooleanArray,
    val anchoHombros: Double, val centroHombros: P, val alturaHombros: Double,
    val cara: Array<P>, val caraOk: Boolean
)

// los 11 puntos de referencia de la cara (CARA_REF) en pixeles
internal fun caraRefPixeles(det: Deteccion, ancho: Int, alto: Int): Pair<Array<P>, Boolean> {
    if (det.cara.size <= CARA_REF.max()) return Array(CANTIDAD_CARA_REF) { P(0.0, 0.0) } to false
    return Array(CANTIDAD_CARA_REF) { val q = det.cara[CARA_REF[it]]; P(q[0].toDouble() * ancho, q[1].toDouble() * alto) } to true
}

private fun leerFrame(det: Deteccion, ancho: Int, alto: Int): Cuadro? {
    val pose = det.pose
    if (pose.size <= POSE_MUNECA_DER) return null
    fun pixel(i: Int) = P(pose[i][0].toDouble() * ancho, pose[i][1].toDouble() * alto)
    fun visible(i: Int) = pose[i][2] > VISIBILIDAD_MINIMA
    val hi = pixel(POSE_HOMBRO_IZQ); val hd = pixel(POSE_HOMBRO_DER)
    val (mi, md) = asignarManos(
        if (det.manoIzq.isNotEmpty()) aPixeles(det.manoIzq, ancho, alto) else null,
        if (det.manoDer.isNotEmpty()) aPixeles(det.manoDer, ancho, alto) else null,
        if (visible(POSE_MUNECA_IZQ)) pixel(POSE_MUNECA_IZQ) else null,
        if (visible(POSE_MUNECA_DER)) pixel(POSE_MUNECA_DER) else null
    )
    val vacia = Array(CANTIDAD_MANO) { P(0.0, 0.0) }
    val (cara, caraOk) = caraRefPixeles(det, ancho, alto)
    return Cuadro(
        arrayOf(pixel(POSE_MUNECA_IZQ), pixel(POSE_MUNECA_DER)),
        booleanArrayOf(visible(POSE_MUNECA_IZQ), visible(POSE_MUNECA_DER)),
        arrayOf(mi ?: vacia, md ?: vacia), booleanArrayOf(mi != null, md != null),
        arrayOf(pixel(POSE_CODO_IZQ), pixel(POSE_CODO_DER)),
        booleanArrayOf(visible(POSE_CODO_IZQ), visible(POSE_CODO_DER)),
        (hi - hd).norma(), (hi + hd) / 2.0, (hi.y + hd.y) / 2.0,
        cara, caraOk
    )
}

// ---- al RECONOCER (TRADUCIR): grabar hasta bajar las manos (maximo 5 s). GRABAR sigue con 3.4 s fijos ----
const val SEGUNDOS_MAXIMO_GRABACION = 5.0
const val SEGUNDOS_MINIMO_GRABACION = 0.8
const val SEGUNDOS_MANOS_ABAJO = 0.35
const val CUADROS_MUESTRA = 99

// decide cuando termina: despues de SUBIR una mano por encima de su codo, cuando las dos quedan abajo
// (muñeca debajo del codo, o no se ven) un momento. O a los 5 s.
internal class FinDeSena(private val inicio: Double) {
    var subio = false
    private var abajoDesde: Double? = null

    fun actualizar(c: Cuadro?, ahora: Double): Boolean {
        val t = ahora - inicio
        if (t >= SEGUNDOS_MAXIMO_GRABACION) return true
        var arriba = false; var abajo = true
        if (c != null) for (l in 0..1) {
            if (!(c.validas[l] && c.codosOk[l])) continue          // no se ve: cuenta como abajo
            val yM = c.munecas[l].y; val yC = c.codos[l].y
            if (yM < yC) arriba = true
            if (yM < yC + 0.10 * c.anchoHombros) abajo = false
        }
        if (arriba) subio = true
        if (!(subio && abajo && t >= SEGUNDOS_MINIMO_GRABACION)) { abajoDesde = null; return false }
        val desde = abajoDesde ?: ahora.also { abajoDesde = it }
        return ahora - desde >= SEGUNDOS_MANOS_ABAJO
    }
}

// grabacion en curso
internal class Registro {
    val munecas = arrayOf(mutableListOf<P>(), mutableListOf())
    val validas = arrayOf(mutableListOf<Boolean>(), mutableListOf())
    val manos = arrayOf(mutableListOf<Mano>(), mutableListOf())
    val manosOk = arrayOf(mutableListOf<Boolean>(), mutableListOf())
    val codos = arrayOf(mutableListOf<P>(), mutableListOf())
    val codosOk = arrayOf(mutableListOf<Boolean>(), mutableListOf())
    val anchos = mutableListOf<Double>()
    val centros = mutableListOf<P>()
    val caras = mutableListOf<Array<P>>()
    val carasOk = mutableListOf<Boolean>()

    fun agregar(c: Cuadro?) {
        for (l in 0..1) {
            munecas[l].add(c?.munecas?.get(l) ?: P(0.0, 0.0))
            validas[l].add(c?.validas?.get(l) ?: false)
            manos[l].add(c?.manos?.get(l) ?: Array(CANTIDAD_MANO) { P(0.0, 0.0) })
            manosOk[l].add(c?.manosOk?.get(l) ?: false)
            codos[l].add(c?.codos?.get(l) ?: P(0.0, 0.0))
            codosOk[l].add(c?.codosOk?.get(l) ?: false)
        }
        caras.add(c?.cara ?: Array(CANTIDAD_CARA_REF) { P(0.0, 0.0) })
        carasOk.add(c?.caraOk ?: false)
        if (c != null) { anchos.add(c.anchoHombros); centros.add(c.centroHombros) }
    }

    // la grabacion (dure lo que dure) se lleva a 99 cuadros como las muestras del entrenamiento
    // (cuadro mas cercano, no se mezclan puntos vistos con no vistos)
    fun normalizar(cuadros: Int = CUADROS_MUESTRA) {
        val n = carasOk.size
        if (n == 0 || n == cuadros) return
        val elegidos = IntArray(cuadros) { redondear(it * (n - 1).toDouble() / (cuadros - 1)).coerceIn(0, n - 1) }
        fun <T> tomar(lista: MutableList<T>) { val copia = lista.toList(); lista.clear(); elegidos.forEach { lista.add(copia[it]) } }
        for (l in 0..1) { tomar(munecas[l]); tomar(validas[l]); tomar(manos[l]); tomar(manosOk[l]); tomar(codos[l]); tomar(codosOk[l]) }
        tomar(caras); tomar(carasOk)
    }

    fun brazos() = Array(2) { l ->
        Brazo(
            munecas[l].toTypedArray(), validas[l].toBooleanArray(), manos[l].toTypedArray(),
            manosOk[l].toBooleanArray(), codos[l].toTypedArray(), codosOk[l].toBooleanArray()
        )
    }
}

// guarda los filtros y el estado del traductor o de grabar entre cuadros
class SeguimientoEnVivo(contexto: Context, val senas: Senas) {
    private val manosEnVivo = ManosEnVivo()
    private val antebrazosEnVivo = AntebrazosEnVivo()
    private val caraEnVivo = FiltroCara()
    private var nivelCara = 0.0
    private val aristas = cargarAristas(contexto)
    // el modelo v17 no viene en esta app (solo se carga si alguien lo pide)
    val cascada by lazy { Cascada(contexto) }

    private var estado = Estado.PREPARACION
    private var inicioEstado = 0.0
    private var destelloHasta = 0.0
    private var registro = Registro()
    private var finSena = FinDeSena(0.0)
    // INTENTAR SEÑA: quién analiza la grabación (la versión elegida) y la última grabación (para re-analizar
    // con otra versión sin volver a hacer la seña)
    @Volatile internal var analizadorExterno: ((Registro, Int, Int) -> Resultado)? = null
    @Volatile internal var ultimoRegistro: Registro? = null
    @Volatile var ultimoAncho = 0
    @Volatile var ultimoAlto = 0
    private var resultado: Resultado? = null
    private var framesBocaAbierta = 0
    private var listoParaDisparar = false
    private val frase = mutableListOf<String>()
    private val pedidos = java.util.concurrent.ConcurrentLinkedQueue<() -> Unit>()

    // ---- GRABAR ----
    private var modo = Modo.TRADUCIR
    private var senaActual: Senas.Sena? = null
    private var cantidadActual = 0
    private var mensaje: String? = null
    private var mensajeHasta = 0.0
    private var ultimoAhora = 0.0
    private val ultimaGuardada = mutableMapOf<Int, java.io.File>()
    // TRADUCIR: la ultima seña reconocida, el muñeco la repite hasta BORRAR ULTIMA / BORRAR TODO
    private var ultimaSena: Triple<Array<Brazo>, Double, P>? = null
    private var ultimaSenaId = 0

    private fun avisar(texto: String, segundos: Double) { mensaje = texto; mensajeHasta = ultimoAhora + segundos }

    // acciones de los botones, se ejecutan en el siguiente cuadro
    fun borrarUltima() = pedidos.add {
        if (estado == Estado.GRABANDO) return@add
        if (modo == Modo.TRADUCIR) { if (frase.isNotEmpty()) frase.removeAt(frase.size - 1); ultimaSena = null; return@add }
        val s = senaActual ?: return@add
        val archivo = ultimaGuardada.remove(s.indice)
        if (archivo != null && archivo.delete()) { cantidadActual = senas.contar(s); avisar("ULTIMA BORRADA", 1.5) }
    }
    fun borrarMuestra(archivo: java.io.File, numero: Int) = pedidos.add {
        if (estado == Estado.GRABANDO) return@add
        val s = senaActual ?: return@add
        if (senas.borrarUna(s, archivo)) {
            ultimaGuardada.remove(s.indice)
            cantidadActual = senas.contar(s)
            avisar("MUESTRA #$numero BORRADA", 2.0)
        }
    }
    fun borrarTodo() = pedidos.add {
        if (estado == Estado.GRABANDO) return@add
        if (modo == Modo.TRADUCIR) { frase.clear(); ultimaSena = null; return@add }
        val s = senaActual ?: return@add
        val n = senas.borrarTodas(s)
        ultimaGuardada.remove(s.indice)
        cantidadActual = senas.contar(s)
        avisar("${textoAscii(s.nombre)}: $n MUESTRAS BORRADAS (DEL CELULAR)", 2.5)
    }
    fun cambiarModo(nuevo: Modo, sena: Senas.Sena? = null) = pedidos.add {
        modo = nuevo
        estado = Estado.PREPARACION
        resultado = null
        listoParaDisparar = false
        if (sena != null) { senaActual = sena; cantidadActual = senas.contar(sena) }
    }
    fun elegirSena(sena: Senas.Sena, aviso: String? = null) = pedidos.add {
        if (estado == Estado.GRABANDO) return@add
        senaActual = sena; cantidadActual = senas.contar(sena)
        if (aviso != null) avisar(aviso, 3.0)
    }

    fun procesar(det: Deteccion, ancho: Int, alto: Int, ahora: Double): Escena {
        ultimoAhora = ahora
        while (true) (pedidos.poll() ?: break).invoke()

        val datos = leerFrame(det, ancho, alto)
        val apertura = aperturaBoca(det, ancho, alto)

        if (apertura != null && apertura < UMBRAL_BOCA_CERRADA) listoParaDisparar = true
        framesBocaAbierta = if (apertura != null && apertura >= UMBRAL_BOCA_ABIERTA) framesBocaAbierta + 1 else 0

        val puedeDisparar = estado == Estado.PREPARACION ||
            (modo != Modo.GRABAR && estado == Estado.RESULTADO)
        if (puedeDisparar && (modo != Modo.GRABAR || senaActual != null) && listoParaDisparar &&
            framesBocaAbierta >= FRAMES_BOCA_ABIERTA
        ) {
            estado = Estado.GRABANDO
            inicioEstado = ahora
            destelloHasta = ahora + DURACION_DESTELLO
            registro = Registro()
            finSena = FinDeSena(ahora)
            resultado = null
            listoParaDisparar = false
            framesBocaAbierta = 0
        }

        val manos = manosEnVivo.actualizar(det, ancho, alto, ahora)
        val caraSuave = caraEnVivo.actualizar(det, ancho, alto, ahora)
        val antebrazos = antebrazosEnVivo.actualizar(det, ancho, alto, ahora, manos)
        nivelCara = 0.6 * nivelCara + 0.4 * nivelBoca(apertura, UMBRAL_BOCA_CERRADA, UMBRAL_BOCA_ABIERTA)

        var restante = 0.0
        var tramos = emptyList<Tramo>()
        var puntas = emptyList<Punta>()
        when (estado) {
            Estado.PREPARACION -> {}
            Estado.GRABANDO -> {
                registro.agregar(datos)
                val r = recorridoEnVivo(registro, ahora - inicioEstado); tramos = r.first; puntas = r.second
                // GRABAR: 3.4 s fijos (cuenta atras). TRADUCIR: hasta bajar las manos (cuenta hacia arriba)
                val termino = if (modo == Modo.GRABAR) {
                    restante = SEGUNDOS_GRABACION - (ahora - inicioEstado); restante <= 0
                } else {
                    restante = ahora - inicioEstado
                    finSena.actualizar(datos, ahora).also { if (it) registro.normalizar() }
                }
                if (termino) {
                    destelloHasta = ahora + DURACION_DESTELLO
                    val s = senaActual
                    if (modo == Modo.GRABAR && s != null) {
                        val (archivo, msg) = senas.guardar(registro, s)
                        if (archivo != null) ultimaGuardada[s.indice] = archivo
                        cantidadActual = senas.contar(s)
                        avisar(msg, 1.5)
                        estado = Estado.PREPARACION
                    } else {
                        estado = Estado.ANALISIS
                    }
                }
            }
            Estado.ANALISIS -> {
                // INTENTAR SEÑA (4 versiones): el análisis lo hace la versión elegida, no el modelo v17
                val externo = analizadorExterno
                ultimoRegistro = registro; ultimoAncho = ancho; ultimoAlto = alto
                val res = if (modo == Modo.PRACTICAR && externo != null) externo(registro, ancho, alto)
                    else analizar(registro, modo == Modo.PRACTICAR)
                resultado = res
                estado = Estado.RESULTADO
                if (res.error == null && modo == Modo.TRADUCIR) {
                    val (nombre, p) = res.top[0]
                    if (p >= UMBRAL_CONFIANZA) {
                        frase.add(nombre)
                        if (registro.anchos.isNotEmpty()) {
                            ultimaSena = Triple(registro.brazos(), mediana(registro.anchos),
                                P(mediana(registro.centros.map { it.x }), mediana(registro.centros.map { it.y })))
                            ultimaSenaId++
                        }
                    }
                }
            }
            Estado.RESULTADO -> {
                val r = recorridoEnVivo(registro, SEGUNDOS_GRABACION); tramos = r.first; puntas = r.second
            }
        }

        val torso = datos?.let { it.alturaHombros + UMBRAL_TORSO * it.anchoHombros }
        return Escena(
            ancho, alto, manos.copyOf(), manosEnVivo.fantasma.copyOf(), antebrazos,
            antebrazosEnVivo.giro.copyOf(), caraSuave?.let { puntosDensosCara(it, aristas) }, caraSuave,
            nivelCara,
            if (modo == Modo.GRABAR || estado == Estado.PREPARACION || estado == Estado.GRABANDO) torso else null,
            estado, datos != null, restante, resultado, frase.toList(), tramos, puntas,
            if (ahora < destelloHasta) 1.0 - (destelloHasta - ahora) / DURACION_DESTELLO else null,
            modo, senaActual?.let { textoAscii(it.nombre) } ?: "", cantidadActual,
            if (ahora < mensajeHasta) mensaje else null ?: if (estado == Estado.GRABANDO && modo != Modo.GRABAR)
                (if (finSena.subio) "BAJA LAS MANOS PARA TERMINAR" else "HAZ LA SENA") else null,
            apertura != null,
            datos?.let { it.centroHombros to it.anchoHombros },
            if (modo == Modo.TRADUCIR) ultimaSena else null, ultimaSenaId
        )
    }

    private fun analizar(r: Registro, conParecidas: Boolean = false): Resultado {
        if (r.anchos.isEmpty()) return Resultado(error = "NO SE VE EL CUERPO")
        val muestra = construirMuestraCompleta(
            r.brazos(), mediana(r.anchos), P(mediana(r.centros.map { it.x }), mediana(r.centros.map { it.y })),
            true, r.caras.toTypedArray(), r.carasOk.toBooleanArray()
        )
        // la que gana: igual que en el traductor (sus mismas candidatas)
        val res = cascada.analizar(muestra)
        if (!conParecidas || res.error != null) return res
        // las 7 parecidas: se mira con mas candidatas, sin cambiar la que gano
        val ganadora = res.top[0].first
        val mas = cascada.analizar(muestra, CANDIDATAS_PRACTICA)
        return Resultado(res.error, res.top, res.candidatas,
            mas.candidatas.map { it.nombre }.filter { it != ganadora }.take(CANDIDATAS_PRACTICA - 1))
    }

    // trazo limpio de la grabacion para dibujarlo en vivo
    private fun recorridoEnVivo(r: Registro, segundos: Double): Pair<List<Tramo>, List<Punta>> {
        val tramos = mutableListOf<Tramo>(); val puntas = mutableListOf<Punta>()
        if (r.anchos.isEmpty()) return tramos to puntas
        val anchoHombros = mediana(r.anchos)
        val b = r.brazos()
        val total = b[0].munecas.size
        val fps = total / max(segundos, 0.1)
        val maxHueco = max(1, redondear(SEGUNDOS_HUECO * fps))

        fun cortar(p: Array<P>, v: BooleanArray, color: Int, grosor: Float) {
            var i = 0
            while (i < p.size) {
                if (!v[i]) { i++; continue }
                var j = i
                while (j < p.size && v[j]) j++
                tramos.add(Tramo(p.slice(i until j), color, grosor))
                i = j
            }
        }
        for (lado in 0..1) {
            val br = b[lado]
            val (pB, vB) = recorridoAntebrazo(br.munecas, br.validas, br.codos, br.codosOk, anchoHombros, fps)
            if (vB.any { it }) cortar(pB, vB, 2, 3f)

            val (p, v) = limpiarRecorrido(
                br.munecas, br.validas, anchoHombros, fps, suavizarManos(br.manos, br.manosOk, fps), br.manosOk
            )
            if (v.none { it }) continue
            cortar(p, v, lado, 3f)
            val ultimo = v.indices.last { v[it] }
            val perdidos = total - 1 - ultimo
            if (perdidos == 0) puntas.add(Punta(p[ultimo], lado, true))
            else if (perdidos <= maxHueco) puntas.add(Punta(p[ultimo], lado, false))
        }
        return tramos to puntas
    }

    private fun cargarAristas(contexto: Context): AristasCara {
        val texto = contexto.assets.open("aristas_cara.json").bufferedReader().use { it.readText() }
        val json = JSONObject(texto)
        fun lista(clave: String): Array<IntArray> {
            val a = json.getJSONArray(clave)
            return Array(a.length()) { val par = a.getJSONArray(it); intArrayOf(par.getInt(0), par.getInt(1)) }
        }
        return AristasCara(lista("malla"), lista("detalle"), lista("nariz"))
    }
}
