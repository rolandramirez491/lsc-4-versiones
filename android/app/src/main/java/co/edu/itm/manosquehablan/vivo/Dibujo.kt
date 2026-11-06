package co.edu.itm.manosquehablan.vivo

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PointMode
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.PI
import kotlin.math.roundToInt

// colores en rgb
val COLOR_IZQ = Color(0, 255, 0)          // verde
val COLOR_DER = Color(255, 0, 255)        // magenta
private val COLOR_DORADO = Color(255, 215, 0)
private val COLOR_HUESO_CLARO = Color(255, 255, 255)
private val COLOR_HUESO_OSCURO = Color(40, 40, 40)
private val COLOR_TORSO = Color(255, 200, 0)

const val HUESO_ALFA = 0.45f
const val HUESO_GROSOR = 0.16
const val MANO_ALFA = 0.45f

private val DEDOS = listOf(
    intArrayOf(0, 1, 2, 3, 4), intArrayOf(5, 6, 7, 8), intArrayOf(9, 10, 11, 12),
    intArrayOf(13, 14, 15, 16), intArrayOf(17, 18, 19, 20)
)
private val PALMA = intArrayOf(0, 1, 5, 9, 13, 17)
private val CONEXIONES_MANO = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8,
    5 to 9, 9 to 10, 10 to 11, 11 to 12, 9 to 13, 13 to 14, 14 to 15, 15 to 16,
    13 to 17, 17 to 18, 18 to 19, 19 to 20, 0 to 17
)

private fun mezclar(a: Color, b: Color, t: Float) = Color(
    a.red + (b.red - a.red) * t, a.green + (b.green - a.green) * t, a.blue + (b.blue - a.blue) * t
)

fun colorPorGiro(giro: Double) = mezclar(COLOR_HUESO_CLARO, COLOR_HUESO_OSCURO, giro.toFloat())
fun colorAntebrazo(c: Color) = mezclar(c, Color.White, 0.55f)

// envolvente convexa de los puntos
internal fun envolvente(puntos: List<Offset>): List<Offset> {
    val p = puntos.sortedWith(compareBy({ it.x }, { it.y }))
    if (p.size < 3) return p
    fun cruz(o: Offset, a: Offset, b: Offset) = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    val abajo = mutableListOf<Offset>()
    for (q in p) { while (abajo.size >= 2 && cruz(abajo[abajo.size - 2], abajo.last(), q) <= 0) abajo.removeAt(abajo.size - 1); abajo.add(q) }
    val arriba = mutableListOf<Offset>()
    for (q in p.reversed()) { while (arriba.size >= 2 && cruz(arriba[arriba.size - 2], arriba.last(), q) <= 0) arriba.removeAt(arriba.size - 1); arriba.add(q) }
    return abajo.dropLast(1) + arriba.dropLast(1)
}

// dibuja todo encima de la camara, la imagen se escala como FILL_CENTER
fun DrawScope.dibujarEscena(e: Escena) {
    if (e.ancho == 0) return
    val esc = max(size.width / e.ancho, size.height / e.alto)
    val dx = (size.width - e.ancho * esc) / 2f
    val dy = (size.height - e.alto * esc) / 2f
    // de pixeles de la imagen a pixeles de la pantalla
    fun pant(x: Double, y: Double) = Offset((x * esc + dx).toFloat(), (y * esc + dy).toFloat())
    fun espejo(p: P) = pant(e.ancho - 1 - p.x, p.y)
    // los textos se ubican sobre la pantalla (720 de alto como referencia)
    val hp = size.height / 720f
    // margen arriba y abajo para que no queden pegados al borde
    fun hud(x: Double, y: Double) = Offset((x * hp).toFloat(), ((y + MARGEN_HUD) * hp).toFloat())
    val px = esc   // 1 píxel de la cámara en la pantalla

    dibujarAntebrazos(e, ::espejo, px)
    dibujarCara(e, ::espejo, px)
    dibujarSiluetas(e, ::espejo, px)

    // esqueleto de la mano
    for (lado in 0..1) {
        val mano = e.manos[lado] ?: continue
        val color = if (lado == 0) COLOR_IZQ else COLOR_DER
        val p = mano.map(::espejo)
        for ((a, b) in CONEXIONES_MANO) drawLine(color, p[a], p[b], 2 * px, cap = StrokeCap.Round)
        for (q in p) {
            drawCircle(Color.White, 4 * px, q)
            drawCircle(color, 4 * px, q, style = Stroke(1 * px))
        }
    }

    // linea del torso
    e.alturaTorso?.let { yT ->
        val y = pant(0.0, yT).y
        drawLine(COLOR_TORSO, Offset(dx, y), Offset(dx + e.ancho * esc, y), 1 * px)
        texto("SUBE LAS MANOS SOBRE ESTA LINEA", pant(e.ancho - 430.0, yT - 8), 0.5f, COLOR_TORSO, px)
    }

    // trazo de la grabacion
    for (t in e.tramos) {
        if (t.puntos.size < 2) continue
        val color = when (t.color) { 0 -> COLOR_IZQ; 1 -> COLOR_DER; else -> COLOR_TRAZO_ANTEBRAZO }
        val camino = Path().apply {
            val p0 = espejo(t.puntos[0]); moveTo(p0.x, p0.y)
            for (q in t.puntos.drop(1)) espejo(q).let { lineTo(it.x, it.y) }
        }
        drawPath(camino, color, style = Stroke(t.grosor * px, cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
    for (pu in e.puntas) {
        val color = if (pu.lado == 0) COLOR_IZQ else COLOR_DER
        if (pu.lleno) drawCircle(color, 6 * px, espejo(pu.punto))
        else drawCircle(color, 10 * px, espejo(pu.punto), style = Stroke(2 * px))
    }

    val rojo = Color(255, 0, 0); val naranja = Color(255, 165, 0); val verde = Color(0, 255, 0)
    val gris = Color(220, 220, 220)
    if (e.modo == Modo.GRABAR) {
        if (e.estado == Estado.GRABANDO) {
            drawCircle(rojo, 14 * hp, hud(35.0, 38.0))
            texto("GRABANDO ${e.sena} %.1f".format(max(e.restante, 0.0)), hud(60.0, 50.0), 0.9f, rojo, hp, 2)
        } else {
            // FPS, SEÑA y MUESTRA x DE y van arriba a la izquierda (PantallaCamara)
            if (!e.hayCara) texto("NO SE VE LA CARA", hud(20.0, 200.0), 0.7f, rojo, hp)
            e.mensaje?.let {
                texto(it, hud(20.0, 245.0), 1.0f, if (it == "GUARDADA" || it.startsWith("NUEVA")) verde else rojo, hp, 3)
            }
        }
        e.destello?.let { dibujarDestello(e, it, ::espejo) }
        return
    }
    // PRACTICAR: lo que reconocio va ARRIBA de la camara (PantallaAprender), aqui solo el aviso
    if (e.modo == Modo.PRACTICAR) {
        when (e.estado) {
            Estado.GRABANDO -> {
                drawCircle(rojo, 14 * hp, hud(35.0, 38.0))
                texto("GRABANDO %.1f s".format(max(e.restante, 0.0)), hud(60.0, 50.0), 0.9f, rojo, hp, 2)
                e.mensaje?.let { texto(it, hud(60.0, 90.0), 0.7f, Color.White, hp, 2) }
            }
            Estado.ANALISIS -> texto("ANALIZANDO...", hud(20.0, 45.0), 0.9f, Color(255, 255, 0), hp, 2)
            Estado.RESULTADO -> texto("ABRE LA BOCA PARA OTRA VEZ", hud(20.0, 45.0), 0.8f, Color.White, hp, 2)
            Estado.PREPARACION -> texto(if (e.hayCuerpo) "ABRE LA BOCA Y HAZ LA SENA" else "NO SE VE EL CUERPO",
                hud(20.0, 45.0), 0.8f, if (e.hayCuerpo) verde else rojo, hp, 2)
        }
        e.destello?.let { dibujarDestello(e, it, ::espejo) }
        return
    }
    when (e.estado) {
        Estado.PREPARACION -> if (!e.hayCuerpo) texto("NO SE VE EL CUERPO", hud(20.0, 125.0), 0.7f, rojo, hp)
        Estado.GRABANDO -> {
            drawCircle(rojo, 14 * hp, hud(35.0, 38.0))
            texto("GRABANDO %.1f s".format(max(e.restante, 0.0)), hud(60.0, 50.0), 0.9f, rojo, hp, 2)
            e.mensaje?.let { texto(it, hud(60.0, 90.0), 0.7f, Color.White, hp, 2) }   // "BAJA LAS MANOS..."
        }
        Estado.ANALISIS -> texto("ANALIZANDO...", hud(20.0, 45.0), 0.9f, Color(255, 255, 0), hp, 2)
        Estado.RESULTADO -> e.resultado?.let { r ->
            if (r.error != null) {
                texto(r.error, hud(20.0, 60.0), 1.1f, rojo, hp, 3)
                return@let
            }
            val (nombre, prob) = r.top[0]
            if (prob < UMBRAL_CONFIANZA) {
                texto("NO SEGURO", hud(20.0, 60.0), 1.1f, naranja, hp, 3)
                texto("$nombre? %.0f%%".format(prob * 100), hud(20.0, 105.0), 0.9f, naranja, hp, 2)
            } else {
                texto("$nombre %.0f%%".format(prob * 100), hud(20.0, 60.0), 1.3f, verde, hp, 3)
            }
            if (r.candidatas.isNotEmpty()) {
                r.candidatas.forEachIndexed { i, c ->
                    val elegida = c.nombre == nombre
                    val brazo = c.pBrazo?.let { "  Angulo codo %.0f%%".format(it * 100) } ?: ""
                    texto(
                        (if (elegida) "> " else "  ") + "${c.nombre} %.0f%%  Dibujo1 %.0f%%".format(c.pCoord * 100, c.pDibujo * 100) + brazo,
                        hud(20.0, 150.0 + i * 32), 0.6f, if (elegida) verde else gris, hp
                    )
                }
            } else {
                r.top.drop(1).forEachIndexed { i, (otro, p) ->
                    texto("$otro %.0f%%".format(p * 100), hud(20.0, 145.0 + i * 35), 0.7f, gris, hp, 2)
                }
            }
        }
    }

    if (e.estado != Estado.GRABANDO) dibujarFrase(e, ::hud, hp, (size.width / hp).toDouble(), 720.0)
    e.destello?.let { dibujarDestello(e, it, ::espejo) }
}

private val COLOR_TRAZO_ANTEBRAZO = Color(205, 175, 50)

// margen de los textos (en una pantalla de 720 de alto)
const val MARGEN_HUD = 40.0

// barra de abajo con la frase (ultimas 2 lineas)
private fun DrawScope.dibujarFrase(e: Escena, pant: (Double, Double) -> Offset, px: Float, anchoHud: Double, altoHud: Double) {
    val escala = 0.9f
    val medidor = android.graphics.Paint().apply { textSize = 30f * escala; typeface = android.graphics.Typeface.DEFAULT_BOLD }
    // espacio libre a la derecha
    val anchoMax = (anchoHud - 40 - 120).toFloat()
    val lineas = mutableListOf("")
    for (palabra in e.frase.ifEmpty { listOf("...") }) {
        val prueba = (lineas.last() + " " + palabra).trim()
        if (medidor.measureText(prueba) > anchoMax && lineas.last().isNotEmpty()) lineas.add(palabra)
        else lineas[lineas.size - 1] = prueba
    }
    val ultimas = lineas.takeLast(2)
    val altoLinea = 38.0
    val yBase = altoHud - 25.0 - 2 * MARGEN_HUD
    val yInicio = yBase - altoLinea * ultimas.size - 10
    val arriba = pant(0.0, yInicio).y; val abajo = pant(0.0, yBase + 5).y
    drawRect(Color.Black.copy(alpha = 0.55f), Offset(0f, arriba), androidx.compose.ui.geometry.Size(size.width, abajo - arriba))
    val color = if (e.frase.isNotEmpty()) Color.White else Color(150, 150, 150)
    ultimas.forEachIndexed { i, linea -> texto(linea, pant(20.0, yInicio + altoLinea * (i + 1)), escala, color, px, 2) }
}

// destello blanco desde la cara
private fun DrawScope.dibujarDestello(e: Escena, progreso: Double, espejo: (P) -> Offset) {
    val intensidad = Math.pow(1.0 - progreso.coerceIn(0.0, 1.0), 1.5).toFloat()
    val cara = e.caraSuave
    if (cara == null || cara.size < 3) {
        drawRect(Color.White.copy(alpha = 0.5f * intensidad))
        return
    }
    val p = cara.map(espejo)
    val casco = envolvente(p)
    val minX = p.minOf { it.x }; val maxX = p.maxOf { it.x }
    val minY = p.minOf { it.y }; val maxY = p.maxOf { it.y }
    val centro = Offset((minX + maxX) / 2, (minY + maxY) / 2)
    val tamano = max(maxX - minX, maxY - minY)
    val sigma = tamano * (0.2f + 1.3f * progreso.toFloat())
    val radio = tamano / 2 + 2.5f * sigma
    drawCircle(
        androidx.compose.ui.graphics.Brush.radialGradient(
            0f to Color.White.copy(alpha = intensidad),
            (tamano / 2 / radio) to Color.White.copy(alpha = intensidad),
            1f to Color.White.copy(alpha = 0f),
            center = centro, radius = radio
        ),
        radio, centro
    )
    val camino = Path().apply { moveTo(casco[0].x, casco[0].y); casco.drop(1).forEach { lineTo(it.x, it.y) }; close() }
    drawPath(camino, Color.White.copy(alpha = intensidad))
}

private fun DrawScope.dibujarAntebrazos(e: Escena, espejo: (P) -> Offset, px: Float) {
    for (lado in 0..1) {
        val (muneca, codo) = e.antebrazos[lado] ?: continue
        val color = if (lado == 0) COLOR_IZQ else COLOR_DER
        val giro = e.giros[lado]
        val m = espejo(muneca); val c = espejo(codo)
        val largoCam = (muneca - codo).norma()
        if (largoCam < 5) continue

        val grosor = max(8, (largoCam * HUESO_GROSOR).roundToInt()) * px
        val radioCabeza = grosor * 0.75f
        val radioPie = grosor * 0.55f

        // hueso semitransparente
        drawIntoCanvas { canvas ->
            canvas.saveLayer(Rect(Offset.Zero, size), Paint().apply { alpha = HUESO_ALFA })
            val col = colorPorGiro(giro)
            drawLine(col, m, c, grosor, cap = StrokeCap.Round)
            drawCircle(col, radioPie, m)
            drawCircle(col, radioCabeza, c)
            canvas.restore()
        }

        // puntos del codo, los de los lados se cruzan cuando gira el antebrazo
        val largo = (c - m).getDistance()
        val dir = (c - m) / largo
        val perp = Offset(-dir.y, dir.x)
        val apertura = cos(giro * PI).toFloat()
        val ladoA = c + perp * (radioCabeza * apertura)
        val ladoB = c - perp * (radioCabeza * apertura)
        val claro = colorAntebrazo(color)
        for ((punto, radio, relleno) in listOf(
            Triple(m, 5f, claro), Triple(ladoA, 4f, Color.White),
            Triple(ladoB, 4f, Color(60, 60, 60)), Triple(c, 6f, claro)
        )) {
            drawCircle(relleno, radio * px, punto)
            drawCircle(color, radio * px, punto, style = Stroke(1 * px))
        }
    }
}

private fun DrawScope.formaMano(p: List<Offset>, grosorDedo: Float, color: Color, extra: Float, modo: BlendMode) {
    val casco = envolvente(PALMA.map { p[it] })
    val camino = Path().apply {
        moveTo(casco[0].x, casco[0].y)
        for (q in casco.drop(1)) lineTo(q.x, q.y)
        close()
    }
    drawPath(camino, color, style = Fill, blendMode = modo)
    if (extra > 0) drawPath(camino, color, style = Stroke(2 * extra, join = StrokeJoin.Round), blendMode = modo)
    for (dedo in DEDOS) {
        for (i in 0 until dedo.size - 1)
            drawLine(color, p[dedo[i]], p[dedo[i + 1]], grosorDedo + 2 * extra, cap = StrokeCap.Round, blendMode = modo)
        drawCircle(color, grosorDedo / 2 + extra, p[dedo.last()], blendMode = modo)
    }
}

private fun DrawScope.dibujarSiluetas(e: Escena, espejo: (P) -> Offset, px: Float) {
    for (lado in 0..1) {
        val mano = e.manos[lado] ?: continue
        val tamano = (mano[MANO_BASE_MEDIO] - mano[MANO_MUNECA]).norma()
        if (tamano < 5) continue
        val p = mano.map(espejo)
        val grosorDedo = max(4, (tamano * 0.28).roundToInt()) * px

        drawIntoCanvas { canvas ->
            // relleno segun el giro de la mano
            canvas.saveLayer(Rect(Offset.Zero, size), Paint().apply { alpha = MANO_ALFA })
            formaMano(p, grosorDedo, colorPorGiro(e.giros[lado]), 0f, BlendMode.SrcOver)
            canvas.restore()
            // contorno que sigue la mano y cada dedo: negro si se ve la mano,
            // blanco brillante con resplandor si se perdió el seguimiento (mano estimada)
            val fantasma = e.fantasma[lado]
            if (fantasma) {
                canvas.saveLayer(Rect(Offset.Zero, size), Paint().apply { alpha = 0.45f })
                formaMano(p, grosorDedo, Color.White, 5 * px, BlendMode.SrcOver)
                formaMano(p, grosorDedo, Color.White, 0f, BlendMode.Clear)
                canvas.restore()
            }
            canvas.saveLayer(Rect(Offset.Zero, size), Paint())
            formaMano(p, grosorDedo, if (fantasma) Color.White else Color.Black, 2 * px, BlendMode.SrcOver)
            formaMano(p, grosorDedo, Color.Black, 0f, BlendMode.Clear)
            canvas.restore()
        }
    }
}

private fun DrawScope.dibujarCara(e: Escena, espejo: (P) -> Offset, px: Float) {
    val cara = e.cara ?: return
    val color = mezclar(COLOR_DORADO, Color.White, e.nivelCara.toFloat())
    drawPoints(cara.map(espejo), PointMode.Points, color, strokeWidth = max(2 * px, 1.5f), cap = StrokeCap.Square)
}

// texto con borde negro
fun DrawScope.texto(contenido: String, pos: Offset, escala: Float, color: Color, px: Float, grosor: Int = 1) {
    drawIntoCanvas { canvas ->
        val pintura = android.graphics.Paint().apply {
            isAntiAlias = true
            textSize = 30f * escala * px
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
        pintura.style = android.graphics.Paint.Style.STROKE
        pintura.strokeWidth = (grosor + 3) * px
        pintura.color = android.graphics.Color.BLACK
        canvas.nativeCanvas.drawText(contenido, pos.x, pos.y, pintura)
        pintura.style = android.graphics.Paint.Style.FILL
        pintura.color = color.toArgb()
        canvas.nativeCanvas.drawText(contenido, pos.x, pos.y, pintura)
    }
}
