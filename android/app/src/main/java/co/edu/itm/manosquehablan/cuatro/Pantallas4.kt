package co.edu.itm.manosquehablan.cuatro

import android.Manifest
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import co.edu.itm.manosquehablan.aprender.AvatarPc
import co.edu.itm.manosquehablan.aprender.AvatarSena
import co.edu.itm.manosquehablan.aprender.areaPorAlto
import co.edu.itm.manosquehablan.aprender.dibujarMunieco
import co.edu.itm.manosquehablan.aprender.leerCuadrosClip
import co.edu.itm.manosquehablan.camara.CamaraIntentar
import co.edu.itm.manosquehablan.vivo.COLOR_DER
import co.edu.itm.manosquehablan.vivo.COLOR_IZQ
import co.edu.itm.manosquehablan.vivo.InfoVersiones
import co.edu.itm.manosquehablan.vivo.Registro
import co.edu.itm.manosquehablan.vivo.Resultado
import co.edu.itm.manosquehablan.vivo.Senas
import co.edu.itm.manosquehablan.vivo.Versiones4
import co.edu.itm.manosquehablan.vivo.textoAscii
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

// LSC 4 VERSIONES: elegir versión, MOSTRAR SEÑAS (las 90, una a una) e INTENTAR SEÑA con la versión elegida.

private val FONDO = Color(28, 28, 32)
private val FONDO_LISTA = Color(36, 36, 42)
private val DORADO = Color(255, 215, 0)
private val COLOR_VERSION = mapOf(1 to Color(90, 160, 90), 2 to Color(60, 140, 200), 3 to Color(220, 120, 60), 4 to Color(200, 40, 40))
private val CONEXIONES = listOf(0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11,
    11 to 12, 9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 17 to 18, 18 to 19, 19 to 20, 0 to 17)

private enum class Pantalla { INICIO, MOSTRAR, INTENTAR }

class Sena90(val clase: Int, val nombre: String, val video: Boolean)

@Composable
fun App4(senas: Senas) {
    val contexto = LocalContext.current
    val pedir = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
    LaunchedEffect(Unit) { pedir.launch(Manifest.permission.CAMERA) }
    val lista = remember {
        val a = JSONArray(contexto.assets.open("senas_90.json").bufferedReader().readText())
        List(a.length()) { val o = a.getJSONObject(it); Sena90(o.getInt("clase"), o.getString("nombre"), o.getBoolean("video")) }
    }
    var versiones by remember { mutableStateOf<Versiones4?>(null) }
    LaunchedEffect(Unit) { versiones = withContext(Dispatchers.IO) { Versiones4(contexto) } }
    var pantalla by remember { mutableStateOf(Pantalla.INICIO) }
    var version by remember { mutableIntStateOf(4) }
    var indice by remember { mutableIntStateOf(0) }

    when (pantalla) {
        Pantalla.INICIO -> Inicio(version, { version = it }) { pantalla = Pantalla.MOSTRAR }
        Pantalla.MOSTRAR -> {
            BackHandler { pantalla = Pantalla.INICIO }
            Mostrar(lista, indice, version, { indice = it }, { version = it }, alVolver = { pantalla = Pantalla.INICIO }) { pantalla = Pantalla.INTENTAR }
        }
        Pantalla.INTENTAR -> {
            BackHandler { pantalla = Pantalla.MOSTRAR }
            val v = versiones
            if (v == null) Box(Modifier.fillMaxSize().background(FONDO), contentAlignment = Alignment.Center) {
                Text("Cargando las 4 versiones...", color = Color.White)
            } else Intentar(senas, v, lista[indice], version, { version = it }) { pantalla = Pantalla.MOSTRAR }
        }
    }
}

// ---------------------------------------------------------------- inicio: las 4 versiones

@Composable
private fun Inicio(version: Int, alElegir: (Int) -> Unit, alMostrar: () -> Unit) {
    Column(Modifier.fillMaxSize().background(FONDO).padding(16.dp)) {
        Text("LSC 4 versiones", color = Color.White, fontSize = 28.sp, fontWeight = FontWeight.Bold)
        Text("Elige una versión del reconocedor. Cada una agrega algo a la anterior.", color = Color(200, 200, 200), fontSize = 14.sp)
        Spacer(Modifier.height(10.dp))
        Row(Modifier.fillMaxWidth().weight(1f), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            for (v in 1..4) TarjetaVersion(v, v == version, Modifier.weight(1f).fillMaxHeight()) { alElegir(v) }
        }
        Spacer(Modifier.height(12.dp))
        Button(onClick = alMostrar, colors = ButtonDefaults.buttonColors(containerColor = Color(0, 110, 180)),
            modifier = Modifier.fillMaxWidth().height(56.dp)) {
            Text("MOSTRAR SEÑAS", fontSize = 20.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun TarjetaVersion(v: Int, elegida: Boolean, modifier: Modifier, alClic: () -> Unit) {
    val color = COLOR_VERSION.getValue(v)
    Column(
        modifier.background(if (elegida) color else FONDO_LISTA, RoundedCornerShape(14.dp))
            .border(if (elegida) 3.dp else 1.dp, if (elegida) Color.White else color, RoundedCornerShape(14.dp))
            .clickable(onClick = alClic).padding(12.dp)
    ) {
        Text(InfoVersiones.titulos.getValue(v), color = Color.White, fontSize = 17.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(6.dp))
        Text(InfoVersiones.detalles.getValue(v), color = Color.White, fontSize = 13.sp, modifier = Modifier.weight(1f))
        Text("Acierto con personas nuevas: %.1f %%".format(InfoVersiones.acierto.getValue(v)), color = Color.White,
            fontSize = 12.sp, fontWeight = FontWeight.Bold)
        if (elegida) Text("✔ ELEGIDA", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun SelectorVersion(version: Int, alElegir: (Int) -> Unit, modifier: Modifier = Modifier) {
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        for (v in 1..4) Text("V$v", color = Color.White, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
            modifier = Modifier.weight(1f).background(if (v == version) COLOR_VERSION.getValue(v) else Color(70, 70, 70), RoundedCornerShape(8.dp))
                .clickable { alElegir(v) }.padding(vertical = 8.dp))
    }
}

// ---------------------------------------------------------------- mostrar señas: el muñeco y el video

@Composable
private fun Mostrar(lista: List<Sena90>, indice: Int, version: Int, alCambiar: (Int) -> Unit, alVersion: (Int) -> Unit,
                    alVolver: () -> Unit, alIntentar: () -> Unit) {
    val sena = lista[indice]
    Row(Modifier.fillMaxSize().background(FONDO)) {
        ListaSenas(lista, indice, alCambiar, Modifier.width(230.dp).fillMaxHeight())
        Column(Modifier.weight(1f).fillMaxHeight().padding(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("◀", color = Color.White, fontSize = 26.sp, modifier = Modifier.clickable { alCambiar((indice - 1 + lista.size) % lista.size) }.padding(8.dp))
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(textoAscii(sena.nombre), color = Color(0, 230, 0), fontSize = 24.sp, fontWeight = FontWeight.Bold)
                    Text("seña ${indice + 1} de ${lista.size}", color = Color(200, 200, 200), fontSize = 12.sp)
                }
                Text("▶", color = Color.White, fontSize = 26.sp, modifier = Modifier.clickable { alCambiar((indice + 1) % lista.size) }.padding(8.dp))
                Spacer(Modifier.width(10.dp))
                SelectorVersion(version, alVersion, Modifier.width(220.dp))
                Spacer(Modifier.weight(1f))
                Button(onClick = alIntentar, colors = ButtonDefaults.buttonColors(containerColor = Color(200, 40, 40))) {
                    Text("INTENTAR SEÑA", fontWeight = FontWeight.Bold)
                }
                Spacer(Modifier.width(8.dp))
                Button(onClick = alVolver, colors = ButtonDefaults.buttonColors(containerColor = Color(70, 70, 70))) { Text("◀ VERSIONES") }
            }
            Text(InfoVersiones.titulos.getValue(version) + ": encima se ve lo que ella mira", color = COLOR_VERSION.getValue(version),
                fontSize = 14.sp, fontWeight = FontWeight.Bold)
            MunecoYVideo(sena, version, conVideo = true, Modifier.weight(1f).fillMaxWidth())
        }
    }
}

@Composable
private fun ListaSenas(lista: List<Sena90>, indice: Int, alCambiar: (Int) -> Unit, modifier: Modifier) {
    val estado = rememberLazyListState()
    LaunchedEffect(indice) { if (indice !in estado.firstVisibleItemIndex until estado.firstVisibleItemIndex + 8) estado.animateScrollToItem(maxOf(0, indice - 3)) }
    LazyColumn(state = estado, modifier = modifier.background(FONDO_LISTA)) {
        itemsIndexed(lista) { i, s ->
            Text("${i + 1}. ${textoAscii(s.nombre)}" + if (s.video) "" else "  ·",
                color = Color.White, fontSize = 15.sp,
                modifier = Modifier.fillMaxWidth().background(if (i == indice) Color(160, 120, 0) else Color.Transparent)
                    .clickable { alCambiar(i) }.padding(horizontal = 12.dp, vertical = 8.dp))
        }
    }
}

// los puntos de la muestra (puntos.json) para dibujar encima del video lo que ve cada versión
private class PuntosSena(o: JSONObject) {
    private fun pts(a: JSONArray) = List(a.length()) { val q = a.getJSONArray(it); Offset(q.getDouble(0).toFloat(), q.getDouble(1).toFloat()) }
    private fun bools(a: JSONArray) = List(a.length()) { a.getBoolean(it) }
    val anchoRef = o.getDouble("ancho_ref").toFloat(); val altoRef = o.getDouble("alto_ref").toFloat()
    val hombros = pts(o.getJSONArray("hombros"))
    val cara = o.getJSONArray("cara").let { a -> List(a.length()) { pts(a.getJSONArray(it)) } }
    val caraOk = bools(o.getJSONArray("cara_ok"))
    class Lado(o: JSONObject, p: (JSONArray) -> List<Offset>, b: (JSONArray) -> List<Boolean>) {
        val mano = o.getJSONArray("mano").let { a -> List(a.length()) { p(a.getJSONArray(it)) } }
        val manoOk = b(o.getJSONArray("mano_ok"))
        val muneca = p(o.getJSONArray("muneca")); val codo = p(o.getJSONArray("codo"))
        val palma = p(o.getJSONArray("palma")); val palmaOk = b(o.getJSONArray("palma_ok"))
        val trazoCodo = p(o.getJSONArray("trazo_codo")); val trazoCodoOk = b(o.getJSONArray("trazo_codo_ok"))
    }
    val lados = o.getJSONArray("lados").let { a -> List(a.length()) { Lado(a.getJSONObject(it), ::pts, ::bools) } }
}

// encima del video: version 1 los 21 puntos de cada mano; 2 + hombros, codos y muñecas; 3 manos + trazo de la palma,
// trazo del codo y los lugares de la cara y el pecho; 4 todo. Los colores son los del muñeco (sin espejo).
private fun DrawScope.dibujarPuntosVideo(p: PuntosSena, f: Int, version: Int, zona: Rect) {
    fun a(q: Offset) = Offset(zona.left + q.x / p.anchoRef * zona.width, zona.top + q.y / p.altoRef * zona.height)
    val g = zona.height / 360f
    for ((l, lado) in p.lados.withIndex()) {
        val color = if (l == 0) COLOR_DER else COLOR_IZQ            // el brazo izquierdo de la persona: igual que en el muñeco
        if (version == 2 || version == 4) {
            val h = a(p.hombros[l]); val c = a(lado.codo[f]); val m = a(lado.muneca[f])
            drawLine(Color.White, h, c, 3 * g); drawLine(Color.White, c, m, 3 * g)
            for (q in listOf(h, c, m)) drawCircle(Color(0, 200, 255), 5 * g, q)
        }
        if (version >= 3) {
            fun trazo(t: List<Offset>, ok: List<Boolean>, col: Color, ancho: Float) {
                var camino: Path? = null
                for (i in 0..f) {
                    if (!ok[i]) { camino?.let { drawPath(it, col, style = Stroke(ancho, cap = StrokeCap.Round)) }; camino = null; continue }
                    val q = a(t[i])
                    camino = (camino ?: Path().apply { moveTo(q.x, q.y) }).apply { lineTo(q.x, q.y) }
                }
                camino?.let { drawPath(it, col, style = Stroke(ancho, cap = StrokeCap.Round)) }
            }
            trazo(lado.trazoCodo, lado.trazoCodoOk, DORADO, 3 * g)
            trazo(lado.palma, lado.palmaOk, color, 4 * g)
        }
        if (lado.manoOk[f]) {
            val q = lado.mano[f].map(::a)
            for ((i, j) in CONEXIONES) drawLine(color, q[i], q[j], 1.5f * g)
            for (o in q) drawCircle(Color.White, 2.2f * g, o)
        }
    }
    if (version >= 3) {
        if (p.caraOk[f]) for (q in p.cara[f]) drawCircle(DORADO, 3 * g, a(q))
        val pecho = Offset((p.hombros[0].x + p.hombros[1].x) / 2, (p.hombros[0].y + p.hombros[1].y) / 2 +
            0.45f * kotlin.math.abs(p.hombros[0].x - p.hombros[1].x))
        drawCircle(DORADO, 4 * g, a(pecho))
    }
}

// el muñeco hace la seña sin parar; a su lado el video real (solo en las señas de LSC50), con lo que ve la versión
@Composable
private fun MunecoYVideo(sena: Sena90, version: Int, conVideo: Boolean, modifier: Modifier) {
    val contexto = LocalContext.current
    val avatares = remember { AvatarPc(contexto) }
    var avatar by remember { mutableStateOf<AvatarSena?>(null) }
    var puntos by remember { mutableStateOf<PuntosSena?>(null) }
    var cuadros by remember { mutableStateOf<List<ImageBitmap>>(emptyList()) }
    var frame by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(sena.nombre) {
        cuadros = emptyList(); frame = 0f; puntos = null
        val a = withContext(Dispatchers.IO) { avatares[sena.nombre] }
        avatar = a
        val clip = a?.clip
        if (conVideo && clip != null) {
            cuadros = withContext(Dispatchers.IO) { leerCuadrosClip(contexto, clip, a.T) }
            puntos = withContext(Dispatchers.IO) {
                runCatching { PuntosSena(JSONObject(contexto.assets.open(clip.replace("clip.mp4", "puntos.json")).bufferedReader().readText())) }.getOrNull()
            }
        }
        while (a != null) {
            delay(33)
            frame += (a.fps / 30.0).toFloat()
            if (frame >= a.T + a.fps * 0.8) frame = 0f            // se repite sin fin
        }
    }
    Canvas(modifier) {
        val a = avatar ?: return@Canvas
        val f = minOf(frame.toInt(), a.T - 1)
        val hayVideo = conVideo && cuadros.isNotEmpty()
        dibujarMunieco(a, f, areaPorAlto(size.width * (if (hayVideo) 0.25f else 0.5f), 0f, size.height), true, version = version)
        if (hayVideo) {
            val img = cuadros[f.coerceIn(0, cuadros.size - 1)]
            val zona = Rect(size.width * 0.5f, 0f, size.width, size.height)
            val k = minOf(zona.width / img.width, zona.height / img.height)
            val w = (img.width * k).toInt(); val h = (img.height * k).toInt()
            val x0 = (zona.left + (zona.width - w) / 2).toInt(); val y0 = (zona.top + (zona.height - h) / 2).toInt()
            drawImage(img, dstOffset = IntOffset(x0, y0), dstSize = IntSize(w, h))
            puntos?.let { dibujarPuntosVideo(it, f.coerceIn(0, it.lados[0].muneca.size - 1), version,
                Rect(x0.toFloat(), y0.toFloat(), (x0 + w).toFloat(), (y0 + h).toFloat())) }
        }
    }
}

// ---------------------------------------------------------------- intentar seña con la versión elegida

@Composable
private fun Intentar(senas: Senas, versiones: Versiones4, sena: Sena90, version: Int, alElegir: (Int) -> Unit, alVolver: () -> Unit) {
    var resultado by remember(sena.nombre) { mutableStateOf<Resultado?>(null) }
    var grabacion by remember(sena.nombre) { mutableStateOf<Triple<Registro, Int, Int>?>(null) }
    var camara by remember(sena.nombre) { mutableStateOf(true) }
    var intentos by remember(sena.nombre) { mutableIntStateOf(0) }
    // otra versión con el resultado en pantalla: la misma seña analizada otra vez (sin repetirla)
    LaunchedEffect(version) {
        val g = grabacion ?: return@LaunchedEffect
        if (camara) return@LaunchedEffect
        resultado = withContext(Dispatchers.Default) { versiones.reconocer(g.first, g.second, g.third, version, sena.nombre) }
    }
    Row(Modifier.fillMaxSize().background(FONDO)) {
        Column(Modifier.width(300.dp).fillMaxHeight().background(FONDO_LISTA).padding(10.dp)) {
            Button(onClick = alVolver, colors = ButtonDefaults.buttonColors(containerColor = Color(70, 70, 70)), modifier = Modifier.fillMaxWidth()) {
                Text("◀ VOLVER A LAS SEÑAS", fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(8.dp))
            SelectorVersion(version, alElegir, Modifier.fillMaxWidth())
            Text(InfoVersiones.titulos.getValue(version), color = COLOR_VERSION.getValue(version), fontSize = 17.sp,
                fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 8.dp))
            Text(InfoVersiones.detalles.getValue(version), color = Color(200, 200, 200), fontSize = 12.sp)
            Spacer(Modifier.height(10.dp))
            Text("Intenta: ${textoAscii(sena.nombre)}", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            if (camara) Text("1. Abre la boca\n2. Haz la seña\n3. Baja las manos", color = Color(200, 200, 200), fontSize = 14.sp,
                modifier = Modifier.padding(top = 6.dp))
            else Text("Intento $intentos", color = Color(200, 200, 200), fontSize = 13.sp)
        }
        // al terminar la seña la cámara se va y queda el resultado
        Box(Modifier.weight(1f).fillMaxHeight()) {
            if (camara) CamaraIntentar(senas, versiones, version, sena.nombre, Modifier.fillMaxSize()) { r, reg, w, h ->
                resultado = r; grabacion = reg?.let { Triple(it, w, h) }; intentos++; camara = false
            } else resultado?.let { PanelResultado(it, sena, version, Modifier.fillMaxSize()) { camara = true } }
        }
        // el muñeco haciendo la seña que hay que intentar (con lo que ve la versión)
        MunecoYVideo(sena, version, conVideo = false, Modifier.width(190.dp).fillMaxHeight().background(FONDO))
    }
}

@Composable
private fun PanelResultado(r: Resultado, sena: Sena90, version: Int, modifier: Modifier, alOtraVez: () -> Unit) {
    val nombre = textoAscii(sena.nombre)
    Column(modifier.background(Color(20, 20, 24)).padding(16.dp).verticalScroll(rememberScrollState())) {
        if (r.error != null) {
            Text(r.error, color = Color(255, 90, 90), fontSize = 26.sp, fontWeight = FontWeight.Bold)
        } else {
            val acerto = r.objetivoPuesto == 1
            Text(if (acerto) "✔ ¡ACERTASTE!" else "✘ NO LA RECONOCIÓ", color = if (acerto) Color(0, 230, 0) else Color(255, 165, 0),
                fontSize = 30.sp, fontWeight = FontWeight.Bold)
            Text(
                if (acerto) "La ${InfoVersiones.titulos.getValue(version).substringBefore(":").lowercase()} reconoció $nombre con %.0f %%".format(r.objetivoProb * 100)
                else "$nombre quedó en el puesto ${r.objetivoPuesto} con %.1f %%".format(r.objetivoProb * 100),
                color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            if (r.objetivoPartes.size > 1 || version >= 3) for ((parte, p) in r.objetivoPartes)
                Text("   $parte: %.0f %%".format(p * 100), color = Color(200, 200, 200), fontSize = 14.sp)
            Spacer(Modifier.height(12.dp))
            Text("Las 4 señas que más se parecieron:", color = Color(200, 200, 200), fontSize = 15.sp)
            r.top.forEachIndexed { i, (n, p) ->
                val esLaMia = textoAscii(n) == nombre
                Column(Modifier.fillMaxWidth().padding(top = 8.dp)
                    .background(if (esLaMia) Color(30, 80, 30) else Color(40, 40, 46), RoundedCornerShape(8.dp)).padding(8.dp)) {
                    Row {
                        Text("${i + 1}. ${textoAscii(n)}", color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                        Text("%.0f %%".format(p * 100), color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold)
                    }
                    Box(Modifier.fillMaxWidth(p.coerceIn(0.01f, 1f)).height(6.dp).background(COLOR_VERSION.getValue(version)))
                    for ((parte, valores) in r.partes) Text("  $parte: %.0f %%".format(valores[i] * 100), color = Color(190, 190, 190), fontSize = 13.sp)
                }
            }
        }
        Spacer(Modifier.height(14.dp))
        Button(onClick = alOtraVez, colors = ButtonDefaults.buttonColors(containerColor = Color(200, 40, 40)), modifier = Modifier.fillMaxWidth().height(52.dp)) {
            Text("INTENTAR OTRA VEZ", fontSize = 18.sp, fontWeight = FontWeight.Bold)
        }
    }
}
