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
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
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
import co.edu.itm.manosquehablan.vivo.InfoVersiones
import co.edu.itm.manosquehablan.vivo.Resultado
import co.edu.itm.manosquehablan.vivo.Senas
import co.edu.itm.manosquehablan.vivo.Versiones4
import co.edu.itm.manosquehablan.vivo.textoAscii
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray

// LSC 4 VERSIONES: elegir versión, MOSTRAR SEÑAS (las 90, una a una) e INTENTAR SEÑA con la versión elegida.

private val FONDO = Color(28, 28, 32)
private val FONDO_LISTA = Color(36, 36, 42)
private val COLOR_VERSION = mapOf(1 to Color(90, 160, 90), 2 to Color(60, 140, 200), 3 to Color(220, 120, 60), 4 to Color(200, 40, 40))

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
            Mostrar(lista, indice, { indice = it }, alVolver = { pantalla = Pantalla.INICIO }) { pantalla = Pantalla.INTENTAR }
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

// ---------------------------------------------------------------- mostrar señas: el muñeco y el video

@Composable
private fun Mostrar(lista: List<Sena90>, indice: Int, alCambiar: (Int) -> Unit, alVolver: () -> Unit, alIntentar: () -> Unit) {
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
                Spacer(Modifier.weight(1f))
                Button(onClick = alIntentar, colors = ButtonDefaults.buttonColors(containerColor = Color(200, 40, 40))) {
                    Text("INTENTAR SEÑA", fontWeight = FontWeight.Bold)
                }
                Spacer(Modifier.width(8.dp))
                Button(onClick = alVolver, colors = ButtonDefaults.buttonColors(containerColor = Color(70, 70, 70))) { Text("VERSIONES") }
            }
            MunecoYVideo(sena, conVideo = true, Modifier.weight(1f).fillMaxWidth())
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

// el muñeco hace la seña sin parar; a su lado el video real (solo en las señas de LSC50)
@Composable
private fun MunecoYVideo(sena: Sena90, conVideo: Boolean, modifier: Modifier) {
    val contexto = LocalContext.current
    val avatares = remember { AvatarPc(contexto) }
    var avatar by remember { mutableStateOf<AvatarSena?>(null) }
    var cuadros by remember { mutableStateOf<List<ImageBitmap>>(emptyList()) }
    var frame by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(sena.nombre) {
        cuadros = emptyList(); frame = 0f
        val a = withContext(Dispatchers.IO) { avatares[sena.nombre] }
        avatar = a
        val clip = a?.clip
        if (conVideo && clip != null) cuadros = withContext(Dispatchers.IO) { leerCuadrosClip(contexto, clip, a.T) }
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
        dibujarMunieco(a, f, areaPorAlto(size.width * (if (hayVideo) 0.25f else 0.5f), 0f, size.height), false)
        if (hayVideo) {
            val img = cuadros[f.coerceIn(0, cuadros.size - 1)]
            val zona = Rect(size.width * 0.5f, 0f, size.width, size.height)
            val k = minOf(zona.width / img.width, zona.height / img.height)
            val w = (img.width * k).toInt(); val h = (img.height * k).toInt()
            drawImage(img, dstOffset = IntOffset((zona.left + (zona.width - w) / 2).toInt(), (zona.top + (zona.height - h) / 2).toInt()),
                dstSize = IntSize(w, h))
        }
    }
}

// ---------------------------------------------------------------- intentar seña con la versión elegida

@Composable
private fun Intentar(senas: Senas, versiones: Versiones4, sena: Sena90, version: Int, alElegir: (Int) -> Unit, alVolver: () -> Unit) {
    var resultado by remember(sena.nombre) { mutableStateOf<Resultado?>(null) }
    var intentos by remember(sena.nombre) { mutableIntStateOf(0) }
    Row(Modifier.fillMaxSize().background(FONDO)) {
        // menú de la izquierda: la versión y las 4 señas más parecidas con lo que opina cada parte
        Column(Modifier.width(330.dp).fillMaxHeight().background(FONDO_LISTA).padding(10.dp).verticalScroll(rememberScrollState())) {
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                for (v in 1..4) Text("V$v", color = Color.White, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
                    modifier = Modifier.weight(1f).background(if (v == version) COLOR_VERSION.getValue(v) else Color(70, 70, 70), RoundedCornerShape(8.dp))
                        .clickable { alElegir(v) }.padding(vertical = 8.dp))
            }
            Text(InfoVersiones.titulos.getValue(version), color = COLOR_VERSION.getValue(version), fontSize = 17.sp,
                fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 8.dp))
            Text("Intenta: ${textoAscii(sena.nombre)}", color = Color.White, fontSize = 15.sp)
            Spacer(Modifier.height(8.dp))
            val r = resultado
            when {
                r == null -> Text("Abre la boca, haz la seña y baja las manos.", color = Color(200, 200, 200), fontSize = 14.sp)
                r.error != null -> Text(r.error, color = Color(255, 90, 90), fontSize = 16.sp, fontWeight = FontWeight.Bold)
                else -> {
                    val acerto = textoAscii(r.top.first().first) == textoAscii(sena.nombre)
                    Text(if (acerto) "✔ ¡RECONOCIÓ TU SEÑA!" else "✘ No fue la primera", color = if (acerto) Color(0, 230, 0) else Color(255, 165, 0),
                        fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    Text("Intento $intentos · las 4 más parecidas:", color = Color(200, 200, 200), fontSize = 12.sp)
                    r.top.forEachIndexed { i, (nombre, p) ->
                        val esLaMia = textoAscii(nombre) == textoAscii(sena.nombre)
                        Column(Modifier.fillMaxWidth().padding(top = 8.dp)
                            .background(if (esLaMia) Color(30, 80, 30) else Color.Transparent, RoundedCornerShape(8.dp)).padding(6.dp)) {
                            Row {
                                Text("${i + 1}. ${textoAscii(nombre)}", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                                Text("%.0f %%".format(p * 100), color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                            }
                            Box(Modifier.fillMaxWidth(p.coerceIn(0.01f, 1f)).height(5.dp).background(COLOR_VERSION.getValue(version)))
                            for ((parte, valores) in r.partes) if (r.partes.size > 1 || parte != r.partes.keys.first() || true)
                                Text("  $parte: %.0f %%".format(valores[i] * 100), color = Color(190, 190, 190), fontSize = 12.sp)
                        }
                    }
                }
            }
            Spacer(Modifier.height(10.dp))
            Button(onClick = alVolver, colors = ButtonDefaults.buttonColors(containerColor = Color(70, 70, 70)), modifier = Modifier.fillMaxWidth()) {
                Text("VOLVER A LAS SEÑAS")
            }
        }
        // la cámara, y a la derecha el muñeco haciendo la seña que hay que intentar
        CamaraIntentar(senas, versiones, version, Modifier.weight(1f).fillMaxHeight()) { r -> resultado = r; if (r.version == version) intentos++ }
        MunecoYVideo(sena, conVideo = false, Modifier.width(190.dp).fillMaxHeight().background(FONDO))
    }
}
