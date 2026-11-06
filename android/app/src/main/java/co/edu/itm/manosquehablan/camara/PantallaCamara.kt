package co.edu.itm.manosquehablan.camara

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.camera2.CaptureRequest
import android.util.Range
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.ExtendableBuilder
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.geometry.Rect
import co.edu.itm.manosquehablan.aprender.AvatarPc
import co.edu.itm.manosquehablan.aprender.dibujarMunieco
import co.edu.itm.manosquehablan.aprender.avatarDeArchivo
import co.edu.itm.manosquehablan.aprender.areaPorAlto
import co.edu.itm.manosquehablan.aprender.avatarDeMuestra
import co.edu.itm.manosquehablan.aprender.avatarEnVivo
import androidx.compose.material3.Slider
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.size
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.TextField
import androidx.compose.runtime.mutableIntStateOf
import co.edu.itm.manosquehablan.vivo.Escena
import co.edu.itm.manosquehablan.vivo.Estado
import co.edu.itm.manosquehablan.vivo.Modo
import co.edu.itm.manosquehablan.vivo.META_POR_SENA
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import co.edu.itm.manosquehablan.vivo.Senas
import co.edu.itm.manosquehablan.vivo.textoAscii
import co.edu.itm.manosquehablan.vivo.dibujarEscena
import co.edu.itm.manosquehablan.vivo.texto
import java.util.concurrent.Executors

@Composable
fun PantallaCamara(
    senas: Senas, modo: Modo, senaInicial: Senas.Sena?, avisoInicial: String?,
    alCambiarModo: (Modo) -> Unit, alAprender: () -> Unit, alAnalizar: (List<String>) -> Unit = {}
) {
    val contexto = LocalContext.current
    var permiso by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(contexto, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val pedir = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        permiso = it
    }
    LaunchedEffect(Unit) { if (!permiso) pedir.launch(Manifest.permission.CAMERA) }

    if (!permiso) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Button(onClick = { pedir.launch(Manifest.permission.CAMERA) }) {
                Text("Permitir cámara")
            }
        }
        return
    }
    VistaCamara(senas, modo, senaInicial, avisoInicial, alCambiarModo, alAprender, alAnalizar)
}

// resolucion 1280x720 (16:9), la misma con la que se grabaron las muestras
private val RESOLUCION = ResolutionSelector.Builder()
    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
    .setResolutionStrategy(
        ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
    )
    .build()

// fija la camara a 30 fps (con poca luz baja sola a 10)
@OptIn(ExperimentalCamera2Interop::class)
private fun a30Fps(constructor: ExtendableBuilder<*>) {
    Camera2Interop.Extender(constructor).setCaptureRequestOption(
        CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(30, 30)
    )
}

@Composable
private fun VistaCamara(
    senas: Senas, modo: Modo, senaInicial: Senas.Sena?, avisoInicial: String?,
    alCambiarModo: (Modo) -> Unit, alAprender: () -> Unit, alAnalizar: (List<String>) -> Unit = {}
) {
    val contexto = LocalContext.current
    val ciclo = LocalLifecycleOwner.current
    var escena by remember { mutableStateOf<Escena?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var fps by remember { mutableFloatStateOf(0f) }

    // estado de grabar: seña elegida y dialogos
    var lista by remember { mutableStateOf(senas.lista()) }
    var posicion by remember {
        mutableIntStateOf(senaInicial?.let { s -> lista.indexOfFirst { it.indice == s.indice }.coerceAtLeast(0) } ?: 0)
    }
    var nuevaAbierta by remember { mutableStateOf(false) }
    var escribiendo by remember { mutableStateOf("") }
    var errorNombre by remember { mutableStateOf<String?>(null) }
    var confirmarBorrado by remember { mutableStateOf(false) }
    var confirmarEliminar by remember { mutableStateOf(false) }
    val avatarPc = remember { AvatarPc(contexto) }
    val ejecutor = remember { Executors.newSingleThreadExecutor() }
    val detector = remember {
        var anterior = 0L
        DetectorHolistico(
            contexto, senas,
            alResultado = {
                val ahora = System.nanoTime()
                if (anterior != 0L) fps = 0.9f * fps + 0.1f * (1e9f / (ahora - anterior))
                if (ahora / 1_000_000_000L != anterior / 1_000_000_000L) android.util.Log.i("ManosFPS", "fps=%.1f".format(fps))
                anterior = ahora
                escena = it
            },
            alError = { error = it }
        )
    }
    LaunchedEffect(modo) {
        lista = senas.lista()
        detector.seguimiento.cambiarModo(modo, if (modo == Modo.GRABAR) lista[posicion.coerceIn(0, lista.size - 1)] else null)
        if (modo == Modo.GRABAR && avisoInicial != null) detector.seguimiento.elegirSena(lista[posicion], avisoInicial)
    }
    DisposableEffect(Unit) {
        onDispose {
            ejecutor.shutdown()
            detector.cerrar()
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val vista = PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
                val futuro = ProcessCameraProvider.getInstance(ctx)
                futuro.addListener({
                    val proveedor = futuro.get()
                    val vistaPrevia = Preview.Builder().setResolutionSelector(RESOLUCION)
                        .also { a30Fps(it) }.build().also {
                        it.surfaceProvider = vista.surfaceProvider
                    }
                    val analisis = ImageAnalysis.Builder()
                        .setResolutionSelector(RESOLUCION)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .also { a30Fps(it) }
                        .build()
                        .also { it.setAnalyzer(ejecutor) { img -> detector.procesar(img) } }
                    proveedor.unbindAll()
                    proveedor.bindToLifecycle(ciclo, CameraSelector.DEFAULT_FRONT_CAMERA, vistaPrevia, analisis)
                }, ContextCompat.getMainExecutor(ctx))
                vista
            }
        )

        Canvas(Modifier.fillMaxSize()) {
            val e = escena ?: return@Canvas
            dibujarEscena(e)
            if (error != null) {
                val hp = size.height / 720f
                texto(error!!, Offset(20 * hp, size.height / 2), 0.8f, Color.Red, hp, 2)
            }
        }

        // TRADUCIR: debajo de las señas candidatas, boton para verlas en APRENDER
        val parecidas = escena?.resultado?.candidatas?.map { it.nombre }.orEmpty()
        if (modo == Modo.TRADUCIR && escena?.estado == Estado.RESULTADO && parecidas.isNotEmpty()) {
            androidx.compose.foundation.layout.BoxWithConstraints(Modifier.fillMaxSize()) {
                // misma altura que los textos del HUD (pantalla de 720 de alto)
                val y = (150.0 + parecidas.size * 32 + co.edu.itm.manosquehablan.vivo.MARGEN_HUD - 12) / 720.0
                Text(
                    "ANALIZAR\nSEÑAS\nPARECIDAS", color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.Bold,
                    lineHeight = 17.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                    modifier = Modifier.padding(start = 20.dp, top = maxHeight * y.toFloat())
                        .background(AZUL, RoundedCornerShape(12.dp))
                        .clickable { alAnalizar(parecidas) }
                        .padding(horizontal = 14.dp, vertical = 8.dp)
                )
            }
        }


        // ---- GRABAR: muestras de la seña elegida (la del PC + las grabadas en el celular) ----
        val actual = lista.getOrNull(posicion)
        val cantidadEscena = escena?.cantidad ?: 0
        val archivos = remember(posicion, lista, cantidadEscena, modo) {
            if (modo == Modo.GRABAR) actual?.let { senas.muestrasCelular(it) } ?: emptyList() else emptyList()
        }
        // primero todas las muestras del PC (vienen en el apk) y despues las grabadas en el celular
        val desfase = remember(posicion, lista) { actual?.let { avatarPc.cantidadPc(it.nombre) } ?: 0 }
        val nMuestras = archivos.size + desfase
        var kMuestra by remember(posicion) { mutableIntStateOf(0) }
        LaunchedEffect(posicion, nMuestras) { kMuestra = (nMuestras - 1).coerceAtLeast(0) }   // al grabar, la nueva
        val k = kMuestra.coerceIn(0, (nMuestras - 1).coerceAtLeast(0))
        val archivoK = if (k < desfase) null else archivos.getOrNull(k - desfase)
        val avatarSena = remember(actual, archivoK, k, modo) {
            if (modo != Modo.GRABAR || actual == null) null
            else if (archivoK == null) avatarPc.muestraPc(actual.nombre, k) else avatarDeArchivo(senas, archivoK)
        }
        var cuadroAvatar by remember { mutableFloatStateOf(0f) }
        LaunchedEffect(avatarSena) {
            cuadroAvatar = 0f
            val sa = avatarSena ?: return@LaunchedEffect
            while (true) {
                kotlinx.coroutines.delay(33)
                cuadroAvatar += (sa.fps / 30.0).toFloat()
                if (cuadroAvatar >= sa.T + sa.fps * 0.5) cuadroAvatar = 0f      // se repite sin fin
            }
        }

        // TRADUCIR: muñeco que imita en vivo; tras reconocer una seña, la repite hasta BORRAR
        val repetir = escena?.repetir
        val repetirId = escena?.repetirId ?: 0
        val avatarRepetir = remember(repetirId, repetir != null) {
            repetir?.let { avatarDeMuestra(it.first, it.second, it.third) }
        }
        var cuadroRepetir by remember { mutableFloatStateOf(0f) }
        LaunchedEffect(avatarRepetir) {
            cuadroRepetir = 0f
            val sa = avatarRepetir ?: return@LaunchedEffect
            while (true) {
                kotlinx.coroutines.delay(33)
                cuadroRepetir += (sa.fps / 30.0).toFloat()
                if (cuadroRepetir >= sa.T + sa.fps * 0.5) cuadroRepetir = 0f
            }
        }

        // FPS: en GRABAR abajo a la izquierda, en TRADUCIR arriba a la izquierda
        Text(
            "FPS %.0f".format(fps), color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold,
            modifier = Modifier.align(if (modo == Modo.GRABAR) Alignment.BottomStart else Alignment.TopStart)
                .padding(start = 16.dp, top = 8.dp, bottom = 8.dp)
                .background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(10.dp))
                .padding(horizontal = 10.dp, vertical = 2.dp)
        )
        // arriba a la izquierda en GRABAR: la seña y la muestra
        Column(Modifier.align(Alignment.TopStart).padding(start = 16.dp, top = 8.dp)) {
            if (modo == Modo.GRABAR && escena?.estado != Estado.GRABANDO && actual != null) {
                val fondo = Modifier.padding(top = 4.dp).background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 8.dp, vertical = 1.dp)
                Text("SEÑA: ${textoAscii(actual.nombre)}  $cantidadEscena/$META_POR_SENA",
                    color = if (cantidadEscena >= META_POR_SENA) Color.Green else Color.Yellow,
                    fontSize = 22.sp, fontWeight = FontWeight.Bold, modifier = fondo)
                Text(if (nMuestras == 0) "SIN MUESTRAS" else "MUESTRA #${k + 1} DE $nMuestras" + if (k < desfase) " (PC)" else " (CELULAR)",
                    color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold, modifier = fondo)
            }
        }

        // botones a la derecha (se esconden al grabar); en GRABAR el muñeco va debajo
        if (escena?.estado != Estado.GRABANDO) Column(
            Modifier.align(Alignment.TopEnd).fillMaxHeight().padding(top = 8.dp, end = 16.dp, bottom = 8.dp),
            verticalArrangement = Arrangement.spacedBy(5.dp), horizontalAlignment = Alignment.End
        ) {
            val s = detector.seguimiento
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (modo == Modo.TRADUCIR) Boton("GRABAR", AZUL) { alCambiarModo(Modo.GRABAR) }
                else Boton("TRADUCIR", AZUL) { alCambiarModo(Modo.TRADUCIR) }
                Boton("APRENDER", AZUL, alClic = alAprender)
            }
            if (modo == Modo.TRADUCIR) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Boton("BORRAR ÚLTIMA") { s.borrarUltima() }
                    Boton("BORRAR TODO") { s.borrarTodo() }
                }
                // el muñeco: debajo de estos botones hasta abajo de la pantalla
                val e = escena
                Canvas(Modifier.width(320.dp).weight(1f).alpha(0.85f)) {
                    // un cuerpo más a la derecha, pegado al borde derecho de la pantalla
                    val area = areaPorAlto(size.width / 2 + size.height * 0.08f, 0f, size.height)
                    val ar = avatarRepetir
                    if (ar != null) {
                        dibujarMunieco(ar, minOf(cuadroRepetir.toInt(), ar.T - 1), area, false, mostrarAngulo = true)
                    } else if (e != null) {
                        avatarEnVivo(e)?.let { (av, dx, g) -> dibujarMunieco(av, 0, area, false, cara = g, dx = dx, mostrarAngulo = true) }
                    }
                }
            } else {
                // flechas para cambiar de seña
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Boton("◀") { posicion = (posicion - 1 + lista.size) % lista.size; s.elegirSena(lista[posicion]) }
                    Column(
                        Modifier.background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                            .padding(horizontal = 14.dp, vertical = 2.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(actual?.let { textoAscii(it.nombre) } ?: "", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                        Text("$cantidadEscena/$META_POR_SENA   (${posicion + 1} de ${lista.size})",
                            color = if (cantidadEscena >= META_POR_SENA) Color.Green else Color.Yellow, fontSize = 12.sp)
                    }
                    Boton("▶") { posicion = (posicion + 1) % lista.size; s.elegirSena(lista[posicion]) }
                }
                // el muñeco grande a la izquierda (desde + NUEVA SEÑA hasta abajo) y los botones a su derecha
                Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (avatarSena != null) Canvas(Modifier.width(320.dp).fillMaxHeight().alpha(0.75f)) {
                        dibujarMunieco(avatarSena, minOf(cuadroAvatar.toInt(), avatarSena.T - 1),
                            // un cuerpo (≈ ancho del torso) más a la derecha
                            areaPorAlto(size.width / 2 + size.height * 0.36f, 0f, size.height), false)
                    }
                    Column(verticalArrangement = Arrangement.spacedBy(5.dp), horizontalAlignment = Alignment.End) {
                        Boton("+ NUEVA SEÑA") { escribiendo = ""; errorNombre = null; nuevaAbierta = true }
                        Boton("BORRAR SEÑA", Color(120, 0, 0)) { confirmarEliminar = true }
                        Boton("BORRAR\nMUESTRAS", ROJO, alto = 48) { confirmarBorrado = true }
                        Boton("BORRAR\nMUESTRA #${k + 1}", if (archivoK == null) Color(90, 90, 90) else ROJO, alto = 48) {
                            if (archivoK != null) s.borrarMuestra(archivoK, k + 1)
                        }
                        // flechas para pasar de muestra en muestra (el muñeco hace la elegida)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Boton("◀") { if (nMuestras > 0) kMuestra = (k - 1 + nMuestras) % nMuestras }
                            Boton("▶") { if (nMuestras > 0) kMuestra = (k + 1) % nMuestras }
                        }
                    }
                }
            }
        }
    }

    if (nuevaAbierta) {
        AlertDialog(
            onDismissRequest = { nuevaAbierta = false },
            title = { Text("NUEVA SEÑA") },
            text = {
                Column {
                    TextField(escribiendo, { escribiendo = it.take(30); errorNombre = null }, singleLine = true)
                    errorNombre?.let { Text(it, color = Color.Red) }
                }
            },
            confirmButton = {
                Button(onClick = {
                    val error = senas.agregar(escribiendo)
                    if (error != null) { errorNombre = error; return@Button }
                    lista = senas.lista()
                    posicion = lista.size - 1
                    detector.seguimiento.elegirSena(lista[posicion], "NUEVA: ${textoAscii(lista[posicion].nombre)}")
                    nuevaAbierta = false
                }) { Text("GUARDAR") }
            },
            dismissButton = { OutlinedButton(onClick = { nuevaAbierta = false }) { Text("CANCELAR") } }
        )
    }

    if (confirmarEliminar) {
        val sena = lista[posicion.coerceIn(0, lista.size - 1)]
        AlertDialog(
            onDismissRequest = { confirmarEliminar = false },
            title = { Text("¿BORRAR LA SEÑA ${textoAscii(sena.nombre)} DEL TODO?") },
            text = { Text("Se borra el nombre de la lista y todas sus muestras del celular (se guardan en datos_propios_borradas por si te arrepientes). Para volver a usarla habría que crearla y grabarla de nuevo.") },
            confirmButton = {
                Button(onClick = {
                    senas.eliminar(sena)
                    lista = senas.lista()
                    posicion = posicion.coerceIn(0, lista.size - 1)
                    detector.seguimiento.elegirSena(lista[posicion], "${textoAscii(sena.nombre)}: SEÑA BORRADA")
                    confirmarEliminar = false
                }, colors = ButtonDefaults.buttonColors(containerColor = Color(120, 0, 0))) { Text("BORRAR SEÑA") }
            },
            dismissButton = { OutlinedButton(onClick = { confirmarEliminar = false }) { Text("CANCELAR") } }
        )
    }

    if (confirmarBorrado) {
        AlertDialog(
            onDismissRequest = { confirmarBorrado = false },
            title = { Text("¿BORRAR TODAS LAS MUESTRAS DE ${textoAscii(lista[posicion].nombre)}?") },
            text = { Text("Solo las grabadas en el celular. Se mueven a datos_propios_borradas por si te arrepientes.") },
            confirmButton = {
                Button(onClick = { detector.seguimiento.borrarTodo(); confirmarBorrado = false },
                    colors = ButtonDefaults.buttonColors(containerColor = ROJO)) { Text("BORRAR") }
            },
            dismissButton = { OutlinedButton(onClick = { confirmarBorrado = false }) { Text("CANCELAR") } }
        )
    }
}

// PRACTICAR (en APRENDER): la camara con el mismo reconocedor del traductor, en un recuadro.
// Cada vez que reconoce una seña avisa con alResultado (lo que gano y las parecidas).
@Composable
fun CamaraPractica(senas: Senas, modifier: Modifier, alResultado: (co.edu.itm.manosquehablan.vivo.Resultado) -> Unit) {
    val contexto = LocalContext.current
    val ciclo = LocalLifecycleOwner.current
    if (ContextCompat.checkSelfPermission(contexto, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
        Box(modifier.background(Color.Black), contentAlignment = Alignment.Center) { Text("SIN PERMISO DE CÁMARA", color = Color.White) }
        return
    }
    var escena by remember { mutableStateOf<Escena?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val ejecutor = remember { Executors.newSingleThreadExecutor() }
    val detector = remember { DetectorHolistico(contexto, senas, alResultado = { escena = it }, alError = { error = it }) }
    LaunchedEffect(Unit) { detector.seguimiento.cambiarModo(Modo.PRACTICAR) }
    DisposableEffect(Unit) {
        onDispose { ejecutor.shutdown(); detector.cerrar() }
    }
    // un resultado nuevo cada vez que pasa de ANALISIS a RESULTADO
    val resultado = escena?.resultado
    LaunchedEffect(resultado) { resultado?.let(alResultado) }

    Box(modifier.background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val vista = PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
                val futuro = ProcessCameraProvider.getInstance(ctx)
                futuro.addListener({
                    val proveedor = futuro.get()
                    val vistaPrevia = Preview.Builder().setResolutionSelector(RESOLUCION)
                        .also { a30Fps(it) }.build().also { it.surfaceProvider = vista.surfaceProvider }
                    val analisis = ImageAnalysis.Builder()
                        .setResolutionSelector(RESOLUCION)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .also { a30Fps(it) }
                        .build()
                        .also { it.setAnalyzer(ejecutor) { img -> detector.procesar(img) } }
                    proveedor.unbindAll()
                    proveedor.bindToLifecycle(ciclo, CameraSelector.DEFAULT_FRONT_CAMERA, vistaPrevia, analisis)
                }, ContextCompat.getMainExecutor(ctx))
                vista
            }
        )
        Canvas(Modifier.fillMaxSize()) {
            escena?.let { dibujarEscena(it) }
            error?.let { texto(it, Offset(20f, size.height / 2), 0.6f, Color.Red, size.height / 720f, 2) }
        }
    }
}

// INTENTAR SEÑA (4 versiones): la cámara con el grabador del traductor; al terminar la seña la analiza la
// versión elegida. Si se cambia de versión, la última seña se vuelve a analizar sin repetirla.
@Composable
fun CamaraIntentar(
    senas: Senas, versiones: co.edu.itm.manosquehablan.vivo.Versiones4, version: Int, modifier: Modifier,
    alResultado: (co.edu.itm.manosquehablan.vivo.Resultado) -> Unit
) {
    val contexto = LocalContext.current
    val ciclo = LocalLifecycleOwner.current
    if (ContextCompat.checkSelfPermission(contexto, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
        Box(modifier.background(Color.Black), contentAlignment = Alignment.Center) { Text("SIN PERMISO DE CÁMARA", color = Color.White) }
        return
    }
    var escena by remember { mutableStateOf<Escena?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val versionActual = androidx.compose.runtime.rememberUpdatedState(version)
    val ejecutor = remember { Executors.newSingleThreadExecutor() }
    val detector = remember {
        DetectorHolistico(contexto, senas, alResultado = { escena = it }, alError = { error = it }).also { d ->
            d.seguimiento.analizadorExterno = { r, w, h -> versiones.reconocer(r, w, h, versionActual.value) }
        }
    }
    LaunchedEffect(Unit) { detector.seguimiento.cambiarModo(Modo.PRACTICAR) }
    DisposableEffect(Unit) { onDispose { ejecutor.shutdown(); detector.cerrar() } }
    val resultado = escena?.resultado
    LaunchedEffect(resultado) { resultado?.let(alResultado) }
    // otra versión: la misma seña analizada otra vez (sin volver a hacerla)
    LaunchedEffect(version) {
        val s = detector.seguimiento
        val r = s.ultimoRegistro ?: return@LaunchedEffect
        alResultado(withContext(Dispatchers.Default) { versiones.reconocer(r, s.ultimoAncho, s.ultimoAlto, version) })
    }

    Box(modifier.background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val vista = PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
                val futuro = ProcessCameraProvider.getInstance(ctx)
                futuro.addListener({
                    val proveedor = futuro.get()
                    val vistaPrevia = Preview.Builder().setResolutionSelector(RESOLUCION)
                        .also { a30Fps(it) }.build().also { it.surfaceProvider = vista.surfaceProvider }
                    val analisis = ImageAnalysis.Builder()
                        .setResolutionSelector(RESOLUCION)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .also { a30Fps(it) }
                        .build()
                        .also { it.setAnalyzer(ejecutor) { img -> detector.procesar(img) } }
                    proveedor.unbindAll()
                    proveedor.bindToLifecycle(ciclo, CameraSelector.DEFAULT_FRONT_CAMERA, vistaPrevia, analisis)
                }, ContextCompat.getMainExecutor(ctx))
                vista
            }
        )
        Canvas(Modifier.fillMaxSize()) {
            escena?.let { dibujarEscena(it) }
            error?.let { texto(it, Offset(20f, size.height / 2), 0.6f, Color.Red, size.height / 720f, 2) }
        }
    }
}

// tiempo aproximado de entrenamiento por muestra en el celular

private val AZUL = Color(0, 150, 200)
private val ROJO = Color(180, 40, 40)

@Composable
private fun Boton(texto: String, color: Color = Color(70, 70, 70), alto: Int = 34, alClic: () -> Unit) {
    Button(
        onClick = alClic, colors = ButtonDefaults.buttonColors(containerColor = color.copy(alpha = 0.85f)),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 14.dp, vertical = 0.dp),
        modifier = Modifier.height(alto.dp)
    ) { Text(texto, fontSize = 13.sp, lineHeight = 15.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center) }
}
