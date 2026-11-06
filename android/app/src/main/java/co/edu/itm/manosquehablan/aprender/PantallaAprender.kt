package co.edu.itm.manosquehablan.aprender

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.speech.RecognizerIntent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import co.edu.itm.manosquehablan.vivo.Senas
import co.edu.itm.manosquehablan.vivo.textoAscii
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.launch
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown

// Pantalla APRENDER: lista de señas, muñeco y cuadro para escribir o hablar
@Composable
fun PantallaAprender(
    senas: Senas, alEntrenar: (Senas.Sena) -> Unit, alVolver: () -> Unit,
    parecidas: List<String>? = null
) {
    // si viene del traductor, la lista muestra solo las señas parecidas hasta quitar el filtro
    var filtroParecidas by remember(parecidas) { mutableStateOf(parecidas) }
    val contexto = LocalContext.current
    var lista by remember { mutableStateOf(senas.lista()) }
    val dic = remember(lista) { diccionario(lista) }
    // muñecos y videos que manda el PC (cel_enviar.py); si no hay, las grabaciones del celular
    val avatarPc = remember { AvatarPc(contexto) }
    var reposo by remember { mutableStateOf<AvatarSena?>(null) }
    var sinDatosPc by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        reposo = withContext(Dispatchers.IO) { avatarPc.cualquiera() ?: lista.firstNotNullOfOrNull { cargarAvatarCelular(senas, it) } }
        sinDatosPc = withContext(Dispatchers.IO) { avatarPc.cantidad == 0 }
    }
    val cache = remember { mutableStateMapOf<Int, AvatarSena?>() }

    fun senaDe(s: Senas.Sena): AvatarSena? = cache.getOrPut(s.indice) {
        avatarPc[s.nombre] ?: cargarAvatarCelular(senas, s)
    }

    // video de la seña al lado del muñeco
    var cuadrosClip by remember { mutableStateOf<Pair<String, List<androidx.compose.ui.graphics.ImageBitmap>>?>(null) }

    var elegida by remember { mutableStateOf<Int?>(null) }
    var sena by remember { mutableStateOf<AvatarSena?>(null) }
    var frame by remember { mutableDoubleStateOf(0.0) }
    val velocidad = 1.0
    val pausa = false
    val puntos = true
    var mensaje by remember { mutableStateOf("ELIGE UNA SENA O ESCRIBE UNA FRASE") }
    var acumulado by remember { mutableStateOf<String?>(null) }   // palabra que se va deletreando
    var escrito by remember { mutableStateOf("") }
    var aviso by remember { mutableStateOf<String?>(null) }
    val reproductor = remember { Reproductor() }
    // lo que se dibuja
    var figuras by remember { mutableStateOf<List<Figura>>(emptyList()) }
    var corazon by remember { mutableStateOf<CorazonEscena?>(null) }
    var estaQuieto by remember { mutableStateOf(true) }
    var asomo by remember { mutableStateOf<Triple<Apariencia, Float, Boolean>?>(null) }   // quien, cuanto se ve y si guiña
    var tic by remember { mutableIntStateOf(0) }
    var repeticiones by remember { mutableIntStateOf(0) }
    var relevoPedido by remember { mutableStateOf<TipoEscena?>(null) }
    var senaPendiente by remember { mutableStateOf<AvatarSena?>(null) }
    val azarToque = remember { kotlin.random.Random(System.nanoTime()) }

    // PRACTICAR: la camara en vez del video; dice que seña reconocio y la lista muestra las parecidas
    var practicando by remember { mutableStateOf(false) }
    var reconocida by remember { mutableStateOf<Pair<String, Float>?>(null) }
    var errorPractica by remember { mutableStateOf<String?>(null) }

    var interpretarPendiente by remember { mutableStateOf(false) }
    // dictado por voz
    val dictado = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        val dicho = r.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()
        if (r.resultCode == Activity.RESULT_OK && !dicho.isNullOrBlank()) {
            escrito = dicho.trim() + " "
            interpretarPendiente = true
        }
    }
    fun escuchar() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE, "es-CO")
            .putExtra(RecognizerIntent.EXTRA_PROMPT, "Habla y el muñeco lo dirá en señas")
        try { dictado.launch(intent) } catch (_: ActivityNotFoundException) {
            mensaje = "ESTE CELULAR NO TIENE DICTADO POR VOZ"
        }
    }

    // convierte el texto en señas, las palabras en mayusculas se deletrean
    fun interpretar(): Boolean {
        val palabras = escrito.split(Regex("\\s+")).filter { it.isNotEmpty() }
        val entradas = mutableListOf<Entrada>()
        var i = 0
        while (i < palabras.size) {
            if (esMayusculaSostenida(palabras[i])) {
                var formada = ""
                for ((letra, clave) in letrasDe(palabras[i])) {
                    formada += letra
                    val s = dic[listOf(clave)]?.let { senaDe(it) } ?: continue
                    entradas += Entrada(letra, formada, s)
                }
                i++
                continue
            }
            var j = i
            while (j < palabras.size && !esMayusculaSostenida(palabras[j])) j++
            val tramo = palabras.subList(i, j)
            for (t in analizarTexto(tramo, dic)) {
                val s = t.sena?.let { senaDe(it) } ?: continue
                entradas += Entrada(tramo.subList(t.desde, t.hasta).joinToString(" ").uppercase(), null, s)
            }
            i = j
        }
        reproductor.cargar(entradas)
        if (reproductor.activo) elegida = null
        return reproductor.activo
    }

    // 1.5 s sin escribir: el muñeco hace la frase escrita (como en el PC)
    LaunchedEffect(escrito) {
        if (escrito.isBlank()) return@LaunchedEffect
        delay(1500)
        interpretar()
    }

    LaunchedEffect(interpretarPendiente) {
        if (interpretarPendiente) { interpretar(); interpretarPendiente = false }
    }

    // animacion a 30 cuadros por segundo
    var fps by remember { mutableDoubleStateOf(0.0) }
    LaunchedEffect(Unit) {
        var anterior = System.nanoTime()
        val azar = kotlin.random.Random(System.currentTimeMillis())
        var apariencia = HOMBRE
        var rascarse: AvatarSena? = null        // gesto o animacion corta en curso
        var cuadroRascarse = 0
        var escena: Escena? = null
        var cuadroEscena = 0
        var pose: AvatarSena? = null            // ultima pose
        var poseCuadro = 0
        var quietoDesde = System.currentTimeMillis()
        var asomoCuadro = -1
        var sinChica = 0
        fun quieto(s: AvatarSena, c: Int) { pose = s; poseCuadro = c }
        fun poseActual() = pose?.let { poseEn(it, poseCuadro) }
        fun quizasRascarse() {
            val p = poseActual() ?: return
            if (azar.nextDouble() < PROBABILIDAD_RASCARSE) { rascarse = gestoRascarse(p, azar); cuadroRascarse = 0 }
        }
        fun uno(s: AvatarSena, c: Int) { figuras = listOf(Figura(apariencia, s, c)); corazon = null; quieto(s, c) }

        while (true) {
            delay(33)
            val ahora = System.nanoTime()
            fps = 0.9 * fps + 0.1 * (1e9 / (ahora - anterior)); anterior = ahora
            val ms = System.currentTimeMillis()

            // pose inicial con los brazos abajo
            if (pose == null) reposo?.let { quieto(poseReposo(it), 0); quietoDesde = ms }

            // relevo pedido al tocar una seña
            relevoPedido?.let { tipo ->
                relevoPedido = null
                val p = poseActual()
                if (p != null && escena == null) { escena = escenaRelevo(tipo, apariencia, p); cuadroEscena = 0; rascarse = null }
                else senaPendiente?.let { sena = it; frame = 0.0; repeticiones = 0; senaPendiente = null }
            }

            val e = escena
            val r = rascarse
            // quieto = no esta haciendo una seña ni una frase
            estaQuieto = e == null && !reproductor.activo && !(sena != null && elegida != null)
            when {
                // escena con los dos personajes
                e != null -> {
                    figuras = e.actores.map { a -> Figura(a.apariencia, a.sena, cuadroEscena, a.dx[cuadroEscena], a.sonrie[cuadroEscena], a.beso[cuadroEscena]) }
                    corazon = e.corazon(cuadroEscena)
                    if (++cuadroEscena >= e.total) {
                        escena = null; apariencia = e.nueva; quieto(e.poseFinal, 0); quietoDesde = ms
                        senaPendiente?.let { sena = it; frame = 0.0; repeticiones = 0; senaPendiente = null }
                    }
                }
                // gesto corto
                r != null -> {
                    figuras = listOf(Figura(apariencia, r, cuadroRascarse)); corazon = null
                    if (++cuadroRascarse >= r.T) { rascarse = null; quieto(r, r.T - 1); quietoDesde = ms }
                }
                // frase
                reproductor.activo -> {
                    val p = reproductor.paso(if (pausa) 0.0 else velocidad)
                    p.sena?.let { uno(it, p.cuadro) }
                    mensaje = p.texto
                    acumulado = p.acumulado
                    // al terminar baja las manos
                    if (!reproductor.activo) { poseActual()?.let { rascarse = bajarBrazos(it); cuadroRascarse = 0 }; quietoDesde = ms }
                }
                // seña de la lista: se repite sin fin (muñeco y video) hasta elegir otra cosa
                sena != null && elegida != null -> {
                    val s = sena!!
                    uno(s, minOf(frame.toInt(), s.T - 1))
                    if (!pausa) frame += velocidad * s.fps / 30.0
                    if (frame >= s.T + s.fps * 0.8) {
                        repeticiones++
                        frame = 0.0
                        quietoDesde = ms
                    }
                }
                // quieto, cada cierto tiempo hace algo
                else -> {
                    val p = pose
                    if (p != null) { figuras = listOf(Figura(apariencia, p, poseCuadro)); corazon = null }
                    if (p != null && asomoCuadro < 0 && ms - quietoDesde > SEGUNDOS_QUIETO_PARA_DESPEDIRSE * 1000) {
                        quietoDesde = ms
                        // si van muchos eventos sin la chica, esta vez sale
                        val suerte = if (sinChica >= MAXIMO_SIN_CHICA) azar.nextDouble() * (PROBABILIDAD_DESPEDIRSE + PROBABILIDAD_ASOMARSE)
                            else azar.nextDouble()
                        sinChica = if (suerte < PROBABILIDAD_DESPEDIRSE + PROBABILIDAD_ASOMARSE) 0 else sinChica + 1
                        android.util.Log.i("Aprender", "evento a los ${SEGUNDOS_QUIETO_PARA_DESPEDIRSE.toInt()} s: " + when {
                            suerte < PROBABILIDAD_DESPEDIRSE -> "llega el otro (beso)"
                            suerte < PROBABILIDAD_DESPEDIRSE + PROBABILIDAD_ASOMARSE -> "el otro se asoma"
                            else -> "gesto"
                        } + "  (sin chica: $sinChica)")
                        when {
                            // llega el otro personaje
                            suerte < PROBABILIDAD_DESPEDIRSE ->
                                { escena = escenaRelevo(TipoEscena.DESPEDIDA, apariencia, poseEn(p, poseCuadro)); cuadroEscena = 0 }
                            // el otro se asoma
                            suerte < PROBABILIDAD_DESPEDIRSE + PROBABILIDAD_ASOMARSE -> asomoCuadro = 0
                            // gesto al azar
                            else -> {
                                val tipo = GestoQuieto.entries[azar.nextInt(GestoQuieto.entries.size)]
                                rascarse = gestoQuieto(tipo, poseEn(p, poseCuadro), azar); cuadroRascarse = 0
                            }
                        }
                    }
                }
            }
            // asomo
            if (asomoCuadro >= 0) {
                val f = asomoCuadro
                val visible = when {
                    f < 15 -> f / 15f
                    f < 60 -> 1f
                    else -> 1f - (f - 60) / 24f
                }.coerceIn(0f, 1f).let { it * it * (3 - 2 * it) }
                asomo = Triple(if (apariencia.mujer) HOMBRE else MUJER, visible, f in 24 until 50)
                if (++asomoCuadro >= CUADROS_ASOMO || escena != null) { asomoCuadro = -1; asomo = null }
            }
            tic++
        }
    }

    // cada 2 s revisa si hay palabras que no existen
    LaunchedEffect(Unit) {
        while (true) {
            delay((SEGUNDOS_REVISAR * 1000).toLong())
            if (aviso != null) continue
            val palabras = escrito.split(Regex("\\s+")).filter { it.isNotEmpty() }.toMutableList()
            val completas = if (escrito.endsWith(" ")) palabras.toList() else palabras.dropLast(1)
            if (completas.isEmpty()) continue
            for (t in analizarTexto(completas, dic)) {
                if (t.sena != null) continue
                if (esMayusculaSostenida(completas[t.desde])) continue
                if (t.hasta == completas.size && esPrefijo(completas.drop(t.desde).map(::normalizar), dic)) continue
                aviso = completas[t.desde]
                val terminaEnEspacio = escrito.endsWith(" ")
                palabras.removeAt(t.desde)
                escrito = palabras.joinToString(" ") + if (terminaEnEspacio && palabras.isNotEmpty()) " " else ""
                break
            }
        }
    }

    Row(Modifier.fillMaxSize().background(FONDO)) {
        // lista de señas (como en el PC): buscador arriba, lista y la tira A-Z a la derecha
        // la palabra que se esta escribiendo filtra la lista
        val ordenada = remember(lista) { lista.sortedBy { textoAscii(it.nombre) } }
        val cola = if (escrito.endsWith(" ")) "" else escrito.split(Regex("\\s+")).lastOrNull().orEmpty()
        val visibles = remember(ordenada, cola, filtroParecidas) {
            val f = filtroParecidas
            if (f != null) f.mapNotNull { n -> ordenada.firstOrNull { normalizar(it.nombre) == normalizar(n) } }
            else if (cola.isBlank()) ordenada else normalizar(cola).let { c -> ordenada.filter { normalizar(it.nombre).startsWith(c) } }
        }
        val estadoLista = androidx.compose.foundation.lazy.rememberLazyListState()
        val alcance = androidx.compose.runtime.rememberCoroutineScope()
        Row(Modifier.width(250.dp).fillMaxHeight().background(FONDO_LISTA)) {
          Column(Modifier.weight(1f).fillMaxHeight()) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 6.dp, top = 6.dp, end = 4.dp)) {
                TextField(
                    escrito, { escrito = it.take(200) },
                    placeholder = { Text("Buscar seña o frase...", fontSize = 13.sp) },
                    singleLine = true,
                    textStyle = androidx.compose.ui.text.TextStyle(fontSize = 15.sp),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Go),
                    keyboardActions = KeyboardActions(onGo = { interpretar() }),
                    trailingIcon = if (escrito.isNotEmpty()) ({
                        Text("✕", color = Color.Gray, fontSize = 18.sp, modifier = Modifier.clickable { escrito = "" }.padding(6.dp))
                    }) else null,
                    modifier = Modifier.weight(1f)
                )
                Text("🎤", fontSize = 26.sp, modifier = Modifier.clickable { escuchar() }.padding(6.dp))
            }
            // aviso del filtro de señas parecidas, con ✕ para ver todas otra vez
            if (filtroParecidas != null) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth().padding(6.dp)
                        .background(Color(0, 150, 200), androidx.compose.foundation.shape.RoundedCornerShape(8.dp))
                        .clickable { filtroParecidas = null }
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text("SEÑAS PARECIDAS", color = Color.White, fontSize = 13.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold,
                        modifier = Modifier.weight(1f))
                    Text("✕ VER TODAS", color = Color.White, fontSize = 11.sp)
                }
            }
                LazyColumn(state = estadoLista, modifier = Modifier.weight(1f).fillMaxWidth()) {
                    itemsIndexed(visibles, key = { _, s -> s.indice }) { _, s ->
                        Text(
                            textoAscii(s.nombre), color = Color.White, fontSize = 16.sp,
                            modifier = Modifier.fillMaxWidth()
                                .background(if (s.indice == elegida) Color(160, 120, 0) else Color.Transparent)
                                .clickable {
                                    elegida = s.indice
                                    reproductor.cargar(emptyList())
                                    acumulado = null
                                    val nueva = senaDe(s)
                                    mensaje = if (nueva != null) textoAscii(s.nombre) else "${textoAscii(s.nombre)}: SIN GRABACIONES"
                                    if (nueva != null && azarToque.nextDouble() < PROBABILIDAD_RELEVO_AL_TOCAR) {
                                        // a veces llega el otro personaje y hace la seña
                                        senaPendiente = nueva
                                        sena = null
                                        relevoPedido = if (azarToque.nextBoolean()) TipoEscena.EMPUJON else TipoEscena.BESO
                                    } else {
                                        sena = nueva; frame = 0.0; repeticiones = 0
                                    }
                                }
                                .padding(horizontal = 12.dp, vertical = 8.dp)
                        )
                    }
                }
          }
                // tira A-Z (todo el alto de la pantalla, de la A a la Z): tocar o deslizar el dedo lleva la lista a esa letra
                val letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                var letraActiva by remember { mutableStateOf<Char?>(null) }
                fun irA(y: Float, alto: Int) {
                    val k = ((y / alto.coerceAtLeast(1)) * letras.length).toInt().coerceIn(0, letras.length - 1)
                    val letra = letras[k]
                    letraActiva = letra
                    val i = visibles.indexOfFirst { textoAscii(it.nombre).firstOrNull()?.let { c -> c >= letra } == true }
                    if (i >= 0) alcance.launch { estadoLista.scrollToItem(i) }
                }
                Column(
                    Modifier.width(26.dp).fillMaxHeight().background(Color(40, 40, 46))
                        .pointerInput(visibles) {
                            awaitEachGesture {
                                val abajo = awaitFirstDown()
                                irA(abajo.position.y, size.height)
                                do {
                                    val ev = awaitPointerEvent()
                                    ev.changes.firstOrNull()?.let { if (it.pressed) { irA(it.position.y, size.height); it.consume() } }
                                } while (ev.changes.any { it.pressed })
                                letraActiva = null
                            }
                        },
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // cada letra ocupa la misma parte del alto: siempre caben las 26
                    for (l in letras) Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                        Text(l.toString(), fontSize = 10.sp, lineHeight = 10.sp, fontWeight = FontWeight.Bold,
                            color = if (l == letraActiva) Color(255, 215, 0) else Color(200, 200, 200))
                    }
                }
        }

        // muñeco y controles
        Column(Modifier.fillMaxSize().padding(start = 8.dp, end = 0.dp, top = 4.dp, bottom = 0.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(end = 8.dp)) {
                // junto al micrófono (aquí antes iba el nombre de la seña en verde)
                Button(onClick = { interpretar() }, colors = ButtonDefaults.buttonColors(containerColor = Color(60, 150, 0))) { Text("INTERPRETAR FRASE") }
                // palabra deletreada
                acumulado?.let {
                    Text(it, color = Color(190, 90, 255), fontSize = 26.sp, fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(start = 28.dp))
                }
                Spacer(Modifier.weight(1f))
                if (sinDatosPc) Text("sin videos: cel_preparar_aprender.py", color = Color.Gray)
                // donde iba el FPS: flechas para pasar por las muestras de la seña elegida
                val senaElegida = lista.firstOrNull { it.indice == elegida }
                val totalMuestras = remember(senaElegida) { senaElegida?.let { avatarPc.cantidadPc(it.nombre) } ?: 0 }
                var kMuestra by remember(senaElegida) { mutableIntStateOf(0) }
                fun verMuestra(k: Int) {
                    val s = senaElegida ?: return
                    val nueva = avatarPc.muestraPc(s.nombre, k) ?: return
                    kMuestra = k
                    reproductor.cargar(emptyList()); acumulado = null
                    sena = nueva; frame = 0.0; repeticiones = 0
                }
                if (totalMuestras > 1) {
                    Text("◀", color = Color.White, fontSize = 22.sp,
                        modifier = Modifier.clickable { verMuestra((kMuestra - 1 + totalMuestras) % totalMuestras) }.padding(horizontal = 10.dp))
                    Text("${kMuestra + 1}/$totalMuestras", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                    Text("▶", color = Color.White, fontSize = 22.sp,
                        modifier = Modifier.clickable { verMuestra((kMuestra + 1) % totalMuestras) }.padding(horizontal = 10.dp))
                }
                Button(
                    onClick = { practicando = !practicando; reconocida = null; errorPractica = null },
                    colors = ButtonDefaults.buttonColors(containerColor = if (practicando) Color(180, 40, 40) else Color(0, 110, 180)),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp),
                    modifier = Modifier.padding(end = 6.dp)
                ) { Text(if (practicando) "CERRAR" else "PRACTICAR", maxLines = 1) }
                var dedos by remember { mutableStateOf(mostrarDedos) }
                val compacto = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp)
                OutlinedButton(onClick = { dedos = !dedos; mostrarDedos = dedos }, contentPadding = compacto,
                    modifier = Modifier.padding(end = 6.dp)) {
                    Text("TRAZO", color = if (dedos) Color(255, 215, 0) else Color.White, maxLines = 1)
                }
                OutlinedButton(onClick = alVolver, contentPadding = compacto) { Text("LSC", color = Color.White, maxLines = 1) }
            }
            // el clip de la seña que se esta haciendo (se lee aparte)
            val clipActual = figuras.firstNotNullOfOrNull { it.sena.clip }
            LaunchedEffect(clipActual) {
                if (clipActual != null && cuadrosClip?.first != clipActual) {
                    val t = figuras.firstOrNull { it.sena.clip == clipActual }?.sena?.T ?: 102
                    cuadrosClip = clipActual to withContext(Dispatchers.IO) { leerCuadrosClip(contexto, clipActual, t) }
                }
            }
            // muñeco (y el video a la derecha si la seña lo tiene)
            Box(Modifier.weight(1f).fillMaxWidth()) {
                Canvas(Modifier.fillMaxSize()) {
                    @Suppress("UNUSED_EXPRESSION") tic
                    // practicando: la camara ocupa el lugar del video (el muñeco queda a la izquierda)
                    val conVideo = if (practicando) figuras.firstOrNull()
                        else figuras.firstOrNull { it.sena.clip != null && it.sena.clip == cuadrosClip?.first }
                    val cuadrosV = cuadrosClip?.second.orEmpty()
                    // el muñeco ocupa todo el alto: la cabeza debajo de INTERPRETAR FRASE y los pies abajo
                    // practicando: el muñeco más a la izquierda (cerca de la tira A-Z) para que la camara sea grande
                    // casi pegado a la tira A-Z
                    val area = if (practicando) areaPorAlto(size.width * 0.14f, 0f, size.height)
                        else if (conVideo != null && cuadrosV.isNotEmpty()) areaPorAlto(size.width * 0.25f, 0f, size.height)
                        else areaPorAlto(size.width * 0.5f, 0f, size.height)
                    if (!practicando && conVideo != null && cuadrosV.isNotEmpty()) {
                        val img = cuadrosV[conVideo.cuadro.coerceIn(0, cuadrosV.size - 1)]
                        val zona = Rect(size.width * 0.5f, 0f, size.width, size.height)
                        val k = minOf(zona.width / img.width, zona.height / img.height)
                        val w = (img.width * k).toInt(); val h = (img.height * k).toInt()
                        drawImage(img, dstOffset = androidx.compose.ui.unit.IntOffset(
                            (zona.left + (zona.width - w) / 2).toInt(), (zona.top + (zona.height - h) / 2).toInt()),
                            dstSize = androidx.compose.ui.unit.IntSize(w, h))
                    }
                    // los gestos de la cara solo cuando esta quieto
                    val cara = gestos(System.currentTimeMillis()).let { g ->
                        if (estaQuieto) g else Gestos(g.parpadeo, 0f, floatArrayOf(0f, 0f))
                    }
                    for (f in figuras) {
                        // cara propia del gesto si tiene
                        val caraF = cara.con(f.sena.caras?.getOrNull(f.cuadro))
                        dibujarMunieco(f.sena, f.cuadro, area, puntos, caraF, f.apariencia, f.dx, f.sonrie, f.beso)
                    }
                    asomo?.let { (quien, visible, guino) -> dibujarAsomo(area, quien, visible, guino) }
                    corazon?.let { c ->
                        val esc = escalaMunieco(area)
                        dibujarCorazon(Offset(area.center.x + c.dx * esc * ANCHO_MUNIECO, area.top + (1.25f + c.y) * esc), c.tam * esc, c.alfa)
                    }
                }
                if (practicando) {
                    // recuadro angosto: la camara se recorta por los lados y se ve solo la parte del centro (tú)
                    // 30 % más angosto (0.46 -> 0.32): el borde izquierdo se corre a la derecha, pegado al borde derecho
                    Column(Modifier.align(Alignment.TopEnd).fillMaxWidth(0.32f).fillMaxHeight()) {
                        // arribita de la camara: la seña que reconocio (verde si es la elegida)
                        val elegidaNombre = lista.firstOrNull { it.indice == elegida }?.let { textoAscii(it.nombre) }
                        val (txt, color) = when {
                            errorPractica != null -> errorPractica!! to Color(255, 80, 80)
                            reconocida == null -> "HAZ LA SEÑA FRENTE A LA CÁMARA" to Color.White
                            else -> {
                                val (n, p) = reconocida!!
                                val bien = elegidaNombre != null && normalizar(n) == normalizar(elegidaNombre)
                                "RECONOCIÓ: $n %.0f%%".format(p * 100) + (if (elegidaNombre == null) "" else if (bien) "  ✔" else "  ✘") to
                                    (if (elegidaNombre == null) Color.White else if (bien) Color(0, 230, 0) else Color(255, 165, 0))
                            }
                        }
                        // nombres largos (HACE MUCHO TIEMPO 2) se achican para que quepan completos
                        Text(txt, color = color, fontSize = if (txt.length > 28) 15.sp else 20.sp, fontWeight = FontWeight.Bold,
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center, maxLines = 2)
                        co.edu.itm.manosquehablan.camara.CamaraPractica(senas, Modifier.weight(1f).fillMaxWidth()) { r ->
                            if (r.error != null) { errorPractica = r.error; reconocida = null; return@CamaraPractica }
                            errorPractica = null
                            val gano = r.top[0]
                            reconocida = gano
                            // la lista de la izquierda: la elegida, la que reconocio y hasta 7 parecidas
                            filtroParecidas = (listOfNotNull(elegidaNombre, gano.first) + r.parecidas).distinctBy { normalizar(it) }
                        }
                    }
                }
            }
        }
    }

    aviso?.let { palabra ->
        AlertDialog(
            onDismissRequest = { aviso = null },
            title = { Text("\"${textoAscii(palabra)}\" AÚN NO ESTÁ EN EL DICCIONARIO DE SEÑAS") },
            text = { Text("¿DESEA ENTRENARLA?") },
            confirmButton = {
                Button(onClick = {
                    senas.agregar(palabra)
                    lista = senas.lista()
                    val nueva = lista.firstOrNull { textoAscii(it.nombre) == textoAscii(palabra) }
                    aviso = null
                    if (nueva != null) alEntrenar(nueva)
                }) { Text("SÍ") }
            },
            dismissButton = { OutlinedButton(onClick = { aviso = null }) { Text("NO") } }
        )
    }
}
