package co.edu.itm.manosquehablan

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import co.edu.itm.manosquehablan.aprender.PantallaAprender
import co.edu.itm.manosquehablan.camara.PantallaCamara
import co.edu.itm.manosquehablan.ui.theme.ManosQueHablanTheme
import co.edu.itm.manosquehablan.vivo.Modo
import co.edu.itm.manosquehablan.vivo.Senas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.Image
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import co.edu.itm.manosquehablan.vivo.textoAscii

private const val SEGUNDOS_LOGO_MS = 2000L

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // pantalla completa y que no se apague
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // usar también la franja negra del borde (donde está la cámara frontal)
        if (android.os.Build.VERSION.SDK_INT >= 28) window.attributes = window.attributes.apply {
            layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }
        WindowCompat.getInsetsController(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
        val senas = Senas(this)
        setContent {
            ManosQueHablanTheme {
                // al abrir: el logo en toda la pantalla mientras carga la app
                var cargando by remember { mutableStateOf(true) }
                if (cargando) {
                    LaunchedEffect(Unit) {
                        val inicio = System.currentTimeMillis()
                        withContext(Dispatchers.IO) { senas.lista() }
                        val falta = SEGUNDOS_LOGO_MS - (System.currentTimeMillis() - inicio)
                        if (falta > 0) kotlinx.coroutines.delay(falta)
                        cargando = false
                    }
                    Image(
                        painterResource(R.drawable.logo_manos), contentDescription = "Manos que Hablan",
                        modifier = Modifier.fillMaxSize().background(Color.White),
                        contentScale = ContentScale.Crop   // llena toda la pantalla (sin franjas blancas a los lados)
                    )
                    return@ManosQueHablanTheme
                }

                // LSC 4 versiones: elegir versión, mostrar las 90 señas e intentarlas
                co.edu.itm.manosquehablan.cuatro.App4(senas)
            }
        }
    }
}
