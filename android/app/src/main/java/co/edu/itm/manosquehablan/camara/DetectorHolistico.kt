package co.edu.itm.manosquehablan.camara

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import co.edu.itm.manosquehablan.vivo.Deteccion
import co.edu.itm.manosquehablan.vivo.Escena
import co.edu.itm.manosquehablan.vivo.SeguimientoEnVivo
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult

// Detector holistic de mediapipe (pose, manos y cara).
// La imagen entra sin espejo para que izquierda y derecha queden bien.
class DetectorHolistico(
    contexto: Context,
    senas: co.edu.itm.manosquehablan.vivo.Senas,
    private val alResultado: (Escena) -> Unit,
    private val alError: (String) -> Unit
) {
    private val landmarker: HolisticLandmarker
    val seguimiento = SeguimientoEnVivo(contexto, senas)
    private var ultimoTiempo = 0L
    @Volatile private var ancho = 0
    @Volatile private var alto = 0

    init {
        val opciones = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath("holistic_landmarker.task")
                    .setDelegate(Delegate.GPU)   // con gpu va mas rapido que con cpu
                    .build()
            )
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinHandLandmarksConfidence(0.5f)
            .setResultListener { r: HolisticLandmarkerResult, _ ->
                alResultado(seguimiento.procesar(convertir(r), ancho, alto, r.timestampMs() / 1000.0))
            }
            .setErrorListener { e -> alError(e.message ?: "error MediaPipe") }
            .build()
        landmarker = HolisticLandmarker.createFromOptions(contexto, opciones)
    }

    // se llama con cada imagen de la camara
    private var cuadros = 0
    private var inicioConteo = 0L

    @Volatile private var cerrado = false

    fun procesar(imagen: ImageProxy) {
        // si ya se cerro el detector se ignora la imagen
        if (cerrado) { imagen.close(); return }
        val t0 = SystemClock.uptimeMillis()
        cuadros++
        if (t0 - inicioConteo >= 1000) {
            android.util.Log.i("ManosFPS", "camara=$cuadros cuadros/s")
            cuadros = 0; inicioConteo = t0
        }
        val bitmap = imagen.toBitmap()
        val rotacion = imagen.imageInfo.rotationDegrees
        imagen.close()

        val derecho = if (rotacion == 0) bitmap else Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height,
            Matrix().apply { postRotate(rotacion.toFloat()) }, true
        )
        ancho = derecho.width
        alto = derecho.height

        var tiempo = SystemClock.uptimeMillis()
        if (tiempo <= ultimoTiempo) tiempo = ultimoTiempo + 1
        ultimoTiempo = tiempo
        synchronized(this) {
            if (cerrado) return
            try {
                landmarker.detectAsync(BitmapImageBuilder(derecho).build(), tiempo)
            } catch (e: Exception) {
                alError(e.message ?: "error MediaPipe")
            }
        }
    }

    private fun convertir(r: HolisticLandmarkerResult): Deteccion {
        fun lista(puntos: List<NormalizedLandmark>?) =
            puntos.orEmpty().map { p -> floatArrayOf(p.x(), p.y(), p.visibility().orElse(1f)) }
        return Deteccion(
            pose = lista(r.poseLandmarks()),
            manoIzq = lista(r.leftHandLandmarks()),
            manoDer = lista(r.rightHandLandmarks()),
            cara = lista(r.faceLandmarks())
        )
    }

    fun cerrar() = synchronized(this) {
        cerrado = true
        landmarker.close()
    }
}
