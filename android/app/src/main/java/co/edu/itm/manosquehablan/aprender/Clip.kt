package co.edu.itm.manosquehablan.aprender

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import android.content.Context

// Cuadros del clip de una seña (el pedazo del video con la cara y el tracking ya dibujados).
// El clip tiene los mismos cuadros que la muestra: el cuadro i va con el cuadro i del muñeco.
fun leerCuadrosClip(contexto: Context, clip: String, cantidad: Int): List<ImageBitmap> {
    val r = MediaMetadataRetriever()
    return try {
        // el mp4 va sin comprimir dentro del APK (noCompress), se lee directo
        contexto.assets.openFd(clip).use { fd -> r.setDataSource(fd.fileDescriptor, fd.startOffset, fd.length) }
        val total = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: cantidad
        val n = minOf(total, cantidad).coerceAtLeast(1)
        val cuadros: List<Bitmap?> = if (Build.VERSION.SDK_INT >= 28) {
            val op = MediaMetadataRetriever.BitmapParams().apply { preferredConfig = Bitmap.Config.RGB_565 }
            r.getFramesAtIndex(0, n, op)
        } else {
            val duracion = (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 3400L) * 1000
            List(n) { i -> r.getFrameAtTime(duracion * i / n, MediaMetadataRetriever.OPTION_CLOSEST) }
        }
        cuadros.filterNotNull().map { it.asImageBitmap() }
    } catch (_: Exception) {
        emptyList()
    } finally {
        r.release()
    }
}
