"""
EXPORTAR AL CELULAR: las 4 versiones en ONNX + su configuración, y casos de prueba para comprobar
que el cálculo del celular (Kotlin) da lo mismo que Python.

    android/app/src/main/assets/v1_manos.onnx, v2_cuerpo.onnx, v3_coord.onnx, v3_dibujo.onnx, v3_codo.onnx
    android/app/src/main/assets/cuatro_versiones.json   (nombres, medias y desviaciones, peso de la fusión)
    android/app/src/test/resources/casos.json           (muestras del banco y lo que Python calcula con ellas)

Uso:  python exportar_celular.py   (después de  python versiones.py entrenar)
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

import dibujo_senas as D
import experimento_versiones as E
import versiones as V

ANDROID = V.PAQUETE / "android" / "app" / "src"
ASSETS = ANDROID / "main" / "assets"
CASOS = ANDROID / "test" / "resources" / "casos.json"
CUADROS = 99


class SoloImagen(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, entrada):
        return self.m(entrada, None)


class SoloExtras(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, entrada):
        return self.m(None, entrada)


def exportar(modelo, ejemplo, nombre):
    ruta = ASSETS / nombre
    torch.onnx.export(modelo.eval(), ejemplo, str(ruta), input_names=["entrada"], output_names=["salida"],
                      opset_version=17, dynamo=False)
    s = ort.InferenceSession(str(ruta))
    with torch.no_grad():
        a = modelo(ejemplo).numpy()
    b = s.run(None, {"entrada": ejemplo.numpy()})[0]
    print(f"  {nombre}: diferencia máxima ONNX vs PyTorch = {np.abs(a - b).max():.2e}")


def lista(a):
    return np.round(np.asarray(a, np.float64), 6).tolist()


def main():
    R = V.Reconocedor()
    g = R.g
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("ONNX:")
    exportar(R.gru["v1"], torch.randn(1, CUADROS, g["v1"]["entradas"]), "v1_manos.onnx")
    exportar(R.gru["v2"], torch.randn(1, CUADROS, g["v2"]["entradas"]), "v2_cuerpo.onnx")
    exportar(SoloExtras(R.coord), torch.randn(1, D.CANTIDAD_EXTRAS), "v3_coord.onnx")
    exportar(SoloImagen(R.dib), torch.randn(1, D.CANALES_DIBUJO, D.TAMANO_IMAGEN, D.TAMANO_IMAGEN), "v3_dibujo.onnx")
    exportar(SoloImagen(R.dib2), torch.randn(1, 2, D.TAMANO_IMAGEN, D.TAMANO_IMAGEN), "v3_codo.onnx")

    v3 = g["v3"]
    config = {
        "nombres": g["nombres"], "peso_fusion": g["peso_fusion"], "candidatas_k": E.CANDIDATAS_K, "cuadros": CUADROS,
        "ancho_ref": E.ANCHO, "alto_ref": E.ALTO, "semilla": g["semilla"],
        "v1": {"media": lista(g["v1"]["media"]), "desv": lista(g["v1"]["desv"])},
        "v2": {"media": lista(g["v2"]["media"]), "desv": lista(g["v2"]["desv"])},
        "v3": {k: lista(v3[k]) for k in ("media_imagen", "desv_imagen", "media_extras", "desv_extras",
                                          "media_imagen2", "desv_imagen2")},
    }
    (ASSETS / "cuatro_versiones.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    print(f"Configuración: {ASSETS / 'cuatro_versiones.json'}")

    # ---- casos de prueba: 8 muestras del banco (de distintas señas y de los dos datasets) ----
    muestras, y, _, _, nombres = E.cargar_banco()
    elegidas = [int(np.where(y == c)[0][0]) for c in (0, 7, 25, 44, 50, 63, 77, 89)]
    casos = []
    for i in elegidas:
        m = muestras[i]
        t, val, e, tb, vb = E.construir(m, V.X.CONFIG_M2)
        casos.append({
            "nombre": nombres[int(y[i])],
            "muestra": {k: lista(np.asarray(m[k], np.float64)) if np.asarray(m[k]).dtype != bool else np.asarray(m[k]).tolist()
                        for k in ("muneca_izq", "muneca_der", "valida_izq", "valida_der", "mano_izq", "mano_der",
                                  "mano_izq_ok", "mano_der_ok", "codo_izq", "codo_der", "codo_izq_ok", "codo_der_ok",
                                  "cara", "cara_ok", "ancho_hombros", "centro_hombros")},
            "seq_v1": lista(E.secuencia(m, g["v1"]["config"])),
            "seq_v2": lista(E.secuencia(m, g["v2"]["config"])),
            "trazos": lista(t), "validos": val.tolist(), "extras": lista(e),
            "trazos_codo": lista(tb), "validos_codo": vb.tolist(),
            "imagen": lista(D.dibujar_trazos(t, val)), "imagen_codo": lista(D.dibujar_trazos(tb, vb)),
            "final": {str(v): lista(R.probabilidades(m, v)[0]) for v in (1, 2, 3, 4)},
        })
    CASOS.parent.mkdir(parents=True, exist_ok=True)
    CASOS.write_text(json.dumps(casos), encoding="utf-8")
    print(f"Casos de prueba: {len(casos)} -> {CASOS}")


if __name__ == "__main__":
    main()
