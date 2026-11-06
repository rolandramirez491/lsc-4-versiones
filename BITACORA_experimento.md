# Bitácora del experimento: evolución del reconocedor de señas LSC

## 1. Datos (conjunto reducido)
- **Fuente:** datasets oficiales LSC50 (50 señas, 5 personas) y LSC-54 (54 señas, 25 personas). Unidos dan 90 señas (14 están en los dos).
- **Banco balanceado:** 90 señas × 5 personas × 4 muestras = **1800 muestras**.
  - LSC50: se usan sus 5 personas × 4 repeticiones.
  - LSC-54: 5 personas al azar entre las que tienen al menos 4 muestras, y 4 muestras al azar de cada una.
  - Las 14 señas comunes se toman de LSC50.
- Todas las muestras tienen 99 cuadros, los mismos puntos de MediaPipe (manos, muñecas, codos, hombros y 11 puntos de la cara) y la persona anotada.
- **Semilla:** 42 en todo (selección de muestras, partición, pesos iniciales y aumento de datos).
- Script: `codigo/crear_dataset_90.py`. Lo que se escogió queda en `dataset 90 senas/seleccion.json`.

## 2. Partición entrenamiento / prueba
- **5 vueltas dejando una persona por fuera:** en la vuelta *k* se prueba con la persona *k* de cada seña.
- En cada vuelta: **1260 de entrenamiento / 180 de validación / 360 de prueba**. La validación se usa solo para la parada temprana.
- Cada una de las 1800 muestras se prueba **una sola vez**, siempre con una persona que el modelo no vio.
- Mismas reglas para todas las configuraciones: hasta 150 épocas, paciencia 25, lote 32, AdamW (tasa 1e-3), label smoothing 0,1 y 24 copias aumentadas por muestra (rotación, escala, ruido).

## 3. Configuraciones (de la más sencilla a la más completa)

| | Qué se agrega | Red |
|---|---|---|
| **B0** | Línea base: solo **Hand Landmarker** (21 puntos de cada mano), como en el estado del arte | GRU |
| **M1** | **+ cuerpo como referencia:** muñecas y codos (Pose), todo medido desde los hombros | GRU |
| **M2** | **+ la seña como dibujo** (trazo de la palma), **lugar** respecto al cuerpo y la cara (Face) y **forma de los dedos** al inicio, la mitad y el final | Cascada CNN + densa |
| **M3** | **Fusión M1 + M2:** las dos redes votan con el mismo peso (0,5, fijado antes de ver los resultados) | GRU + cascada |

## 4. Resultados (1800 pruebas con personas no vistas)

| | Acierto | IC 95 % | Entre las 3 primeras | vs. anterior |
|---|---|---|---|---|
| **B0** | 55,3 % | 53,0–57,6 | 69,9 % | — |
| **M1** | 59,2 % | 56,9–61,4 | 70,7 % | **+3,9** (p = 0,0001) |
| **M2** | 57,5 % | 55,2–59,8 | 73,5 % | −1,7 frente a M1 (p = 0,17, no significativa) |
| **M3** | **62,8 %** | 60,6–65,0 | **75,3 %** | **+3,6 sobre M1** (p = 0,0002) · **+7,5 sobre B0** (p < 0,0001) |

Significancia: prueba de McNemar exacta, sobre las mismas muestras de prueba.

**Lo que se resalta:**
- Agregar el **cuerpo como referencia** mejora a la línea base de forma significativa (+3,9).
- El **dibujo con lugar y forma de la mano** solo rinde igual que M1 en el primer puesto, pero acierta más **entre las 3 primeras** (73,5 % contra 70,7 %).
- **Combinar las dos formas de ver la seña** es lo mejor: **62,8 %**, +7,5 sobre la línea base. Las dos redes se equivocan en señas distintas.

## 5. Límites
- Solo 5 personas por seña. La variación entre personas es grande (±11–13 puntos), aunque las comparaciones de McNemar sí son significativas.
- En LSC-54 las 5 personas de prueba cambian de una seña a otra.
- Los datasets son de estudio: la mano se ve en todos los cuadros y los clips vienen recortados. Las mejoras de robustez para cámara en vivo no se pueden medir aquí.
- Falta confirmar en las publicaciones de LSC50 y LSC-54 si las personas son sordas.

## 6. Archivos
- `codigo/experimento_4.py`: las 4 configuraciones.
- `resultados versiones/resultados_4.csv`, `resultados_4.json` (con cada vuelta), `resultados_4.png` y `predicciones_4.npz` (predicción de cada muestra).
- Anexo: `codigo/experimento_versiones.py` y `resultados.csv`, con las 17 variantes probadas antes (V1–V17). M1 = V4, M2 = V14.
