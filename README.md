# Predicción de pCR en resonancia DCE de mama

Una red convolucional propia, entrenada desde cero en PyTorch, estima la probabilidad de **respuesta patológica completa (pCR)**
a la quimioterapia neoadyuvante a partir de **un solo corte** de resonancia magnética con contraste dinámico (DCE), tomado antes
del tratamiento. Trabajo académico sobre el conjunto de datos **BreastDCEDL**.

**App:** https://pablojimenezgirardeau.github.io/Caso-Cancer-de-mama/

> Uso exclusivamente educativo. Sin validez clínica: no debe usarse para tomar decisiones sobre pacientes.

## El problema

La pCR significa que, tras la quimioterapia previa a la cirugía, no queda tumor invasivo. Cada muestra son tres PNG de 256 × 256
del mismo corte: antes del contraste (PRE), poco después (EARLY) y un rato después (LATE). El conjunto tiene 1.273 pacientes de tres
cohortes (I-SPY1, I-SPY2 y Duke), con un 29 % de pCR; la evaluación es siempre **por paciente** y con la partición oficial
(1.097 pacientes en 5 folds de validación cruzada y 176 en el test).

## El modelo final

- **Arquitectura:** cuatro bloques convolución 3 × 3 + BatchNorm + ReLU + max pooling (16, 32, 64 y 128 filtros), media global,
  capa densa de 64 con dropout y una salida. La red mira solo los **128 × 128 píxeles centrales**, donde suele estar el tumor.
  106.001 parámetros.
- **Entrenamiento:** entropía cruzada binaria, Adam con tasa 3·10⁻⁵, lotes de 128 y parada temprana sobre el AUC de validación
  (media móvil de 3 épocas, paciencia 10, máximo 60).
- **Conjunto:** 15 redes (5 folds × 3 semillas). Para predecir se promedian sus probabilidades y se calibran con Platt.
- **Umbral:** 0,316, el de máxima F1 equilibrada en validación. pCR = 1 si la probabilidad calibrada llega al umbral.

El modelo principal usa **solo la imagen**. Los datos clínicos (receptores, edad y volumen tumoral) se usan únicamente como
comparación.

## Resultados

AUC por paciente (intervalo de confianza del 95 % en el test):

| Modelo | Validación (1.097) | Test (176) |
|---|---|---|
| **CNN solo imagen (principal)** | **0,573** | **0,591** (0,50-0,68) |
| Solo datos clínicos | 0,704 | 0,695 |
| Imagen y datos clínicos | 0,710 | 0,705 |
| Solo cohorte (referencia) | 0,523 | 0,544 |

Con el umbral, en el test: sensibilidad 30 %, especificidad 76 %, precisión 35 % y, de media en las dos clases, precisión 0,53,
recall 0,53 y F1 0,53. El test se evaluó **una sola vez**, con el modelo ya cerrado (`figuras/test/`). La imagen apenas añade a los
datos clínicos (+0,010 en el test). Según Grad-CAM, la red mira sobre todo la zona del tumor, aunque no solo: el 41 % de su mapa
cae en la caja del tumor, frente al 34 % de un mapa uniforme.

Cada decisión, con sus cifras y su justificación, está en [`notas/DECISIONES.md`](notas/DECISIONES.md).

## La app

Página estática en GitHub Pages (carpeta `docs/`). Las 15 redes, exportadas a ONNX, se ejecutan **en el navegador** con
ONNX Runtime Web, con la misma lectura de los PNG que en Python. Permite cargar una muestra o elegir un ejemplo, ver las tres fases y
el realce, y muestra la probabilidad, la clase, el umbral, el checksum del modelo, el dispositivo, el tiempo de respuesta y los
resultados de validación y test. La vista **Verificación** comprueba que la página reproduce las
probabilidades de Python.

## Estructura

```
├── utils_caso.py            lectura del dataset y de los PNG
├── modelos.py               la red (RedBase)
├── parada.py                parada temprana y elección de la época
├── entrenar.py              entrena una red (un fold y una semilla)
├── umbral.py                agregación, calibración de Platt y umbral, con la validación
├── predecir.py              el modelo final: 15 redes, calibración y umbral (lo usan el test y la app)
├── evaluar_test.py          validación y test: CNN, datos clínicos, combinación y referencia
├── clinicas.py              variables clínicas de la comparación
├── cajas_tumor.py           caja del tumor llevada a los PNG
├── mapas_gradcam.py         dónde mira el modelo (Grad-CAM)
├── curvas_entrenamiento.py  figuras de las curvas de entrenamiento
├── graficas.py              utilidades de gráficas
├── exportar_web.py          modelos ONNX y ejemplos para la app (necesita PyTorch)
├── datos_web.py             datos de la app (no necesita PyTorch)
├── descargar_datos.py       descarga el dataset
├── ver_muestras.py          visor de muestras del dataset
├── analisis/                análisis puntuales: comparación de experimentos, cohortes, estabilidad, control positivo...
├── modelo/                  el modelo final: 15 pesos .pt, predicciones de validación y umbral_final.json
├── referencia/              curvas por época de todos los experimentos
├── figuras/                 figuras de validación, test, entrenamiento, Grad-CAM y estabilidad
├── docs/                    la app web (GitHub Pages)
├── documentation/           material del caso
├── notas/                   DECISIONES.md y HOJA_DE_RUTA.md
├── GUIA.md                  guía del caso
└── requirements.txt
```

Los datos (`breastdcedl/`) y las salidas intermedias de los entrenamientos (`resultados/`) no se suben al repositorio.

## Cómo reproducirlo

```bash
pip install -r requirements.txt
python descargar_datos.py                  # el dataset, en breastdcedl/

# las 15 redes del modelo final (3 semillas × 5 folds); salen en resultados/
for s in 42 1 2; do for k in 0 1 2 3 4; do
  python entrenar.py --batchnorm --recorte 128 --lr 3e-5 --semilla $s --fold $k
done; done

python umbral.py                           # calibración y umbral con la validación (resultados/)
# copiar a modelo/ los 15 .pt, sus *_probs_val.csv y *_resumen.json, y umbral_final.json

python predecir.py --verificar 20          # comprueba que el modelo reproduce la validación
python predecir.py --muestra ISPY1_1001_z012
python evaluar_test.py                     # ensayo con la validación; con --test, el test (una sola vez)
python mapas_gradcam.py                    # Grad-CAM
python exportar_web.py                     # la app: modelos ONNX y ejemplos
python datos_web.py                        # la app: métricas, curvas y resultados
```

Los scripts de `analisis/` se ejecutan desde la raíz, por ejemplo `python analisis/comparar.py bn_normal_c128_lr3e-05`.

## Limitaciones

- Capacidad de discriminación limitada: con 176 pacientes, el intervalo del test roza el 0,5.
- Un solo corte 2D: se pierde la forma tridimensional del tumor.
- Sensible al brillo de la imagen; probado solo con PNG del conjunto BreastDCEDL.
- Tres cohortes con escáneres y protocolos distintos, sin armonizar.

## Datos

BreastDCEDL (Fridman y colaboradores). Consulta la licencia del conjunto de datos en su fuente oficial antes de reutilizarlo.
