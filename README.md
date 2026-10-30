# Predicción de pCR en resonancia DCE de mama

Una red convolucional propia, entrenada desde cero en PyTorch, estima la probabilidad de **respuesta patológica completa (pCR)**
a la quimioterapia neoadyuvante a partir de **un solo corte** de resonancia magnética con contraste dinámico (DCE), tomado antes
del tratamiento. Trabajo académico sobre el conjunto de datos **BreastDCEDL**.

### App web: [pablojimenezgirardeau.github.io/Caso-Cancer-de-mama](https://pablojimenezgirardeau.github.io/Caso-Cancer-de-mama/)

Se abre en el navegador del ordenador, sin instalar nada: carga las tres fases de un corte (o un ejemplo) y predice.

**Presentación:** [`presentacion/Presentacion_pCR.pdf`](presentacion/Presentacion_pCR.pdf) · **Decisiones:**
[`notas/DECISIONES.md`](notas/DECISIONES.md) · **Licencia:** [CC BY-NC 4.0](LICENSE)

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

Con el umbral, en el test: sensibilidad 30 %, especificidad 76 %, precisión 35 % y, de media en las dos clases, precisión 0,53,
recall 0,53 y F1 0,53 (en validación: 41 %, 70 %, 36 % y 0,55 / 0,56 / 0,55). El test se evaluó **una sola vez**, con el modelo ya
cerrado ([`figuras/test/`](figuras/test/)). La imagen apenas añade a los datos clínicos (+0,010 en el test). Según Grad-CAM, la red
mira sobre todo la zona del tumor, aunque no solo: el 41 % de su mapa cae en la caja del tumor, frente al 34 % de un mapa uniforme.

## La app

Página estática en GitHub Pages: [abrir la app](https://pablojimenezgirardeau.github.io/Caso-Cancer-de-mama/) (carpeta
[`docs/`](docs/)). Las 15 redes, exportadas a ONNX, se ejecutan **en el navegador** con
ONNX Runtime Web, con la misma lectura de los PNG que en Python; las imágenes no salen del ordenador de quien la usa.

- **Predecir:** una casilla para cada fase (PRE, EARLY y LATE; el nombre del fichero da igual) o un ejemplo de la galería. Visor con
  las fases, las tres a la vez, el realce y un comparador. El informe da la probabilidad, la clase, el umbral, si la decisión
  aguanta un 10 % más o menos de brillo, el control de calidad de la entrada (formato, orden de las fases, brillo frente al
  entrenamiento), el checksum del modelo, el dispositivo y el tiempo de respuesta. Historial de la sesión descargable en CSV.
- **Modelo:** métricas de validación y test, comparación con los datos clínicos, arquitectura, umbral, calibración y curvas de
  entrenamiento con la parada temprana.
- **Acerca de:** qué es la pCR y por qué importa, los datos por cohorte, cómo predice el modelo, cómo usar la app y leer el
  informe, limitaciones, reproducibilidad (la página da las mismas probabilidades que Python: diferencia máxima 5·10⁻⁸ en doce
  cortes) y créditos.

Los ejemplos de la galería son pacientes de **entrenamiento** (nunca del test): la mayoría de las redes ya las vieron, así que no
miden el rendimiento.

## Estructura

```
├── src/                 todo el código Python, por etapas
│   ├── datos/           descarga, lectura del dataset y de los PNG, cajas del tumor y variables clínicas
│   ├── entrenamiento/   la red, la parada temprana, el entrenamiento y las figuras de las curvas
│   ├── evaluacion/      calibración y umbral, el modelo final (15 redes) y el test
│   ├── web/             lo que prepara la app: modelos ONNX, datos y galería de ejemplos
│   └── analisis/        estudios puntuales: comparación de experimentos, cohortes, estabilidad, Grad-CAM...
├── docs/                la app web (GitHub Pages): index.html, app.js, estilos.css, modelos/ (ONNX), datos/ y ejemplos/
├── modelo/              el modelo final: 15 pesos .pt, sus predicciones de validación y umbral_final.json
├── figuras/             figuras de datos, entrenamiento, validación, test, Grad-CAM y estabilidad
├── referencia/          curvas por época de todos los experimentos
├── notas/               DECISIONES.md (cada decisión, con sus cifras) y HOJA_DE_RUTA.md
├── presentacion/        las 5 diapositivas de la defensa (.pptx y .pdf) y el script que las genera con PowerPoint
├── documentation/       enunciado oficial y figuras del caso (material del curso)
├── GUIA.md              guía del caso (material del curso)
├── requirements.txt
└── LICENSE
```

| Carpeta | Script | Qué hace |
|---|---|---|
| `src/datos/` | `descargar_datos.py` | descarga el dataset en `breastdcedl/` (material del curso) |
| | `utils_caso.py` | lectura del dataset, de los folds y de los PNG (material del curso) |
| | `ver_muestras.py` | visor de muestras del dataset (material del curso) |
| | `cajas_tumor.py` | la caja del tumor de `patients.csv`, llevada a los PNG |
| | `clinicas.py` | variables clínicas del modelo de comparación |
| `src/entrenamiento/` | `modelos.py` | la red (`RedBase`) |
| | `parada.py` | parada temprana y elección de la época |
| | `entrenar.py` | entrena una red (un fold y una semilla) |
| | `curvas_entrenamiento.py`, `graficas.py` | figuras de las curvas y utilidades de gráficas |
| `src/evaluacion/` | `umbral.py` | calibración de Platt y umbral, con la validación |
| | `predecir.py` | el modelo final: 15 redes, calibración y umbral (lo usan el test y la app) |
| | `evaluar_test.py` | validación y test: CNN, datos clínicos y combinación |
| `src/web/` | `exportar_web.py` | modelos ONNX y cortes de comprobación frente a Python (necesita PyTorch) |
| | `datos_web.py` | métricas, curvas y rango de brillo del entrenamiento (sin PyTorch) |
| | `galeria_web.py` | galería de ejemplos (sin PyTorch) |

Cada carpeta de `src/` es un paquete de Python: los scripts se ejecutan desde la raíz del proyecto y se importan entre sí con
`from datos import utils_caso`, `from entrenamiento import modelos`, etc. Los datos (`breastdcedl/`) y las salidas intermedias
de los entrenamientos (`resultados/`) no se suben al repositorio.

## Cómo reproducirlo

Python 3.12 o posterior. PyTorch se instala según el hardware (ver la nota de [`requirements.txt`](requirements.txt)). Todo se
ejecuta desde la raíz del proyecto.

```bash
pip install -r requirements.txt
python src/datos/descargar_datos.py                 # el dataset, en breastdcedl/

# las 15 redes del modelo final (3 semillas × 5 folds); salen en resultados/
for s in 42 1 2; do for k in 0 1 2 3 4; do
  python src/entrenamiento/entrenar.py --batchnorm --recorte 128 --lr 3e-5 --semilla $s --fold $k
done; done

python src/evaluacion/umbral.py                     # calibración y umbral con la validación (resultados/)
# copiar a modelo/ los 15 .pt, sus *_probs_val.csv y *_resumen.json, y umbral_final.json

python src/evaluacion/predecir.py --verificar 20    # comprueba que el modelo reproduce la validación
python src/evaluacion/predecir.py --muestra ISPY1_1001_z012
python src/evaluacion/evaluar_test.py               # ensayo con la validación; con --test, el test (una sola vez)

python src/web/exportar_web.py                      # la app: modelos ONNX y cortes de comprobación
python src/web/datos_web.py                         # la app: métricas, curvas y resultados (lee los PNG: un minuto)
python src/web/galeria_web.py                       # la app: galería de ejemplos
```

Los estudios de `src/analisis/` se ejecutan igual, por ejemplo `python src/analisis/comparar.py bn_normal_c128_lr3e-05` o
`python src/analisis/mapas_gradcam.py`. Los comandos de `GUIA.md` (`python descargar_datos.py`, `python ver_muestras.py`) son de
antes de la reorganización: ahora esos scripts están en `src/datos/`.

## Limitaciones

- Capacidad de discriminación limitada: con 176 pacientes, el intervalo del test roza el 0,5.
- Un solo corte 2D: se pierde la forma tridimensional del tumor.
- Sensible al brillo de la imagen; probado solo con PNG del conjunto BreastDCEDL.
- Tres cohortes con escáneres y protocolos distintos, sin armonizar.

## Datos y licencia

Imágenes desidentificadas derivadas de **BreastDCEDL** (Fridman y colaboradores), con licencia CC BY-NC 4.0: uso docente y de
investigación. Este repositorio se publica con la misma licencia, [CC BY-NC 4.0](LICENSE).
