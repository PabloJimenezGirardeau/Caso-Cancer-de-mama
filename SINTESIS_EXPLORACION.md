# Síntesis de la exploración — caso BreastDCEDL

Documento de hechos. No contiene decisiones de diseño ni recomendaciones: donde
el enunciado deja algo abierto, solo se listan las opciones que aparecen en la
documentación. Todo lo marcado como **[verificado]** lo he comprobado yo contra
los ficheros reales, no lo he copiado de la documentación.

Fuentes leídas: `GUIA.md`, `documentation/caso_breastdcedl.pdf`, `README.md`
(solo contiene `#caso`), `LICENSE`, `descargar_datos.py`, `utils_caso.py`,
`ver_muestras.py`, los cuatro ficheros de `metadata/` y las imágenes de
`breastdcedl/dataset/`.

---

## 1. El problema, en mis palabras

- **Qué se predice.** `pCR` (*pathological Complete Response*): 1 si, tras la
  quimioterapia neoadyuvante y la cirugía, el patólogo no encuentra cáncer
  invasivo residual; 0 si queda enfermedad invasiva (poca o mucha; la etiqueta
  no dice cuánta). `pCR=1` es el resultado favorable.
- **Con qué entrada.** Una DCE-MRI de mama tomada **antes** del tratamiento. Cada
  *corte* axial se entrega como tres PNG en escala de grises de 256×256 (PRE,
  EARLY, LATE: sin contraste, contraste temprano, contraste tardío), que se
  apilan en un tensor `[3, 256, 256]`. Los tres canales son instantes de tiempo,
  no colores. La señal está sobre todo en la diferencia entre fases (realce).
- **No es detección.** El tumor ya se ve; lo que se estima es una respuesta
  futura, que solo se confirma después con tejido. Una salida de 0,82 es
  P(pCR=1)=0,82, no una probabilidad de curación ni de supervivencia.
- **Nivel corte vs. paciente.** El modelo puntúa cortes, pero la etiqueta es una
  propiedad de la paciente (**[verificado]**: ninguna paciente tiene etiquetas
  mezcladas; 1.273 pacientes, 12.703 cortes, 1.266 pacientes con exactamente 10
  cortes). Los ~10 cortes de una paciente están fuertemente correlacionados (misma
  anatomía, mismo tumor, misma etiqueta), así que:
  - los 12.703 cortes son más filas, no más pacientes independientes;
  - separar cortes de la misma paciente entre entrenamiento y validación
    permitiría reconocer anatomía ya vista (fuga) e inflaría las métricas;
  - la métrica que cuenta es por paciente, lo que obliga a combinar las
    probabilidades de sus cortes en una sola;
  - el tamaño efectivo de la evaluación es 176 pacientes de test (53 con pCR=1),
    no 1.758 cortes; la propia guía advierte de intervalos de confianza anchos.
- **Desbalance.** Por paciente, train: 775 pCR=0 / 322 pCR=1; test: 123 / 53
  (**[verificado]**, coincide con `statistics.json`). Predecir siempre 0 da ~70 %
  de accuracy (**[verificado]** con `evaluar_por_paciente` sobre un fold: 70,8 %,
  sensibilidad 0, AUC no definible).

## 2. Restricciones duras

Reglas que la guía (D1) declara que **invalidan el trabajo**:

1. Mezclar cortes de una misma paciente entre entrenamiento y validación.
2. Usar test para ajustar el modelo o los hiperparámetros (incluido el umbral y el
   método de agregación, según C4 y el PDF §2.3).
3. Modelos preentrenados o transfer learning (y, por el PDF §7, ResNet,
   EfficientNet, VGG, DenseNet).
4. Publicar la validación privada o sus etiquetas.
5. Trabajo no individual.

Otras obligaciones explícitas del enunciado/guía (no las llama "invalidantes", pero
se comprueban en la defensa o están redactadas como obligatorias):

- CNN 2D propia en PyTorch, `in_channels=3` por las 3 fases, **una sola salida
  (un logit)**, `BCEWithLogitsLoss`, comparar pérdida normal frente a ponderada
  con `pos_weight = N0/N1`.
- Documentar y guardar: arquitectura, inicialización, regularización, optimizador,
  tasa de aprendizaje, tamaño de lote, épocas, semilla y criterio de parada.
- Los 3 canales no son RGB: nada de normalización ImageNet, `ColorJitter` ni
  `ImageFolder`; cualquier transformación geométrica se aplica **al tensor
  completo**, nunca canal a canal.
- Cualquier partición interna adicional, también por paciente.
- Reportar más que accuracy (matriz de confusión, sensibilidad, especificidad,
  AUC); el PDF menciona además calibración e intervalo de incertidumbre.
- Evaluar en test **una sola vez**, con el modelo ya cerrado.
- App desplegada y accesible por URL (no solo local): `model.eval()`, sin
  gradientes, resultado reproducible, sin rutas arbitrarias del sistema, sin
  ejecutar contenido subido, límite de tamaño y formato, mensajes claros ante
  errores, aviso de uso educativo sin validez clínica.
- **Normalización idéntica entre entrenamiento y app** (se verifica en la defensa,
  con cinco muestras de cinco pacientes privadas, sin tocar código ni reentrenar).
- Máximo cinco diapositivas; entrega de repositorio reproducible, pesos, informe y
  URL.
- Licencia y ética: uso docente/investigación, sin reidentificación ni
  recomendación terapéutica; las variables raciales no pueden tratarse como
  causales (la guía lo califica de error metodológico, no de opción).

## 3. Qué da resuelto `utils_caso.py` y qué queda por construir

### Funciones y constantes exportadas, tal como están escritas

| Símbolo | Qué hace |
|---|---|
| `RAIZ` | `Path(__file__).resolve().parent`: la carpeta donde vive `utils_caso.py`. Es el valor por defecto de `raiz` en todas las funciones. |
| `FASES` | `("PRE", "EARLY", "LATE")`. |
| `METODOS_AGREGACION` | `("mean", "max", "median", "voto")`. |
| `cargar_samples(raiz)` | `pd.read_csv(raiz/metadata/samples.csv)`. |
| `cargar_patients(raiz)` | `pd.read_csv(raiz/metadata/patients.csv)`. |
| `cargar_fase(ruta_relativa, raiz)` | Abre un PNG, lo convierte a `"L"` y devuelve `(256,256)` `float32` = píxel/255. |
| `cargar_imagen(fila, raiz)` | Lee `path_pre`, `path_early`, `path_late` de una fila y apila en ese orden → `(3,256,256)` `float32` en [0,1]. |
| `particion(samples, fold_val=0)` | Exige `fold_val` en 0..4 (si no, `ValueError`). Toma `split=="train"`; validación = filas con `fold==fold_val`, entrenamiento = el resto de train. Comprueba que no haya `patient_id` en ambos lados (si lo hubiera, `RuntimeError`). Devuelve dos DataFrames que son filtrados del original (conservan el índice original). No toca test. |
| `conjunto_test(samples)` | `samples[split=="test"]`. No hay nada en el código que fuerce "usar una sola vez". |
| `BreastDCEDataset(filas, raiz, transform=None)` | `Dataset` de PyTorch; solo existe si `import torch` funciona (si no, la variable vale `None`). Hace `reset_index` de las filas; `__getitem__` carga los 3 canales, aplica `transform` (si hay) al tensor `(3,256,256)` completo y devuelve `(x, y)` con `y` = `pCR` como `float32`. No aplica normalización ni aumento por defecto. |
| `agregar_por_paciente(patient_id, probabilidad, metodo, umbral)` | Construye un DataFrame `patient_id, prob`. `mean`/`max`/`median`: `groupby(patient_id).prob.<método>()`. `voto`: fracción de cortes con `prob >= umbral`. Devuelve un DataFrame indexado por paciente (ordenado por `patient_id`) con columna `prob`. Método fuera de la lista → `ValueError`. |
| `evaluar_por_paciente(probabilidad, filas, umbral, metodo)` | Comprueba que `len(probabilidad)==len(filas)` (solo la longitud, no el orden). Agrega por paciente; la etiqueta de la paciente es `pCR.first()` de sus filas. Predicción = `prob_agregada >= umbral`. Devuelve un `dict` con `pacientes`, `metodo`, `umbral`, `matriz_confusion` (`VP/VN/FP/FN`), `accuracy`, `sensibilidad`, `especificidad`, `precision` (0/0 → `nan`) y `auc` (`roc_auc_score` sobre la `prob` agregada; **cualquier excepción devuelve `nan` en silencio**, p. ej. si sklearn no está instalado o si solo hay una clase). |
| `pos_weight(filas)` | `N(pCR=0)/N(pCR=1)` sobre las filas (cortes) recibidas; `ValueError` si no hay positivas. |

Detalles de comportamiento que he comprobado ejecutándolo (**[verificado]**):

- Con `voto`, el mismo `umbral` se usa dos veces: para decidir si cada corte
  "vota" y, después, en `evaluar_por_paciente`, para comparar la fracción
  resultante (`prob` = fracción) con ese mismo `umbral`. El AUC de `voto` se
  calcula sobre esa fracción.
- Con `max` y probabilidades aleatorias, casi todas las pacientes superan 0,5
  (154 FP, 0 FN, 64 VP, 1 VN sobre el fold 0): la salida depende de cuántos
  cortes se agregan.
- `particion` da, para `fold_val=0..4`, validaciones de 219/219/220/220/219
  pacientes y ninguna fila de test en ningún lado.
- `pos_weight` sobre el conjunto de entrenamiento de cada fold: 2,4002 / 2,3725 /
  2,3281 / 2,5495 / 2,3744; sobre todo train: 2,4033; sobre test: 2,3170.

### Qué queda por construir (según el propio docstring de `utils_caso.py` y el enunciado)

La arquitectura de la CNN, el bucle de entrenamiento, la regularización, el
umbral de decisión y su justificación; además, lo que el enunciado pide y
`utils_caso.py` no contiene: la comparación de pérdida normal vs. ponderada, el
registro de configuración/semilla/pesos, el informe (incluida discusión de
sesgos, calibración y limitaciones), la aplicación web desplegada, las cinco
diapositivas y, si se usa GPU, el reporte de dispositivo/tiempos que pide el PDF
§8. `utils_caso.py` no incluye transformaciones ni estadísticos de normalización,
ni cálculo de intervalos de confianza, ni curvas ROC, ni nada de calibración.

## 4. Discrepancias entre documentación y datos reales

Ordenadas por impacto práctico.

1. **Ubicación de los ficheros vs. `RAIZ`. [verificado]** La guía (A3) describe
   todo dentro de `breastdcedl/`, y `descargar_datos.py` por defecto lo descarga
   todo ahí (imágenes *y* `metadata/`, scripts, etc.). En este proyecto, en cambio:
   `metadata/`, `utils_caso.py`, `ver_muestras.py` y `GUIA.md` están en la raíz,
   y solo `dataset/` está en `breastdcedl/dataset/`. Como `RAIZ` es la carpeta del
   script, `<raíz>/dataset/...` no existe:
   - `uc.cargar_imagen(fila)` con `raiz` por defecto → `FileNotFoundError`
     (comprobado); con `raiz="breastdcedl"` funciona.
   - `python ver_muestras.py muestra ISPY1_1001_z012` → `FileNotFoundError`
     (comprobado). Para ejecutarlo sin modificarlo lo corrí desde una copia en un
     directorio temporal con enlaces (*junction*) a `dataset/` y `metadata/`; ahí
     funcionó.
   - `BreastDCEDataset` por defecto también fallaría (usa `RAIZ`).
2. **Tabla de folds de la guía (C1) vs. `samples.csv`. [verificado]** Los
   pacientes por fold coinciden (219/219/220/220/219), pero no los cortes ni la
   proporción de pCR:

   | fold | cortes guía | cortes real | pCR (cortes) guía | pCR (cortes) real | pacientes pCR=1 real |
   |---|---|---|---|---|---|
   | 0 | 2.187 | 2.186 | 0,2926 | 0,2928 | 64 de 219 |
   | 1 | 2.186 | 2.190 | 0,2928 | 0,2831 | 62 de 219 |
   | 2 | 2.191 | 2.192 | 0,2948 | 0,2673 | 59 de 220 |
   | 3 | 2.195 | 2.192 | 0,2961 | 0,3422 | 75 de 220 |
   | 4 | 2.186 | 2.185 | 0,2928 | 0,2838 | 62 de 219 |

   La guía habla de folds "estratificados por pCR" con proporciones casi iguales:
   **0,2926–0,2961** (por cortes). En `samples.csv` la proporción real de pCR va de
   **0,2673 (fold 2) a 0,3422 (fold 3)**. Sí están repartidas casi por igual las
   cohortes (≈42 duke / 21 spy1 / 157 spy2 por fold). La suma de cortes (10.945)
   coincide.

   **Estado: confirmado por el usuario.** Son dos asignaciones de folds distintas
   (la de la guía parece estratificada por pCR; la de `samples.csv`, no).
   **`samples.csv` es la fuente de verdad** del proyecto: es el fichero que se
   ejecuta. Consecuencia acordada: cualquier `pos_weight`, proporción de clases o
   composición de fold se calcula en código directamente desde `samples.csv` /
   `patients.csv`, nunca copiando cifras de este documento ni de la guía. Las cifras
   de este documento son una fotografía de la exploración, no valores de entrada.
3. **Licencia. [verificado]** El fichero `LICENSE` del repositorio es **CC BY 4.0**
   (permite uso comercial); la guía (A3, D6) y el PDF §15 dicen **CC BY-NC 4.0**.
4. **Comprobación de fases PRE<EARLY / LATE>EARLY. [verificado]** Guía/PDF: 100 %
   y 83 %, medido sobre 120 pacientes. Sobre las 1.273 (corte central, criterio de
   la guía `tejido = PRE>0.1`): PRE<EARLY en **99,7 %** (1.269/1.273); LATE>EARLY
   en **79,5 %**. Con submuestras aleatorias de 120 obtuve PRE<EARLY 99,2–100 % y
   LATE>EARLY 78–84 %, así que es compatible con variación muestral. Las cuatro que
   incumplen PRE<EARLY: `ACRIN-6698-894673`, `Breast_MRI_056`, `ISPY1_1069`,
   `ISPY1_1077`. Aparte, `ISPY1_1139` tiene EARLY y LATE **idénticas en los 10
   cortes** (PRE media 0,18 vs. 0,46 en las otras dos).
5. **Porcentaje de pCR=0. [verificado]** La guía (A1) dice "el 70,6 % de los casos
   son pCR=0". Es cierto en train (por cortes 7.729/10.945; por pacientes 775/1.097).
   En test es 69,9 % (por cortes y por pacientes) y en el total 70,5 %.
6. **Guía C6 vs. PDF §9 (app).** El PDF pide mostrar también el **mapa de realce**
   y gestionar "fases ausentes o duplicadas" e imágenes que no sean PNG 256×256; la
   lista de C6 de la guía no menciona el mapa de realce y habla de "ficheros
   corruptos o valores no finitos". Tampoco recoge el reporte de GPU del PDF §8.
7. **Dimensiones del dataset (PDF vs. copia local).** El PDF habla de una edición de
   1.450 pacientes/14.470 cortes (con 177 pacientes de validación privada). La copia
   local tiene 1.273/12.703, coherente con quitar esas 177 (1.450−177=1.273). No es
   contradicción, pero conviene saber que la validación privada no está aquí.
   `excluded_patients.csv` lista `Breast_MRI_001` y `Breast_MRI_003` con la columna
   `error` = `('Breast_MRI_001', 4)`; el motivo (fase tardía ausente) solo está en
   el PDF, no en el CSV, y no aparecen ni en `patients.csv` ni en `samples.csv`.

Comprobado y **coincide** con la documentación: 1.097/176 pacientes y 10.945/1.758
cortes; 32.835/5.274 PNG (en disco y en el índice); todas las rutas de
`samples.csv` existen (38.109); patrón de nombres `<patient_id>_zNNN_<FASE>.png`
sin excepciones; las 38.109 imágenes son PNG modo `L` de 256×256, `uint8`, con
mínimo 0 y máximo 255 (1,0 tras dividir); `sample_id` = `patient_id_zNNN`; etiqueta
constante por paciente; ninguna paciente en dos splits ni en dos folds; **ninguna
paciente en train y test a la vez** (comprobado sobre las 1.273 en el índice y sobre
los nombres de carpeta del disco); `patients.csv` (1.273 filas, 30 columnas) cruza
exactamente con `samples.csv` (`pid` = `patient_id`, mismas etiquetas y mismos
splits); recuentos por cohorte y `statistics.json`; carpetas de paciente con 30
ficheros salvo las que tienen menos de 10 cortes.

## 5. Otros hechos de los datos que la documentación no cuantifica

- **Cortes por paciente:** 10 en 1.266 pacientes; menos de 10 en 7 (5, 5, 6, 6, 6,
  7 y 8 cortes): seis de Duke (una en test) y `ISPY2-553012`.
- **Nulos en `patients.csv`:** `menopause` 172 (los 139 de `spy1` + 33 de `spy2`),
  `HER2`/`HR_HER2_STATUS`/`TripleNeg`/`HER2pos`/`HRposHER2neg` 5 (todas `spy1`),
  `HR` 2, `age`/`race_*` 3 (`spy2`), `tum_vol` 1 (`duke`). No hay nulos en
  `samples.csv` ni en la columna `pCR`.
- **Tasa de pCR por cohorte (todas las pacientes):** duke 21,1 % (53/251), spy1
  27,3 % (38/139), spy2 32,2 % (284/883).
- **Adquisición distinta por cohorte** (medias): `n_z` 37 / 59 / 106, `n_times`
  4,4 / 3,4 / 7,2, tamaño original `n_xy` de 128 a 512 (duke/spy1/spy2). Todos los
  PNG son 256×256 igualmente.
- **Intensidad media en tejido (PRE / EARLY / LATE, corte a paciente):** duke
  0,26 / 0,38 / 0,41; spy1 0,28 / 0,39 / 0,42; spy2 0,28 / 0,43 / 0,44.
- **`slice_index` frente a `mask_start`/`mask_end`:** en Duke el 100 % de los
  cortes cae dentro de [`mask_start`, `mask_end`]; en spy1 el 47 % y en spy2 el 29 %.
  En spy, ningún corte supera `mask_end`; el resto queda por debajo de
  `mask_start`. La documentación dice que en spy se eligen por superficie de máscara
  3D, sin explicar el sistema de coordenadas de `mask_start/end` para esas cohortes.
- **Visualización [verificado]:** corrí los tres modos de `ver_muestras.py`
  (`muestra`, `paciente`, `comparar`) con el enlace descrito. PRE se ve más oscuro
  que EARLY; el realce EARLY−PRE resalta la zona que capta contraste; LATE es
  parecido a EARLY; los 10 cortes de una paciente son alturas distintas del mismo
  tejido. Coincide con lo que dice la guía.
- **Entorno:** el `.venv` del proyecto solo tiene `requests` (sin numpy, pandas,
  PIL, matplotlib, scikit-learn ni torch); el Python del sistema (3.12) tampoco los
  tiene. Para explorar instalé numpy, pandas, pillow, matplotlib y scikit-learn en
  un directorio temporal fuera del proyecto; ahí sklearn no llegó a importar (un
  DLL bloqueado por política del sistema) y por eso el AUC me salió `nan`, lo que
  ilustra el `except Exception` silencioso de `evaluar_por_paciente`. `torch` no
  está instalado en ninguna parte, luego `BreastDCEDataset` vale `None` en este
  entorno.
- **Git:** `.gitignore` ignora `train/`, `test/` y `metadata/` (con lo que ni las
  imágenes ni los CSV entran en el repo). Solo hay dos commits; `GUIA.md`, `LICENSE`,
  los scripts y `documentation/` están sin añadir.

No he modificado ningún fichero del proyecto salvo crear este documento.

## 6. Puntos donde el enunciado deja una decisión abierta (solo opciones)

- **Agregación por paciente.** En `utils_caso.py`: `mean`, `max`, `median`, `voto`.
  La guía describe el efecto de cada una (mean/median robustas; max más sensible y
  más falsos positivos; voto = fracción de cortes sobre el umbral).
- **Umbral.** Los ejemplos usan 0,5. El enunciado exige elegirlo con validación
  interna, nunca con test, y justificarlo. La guía y el PDF no fijan ningún valor.
- **Validación interna.** Un solo fold (`fold_val` fijo), los cinco folds y
  promediar (la guía lo menciona), o una partición propia con
  `StratifiedGroupKFold(groups=patient_id)` (también mencionada).
- **Arquitectura.** Libre salvo: propia, 2D, `in_channels=3`, un logit. La guía
  sugiere empezar con tres o cuatro bloques convolucionales y crecer solo si hace
  falta; no dice nada más (anchuras, tipo de normalización, pooling, cabeza, etc.).
- **Inicialización, regularización, optimizador, tasa de aprendizaje, tamaño de
  lote, épocas, semilla, criterio de parada.** Solo se exige justificarlos y
  guardarlos.
- **Pérdida.** Es obligatorio comparar `BCEWithLogitsLoss` normal y ponderada con
  `pos_weight=N0/N1`. Queda abierto sobre qué filas se calcula ese cociente:
  `uc.pos_weight(filas)` acepta cualquier subconjunto (por fold de entrenamiento
  2,33–2,55; todo train 2,40; la guía escribe "2,40").
- **Canales de entrada.** La guía y el PDF describen el tensor como PRE, EARLY,
  LATE (`in_channels=3`). El mapa de realce EARLY−PRE aparece como herramienta de
  visualización y como salida que debe mostrar la app; ni la guía ni el PDF dicen si
  entra o no como canal derivado.
- **Aumento de datos.** Permitido solo geométrico y aplicado al tensor completo
  (`transform` recibe `(3,256,256)`); nada de color/ImageNet. Si se usa o cuál queda
  abierto.
- **Normalización de entrada.** Los datos ya vienen en [0,1] con una ventana común
  a las tres fases. Añadir otra normalización no está ni prohibido ni prescrito,
  salvo ImageNet; si se hace, debe ser idéntica en la app.
- **Muestreo/ponderación por paciente.** Todas las pacientes tienen ~10 cortes
  (algunas 5–8); nada dice si cada corte cuenta igual o si se pondera.
- **Modelo final.** El PDF dice "fijar el modelo usando train y una validación
  interna separada por paciente"; no especifica si el modelo final se reentrena con
  todo train o se conserva el entrenado con un fold.
- **Dispositivo.** GPU recomendada, no obligatoria; comparación CPU/GPU con
  `speedup = tCPU/tGPU` opcional. Colab se cita como alternativa en la guía.
- **Aplicación.** Streamlit, Gradio o equivalente. Entrada: "las tres fases de un
  corte" o un ejemplo (PDF §9).
- **Uso de `patients.csv`.** Se presenta para analizar sesgos entre cohortes y
  contexto clínico; el enunciado no dice explícitamente si puede usarse como entrada
  del modelo (la app solo recibe los tres PNG).

## 7. Preguntas abiertas para ti

1. **Disposición de carpetas.** ¿Es intencionado que `metadata/` y los scripts
   estén en la raíz y `dataset/` dentro de `breastdcedl/`? Hoy `cargar_imagen`,
   `BreastDCEDataset` y `ver_muestras.py` no encuentran las imágenes con sus
   valores por defecto.
2. **Entorno de entrenamiento.** ~~¿Dónde vas a entrenar?~~ **Resuelta:** en GPU,
   conectando a un dispositivo externo (no local). El `.venv` del repo solo tiene
   `requests` y no hay `torch` instalado en ningún sitio de este proyecto; sigue
   pendiente qué dependencias exactas se instalarán en ese dispositivo externo,
   pero eso ya no es una pregunta abierta de la exploración, sino un detalle de
   la fase de entrenamiento.
3. **Folds.** ~~La tabla de C1 de la guía no coincide con la columna `fold` de tu
   `samples.csv`.~~ **Resuelta:** son dos asignaciones distintas y `samples.csv` es
   la fuente de verdad (ver sección 4, punto 2).
4. **Licencia.** El `LICENSE` dice CC BY 4.0 y la documentación dice CC BY-NC 4.0.
   ¿Sabes cuál es la intención?
5. **App: corte o paciente.** La app recibe "las tres fases de un corte", pero la
   evaluación oficial y la agregación son por paciente. ¿Cómo debe interpretarse el
   resultado que muestra la app (por corte o por paciente) en la defensa? El
   enunciado no lo aclara.
6. **`patients.csv` como entrada.** ¿Se permite usar variables clínicas o de
   cohorte como entrada del modelo, o solo las imágenes?
7. **Pacientes anómalas.** `ISPY1_1139` (EARLY == LATE en todos los cortes) y las
   cuatro con PRE ≥ EARLY: ¿son conocidas/esperadas? No las he tocado.
8. **`mask_start`/`mask_end` en spy1/spy2.** El 53 % y el 71 % de sus cortes caen por
   debajo de `mask_start`; no sé en qué sistema de coordenadas están esos campos.
9. **Motivo de las exclusiones.** El CSV solo trae `('Breast_MRI_001', 4)`; ¿tienes
   más contexto del "4"?
10. **Validación privada.** ¿Hay alguna información adicional sobre cómo se
    distribuye (cohortes, proporción de pCR) la validación privada de 177 pacientes
    que se usará en la defensa?
