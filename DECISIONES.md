# Registro de decisiones — Caso BreastDCEDL

Cada entrada recoge qué se decidió, qué alternativas se consideraron y por qué. Es la
base para justificar el diseño en el informe y en las diapositivas 3 y 4.

---

## D1 · 2026-09-23 · Arquitecturas candidatas

**Decisión:** comparar cuatro redes. Todas derivan de la arquitectura de referencia del
profesor (`presentacion_caso.pdf`, pág. 8) y cambian una sola cosa cada vez, para
poder atribuir cada efecto a un cambio concreto.

| Red | Cambio respecto a la referencia | Parámetros | Campo receptivo | Mapa final |
|---|---|---|---|---|
| **R** | Ninguno: 4 bloques conv 3×3 + ReLU (16-32-64-128) con maxpool 2×2, GAP, densa 64 + dropout, 1 logit | 105.761 | 46 px | 16×16 |
| **R+BN** | + BatchNorm tras cada convolución (bloque canónico de los apuntes) | 106.001 | 46 px | 16×16 |
| **R-5** | + BatchNorm + 5.º bloque de 256 canales | 409.617 | 94 px | 8×8 |
| **R×2** | + BatchNorm + doble de canales (32-64-128-256) | 405.409 | 46 px | 16×16 |

Cifras calculadas en código (el recuento de R coincide con el de la diapositiva).

**Por qué estas cuatro:**
- R es el punto de partida que da el profesor ("la tuya la justificas tú").
- R+BN comprueba si BatchNorm, que los apuntes ponen en el bloque canónico, ayuda.
- R-5 y R×2 tienen casi los mismos parámetros pero distinto campo receptivo. Si gana
  R-5, a la referencia le faltaba ver más imagen (46 px parece poco frente a lesiones
  que, en los ejemplos vistos, miden ~50–100 px); si gana R×2, le faltaba capacidad.

**Alternativas consideradas y descartadas:**
- *R-2c* (2 convoluciones por bloque, 302 K parámetros, 76 px) y *A* (6 bloques hasta
  4×4, ~4 M parámetros): por coste; A además sobreajustaría con ~878 pacientes.
- *Conexiones residuales:* innecesarias con 4–5 capas convolucionales (los apuntes
  sitúan la degradación a partir de ~20) y con riesgo de parecer "arquitectura de
  catálogo", que invalida el trabajo.
- *Red a nivel de paciente o 3D:* el caso pide una CNN 2D que puntúe cortes y agregue
  después por paciente; la app recibe un solo corte.
- *"3 bloques" literal:* a 256×256 deja un mapa de 32×32 y un campo receptivo de 22 px.

---

## D2 · 2026-09-23 · Protocolo de comparación

**Decisión:** criba de las 4 redes en un fold; las 2 mejores se confirman en los 5
folds. Total: 4 + 8 = 12 entrenamientos.

**Por qué:** validando con un solo fold (≈219 pacientes) el intervalo de confianza del
95 % del AUC es de ±0,08; juntando los 5 folds baja a ±0,04 (calculado con los
recuentos reales). La criba descarta rápido lo que va mal y la confirmación da rigor a
la elección final.

**Alternativas:** todas en los 5 folds (20 entrenamientos, casi el doble de coste);
solo 1 fold (4 entrenamientos, pero diferencias menores de ~0,08 no serían
concluyentes).

---

## D3 · 2026-09-23 · Pérdida durante la comparación

**Decisión:** las 4 redes se entrenan con `BCEWithLogitsLoss` **ponderada**, con
`pos_weight = N0/N1` calculado en código sobre las filas de entrenamiento de cada fold.
La ganadora se entrena después también con la pérdida normal, para la comparación
obligatoria del caso.

**Por qué:** hacer la comparación normal/ponderada en todas las redes duplicaría los
entrenamientos. Con la pérdida normal y un desbalance ~70/30, la red tiende a predecir
casi siempre «no pCR» con umbral 0,5, lo que deja la sensibilidad sin información
durante la criba.

**Alternativas:** pérdida normal durante la comparación; ambas pérdidas en todas las
redes.

---

## D4 · 2026-09-23 · Protocolo de entrenamiento común

**Decisión:** las cuatro redes se entrenan exactamente igual:

| Aspecto | Valor | Por qué |
|---|---|---|
| Activación tras la densa de 64 | ReLU | La referencia no la especifica; sin ella, densa + salida equivalen a una sola capa lineal |
| Dropout | 0,5, justo antes de la salida | La referencia no da el valor; es el que dan los apuntes para capas densas |
| Entrada | Tal como viene, en [0,1], sin normalización adicional | Las imágenes ya vienen normalizadas con una ventana común; así la app usa exactamente la misma entrada |
| Aumentado | Solo volteo horizontal (probabilidad 0,5), al tensor entero | Una mama volteada es anatómicamente plausible; más aumentado, como prueba posterior sobre la ganadora |
| Optimizador | AdamW, learning rate 0,001, weight decay 0,0001 | Estándar para CNN pequeñas desde cero |
| Bajada del learning rate | ×0,5 si el AUC de validación no mejora en 3 épocas | Los apuntes: bajarlo al final "suele valer uno o dos puntos" |
| Tamaño de lote | 32 | El del ejemplo de `GUIA.md` |
| Épocas y parada | Máximo 40; parar si el AUC por paciente no mejora en 8 épocas; se conserva la mejor época | Evita sobreentrenar y decide con la métrica que importa |
| Métrica de elección | AUC por paciente, agregando con la media (provisional) | No depende del umbral; la agregación definitiva se elige después |
| Fold de la criba | 0 | El que usan por defecto la guía y `utils_caso` |
| Semilla | 42, fijada en Python, NumPy y PyTorch | Reproducibilidad, que el caso exige |

En cada entrenamiento se registran las curvas de pérdida y AUC, el AUC por corte y por
paciente, la sensibilidad y especificidad con umbral 0,5, el tiempo por época y la
memoria de GPU (el enunciado pide estos datos de GPU en el informe).

---

## D5 · 2026-09-23 · Dónde se entrena

**Decisión:** en un portátil Razer con CPU AMD y GPU NVIDIA (CUDA). Se preparan scripts
de Python ejecutables en Windows, no un cuaderno de Colab.

---

## Decisiones previas (fase de exploración)

- `metadata/samples.csv` es la fuente de verdad de los folds; la tabla de `GUIA.md` C1
  no coincide con él.
- Las cifras derivadas de los datos (`pos_weight`, proporciones, folds) se calculan
  siempre en código desde los CSV, nunca se copian de documentos.
- Jerarquía de documentos: `GUIA.md` y el enunciado mandan; `presentacion_caso.pdf` es
  el punto de partida por defecto, mejorable con ideas justificadas; los apuntes de
  clase ceden ante los tres.
- El entrenamiento se hace en una GPU externa.
