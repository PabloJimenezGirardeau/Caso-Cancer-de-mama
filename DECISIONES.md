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

## D6 · 2026-10-01 · Resultado de la criba y descarte de RX2

**Resultados** (fold 0, pérdida ponderada, AUC por paciente con agregación `mean`, en la
mejor época; GPU AMD Radeon RX 6700 XT con ROCm):

| Red | Mejor época | AUC val | Épocas | s/época | Pérdida train (final) |
|---|---|---|---|---|---|
| R | 1 | 0,5654 | 9 | 28 | 0,974 |
| R_BN | 2 | 0,5734 | 10 | 30 | 0,901 |
| R5 | 2 | 0,5591 | 10 | 30 | 0,919 |
| RX2 | 2 | 0,5481 | 10 | 50 | 0,924 |

**Lectura:** las cuatro quedan entre 0,548 y 0,573, dentro del ruido (IC95 de ≈ ±0,08 con
un fold de 219 pacientes) y cerca del azar; ninguna gana. El AUC de "mejor época" es
optimista, porque la época se elige sobre la propia validación (siempre la 1 o la 2).
R apenas aprende (pérdida 0,9795 → 0,9739; una red constante tiene 0,9786 con este
`pos_weight`). Como referencia, identificar solo la cohorte da AUC 0,534 en este fold
(0,526-0,593 en los demás), y el subtipo tumoral solo, 0,662 (no disponible para la app).

**Comprobación del código:** prueba de sobreajuste (paso 6B de los apuntes) con R_BN y 20
pacientes: pérdida de entrenamiento 0,664 → 0,070 en 150 épocas. La red memoriza, luego
el entrenamiento funciona.

**Decisión:** se descarta **RX2**. No por su AUC (la diferencia no es significativa),
sino por coste y ausencia de ventaja: es 1,7 veces más lenta, tiene casi 4 veces más
parámetros que R_BN (más riesgo de sobreajuste con ~878 pacientes) y su papel de
comparación limpia frente a R5 ("¿falta contexto o faltan filtros?") no se puede
resolver con este nivel de ruido. Se mantienen **R** (referencia del profesor), **R_BN**
y **R5**.

**Alternativas:** descartar también R (sin BatchNorm no aprende en 9 épocas) — se
descartó la idea a petición del usuario, que prefiere conservar la arquitectura base;
mantener las cuatro.

---

## D7 · 2026-10-01 · Aumentado geométrico más fuerte (opción `--aumentado fuerte`)

**Decisión:** probar, frente al aumentado básico (solo volteo horizontal), un aumentado
"fuerte": volteo horizontal (p=0,5) + rotación aleatoria de ±15° + desplazamiento de
±10 % del lado + escala entre 0,9 y 1,1. Se sortea **una sola transformación por muestra**
y se aplica con la misma malla a las tres fases (PRE, EARLY, LATE no pueden desalinearse).
El hueco se rellena con 0 (fondo negro). Sin cambios de brillo ni de color: las fases no son
colores. Solo en entrenamiento; validación y test no se aumentan.

**Por qué:** con ~878 pacientes de entrenamiento la red memoriza (prueba de sobreajuste:
pérdida 0,664 → 0,070) sin generalizar (AUC de validación ~0,55, igual que identificar solo la
cohorte). Más variedad geométrica es la forma estándar y barata de regularizar.

**Verificado en código:** los tres canales se transforman siempre igual (canales idénticos
siguen idénticos; cada canal sale igual que transformado por separado con el mismo sorteo);
sin rotación/escala/desplazamiento equivale a `torch.flip` (error 3·10⁻⁵); forma y rango
[0,1] conservados; resultados idénticos entre ejecuciones con la misma semilla.

**Alternativas:** aumentado más suave o más agresivo; aumentos fotométricos (descartados por
la regla del caso sobre color/ImageNet); más aumentado solo sobre la ganadora.

---

## D8 · 2026-10-01 · Canales de realce (opción `--entrada realce`)

**Decisión:** probar como entrada de la red **(PRE, EARLY−PRE, LATE−EARLY)** en lugar de
(PRE, EARLY, LATE). Canal 0: anatomía; canal 1: captación de contraste (la señal del problema);
canal 2: lavado posterior (*washout*). Sigue siendo `in_channels=3`. Es una combinación lineal de
los mismos canales: no se pierde ni se añade información.

**Dónde se calcula:** **dentro del modelo**, como primera operación fija y sin parámetros
(`EntradaRealce` en `modelos.py`), no en el `Dataset`. Así el entrenamiento, la evaluación y la
app le entregan al modelo siempre PRE, EARLY y LATE tal cual, y el cálculo no puede diferir entre
entrenamiento e inferencia (comprobación explícita de la defensa).

**Por qué:** la señal está en la diferencia entre fases; una convolución podría aprender esa
resta, pero con pocos pacientes es más fácil dársela hecha.

**Verificado en código:** el cálculo es exactamente (PRE, EARLY−PRE, LATE−EARLY); los parámetros
de las redes no cambian; los checkpoints antiguos siguen cargando. Rangos en un corte de ejemplo:
canal 1 ∈ [−0,23; 0,64], canal 2 ∈ [−0,37; 0,60].

**Alternativas:** (PRE, EARLY−PRE, LATE−PRE); añadir canales en vez de sustituir
(`in_channels` > 3, contra la guía); reescalar las diferencias.

**Plan de pruebas (fold 0, R_BN, protocolo D4 sin más cambios):**

| Prueba | Aumentado | Entrada |
|---|---|---|
| Referencia (hecha en la criba) | básico | fases |
| 1 | fuerte | fases |
| 2 | básico | realce |
| 3 | fuerte | realce |

Los ficheros de salida incluyen ahora la configuración en el nombre (y las pruebas rápidas
`prueba<N>`), y `entrenar.py` se niega a sobrescribir un resultado existente salvo con
`--sobrescribir`.

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
