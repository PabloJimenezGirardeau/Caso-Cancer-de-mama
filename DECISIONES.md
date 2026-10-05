# Decisiones del proyecto

Cada decisión con su motivo y lo que se descartó, para poder justificarla en el informe y
en la defensa. Se actualiza según avanzamos.

## Punto de partida

- **Arquitectura base: la de referencia del profesor** (`presentacion_caso.pdf`, diapositiva 8):
  4 bloques `conv 3×3 + ReLU + maxpool 2×2` con 16-32-64-128 canales, media global (GAP),
  capa densa de 64 con dropout y un logit. 105.761 parámetros (verificado en código).
  Primero se analiza y se mide; después se cambia **una sola cosa cada vez**.
- **Idea del profesor para probar más adelante:** `conv + conv + pool` en cada bloque (dos
  convoluciones seguidas antes de cada pooling). Es una idea, no una obligación.

## Detalles de la red que la diapositiva no fija

| Decisión | Valor | Motivo |
|---|---|---|
| Activación tras la densa de 64 | **ReLU** | Sin ella, densa + salida equivalen a una sola capa lineal |
| Dropout | **0,5** | Valor habitual para capas densas (apuntes de clase) |
| Kernel | **3×3** (el de la diapositiva) | Estándar; dos 3×3 cubren lo que una 5×5 con menos parámetros; el tamaño del kernel es el hiperparámetro que menos suele cambiar el resultado |
| Padding / stride | 1 / 1 | Implícito en los tamaños de la diapositiva (el mapa conserva su tamaño) |
| Sesgo / BatchNorm | con sesgo, sin BatchNorm | Implícito en los 105.761 parámetros de la diapositiva |
| Forma de salida del modelo | `(N,)`, no `(N,1)` | `BCEWithLogitsLoss` no avisa si recibe `(N,1)` contra `(N,)`: hace broadcasting a `(N,N)` y entrena mal sin error |

## Entrenamiento

| Decisión | Valor | Motivo |
|---|---|---|
| Tamaño de lote | **128** | Indicación del profesor (69 iteraciones por época con 8.759 cortes de entrenamiento) |
| Función de pérdida | **las dos: normal primero, ponderada después** | El caso obliga a compararlas. La normal deja ver con nuestros ojos la trampa del desbalance (accuracy ~70 % sin aprender). `pos_weight = N0/N1`, calculado en código desde las filas de entrenamiento de cada fold (nunca copiado de un documento) |
| Optimizador | **Adam** | El más habitual por defecto; inercia y paso propio por parámetro |
| Learning rate | **0,0003** | En la prueba de memorización, 0,001 produjo varios tirones y un pico fuerte (pérdida 1,26 en la época 88). Algo más cauto, y el lote de 128 da pocos pasos por época, así que no debería quedarse corto |
| Épocas | **40 fijas, sin parada temprana** | Para esta primera mirada, ver la curva entera (cuándo empieza el sobreajuste). Se guarda igualmente el modelo de la mejor época según el AUC por paciente en validación |
| Semilla | 42 | Reproducibilidad |
| Aumentado de datos | ninguno por ahora | La base se mide limpia; se probará después como cambio aislado |

## Evaluación (provisional)

- **Por paciente**, no por corte: se promedian las probabilidades de los cortes de cada
  paciente (método `mean`) y se aplica el umbral **0,5**. El método de agregación y el umbral
  definitivos se decidirán con la validación, nunca con el test.
- **Validación:** primero un fold (el 0). El test no se toca hasta el final.
- **Se miden:** curvas de pérdida (entrenamiento y validación), AUC por paciente, curva ROC y
  matriz de confusión. Cada época se mide también sobre 200 pacientes de entrenamiento para
  detectar el sobreajuste.
- **Ruido:** con un solo fold (~219 pacientes) el AUC tiene un margen de ±0,08; cualquier
  diferencia menor entre dos configuraciones no es concluyente. Para comparar, repetir en varios folds.

## Comprobaciones hechas

- La red tiene exactamente 105.761 parámetros y las formas de cada capa coinciden con la diapositiva.
- **Prueba de memorización** (20 pacientes, 200 cortes, Adam 0,001, lote 32, 150 épocas): la pérdida
  baja de 0,6925 a 0,005 y la accuracy llega al 100 % sobre esos cortes. El código, la pérdida y el
  optimizador funcionan. No demuestra generalización.

## Primeros resultados de la red base (en el PC de la universidad, GPU AMD, 45 s por época)

Pérdida normal, AUC por paciente en validación (media de las probabilidades de los cortes, umbral 0,5):

| Fold | Mejor época | AUC (mejor) | AUC (época 40) | VP / VN / FP / FN | Sensibilidad |
|---|---|---|---|---|---|
| 0 | 30 | 0,602 | 0,600 | 3 / 152 / 3 / 61 | 5 % |
| 1 | 40 | 0,585 | 0,585 | 1 / 152 / 5 / 61 | 2 % |
| 2 | 38 | 0,587 | 0,508 | 0 / 160 / 1 / 59 | 0 % |
| 3 | 38 | 0,495 | 0,491 | 5 / 139 / 6 / 70 | 7 % |
| 4 | 15 | 0,605 | 0,554 | 0 / 157 / 0 / 62 | 0 % |
| **Media (5 folds)** | | **0,575** (desviación entre folds 0,045) | **0,548** (desviación 0,047) | | |

Con los 5 folds, el AUC medio es **0,575** si se toma la mejor época de cada fold (intervalo de confianza del 95 %:
0,52-0,63) y **0,548** si se toma la última (intervalo 0,49-0,61, que **incluye el 0,5**). La mejor época varía mucho de
un fold a otro (15, 30, 38, 38, 40) y elegirla es optimista. Cifra honesta: **≈ 0,55**, por encima del azar por poco
y sin que se pueda descartar que sea ruido.

Sumando los cinco folds (en la mejor época), con umbral 0,5: 9 de las 322 pacientes pCR detectadas (sensibilidad 2,8 %),
760 de las 775 no pCR bien clasificadas (especificidad 98,1 %), accuracy 70,1 %: la red con pérdida normal dice "no pCR"
a casi todas.

Fold 0 con pérdida ponderada (`pos_weight` = 2,4002): mejor AUC 0,578 (época 35), matriz 37 / 78 / 77 / 27
(sensibilidad 58 %, especificidad 50 %, accuracy 52,5 %).

**Qué se concluye:**
- La red base aprende una señal **débil**: AUC ≈ 0,55 de media, con una variación de ±0,05 entre folds
  (de 0,495 a 0,602). El fold 3 no muestra señal. Con esta forma de medir solo se detectarían mejoras grandes.
- **Sobreajuste:** el AUC de entrenamiento sube a 0,65-0,75 y el de validación se queda en 0,5-0,6. Con la pérdida
  ponderada, la pérdida de validación sube de 0,97 a 1,07 mientras la de entrenamiento baja.
- **Elegir la mejor época infla el resultado** (media 0,567 frente a 0,546 en la última época): la cifra honesta
  está cerca de 0,55.
- **Pérdida normal y umbral 0,5:** la red dice "no pCR" a casi todas (sensibilidad 0-7 %, accuracy ≈ proporción de
  negativos). El AUC es casi igual con las dos pérdidas; la ponderada solo mueve el punto de operación
  (sensibilidad 5 % → 58 %, especificidad 98 % → 50 %). El umbral definitivo hay que elegirlo con validación.
- La curva ROC es modesta en las dos pérdidas (algo por encima de la diagonal); para detectar la mitad de las pCR
  habría que aceptar ~30 % de falsos positivos.
- Referencia de datos: identificar solo la cohorte da AUC de 0,53-0,59 según el fold.

**Defectos conocidos a corregir:** la línea de "mejor red sin mirar la imagen" de las gráficas usa la proporción de
pCR del entrenamiento y no la de cada fold de validación; aviso inofensivo de NumPy ("array not writable").

## NOTA OFICIAL DE LA BASE (red del profesor, pérdida normal, 5 folds)

Calculada con `comparar.py base_normal`, métrica oficial (media del AUC por paciente en validación de las últimas 10 épocas):

| Fold | 0 | 1 | 2 | 3 | 4 | **Media** |
|---|---|---|---|---|---|---|
| Media ult. 10 épocas | 0,5982 | 0,5634 | 0,5385 | 0,4898 | 0,5505 | **0,5481** |

Desviación entre folds 0,0395; **intervalo de confianza del 95 %: 0,499 a 0,597** (roza el 0,5). Con la última época: 0,5476
(0,489-0,606); con la mejor época de cada fold (optimista): 0,5749 (0,518-0,631). **Esta es la cifra contra la que se compara
cualquier cambio.**

## Cómo se comparan configuraciones (decidido el 2026-10-02)

El ruido de la medida es del tamaño de lo que queremos detectar (±0,05 entre folds y fluctuaciones de ±0,02-0,04 entre
épocas), así que se fija una forma de comparar que no haga trampa:

- **Métrica oficial: la media del AUC por paciente en validación de las últimas 10 épocas** (31-40), no la mejor época.
  Es estable y no elige nada mirando la validación. En los folds 0-3 de la base da 0,598 / 0,563 / 0,539 / 0,490
  (media 0,547), casi igual que la última época (0,546) y por debajo de "la mejor época" (0,567).
- **Siempre en los mismos 5 folds** y con comparación **emparejada** (diferencia fold a fold), que cancela el ruido del
  reparto: un fold difícil lo es igual para los dos modelos.
- **Semillas repetidas solo para los finalistas** (2-3 semillas), antes de dar una mejora por buena. Pruebas normales con
  una semilla (semilla 42). Cada configuración cuesta 5 entrenamientos (≈ 2,5 h de GPU).
- La pérdida de validación se mira como dato complementario, no para decidir.
- Una mejora solo cuenta si el intervalo de confianza de la diferencia emparejada no incluye el 0.
- Herramienta: `comparar.py` (lee los `*_epocas.csv`; sirve también para los resultados ya obtenidos de la base).
- **Copias de seguridad en git:** `referencia/` guarda los CSV de la base (folds 0-3 con pérdida normal y fold 0 con la
  ponderada), reconstruidos de los registros de entrenamiento porque los originales se borraron del PC por error. Se
  comprobó que el máximo de cada CSV coincide con la mejor época registrada. `comparar.py` busca primero en `resultados/`
  y luego en `referencia/`. Hábito: nunca borrar con comodines (`rm base_normal_fold*`); copiar a `referencia/` lo importante.

## Plan de exploración (decidido el 2026-10-03) y experimento 1: aumentado geométrico

**Plan con límite de tiempo:** dos experimentos sobre la base, de uno en uno, medidos en los mismos 5 folds con la métrica oficial
y la comparación emparejada; después se da la exploración por cerrada y se pasa a lo obligatorio (umbral, agregación por paciente,
comparación de pérdidas, test una sola vez, app, informe, defensa). Un cambio solo se adopta si el intervalo de confianza de la
diferencia emparejada excluye el 0; si no, nos quedamos con la base.

1. **Aumentado geométrico** (`--aumentado geometrico`): volteo horizontal (p = 0,5), rotación de hasta ±15°, desplazamiento de
   hasta ±10 % del lado y escala entre 0,9 y 1,1; una sola transformación por corte aplicada a las tres fases a la vez, relleno
   con 0, sin cambios de brillo ni de color, solo en entrenamiento. **Por qué primero:** el problema diagnosticado es el
   sobreajuste (AUC de entrenamiento 0,65-0,75 frente a ~0,55 en validación), y el aumentado es el remedio más directo; además
   es el más barato (≈ 2,5-3 h de GPU, sin añadir parámetros). Verificado en código: las fases no se desalinean, el volteo
   puro equivale a `torch.flip`, el rango [0,1] se conserva y es reproducible con la misma semilla. Resultados en
   `base_normal_aug_fold<k>_*`; se compara con `python comparar.py base_normal base_normal_aug`.
   **RESULTADO del experimento 1 (aumentado geométrico, pérdida normal, 5 folds; métrica oficial):**

   | Fold | 0 | 1 | 2 | 3 | 4 | Media |
   |---|---|---|---|---|---|---|
   | Base | 0,598 | 0,563 | 0,539 | 0,490 | 0,551 | 0,548 |
   | Aumentado | 0,537 | 0,497 | 0,528 | 0,455 | 0,582 | 0,520 |
   | Diferencia | −0,062 | −0,066 | −0,011 | −0,035 | +0,032 | **−0,028** |

   Diferencia emparejada: **−0,028, IC95 de −0,078 a +0,022: no concluyente** (incluye el 0, con tendencia a peor).
   **No se adopta; seguimos con la base.** Lectura: el aumentado sí frena la memorización (AUC de entrenamiento en la época 40:
   0,555 frente a 0,640 de la base) pero no mejora la validación. Además, el AUC de validación tras **una sola época** ya es
   0,52 en las dos configuraciones, y tras 40 épocas de entrenamiento solo sube a unos 0,55: casi todo lo que mide el AUC
   de validación está ya presente al empezar, y entrenar añade muy poco que generalice. El problema no parece ser solo
   sobreajuste de una señal que existe.

2. **conv + conv + pool** (idea del profesor): dos convoluciones 3×3 + ReLU por bloque, mismos canales (16-32-64-128), pool al
   final; ≈ 301.800 parámetros (casi 3× la base), campo receptivo 76 px, ≈ 4 h de GPU estimadas. Pendiente de implementar.

## Pendiente de decidir

Agregación y umbral definitivos · cuántos folds evaluar · aumentado de datos · primer cambio sobre la
base (conv+conv+pool) · inicialización (por ahora, la de PyTorch por defecto) · método de comparación
final entre pérdida normal y ponderada.
