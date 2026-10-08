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

   **Análisis por cohorte del aumentado** (`analizar_cohortes.py base_normal_aug`, 1.097 pacientes, probabilidades de la mejor época):
   AUC global 0,531 (0,494-0,570); dentro de duke 0,524, spy1 0,543, spy2 0,528 (todos los IC incluyen 0,5); estratificado 0,528
   (0,487-0,570); referencia "solo cohorte" 0,523. Probabilidad media por cohorte casi plana (0,282 / 0,275 / 0,294 frente a pCR real
   21 / 25 / 32 %). **La red no explota la cohorte, pero tampoco muestra señal del tumor.** Prueba débil: en 3 de 5 folds la mejor
   época fue la 1-4 (red casi sin entrenar). Pendiente: guardar también las probabilidades de la última época.

2. **BatchNorm tras cada convolución** (`--batchnorm`, decidido el 2026-10-07; sustituye a conv + conv como siguiente experimento).
   **Por qué:** el diagnóstico apunta a que la red *aprende mal*, no a que le falte capacidad: AUC de entrenamiento de solo 0,64 tras
   40 épocas, meseta inicial, AUC que salta entre épocas y gradiente 12 veces más débil en la primera convolución. BatchNorm
   estabiliza el entrenamiento y mejora el paso del gradiente; es el bloque canónico de los apuntes (Conv → BN → ReLU). Las
   convoluciones pasan a no llevar sesgo: 106.001 parámetros (verificado). Todo lo demás igual que la base (Adam 3e-4, lote 128,
   40 épocas, pérdida normal, sin aumentado) para que la comparación sea limpia. Ficheros `bn_normal_fold<k>_*`; se compara con
   `python comparar.py base_normal bn_normal`.
   **Además:** desde ahora `entrenar.py` guarda las probabilidades de la última época y la media de las 10 últimas
   (`*_probs_val_final.csv`) e imprime el AUC del **ensemble temporal** (promediar las predicciones de las 10 últimas épocas), que
   no cuesta GPU extra. `analizar_cohortes.py --probs ultima|media10` usa esas probabilidades.

   **RESULTADO del experimento 2 (BatchNorm, pérdida normal, 5 folds; métrica oficial):**

   | Fold | 0 | 1 | 2 | 3 | 4 | Media | Desv. |
   |---|---|---|---|---|---|---|---|
   | Base | 0,598 | 0,563 | 0,539 | 0,490 | 0,551 | 0,548 | 0,040 |
   | BatchNorm | 0,549 | 0,556 | 0,549 | 0,530 | 0,584 | 0,554 | 0,019 |
   | BatchNorm, ensemble últimas 10 épocas | 0,558 | 0,565 | 0,559 | 0,533 | 0,588 | 0,561 | 0,020 |

   - BatchNorm − base: **+0,006, IC95 de −0,039 a +0,050: no concluyente** → no se adopta como mejora del AUC.
     Sí reduce a la mitad la variación entre folds y aprende mucho más rápido.
   - **Sobreajuste claro:** AUC de entrenamiento 0,99-1,00 al final; pérdida de validación de ~0,60 a 1,2-1,8 (picos de
     hasta 3,5); el AUC de validación tiene su pico en épocas intermedias (13-17 en tres folds).
   - **Ensemble temporal** (promediar las predicciones de las 10 últimas épocas) frente a la media de sus AUC:
     **+0,007 en los 5 folds, IC95 de +0,003 a +0,011** → mejora pequeña pero consistente y gratuita. Se usará en el modelo final.
   - **Matrices de confusión (mejor época, umbral 0,5) muy distintas entre folds** (sensibilidad del 6 % al 72 %): no reflejan
     la dificultad del fold sino dónde caen las probabilidades respecto al 0,5 en esa época, que oscila de una época a otra
     (p. ej., fold 0: sensibilidad 3 % en la época 20 y 94 % en la 21). Sumando los 5 folds: sensibilidad 45 %, especificidad
     66 %, accuracy 60 %, precisión 36 % (prevalencia 29 %). **El umbral 0,5 no es utilizable: hay que elegirlo con validación.**
   - Copias de los registros en `referencia/bn_normal_fold<k>_epocas.csv`.

3. **conv + conv + pool** (idea del profesor; queda como opcional tras el análisis): dos convoluciones 3×3 + ReLU por bloque, mismos canales (16-32-64-128), pool al
   final; ≈ 301.800 parámetros (casi 3× la base), campo receptivo 76 px, ≈ 4 h de GPU estimadas. Pendiente de implementar.

## Datos clínicos de patients.csv (decidido el 2026-10-07)

**Permiso del profesor:** se puede usar `patients.csv` entero para entrenar, y en la app y en la defensa **cada muestra llega con
las mismas columnas**. Por tanto las variables clínicas pueden ser **entrada del modelo**, no solo ayuda durante el entrenamiento.

**Tarea auxiliar (HR y HER2 como objetivo extra, `--aux`):** implementada y probada (reproducible; sin `--aux` todo da igual que
antes), pero **aparcada sin lanzar**. Servía para meter el subtipo en la red cuando creíamos que en la app solo habría imagen; si el
subtipo llega como dato, que la CNN lo aprenda a deducir es redundante. Lo que interesa de la imagen ahora es lo que el subtipo NO explica.

**Análisis de las columnas** (solo las 1.097 pacientes de entrenamiento; el test no se mira):
- pCR por subtipo: **HR+/HER2− 14 %**, HR+/HER2+ 34 %, triple negativo 38 %, **HR−/HER2+ 57 %**. Es la señal más fuerte.
- AUC de cada columna sola (0,5 = nada; por debajo, relación inversa): HRposHER2neg 0,348 (≈ 0,65 al revés), HER2 0,586,
  edad 0,458, tum_vol 0,454. Las de adquisición (n_xy, n_z, n_times, slice_thick, xy_spacing) y la raza ≈ 0,50-0,55.
- Datos que faltan: menopause en **toda** spy1 y 30 de spy2; HR 2, HER2 3, edad 3, tum_vol 1.
- `tum_vol`: mediana 6 en Duke frente a 15 en I-SPY (en Duke la máscara es incompleta): sesgo entre cohortes para el informe.
- Las imágenes son recortes alrededor del tumor (`sraw…ecol`) llevados a 256×256: la CNN no ve el tamaño real del tumor.

**Variables elegidas** (`clinicas.py`): **HR, HER2, HR×HER2, edad y log(volumen tumoral)**. No se usan: pCR/split/test (respuesta y
reparto), dataset y adquisición (delatan cohorte o escáner), raza (la GUIA lo considera error metodológico), menopause (su hueco
delata la cohorte spy1; la edad recoge lo mismo), las columnas de subtipo (se deducen de HR y HER2) y los límites del recorte.
Huecos → media de entrenamiento; edad y log(volumen) estandarizados con medias de entrenamiento. La app usará los mismos valores.

**RESULTADO: modelo solo clínico** (regresión logística, `python combinar.py`, mismos 5 folds, AUC por paciente):

| Variables | Fold 0 | 1 | 2 | 3 | 4 | Media |
|---|---|---|---|---|---|---|
| Subtipo (3 grupos) | 0,709 | 0,647 | 0,630 | 0,675 | 0,647 | 0,662 |
| HR, HER2, HR×HER2 | 0,723 | 0,649 | 0,652 | 0,703 | 0,675 | 0,680 |
| + edad | 0,723 | 0,694 | 0,662 | 0,705 | 0,685 | 0,694 |
| **+ edad + log(volumen)** | **0,757** | **0,712** | **0,679** | **0,712** | **0,675** | **0,707** (IC95 ±0,041) |
| (+ cohorte, solo como referencia, no se usa) | 0,755 | 0,710 | 0,685 | 0,732 | 0,688 | 0,714 |

- **0,707 frente a 0,561 de la mejor CNN.** Cada variable suma en todos los folds (+ edad +0,014; + volumen +0,013).
- Coeficientes (todo el entrenamiento): HR −1,35 (odds ×0,26), HER2 +0,88 (×2,40), HR×HER2 +0,21, edad −0,25 por desviación (×0,78),
  log(volumen) −0,30 por desviación (×0,74).
- La cohorte apenas añade (+0,007): el modelo clínico no depende de ella.

**¿Y con todas las columnas?** (`python comparar_variables.py`; mismas 5 variables + las columnas añadidas tal cual,
estandarizadas, con indicador de hueco; pCR, split y test nunca)

| Opción | Variables | AUC val | AUC entreno | Diferencia con las 5 (IC95) |
|---|---|---|---|---|
| **Las 5 de `clinicas.py`** | 5 | **0,707** | 0,714 | — |
| TODAS las columnas | 36 | 0,704 | 0,743 | −0,003 (−0,016 a +0,011) no concluyente |
| Todas menos la raza | 32 | 0,706 | 0,742 | −0,001 (−0,014 a +0,012) no concluyente |
| + adquisición (escáner) | 12 | 0,711 | 0,726 | +0,004 (−0,008 a +0,015) no concluyente |
| + coordenadas del recorte | 11 | 0,719 | 0,731 | +0,012 (+0,003 a +0,021) mejora |
| + dimensiones del recorte en mm | 8 | 0,714 | 0,726 | +0,007 (−0,007 a +0,022) no concluyente |
| + dimensiones del recorte en píxeles | 8 | 0,715 | 0,725 | +0,008 (+0,003 a +0,012) mejora |
| + menopausia | 7 | 0,702 | 0,717 | −0,005 (−0,013 a +0,003) no concluyente |
| + cohorte | 7 | 0,715 | 0,724 | +0,008 (−0,005 a +0,020) no concluyente |
| + raza | 9 | 0,703 | 0,716 | −0,004 (−0,009 a +0,002) no concluyente |
| + indicadores de subtipo | 11 | 0,704 | 0,714 | −0,003 (−0,008 a +0,003) no concluyente |

- **Todas las columnas no mejora** (0,704) y el AUC de entrenamiento sube de 0,714 a 0,743: las 31 columnas extra sirven
  para memorizar, no para predecir.
- **Coordenadas del recorte:** la ganancia sale de las dimensiones, no de la posición (centro del recorte −0,003; lado de la
  mama 0,000). Y depende de la unidad: en píxeles pasa la prueba, en milímetros no; el diámetro mayor en mm da −0,001 y el log
  de las 3 dimensiones +0,004 (no concluyentes). Un efecto biológico del tamaño no dependería de contar píxeles o milímetros;
  los píxeles dependen de la resolución del escáner, así que apunta más a la máquina que al tumor. El log(volumen) ya recoge
  el tamaño (correlación 0,8 con el diámetro mayor).
- Se han probado unas 15 opciones: que alguna "gane" por azar es esperable (la misma trampa que la mejor época).
- **Decisión: se mantienen las 5 variables.** Además de los números: pCR es la respuesta, split/test el reparto, la raza
  la excluye la GUIA, y cohorte y escáner dicen de qué hospital viene la imagen, no cómo es el tumor.

**Plan: combinar la CNN con lo clínico ("stacking", `combinar.py`).** Una regresión logística con las 5 variables clínicas + el logit
que la CNN da a cada corte, ajustada **por corte** (en la app y en la defensa llega un solo corte con los datos de su paciente). Para
ajustarla se usan las predicciones de la CNN sobre la validación de cada fold (cada corte, predicho por una CNN que no lo vio). Se
comparan, emparejados por fold, solo clínico / solo CNN / combinado. Matiz para el informe: las CNN de los otros folds sí vieron al
fold que se evalúa; es la práctica habitual del stacking, pero no es una validación perfectamente limpia.
Siguiente: `python combinar.py bn_normal` en el PC (usa los `bn_normal_fold<k>_probs_val_final.csv` que ya existen, sin entrenar).

**RESULTADO (2026-10-08): ¿añade algo la CNN con BatchNorm (media de las 10 últimas épocas) a los datos clínicos?**

| AUC por paciente | Fold 0 | 1 | 2 | 3 | 4 | Media |
|---|---|---|---|---|---|---|
| Solo clínico | 0,757 | 0,712 | 0,679 | 0,712 | 0,675 | 0,707 |
| Solo CNN | 0,558 | 0,565 | 0,559 | 0,533 | 0,588 | 0,561 |
| CNN + clínico | 0,759 | 0,710 | 0,691 | 0,713 | 0,682 | **0,711** |

- CNN + clínico − solo clínico: **+0,004, IC95 −0,003 a +0,011: no concluyente** (positivo en 4 de 5 folds, pero diminuto).
  La regresión da a la CNN un peso muy pequeño (logit_cnn +0,071, odds ×1,07); los coeficientes clínicos casi no cambian.
  Por corte (lo que ve la app): 0,707 → 0,710.
- **Dónde acierta la CNN** (`analizar_cohortes.py bn_normal --probs media10`, 1.097 pacientes):
  - Por subtipo: dentro de HR+/HER2− 0,508 (nada; es el 41 % de las pacientes), HR+/HER2+ 0,603 (0,507-0,692), HR−/HER2+ 0,545,
    triple negativo 0,563 (0,504-0,627). **Estratificado por subtipo 0,545 (0,501-0,589)**, casi igual que el global (0,557):
    la poca señal de la CNN **no es el subtipo** (sus probabilidades medias por subtipo apenas siguen la pCR real). No es
    redundante, es débil.
  - Por tamaño: pequeño 0,579, mediano 0,551, grande 0,532 (intervalos solapados); estratificado 0,555. No depende del tamaño.
  - Por cohorte: estratificado 0,563, igual que el global: el AUC no viene de reconocer la cohorte. Sus probabilidades medias sí
    siguen el orden de la pCR de cada cohorte (0,318 / 0,334 / 0,386 frente a 21 / 25 / 32 %). spy1 0,421 (104 pacientes, IC 0,29-0,55).
- **Conclusión:** la CNN actual tiene una señal propia pero muy débil, y apenas mejora el modelo clínico. Mejor resultado hasta
  ahora: CNN + clínico 0,711. El problema no es que la CNN repita el subtipo, sino que lo que aprende generaliza poco (memoriza:
  AUC de entrenamiento 1,0).

## Experimento 3: frenar la memorización con aumentado sobre BatchNorm (decidido el 2026-10-08)

**Por qué:** la CNN con BatchNorm memoriza (AUC de entrenamiento 1,0, validación 0,56) y su señal casi no añade nada a los datos
clínicos. El problema medido es que lo que aprende no generaliza. **Cambio único:** `--batchnorm --aumentado geometrico` (el mismo
aumentado del experimento 1: volteo, rotación ±15°, desplazamiento ±10 %, escala 0,9-1,1, igual en las tres fases). Todo lo demás
como bn_normal.
**Por qué el aumentado antes que el weight decay:** ataca la memorización de forma directa (cada época la imagen es algo distinta);
con BatchNorm el weight decay de las convoluciones actúa de forma poco intuitiva (sus pesos se pueden escalar sin cambiar la
salida, así que el freno se convierte sobre todo en un cambio de la velocidad de aprendizaje); y ya está implementado y probado.
En el experimento 1 el aumentado no ayudó, pero entonces la red no memorizaba (AUC de entrenamiento 0,64): no había nada que frenar.
**Cómo se mide:** con la CNN sola (`comparar.py bn_normal bn_normal_aug`) y, sobre todo, con el modelo combinado
(`combinar.py bn_normal_aug` frente a los 0,711 de `combinar.py bn_normal`), emparejado por fold.

**RESULTADO del experimento 3 (5 folds):**

| Fold | 0 | 1 | 2 | 3 | 4 | Media |
|---|---|---|---|---|---|---|
| CNN sola, BatchNorm (oficial) | 0,549 | 0,556 | 0,549 | 0,530 | 0,583 | 0,554 |
| CNN sola, BatchNorm + aumentado | 0,509 | 0,523 | 0,544 | 0,529 | 0,504 | 0,522 |
| CNN + clínico, con BatchNorm | 0,759 | 0,710 | 0,691 | 0,713 | 0,682 | 0,711 |
| CNN + clínico, con BatchNorm + aumentado | 0,758 | 0,709 | 0,677 | 0,712 | 0,675 | 0,706 |

- CNN sola: −0,032, IC95 −0,071 a +0,007, no concluyente, pero **peor en los 5 folds**. Ensemble de las 10 últimas épocas 0,524 (antes 0,561).
- CNN + clínico − solo clínico: −0,0006 (IC95 −0,002 a +0,001). La regresión da a esta CNN peso cero (logit_cnn −0,010).
- **El freno sobre la memorización sí funciona** (AUC de entrenamiento final 0,75-0,85 en vez de 0,99-1,0), **pero la validación no
  mejora**. El mejor AUC de validación aparece muy pronto en varios folds (épocas 1-3 en los folds 0, 2 y 4) y luego baja, como en el
  experimento 1: lo poco que generaliza se capta al principio y el resto del entrenamiento aprende cosas que no sirven fuera.
- **No se adopta.** La mejor CNN sigue siendo bn_normal (con el ensemble de las 10 últimas épocas). La memorización no era lo que
  tapaba una señal fuerte: la señal de la imagen, con esta red, es débil.

## Experimento 4: canales de realce y lavado (decidido el 2026-10-08; último intento de la parte de imagen)

**Por qué:** frenar la memorización no destapó señal (experimento 3): la red no encuentra en la imagen información que generalice.
Se le da ya calculada la información que, según la GUIA, contiene la señal: **EARLY−PRE (realce)** y **EARLY−LATE (lavado)**.
**Cambio único:** `--batchnorm --realce` (sin aumentado), todo lo demás como bn_normal. Las restas se calculan **dentro de la red**
(`RedBase(realce=True)`), así que la app las hará exactamente igual. 106.289 parámetros (+288 en la primera convolución).
Verificado: la primera convolución recibe PRE, EARLY, LATE, EARLY−PRE y EARLY−LATE; reproducible; sin `--realce` todo da igual que antes.
**Cómo se mide:** `comparar.py bn_normal bn_normal_realce` y `combinar.py bn_normal_realce` (frente a 0,711).
**Límite acordado:** salga lo que salga, después se pasa a lo obligatorio (pérdidas, umbral, calibración, modelo final, test, app,
informe, diapositivas).

## Ideas guardadas para probar en el futuro (2026-10-08)

Salen de analizar una propuesta de arquitectura externa. Se probarán **de una en una**, medidas con el AUC del modelo combinado
(imagen + datos clínicos) en los mismos 5 folds. Por orden de interés:

1. **Canales de realce y lavado:** entrada de 5 canales = PRE, EARLY, LATE + EARLY−PRE (realce) + EARLY−LATE (lavado). La GUIA
   dice que la señal está en la diferencia entre fases, y el lavado es clínicamente relevante (17 % de las pacientes). Es
   información de la imagen que no está en el subtipo ni en el volumen. Solo cambia la primera capa (~300 parámetros más). La app
   tendría que calcular las restas igual que el entrenamiento.
2. **conv+conv con BatchNorm** (ya estaba en la lista por el profesor; ~294.000 parámetros, campo receptivo 76 px).
3. **Media + máximo en el pooling global** (256 números en vez de 128): una zona pequeña muy marcada puede pesar.
4. **Cabeza más simple:** Dropout → Linear directo a 1 logit, en vez de 128 → 64 → 1.

Descartadas de esa propuesta: attention pooling por paciente (en la defensa llega un solo corte), early stopping sobre validación
(resultado optimista), escalado de intensidad (altera el realce), GroupNorm (nuestros lotes son de 128). Su premisa "no hay
máscara y el tumor ocupa pocos píxeles" no se cumple aquí: las imágenes son recortes del tumor llevados a 256×256.

## Pendiente de decidir

Agregación y umbral definitivos · cuántos folds evaluar · aumentado de datos · primer cambio sobre la
base (conv+conv+pool) · inicialización (por ahora, la de PyTorch por defecto) · método de comparación
final entre pérdida normal y ponderada · si la CNN aporta algo sobre lo clínico (y, si no, qué cambiar en ella).
