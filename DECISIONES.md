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

**RESULTADO del experimento 4 (2026-10-14, 5 folds):**

| Fold | 0 | 1 | 2 | 3 | 4 | Media |
|---|---|---|---|---|---|---|
| CNN sola, BatchNorm (oficial) | 0,549 | 0,556 | 0,549 | 0,530 | 0,583 | 0,554 |
| CNN sola, BatchNorm + realce y lavado (oficial) | 0,551 | 0,580 | 0,569 | 0,507 | 0,603 | 0,562 |
| CNN sola, ensemble 10 últimas, BatchNorm | 0,558 | 0,565 | 0,559 | 0,533 | 0,588 | 0,561 |
| CNN sola, ensemble 10 últimas, + realce y lavado | 0,562 | 0,592 | 0,590 | 0,511 | 0,611 | 0,573 |
| CNN + clínico, BatchNorm | 0,759 | 0,710 | 0,691 | 0,713 | 0,682 | 0,711 |
| CNN + clínico, BatchNorm + realce y lavado | 0,756 | 0,719 | 0,687 | 0,712 | 0,683 | 0,712 |

- CNN sola: **+0,008, IC95 −0,016 a +0,033: no concluyente** (mejora en 4 de 5 folds; el fold 3 baja de 0,530 a 0,507). Ensemble +0,013.
  Es el primer cambio de la parte de imagen que sube algo, y el AUC de validación ya no se hunde con el entrenamiento (en los folds
  1, 2 y 4 sigue subiendo hasta el final). La memorización vuelve (AUC de entrenamiento 0,99-1,0).
- CNN + clínico: 0,712 frente a 0,711 (+0,001; por fold −0,003 / +0,010 / −0,004 / 0,000 / +0,001). Frente a solo clínico, +0,005
  (IC95 −0,001 a +0,011, no concluyente). La regresión da a esta CNN un peso algo mayor (logit_cnn +0,097 frente a +0,071).
- **No se adopta** por la regla (el intervalo incluye el 0). En el modelo combinado las dos CNN son equivalentes. Lectura para el
  informe: dar el realce y el lavado ya calculados ayuda un poco a la CNN sola, pero no añade información nueva sobre los datos clínicos.
- **Fin de la exploración** de la parte de imagen, según el límite acordado.

## REORIENTACIÓN tras el correo del profesor "Pistas caso de uso" (2026-10-14)

- **El modelo principal es la CNN solo con imagen** (PRE, EARLY, LATE). Las variables de patients.csv quedan **solo como
  comparación**: el modelo clínico (0,707) y el combinado CNN + clínico (0,711) pasan a ser referencias para el informe, no el
  modelo final. La app no necesita campos clínicos.
- **Objetivo realista:** con cortes 2D y tres cohortes, un AUC por paciente en test de **0,60 a 0,70** es un buen trabajo
  (Fridman et al. 2025, mismo test: 0,72). Nuestra CNN está en ~0,56-0,57 de media de las últimas épocas.
- **Entrenamiento:** criterio de parada (early stopping) y época elegida documentados; sobreajuste controlado y demostrado con las
  curvas de entrenamiento y validación juntas y la diferencia de AUC en la época elegida; descartar modelos sobreentrenados. Las redes
  con BatchNorm tal como están (AUC de entrenamiento → 1, pérdida de validación disparada) son el ejemplo "MAL" del correo.
- Pendientes que marca: pérdida normal frente a ponderada (qué le pasa a la sensibilidad), umbral con criterio clínico explícito,
  test completo una vez (matriz, AUC con IC, precision/recall/F1 por clase, F1 equilibrada, comparación con "siempre no pCR"),
  resultados por cohorte, calibración, casos fallados con imágenes, contraste con la bibliografía y leer el artículo de Fridman 2025.

**Decisiones de la nueva hoja de ruta (2026-10-14, ver `HOJA_DE_RUTA.md`):**
1. Early stopping sobre el **AUC por paciente de validación**, paciencia **10** épocas, máximo **60**; se anotan época elegida,
   AUC de validación, AUC de entrenamiento y su diferencia.
2. Modelo final = **los 5 modelos de los folds juntos** (promedio de probabilidades).
3. Umbral: uso previsto = **desescalar**; el error grave es el falso positivo → priorizar **especificidad y precisión de pCR**
   (como Fridman: especificidad 0,95, precisión 0,70, sensibilidad 0,27), comparado con máximo de F1 y F1 equilibrada.

## Fase 1 hecha: early stopping (2026-10-14)

- `parada.py`: la regla, en un solo sitio. Se vigila el AUC por paciente de validación suavizado con una **media móvil
  centrada de 3 épocas**; la época elegida es la de su máximo; se para tras **10** épocas sin superarlo o a las **60**.
  En la época elegida se anota la diferencia AUC de entrenamiento − validación; **más de 0,15 = sobreentrenado**.
  Por qué la media móvil: simulada sobre los registros de BatchNorm, elegir el máximo de una sola época da 0,601 ± 0,025, y
  con la media móvil 0,563 ± 0,033. La diferencia es un pico de suerte (p. ej., fold 1: la época 13 da 0,607, pero con sus
  vecinas 0,538). El AUC de una época oscila ±0,04 por puro ruido.
- `entrenar.py`: early stopping con `--paciencia` (0 = sin parada) y `--epocas` como máximo (60); guarda pesos y
  probabilidades de la **época elegida** (`*_probs_val.csv`, `*.pt`); el resumen trae época elegida, AUC de validación (y su
  media móvil), AUC de entrenamiento, diferencia y si está sobreentrenado. `--lr` distinto de 3e-4 añade `_lr<valor>` al nombre.
- `comparar.py`: la métrica pasa a ser el **AUC de validación en la época elegida, media ± desviación de los 5 folds**, con
  la diferencia y los folds sobreentrenados; aplica `parada.py` también a los registros antiguos.
- **Referencia con el nuevo protocolo** (registros de 40 épocas; todos los folds paran antes):

  | Configuración | AUC val en la época elegida | Diferencia media | Folds sobreentrenados |
  |---|---|---|---|
  | Base, sin BatchNorm (folds 0-3) | 0,528 ± 0,057 | −0,031 | 0 de 4 (casi no aprende: elige la época 1 en dos folds) |
  | **BatchNorm** | **0,563 ± 0,033** | **+0,218** | **4 de 5** |

  Conclusión: la red con BatchNorm aprende pero memoriza demasiado deprisa. La fase 2 parte de ella y busca frenar la velocidad.

## Fase 2 · experimento 2.1: learning rate 0,0001 (2026-10-14)

Cambio único sobre `bn_normal`: `--lr 1e-4` (antes 3e-4), con el early stopping nuevo. Ficheros `bn_normal_lr0.0001_*`.

| Fold | 0 | 1 | 2 | 3 | 4 | Media |
|---|---|---|---|---|---|---|
| AUC val en la época elegida, lr 3e-4 | 0,529 | 0,538 | 0,602 | 0,552 | 0,595 | 0,563 ± 0,034 |
| AUC val en la época elegida, lr 1e-4 | 0,553 | 0,565 | 0,593 | 0,568 | 0,557 | **0,567 ± 0,016** |
| Diferencia train − val, lr 3e-4 | 0,21 | 0,23 | 0,29 | 0,29 | 0,06 | 0,218 (4 de 5 sobreentrenados) |
| Diferencia train − val, lr 1e-4 | 0,03 | 0,19 | 0,16 | 0,20 | 0,14 | **0,143 (3 de 5)** |
| Época elegida / entrenadas, lr 1e-4 | 3/14 | 16/27 | 18/29 | 24/35 | 11/22 | 127 épocas |

- AUC: +0,004, IC95 −0,030 a +0,038, **no concluyente**; la desviación entre folds se reduce a la mitad.
- Sobreajuste: la diferencia media baja de 0,22 a 0,14 y los sobreentrenados de 4 a 3 (y por menos margen).
- Las curvas: el AUC de entrenamiento sube despacio (0,50 → 0,90) pero el de validación se queda plano (0,50-0,60) todo el
  entrenamiento; la pérdida de validación sube desde la época ~10. Frenar reduce la memorización, pero no hace aparecer señal.
- **Se adopta lr 1e-4** (mismo AUC, menos sobreajuste y menos variación: mejor en el criterio de sobreajuste del profesor).
- Siguiente (2.2): lr 1e-4 + canales de realce y lavado (`--realce`), lo único que había subido la validación.

## Fase 2 · experimento 2.2: lr 0,0001 + realce y lavado (2026-10-14)

Cambio único sobre 2.1: `--realce`. Ficheros `bn_normal_realce_lr0.0001_*`.

| Fold | 0 | 1 | 2 | 3 | 4 | Media |
|---|---|---|---|---|---|---|
| AUC val en la época elegida, 2.1 (lr 1e-4) | 0,553 | 0,565 | 0,593 | 0,568 | 0,557 | 0,567 ± 0,016 |
| AUC val en la época elegida, 2.2 (+ realce) | 0,560 | 0,559 | 0,565 | 0,528 | 0,572 | 0,557 ± 0,017 |
| Época elegida / entrenadas, 2.2 | 3/14 | 10/21 | **1**/12 | 29/40 | **1**/12 | 99 épocas |
| Diferencia train − val, 2.2 | 0,02 | 0,09 | −0,07 | 0,31 | −0,06 | 1 de 5 sobreentrenado |

- −0,011, IC95 −0,040 a +0,018: **no concluyente**, tendencia a peor. **No se adopta**; la mejor sigue siendo 2.1 (lr 1e-4).
- **Observación clave:** en 2 folds la época elegida es la **1** (AUC de entrenamiento 0,49-0,51: la red aún no distingue nada) y
  la validación ya da 0,56-0,57; después no mejora nunca. Se repite en casi todos los registros: el AUC de validación de la época
  1 ronda 0,55 y entrenar solo lo mueve dentro del ruido. La poca señal (~0,56) aparece sin aprendizaje específico, probablemente
  por rasgos globales simples; lo que la red aprende después no generaliza.
- El "ok" de esos folds es engañoso: no están bien entrenados, es que **no han aprendido**. Se añadirá a `comparar.py` un estado
  "sin aprendizaje" para distinguirlos.
- Las matrices de confusión con umbral 0,5 dicen "no pCR" a casi todas (accuracy ≈ 71 % = siempre "no pCR"): las probabilidades
  rondan 0,3, por debajo de 0,5. Es esperable y es la razón de elegir el umbral en la fase 4.
- Siguiente propuesto (2.3): **red estrecha** (8-16-32-64 filtros) con lr 1e-4: menos capacidad para memorizar.

## Fase 2 · experimento 2.3: red estrecha (decidido el 2026-10-14)

Cambio único sobre 2.1 (BatchNorm, lr 1e-4): `--filtros 8`, es decir, bloques de **8-16-32-64** filtros en vez de 16-32-64-128.
**28.873 parámetros** (verificado) frente a 106.001: menos capacidad para memorizar detalles de cada paciente. Es la dirección
contraria a conv+conv. Ficheros `bn_normal_f8_lr0.0001_*`; se compara con `python comparar.py bn_normal_lr0.0001 bn_normal_f8_lr0.0001`.
Además, `parada.py` y `comparar.py` marcan un fold como **SIN APRENDIZAJE** si en la época elegida el AUC de entrenamiento no llega
a 0,60 (la red aún no distingue ni a sus pacientes de entrenamiento), para no confundirlo con un fold bien entrenado.

## Resultados de la noche del 14 al 15 de octubre (sobre 2.1: BatchNorm, lr 1e-4)

| Configuración | AUC val en la época elegida | Diferencia con 2.1 (IC95) | Estados (de 5) |
|---|---|---|---|
| **2.1 referencia** | **0,567 ± 0,016** | — | 1 sin aprendizaje, 3 sobreentrenados, 1 ok |
| 2.3 red estrecha (8-16-32-64) | 0,541 ± 0,025 | −0,026 (−0,066 a +0,014) | 3 sin aprendizaje, 1 sobreentrenado, 1 ok |
| 2.4 aumentado geométrico | 0,549 ± 0,022 | −0,018 (−0,051 a +0,015) | 4 sin aprendizaje, 1 ok |
| 3.1 pérdida ponderada | 0,554 ± 0,012 | −0,013 (−0,044 a +0,018) | 2 sin aprendizaje, 2 sobreentrenados, 1 ok |

- Ninguno mejora a 2.1 (todos no concluyentes, todos en contra). **2.1 sigue siendo la mejor.**
- **Frenos:** sin freno (lr 3e-4) la red memoriza; con freno fuerte (red estrecha o aumentado, con lr 1e-4) no aprende
  ("sin aprendizaje" en 3-4 folds); el punto medio (2.1) es lo mejor. El techo (~0,57) no depende de cuánto se frena.
- **Fase 3, pérdida normal frente a ponderada** (matrices de los 5 folds sumadas, umbral 0,5, época elegida):

  | | Normal (2.1) | Ponderada (3.1) |
  |---|---|---|
  | AUC | 0,567 | 0,554 |
  | VP / FN | 43 / 279 | 139 / 183 |
  | VN / FP | 694 / 81 | 467 / 308 |
  | Sensibilidad | 13 % | 43 % |
  | Especificidad | 90 % | 60 % |

  La ponderada **no ordena mejor** (mismo AUC): solo sube las probabilidades, y con el umbral 0,5 dice "pCR" muchas más veces
  (3 veces más pCR detectadas, casi 4 veces más falsas alarmas). Como el umbral se elige después (fase 4), el mismo efecto se
  consigue moviendo el umbral. **Se mantiene la pérdida normal.**
- Siguiente (2.5 y 2.6, últimos de la fase 2): normalización de intensidad por corte (`--normalizar`) y conv+conv (`--doble`).

## Fase 2 · experimento 2.5: normalización de intensidad por corte (decidido el 2026-10-15)

**Por qué:** el techo de ~0,57 no depende del freno; una causa concreta no tocada es que las imágenes vienen de más de 22 centros y
3 fabricantes **sin armonizar la intensidad** (artículo del dataset). **Cambio único sobre 2.1:** `--normalizar`: cada corte se
reescala con su propia media y desviación, las mismas para las tres fases: x → (x − media) / desviación. Sin parámetros; dentro de
la red, así que la app lo hace igual. Verificado: la salida es idéntica si el mismo corte llega con otro brillo y contraste, y el
realce (EARLY − PRE) se conserva proporcionalmente. Ficheros `bn_normal_norm_lr0.0001_*`; se compara con
`python comparar.py bn_normal_lr0.0001 bn_normal_norm_lr0.0001`.
Después, uno por uno: **2.6 conv+conv** (`--doble`, 302.321 parámetros, ya programado y verificado en la red).

**RESULTADO 2.5 (2026-10-15):** 0,544 ± 0,032 frente a 0,567 (−0,024, IC95 −0,065 a +0,018, no concluyente; baja en 4 de 5 folds).
Estados: 3 sin aprendizaje, 2 sobreentrenados, 0 ok. **No se adopta.** Pista importante: con la normalización, el AUC de validación
de la **época 1** baja a 0,46-0,54 (antes ~0,55). El "0,55 gratis" que la red tenía antes de aprender parece venir en buena parte del
**brillo absoluto** de la imagen (nivel general de realce o diferencias entre escáneres), no de la forma o textura del tumor.
Las matrices de confusión con umbral 0,5 siguen diciendo "no pCR" a casi todas las pacientes (FP = 0 y FN = todas las pCR): con
probabilidades alrededor de 0,3, nadie supera 0,5. Confirma que el umbral debe elegirse en la fase 4.

## Noche del 15 al 16 de octubre: conv+conv (2.6) y semillas de la configuración finalista (2.1)

- **2.6 conv+conv** (`--doble`, 302.321 parámetros): último experimento de arquitectura de la fase 2 (expectativa baja: la red ya
  memoriza con 106.001 parámetros; sirve para justificar la arquitectura con menos capacidad, la del profesor y más capacidad).
- **2.1 con las semillas 1 y 2** (`--semilla 1`, `--semilla 2`; ficheros `bn_normal_lr0.0001_s1_*` y `_s2_*`): mide cuánto cambia
  el resultado solo por el azar del entrenamiento (orden de lotes, pesos iniciales), para saber qué diferencias son reales; y sus
  modelos podrán sumarse al conjunto final. `entrenar.py` añade `_s<semilla>` al nombre si la semilla no es 42.

**RESULTADOS (2026-10-19):**

| Configuración | AUC val en la época elegida | Frente a 2.1 (IC95) | Estados (de 5) |
|---|---|---|---|
| 2.1 semilla 42 | 0,567 ± 0,016 | — | 1 sin aprendizaje, 3 sobreentrenados, 1 ok |
| 2.1 semilla 1 | 0,547 ± 0,020 | −0,020 (−0,060 a +0,020) | 2 sin aprendizaje, 1 sobreentrenado, 2 ok |
| 2.1 semilla 2 | 0,566 ± 0,017 | −0,001 (−0,022 a +0,020) | 3 sin aprendizaje, 1 sobreentrenado, 1 ok |
| 2.6 conv+conv | 0,551 ± 0,019 | **−0,016 (−0,031 a −0,001): empeora** | 1 sin aprendizaje, 4 sobreentrenados |

- **Ruido de semilla:** la misma configuración en los mismos folds cambia hasta **0,06 en un fold** (fold 2: 0,593 / 0,530 / 0,581) y
  hasta **0,02 en la media** (0,547-0,567). Media de las 3 semillas: **0,560**. Todas las diferencias entre configuraciones de la fase 2
  (−0,026 a +0,008) caben en ese ruido: **las configuraciones razonables son equivalentes (~0,56 ± 0,02)**.
- **conv+conv se descarta:** memoriza muy deprisa (AUC de entrenamiento 1,000 hacia la época 15; pérdida de validación hasta 3,5)
  y 4 de 5 folds sobreentrenados. Su "empeora" es contra una sola semilla de 2.1, así que hay que tomarlo con cautela.

## FASE 2 CERRADA (2026-10-19)

Probado: menos capacidad (red estrecha), la red del profesor y más capacidad (conv+conv); frenos suaves y fuertes (lr, aumentado);
otra información de entrada (realce y lavado, normalización por corte); otra pérdida (ponderada). Todo queda en ~0,55-0,57, dentro
del ruido de semilla. Las redes grandes memorizan y las frenadas no aprenden. **El límite no está en la arquitectura ni en el
entrenamiento, sino en la información que una CNN 2D desde cero saca de estas imágenes pretratamiento.**
**Modelo elegido: 2.1** (RedBase con BatchNorm, 106.001 parámetros, Adam lr 1e-4, lote 128, pérdida normal, early stopping), con
**3 semillas × 5 folds**. Idea para la fase 4: promediar las 3 semillas de cada fold reduce el ruido de semilla y se parece más al
promedio que hará el modelo final (5 folds × 3 semillas).


## Fase 4: agregación y umbral (`umbral.py`, 2026-10-19)

Con las predicciones de validación de la configuración elegida (2.1, 3 semillas × 5 folds; `*_probs_val.csv`, época elegida):
1. por corte, media de las 3 semillas; por paciente, media de sus cortes (la agregación que pide el profesor). Se compara el AUC por
   fold de cada semilla con el del promedio (el modelo final promediará 5 folds × 3 semillas, así que el umbral se elige sobre el promedio);
2. con las 1.097 pacientes: ROC, precision-recall (AP), sensibilidad/especificidad/precisión según el umbral, calibración por deciles y
   Brier (frente al de decir siempre la proporción de pCR);
3. umbral = el más bajo con **especificidad ≥ 0,95** (desescalar; el error grave es el falso positivo), comparado con máximo F1 de pCR,
   máximo F1 equilibrada, Youden y 0,5;
4. matriz, sensibilidad, especificidad, precisión, VPN, F1 de cada clase, F1 equilibrada y accuracy frente a "siempre no pCR", y la
   estabilidad del umbral fold a fold. Guarda `umbral_final.json` (umbral para el test y la app), el CSV por paciente y la figura.
Aviso: el umbral se elige con estas mismas predicciones (algo optimista); el test lo aplicará tal cual.

**Primer resultado (sin calibrar, 2026-10-19):** promedio de las 3 semillas 0,571 ± 0,014 por fold, frente a 0,560 de las semillas
sueltas (**+0,011, mejora en los 5 folds**). Pero con todas las pacientes juntas el AUC baja a 0,565 (IC95 0,527-0,599), las
probabilidades van de 0,01 a 0,98, el Brier (0,2087) es peor que el de referencia (0,2074) y el umbral para especificidad ≥ 0,95
(0,470) no detecta ninguna pCR en los folds 0 y 1: cada modelo habla en su propia escala (los "sin aprendizaje" dan ~0,29 a todas;
los sobreentrenados, valores extremos). Con ese umbral: sensibilidad 7 %, especificidad 95 %, precisión 37 % (prevalencia 29 %).
La curva de calibración es mucho más plana que la diagonal y la ROC está pegada a la diagonal en la zona de especificidad alta: no
hay un grupo de pacientes en el que el modelo esté seguro de la pCR.
**Solución:** calibración de Platt **por modelo** (configuración × fold), a nivel de corte, con su propio fold de validación, antes
de promediar (`umbral.py`, por defecto; `--sin-calibrar` para la versión anterior). Los (a, b) se guardan en `umbral_final.json` para
el test y la app. No crea información: la zona de especificidad alta seguirá siendo pobre.

**Resultado de la calibración por modelo (2026-10-19):** la curva de calibración pasa a seguir la diagonal y el Brier baja a 0,2069
(referencia 0,2074), pero el AUC con todas las pacientes baja a 0,557, la desviación por fold sube a 0,028 (fold 1: 0,554 → 0,529) y la
sensibilidad con el umbral de especificidad 0,95 sigue siendo ~7 % (precisión 39 %). Problema: los modelos "sin aprendizaje" dan casi
la misma probabilidad a todas y la calibración les pone pendientes enormes (media ~6·10⁷), así que su ruido pesa tanto como el de los
modelos que sí aprendieron. **Se cambia a una calibración ÚNICA al final:** se promedian las probabilidades de las semillas sin tocar y
se ajusta un solo Platt (a, b) sobre ese promedio, por corte, con todos los cortes de validación. Conserva el AUC del promedio, arregla la
escala y el test y la app solo aplican dos números.

**RESULTADO FINAL DE LA FASE 4 (calibración única, 2026-10-19). FASE 4 CERRADA.**
- AUC con todas las pacientes juntas **0,564 (IC95 0,527-0,599)**, igual que sin calibrar; media por fold 0,571 ± 0,014; AP 0,345
  (azar 0,294); **Brier 0,2058** (referencia 0,2074; sin calibrar 0,2087). Platt: a = 0,319, b = −0,591 (el promedio exageraba sus
  probabilidades ~3 veces). Curva de calibración pegada a la diagonal. Probabilidades por paciente de 0,12 a 0,67.
- **Umbral elegido 0,348** (especificidad ≥ 0,95): sensibilidad 7 % (22/322), especificidad 95 %, precisión 37 %, VPN 71 %, F1 pCR 0,12,
  F1 no pCR 0,81, F1 equilibrada 0,46, accuracy 0,69. "Siempre no pCR": accuracy 0,71, F1 equilibrada 0,41. Alternativa F1 equilibrada
  (0,307): sensibilidad 40 %, especificidad 71 %, precisión 36 %.
- Estabilidad: con 0,348, los folds 0 y 1 no detectan ninguna pCR (sensibilidad por fold 0,00 / 0,00 / 0,17 / 0,12 / 0,05).
- Lectura para el informe: con AUC ~0,56 el modelo apenas supera a "siempre no pCR" y no serviría para desescalar; la ROC está pegada a la
  diagonal en la zona de especificidad alta. Riesgo: en el test el modelo final promedia 15 modelos (5 folds × 3 semillas) y en validación
  cada paciente tenía 3; el umbral podría no trasladarse exactamente.
- Pipeline fijado: 15 modelos → media de sus probabilidades por corte → Platt (a, b) → media de los cortes de la paciente → umbral 0,348
  (parámetros en `resultados/umbral_final.json` del PC).




## Fase 5: modelo final y módulo de predicción común (`predecir.py`, 2026-10-20)

- Los 15 modelos (2.1: 3 semillas × 5 folds), sus probabilidades de validación, sus resúmenes y `umbral_final.json` están en
  `modelo/` del repositorio (subidos desde el PC, commit 3adc1a6). Los registros por época de todos los experimentos, en `referencia/`.
- `predecir.py`: **la misma función para el test y para la app.** Lee cada fase como en el entrenamiento (PNG → escala de grises →
  /255), rechaza lo que no sea un PNG de 256×256 (sin redimensionar), ficheros vacíos, corruptos o de más de 5 MB; avisa si EARLY no
  es más brillante que PRE en el tejido (posible orden incorrecto). Predice con los 15 modelos en modo evaluación y sin gradientes,
  promedia, calibra (a, b) y compara con el umbral. Devuelve probabilidad, clase, umbral, número de modelos, dispositivo, tiempo y un
  **checksum** de los modelos y el json.
- **Verificado:** con 10 pacientes por modelo, reproduce las probabilidades de validación guardadas por el entrenamiento con una
  diferencia máxima de **2·10⁻⁶** en los 15 modelos (CPU del portátil frente a GPU del PC). Da lo mismo leyendo de disco que de bytes.

## Agregación justificada con la validación (`comparar_agregacion.py`, 2026-10-20)

Modelo final (3 semillas × 5 folds, calibrado), 1.097 pacientes de validación; voto = fracción de cortes con probabilidad ≥ la mediana
de todos los cortes (0,295), para que vote la mitad.

| Agregación | AUC por fold | AUC juntas (IC95) | AP | Con especificidad ≥ 0,95: sens / precisión | Sensibilidad por fold |
|---|---|---|---|---|---|
| **media** | **0,571 ± 0,014** | 0,564 (0,527-0,599) | 0,345 | 0,07 / 0,37 | 0,00 0,00 0,17 0,12 0,05 |
| mediana | 0,571 ± 0,014 | 0,562 (0,524-0,597) | 0,345 | 0,07 / 0,37 | 0,00 0,00 0,17 0,12 0,05 |
| máximo | 0,573 ± 0,020 | 0,568 (0,530-0,607) | 0,352 | 0,09 / 0,42 | 0,00 0,00 0,10 0,28 0,02 |
| voto | 0,559 ± 0,032 | 0,555 (0,520-0,590) | 0,325 | 0,00 / — (solo 12 valores distintos) | todo 0 |

- Frente a la media (emparejado por fold): mediana −0,0002, máximo +0,002, voto −0,012; **todos no concluyentes**.
- **Se mantiene la media:** es la que pide el profesor, rinde igual que las demás (las diferencias caben en el ruido), es más estable
  que el máximo (desviación 0,014 frente a 0,020; el máximo tiene una sensibilidad por fold de 0,02 a 0,28) y la calibración se ajustó
  para ella. El máximo, como avisa la GUIA, depende de un solo corte. El voto no sirve para fijar un umbral: con ~10 cortes solo da
  fracciones (12 valores distintos) y no hay ningún punto con especificidad ≥ 0,95 salvo no decir "pCR" a nadie.
- Con un solo corte (la app y la defensa), las cuatro agregaciones coinciden: la media de un corte es ese corte.
- `umbral.barrido` incluye ahora el umbral "nadie es pCR" (por encima del máximo); el umbral elegido no cambia (0,3477).

## Umbral para UN solo corte (app y defensa) (2026-10-20)

El umbral 0,348 se eligió con la media de ~10 cortes por paciente; en la app llega un corte. Medido en la validación (modelo final
calibrado, 10.945 cortes):

| | Paciente (media de cortes) | Un corte suelto |
|---|---|---|
| AUC | 0,564 | 0,561 |
| Con 0,348: sensibilidad / especificidad / precisión | 0,07 / **0,95** / 0,37 | 0,09 / **0,94** / 0,39 |
| Umbral para especificidad ≥ 0,95 | 0,348 | 0,351 |

- Las probabilidades de un corte suelto están casi igual de repartidas que las de la media de ~10 (desviación 0,037 frente a 0,036):
  **el modelo da casi la misma probabilidad a todos los cortes de una paciente**. Encaja con la pista de la normalización (el modelo se
  fija en rasgos generales de la imagen de esa paciente más que en detalles del tumor que cambian de un corte a otro) y responde a la
  pregunta de la GUIA sobre las diez observaciones correlacionadas: aportan casi la misma información que una.
- **Decisión: un solo umbral (0,348) para el test y para la app.** Con un corte la especificidad pasa del 95 % al 94 %.

## Experimento complementario: recorte central 128×128 (decidido el 2026-10-20, ANTES del test)

**Origen:** idea de la arquitectura de un compañero (no se copia su código ni su red; solo se prueba la idea con nuestra red).
**Qué:** la configuración final (2.1) con `--recorte 128`: la red solo mira el cuadrado central de 128×128 del corte de 256×256
(las imágenes ya son recortes centrados en el tumor, así que el centro es sobre todo tumor). Dentro de la red (la app lo aplicaría
igual); mismos 106.001 parámetros; mapa final de 8×8. Verificado: la salida no cambia si se modifica lo que hay fuera del centro.
**Hipótesis:** obligar a la red a mirar el tumor podría quitarle el atajo del brillo general de la imagen; en contra, pierde el tejido
que rodea al tumor (peritumoral), que la bibliografía considera útil (Braman et al.).
**Diseño:** 3 semillas (42, 1 y 2), las mismas que la 2.1, en los 5 folds. Ficheros `bn_normal_c128_lr0.0001_*` (`_s1_`, `_s2_`).
**REGLA FIJADA ANTES DE VER RESULTADOS:** por fold, media de las 3 semillas del AUC de validación en la época elegida, para el recorte y
para la 2.1; diferencia emparejada por fold. **Solo se cambia el modelo final si la diferencia media supera +0,02 y su IC95 excluye el 0.**
Si no, se cuenta en el informe como experimento complementario y el modelo final sigue siendo la 2.1 (el test se hace al día siguiente).

### Resultado (2026-10-21)

AUC de validación en la época elegida (`comparar.py`), media de las 3 semillas por fold:

| Fold | 0 | 1 | 2 | 3 | 4 | Media |
|---|---|---|---|---|---|---|
| 2.1 | 0,553 | 0,548 | 0,568 | 0,561 | 0,572 | 0,560 |
| Recorte 128 | 0,588 | 0,559 | 0,592 | 0,571 | 0,582 | 0,578 |
| Diferencia | +0,036 | +0,011 | +0,024 | +0,010 | +0,010 | **+0,018 (IC95 +0,004 a +0,032)** |

- Por semilla: 42 → +0,018 (IC −0,001 a +0,037, no concluyente); 1 → +0,035 (+0,001 a +0,069, mejora); 2 → +0,001 (−0,022 a
  +0,024, no concluyente). La ventaja depende de la semilla.
- **Sobreajuste:** 0 modelos sobreentrenados de 15 con el recorte, frente a 5 de 15 con la 2.1; diferencia media
  entrenamiento − validación de ~0,04 frente a ~0,11. Sin aprendizaje: 5 de 15 frente a 6 de 15.
- Con el recorte, el AUC de la época 1 (red casi sin entrenar) sube a 0,57-0,62 en varios folds: parte de la señal está en el centro
  (el tumor) desde el principio.
- Unas 3,5 veces más rápido (13 s por época frente a 47).
- **Por la regla:** el IC excluye el 0, pero +0,018 no llega a +0,02 → la regla decía mantener la 2.1.

### DECISIÓN (2026-10-21, ANTES del test): el modelo final pasa a ser el recorte central 128

Decisión del usuario, **desviándose de forma explícita de la regla fijada antes** (que pedía +0,02 y se obtuvo +0,018). Motivos:

1. Mejora en los 5 folds con la media de las 3 semillas, y el IC95 excluye el 0: no es solo ruido.
2. **Generaliza mejor:** ningún modelo sobreentrenado y la distancia entre entrenamiento y validación baja de ~0,11 a ~0,04. Es el
   comportamiento "BIEN" que pide el profesor, no solo un número de AUC algo mayor.
3. Tiene sentido: las imágenes son recortes centrados en el tumor; mirar el centro quita distracciones del borde.
4. El margen de +0,02 se fijó como el ruido de semilla en la media; +0,018 está en ese borde, y la mejora en sobreajuste inclina la
   balanza.

Riesgos que se asumen y se contarán en el informe: la regla se cambió después de ver el resultado; la ventaja depende de la semilla
(semilla 2: +0,001); hemos probado unas 8 configuraciones con la misma validación, así que la mejor puede parecer algo mejor de lo que
es (maldición del ganador). El juez independiente es el test.

**Cómo queda:**
- `modelo/` = recorte 128 (`bn_normal_c128_lr0.0001` + `_s1` + `_s2`, 15 modelos), con su calibración de Platt y su umbral
  recalculados con `umbral.py` por el mismo criterio (especificidad ≥ 0,95). `predecir.py` lee el recorte de cada modelo.
- `modelo_21/` = la 2.1 tal como estaba, para compararla.
- **Test (una sola vez, cuando lo ordene el usuario):** se evalúan los dos, con el recorte declarado AHORA como modelo principal (el de
  la app pase lo que pase) y la 2.1 como comparación. No se elige el principal mirando el test.

### Modelo final con recorte: calibración y umbral (`umbral.py`, 2026-10-21; commit 8b2300f desde el PC)

| | 2.1 (`modelo_21/`) | Recorte 128 (`modelo/`) |
|---|---|---|
| AUC por fold con las 3 semillas juntas (media ± desv.) | 0,571 ± 0,014 | **0,584 ± 0,013** |
| Ganancia de juntar semillas frente a la media de las sueltas | +0,011 | +0,006 |
| AUC con todas las pacientes, calibrado (IC95) | 0,564 (0,527-0,599) | **0,576 (0,539-0,613)** |
| AP (azar 0,294) | 0,345 | 0,347 |
| Brier (referencia 0,2074) | 0,2058 | **0,2046** |
| Platt (a, b) | 0,319, −0,591 | 0,593, −0,338 |
| Umbral (especificidad ≥ 0,95) | 0,348 | **0,371** |
| Sensibilidad / especificidad / precisión | 0,07 / 0,95 / 0,37 | 0,07 / 0,95 / 0,37 (22 de 322 pCR) |
| Sensibilidad por fold con ese umbral | 0,00 en los folds 0 y 1 | 0,02 / 0,06 / 0,02 / 0,04 / 0,21 |

- Al juntar las semillas la ventaja del recorte baja de +0,018 a ~+0,012: juntar semillas ayudaba más a la 2.1.
- Platt corrige menos (a = 0,59 frente a 0,32): el recorte exagera menos sus probabilidades, coherente con que sobreajuste menos.
  Probabilidades por paciente entre 0,107 y 0,479 (ninguna llega a 0,5).
- **Un solo corte** (app y defensa), medido en la validación: sensibilidad 0,08, especificidad 0,950, precisión 0,41; el umbral ideal
  para un corte sale también 0,371. Desviación de las probabilidades: 0,050 por paciente y 0,051 por corte. **Un solo umbral, 0,371,
  para el test y la app.**
- `predecir.py --verificar 20` en el PC: diferencia máxima 5,6·10⁻⁷ en los 15 modelos → OK (checksum 7a45826b7d822592).

## Fase 6: `evaluar_test.py` y ensayo con la validación (2026-10-21)

- `python evaluar_test.py` = **ensayo con la validación** (predicciones fuera de fold de `modelo/` y `modelo_21/`, modelo clínico
  ajustado en los otros 4 folds). Salida en `figuras/validacion/`: ROC, matrices, precisión-recall, umbral, calibración, cohortes,
  casos, `metricas.json`, `resumen.txt`, `pacientes.csv`. `python evaluar_test.py --test` = el test, **una sola vez** (se niega si
  ya existe `figuras/test/metricas.json`); cada corte pasa por `predecir.ModeloFinal`, como en la app.
- Ensayo (optimista, umbral y calibración elegidos con estos datos): recorte 0,576 (0,539-0,613), 2.1 0,564, clínico 0,704; recorte −
  2.1 = +0,011 (IC95 bootstrap −0,027 a +0,051, incluye el 0); recorte − clínico = −0,128 (excluye el 0). Con el umbral: VP 22, FN 300,
  VN 737, FP 38 (sens 7 % [5-10], espec 95 % [93-96], precisión 37 % [26-49], Wilson). Por cohorte (recorte / 2.1 / clínico): duke
  0,552 / 0,546 / 0,739; spy1 0,599 / 0,507 / 0,784; spy2 0,557 / 0,559 / 0,689.
- **Imagen defectuosa:** ISPY1_1193 (spy1, no pCR) tiene la fase LATE casi blanca (brillo medio 0,955; en el resto de pacientes la
  media de una fase no pasa de 0,66). Es la paciente con mayor probabilidad de pCR de toda la validación (0,479; con la 2.1, 0,67): un
  falso positivo causado por la imagen. No se quita del entrenamiento (ya está entrenado y sería decidir mirando resultados); se cuenta
  en el informe y `predecir.avisos` avisa ahora si una fase tiene brillo medio > 0,8 (no cambia la predicción).

## Curvas de entrenamiento de los modelos finales (2026-10-21; `curvas_entrenamiento.py`, `figuras/entrenamiento/`)

- **Matiz al motivo 2 de la decisión del recorte:** la menor distancia entre entrenamiento y validación se mide en la época elegida, y
  el recorte elige épocas mucho más tempranas (media 5,6 frente a 13,8 de la 2.1). Las curvas muestran que, si se deja entrenar, el
  recorte memoriza a un ritmo parecido (AUC de entrenamiento ~0,8 hacia la época 15 en las dos familias). Lo que cambia es que su AUC
  de validación alcanza el máximo antes (épocas 1-8), en parte por la señal que ya da el centro de la imagen desde la época 1. Por
  tanto: "con early stopping, los modelos elegidos sobreajustan menos porque su mejor época llega antes", no "la arquitectura
  sobreajusta menos". La decisión no cambia (el AUC sigue siendo algo mejor y el early stopping forma parte del método), pero el
  informe lo dirá así.
- La pérdida de validación apenas baja de la pérdida "sin mirar la imagen" (0,606, entropía de la proporción de pCR de los cortes de
  entrenamiento) y después sube: la señal que aprende la red es débil. La 2.1, si se deja entrenar, llega a pérdida de validación 2,25
  y AUC de entrenamiento 0,98 (la figura "MAL" del profesor); el early stopping la corta antes.

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
