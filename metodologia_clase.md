# Metodología de clase — apuntes de Redes Convolucionales

Documento de hechos, separado de `SINTESIS_EXPLORACION.md`. Recoge lo que dicen
tres PDFs de la asignatura (no del caso), encontrados en `CNN/`:

- `CNN/03_CNN1.pdf` — *Redes neuronales convolucionales I* (numeración interna:
  diapositivas 4–29)
- `CNN/04_CNN2.pdf` — *Redes Convolucionales II: cómo diseñar una red
  convolucional y caso práctico* (diapositivas 44–63)
- `CNN/05_CNN3.pdf` — *Redes Convolucionales III: arquitecturas comerciales*
  (diapositivas 31–42)

Los tres llevan la numeración de diapositiva de un mismo temario correlativo (I
termina en 29, II empieza en 44 — hay un salto, quizá por diapositivas no
exportadas a PDF — y termina en 63, III va de 31 a 42, que se solapa con el
rango de II; el orden que uso es el de sus nombres de fichero y numeración
`03/04/05`, no el de las diapositivas). No los he modificado; solo los he leído
con `pdftotext`.

No se toma aquí ninguna decisión de diseño para el caso BreastDCEDL. La fase 3
señala coincidencias, conflictos y opciones abiertas; no resuelve nada.

**Precedencia acordada con el usuario:** `GUIA.md` y
`documentation/caso_breastdcedl.pdf` son la fuente de verdad del proyecto y
priman sobre cualquier otro documento, incluidos estos tres PDFs de la
asignatura. Esto no resuelve por sí solo cada conflicto de la sección 4.2 (qué
arquitectura o qué normalización usar sigue siendo una decisión de diseño
pendiente), pero fija que, ante contradicción, las reglas del caso ganan sobre
la práctica general enseñada en clase.

---

## 1. `CNN/03_CNN1.pdf` — Redes neuronales convolucionales I

### Título y tema
Fundamentos de la convolución: qué es una CNN, por qué no basta una red densa,
la operación de convolución (kernel, stride, padding, dilatación), pooling,
funciones de activación, BatchNorm, dropout, capas densas, cómo aprende una
convolución (retropropagación) y la "anatomía" completa de una CNN en PyTorch.

### Metodología o proceso, tal cual la nombra el profesor
Este PDF no presenta un método numerado propio; enuncia una secuencia de
**"cinco piezas y siempre en el mismo orden"** (diapositiva 27, *Anatomía de una
CNN, pieza a pieza*):

1. Convolución
2. Normalización
3. Activación
4. Pooling
5. Capa densa

Y remarca explícitamente: *"El orden no es negociable. Convolución →
normalización → activación → pooling."* (No incluye la capa densa en esa
fórmula corta, pero sí en la tabla de las cinco piezas.)

También enuncia como receta operativa el **"orden canónico"** de la diapositiva
23: *Conv → BatchNorm → ReLU*, con la nota de que la convolución no lleva sesgo
porque BatchNorm lo sustituye.

### Vocabulario y convenciones de nomenclatura
- **Kernel / filtro**, **mapa de características** (*feature map*), **producto
  punto**, **sesgo** (*bias*), **correlación cruzada** (nombre técnicamente
  correcto de lo que PyTorch llama "convolución": no gira el kernel 180°).
- Hiperparámetros de `Conv2d`, nombrados en ese orden: `in_channels`,
  `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`,
  `bias`.
- Fórmula de tamaño de salida: `Hout = (H + 2p − d(k−1) − 1) / s + 1`.
- Fórmula de parámetros de una capa convolucional: `Parámetros = Cin × k × k ×
  Cout + Cout`.
- **Campo receptivo** (*receptive field*).
- **Max pooling** vs. **average pooling** (y **global average pooling** como
  caso particular al final de la red).
- **ReLU**, **LeakyReLU**, **GELU**, **Softmax**, **Sigmoid**; en PyTorch:
  `nn.ReLU()`, `nn.GELU()`, `CrossEntropyLoss` (aplica `log_softmax`
  internamente), `BCEWithLogitsLoss` para binaria/multietiqueta.
- **BatchNorm** (`BatchNorm2d`), con parámetros aprendidos "γ y β"; distingue
  comportamiento **train vs. eval** (estadística del lote vs. media móvil) y
  cita `GroupNorm`/`LayerNorm` como alternativas con lotes pequeños.
- **Dropout**, **Dropout2d** (apaga canales enteros, no neuronas sueltas, para
  "respetar la coherencia espacial").
- **Flatten**, **capas totalmente conectadas** (densas).
- Términos de la anatomía general: **extractor de características** (parte
  reutilizable) vs. **clasificador** (parte específica del problema).
- Código: `nn.Module`, `forward`, `model(x)` (nunca llamar `forward` a mano),
  `.parameters()`, `AdaptiveAvgPool2d`, patrón de verificación
  `CNN()(torch.randn(2,1,28,28)).shape`.

### Reglas prácticas / heurísticas enunciadas como norma general
- *"¿Cuántos filtros pongo?... La regla práctica: empezar en 32 o 64 y duplicar
  cada vez que se reduce la resolución."*
- *"Con padding=1 el tamaño se conserva y la aritmética de toda la red se
  vuelve trivial."* Frente al "error clásico" de perder píxeles de borde capa a
  capa sin padding.
- Sobre campo receptivo: *"si el objeto que buscas no cabe en el campo
  receptivo de la última capa, la red no puede verlo entero"*; *"el campo
  receptivo de la última capa debe ser, al menos, tan grande como el objeto que
  quieres reconocer dentro de la imagen."*
- Tres convoluciones 3×3 cubren lo mismo que una 7×7 con menos parámetros
  (27C² frente a 49C²) y tres no linealidades en vez de una.
- Pooling: *"el pooling es la forma barata de ver mucho contexto con pocas
  capas."* Kernel 2×2 y stride 2 descarta el 75 % de los valores.
- *"Muchas arquitecturas modernas sustituyen el pooling por convoluciones con
  stride 2, que aprenden cómo reducir en vez de aplicar una regla fija."*
- Sin activación no lineal, apilar convoluciones "equivale a una sola
  convolución": toda la red colapsa en una función lineal.
- Dropout: *"0,5 en capas densas y 0,1–0,25 en convolucionales. Si tu red ya
  lleva BatchNorm y buen augmentation, a menudo puedes bajarlo o quitarlo."*
- Sobre qué hiperparámetro tocar primero: *"El que más cambia el resultado, con
  diferencia, es el learning rate. El que menos, casi siempre, el tamaño del
  kernel."*
- *"El único obligatorio: en `Conv2d`, los canales de entrada deben coincidir
  exactamente con los de salida de la capa anterior. Todo lo demás tiene valor
  por defecto."*
- *"Comprueba siempre las formas"* con un tensor aleatorio antes de entrenar.

### Ejemplos de código / plantillas
- Tabla de hiperparámetros por pieza con "valores habituales": kernel 3,
  stride 1, padding 1, `bias=False` si va seguida de BatchNorm;
  `MaxPool2d(2, 2)`; `out_features` = número de clases; dropout 0,5 antes de la
  última capa.
- Tabla de ejemplo de parámetros:
  - `Conv2d(3, 32, 3)` → 896
  - `Conv2d(32, 64, 3)` → 18.496
  - `Conv2d(64, 128, 3)` → 73.856
  - `Linear(2048, 10)` → 20.490
- Patrón de verificación de formas: `CNN()(torch.randn(2,1,28,28)).shape` debe
  dar `[2, 10]`.
- Menciona `AdaptiveAvgPool2d` como forma de librarse de recalcular el "64×7×7"
  al cambiar el tamaño de entrada.

---

## 2. `CNN/04_CNN2.pdf` — Redes Convolucionales II: cómo diseñar una red convolucional y caso práctico

### Título y tema
Método de diseño de una CNN propia (aplicado a CIFAR-10), verificación antes de
entrenar, un caso práctico manual de clasificación binaria ("melanoma sí o no"
sobre imágenes de 6×6), preprocesado según tipo de imagen, Dataset/transforms/
DataLoader, data augmentation, el bucle de entrenamiento y, al final, transfer
learning / fine-tuning.

### Metodología o proceso, tal cual la nombra el profesor
Diapositiva 44, título literal **"El método: seis pasos"**, presentado como
*"Diseñar una CNN no es inspiración: es responder seis preguntas en orden"*:

1. **¿Qué entra y qué sale?** — Forma del tensor de entrada y número de
   neuronas de salida.
2. **¿Cuántas etapas?** — Cada etapa divide la resolución entre 2. Se para al
   llegar a 4×4.
3. **¿Qué bloque se repite?** — Conv 3×3 → BN → ReLU, una o dos veces por
   etapa.
4. **¿Cómo reduzco?** — MaxPool 2×2 o convolución con stride 2.
5. **¿Qué cabeza pongo?** — Global average pooling + una capa lineal.
6. **¿Cómo lo verifico?** — Un tensor falso, comprobar formas, y sobreajustar
   10 imágenes.

Enunciada como **"regla de oro que atraviesa los seis pasos"**: *"cada vez que
la resolución se divide entre dos, los canales se duplican. Así el coste por
etapa se mantiene aproximadamente constante."*

El caso práctico de melanoma se resuelve en tres pasos, nombrados así:
**Paso 1 — Convolución y ReLU**, **Pasos 2 y 3 — Pooling y decisión**.

El bucle de entrenamiento se enuncia como cinco pasos numerados en la
diapositiva 61: **1. Lote x, y del DataLoader — 2. Forward: logits =
modelo(x) — 3. Pérdida: comparar con y — 4. Backward: gradientes — 5. Step:
actualizar pesos.**

### Vocabulario y convenciones de nomenclatura
- **Etapa** (*stage*): unidad que agrupa uno o dos bloques y termina reduciendo
  la resolución a la mitad. **Mapa de dimensiones**: la tabla resolución ×
  canales por etapa, que el profesor dice dibujar *antes* de programar.
- Notación de tensor: `Cin × H × W`, leída "tantas láminas, de tanto por
  tanto".
- **Cabeza clasificadora**, con dos opciones nombradas explícitamente:
  - **Opción A — Flatten + Linear**
  - **Opción B — Global Average Pooling** (GAP)
- **Logits**: "las puntuaciones en bruto, sin normalizar. Pueden ser
  negativas."
- **`bias=False`** en la convolución cuando va seguida de BatchNorm (motivo:
  "sería redundante").
- Patrón de comentarios de forma en el código: anotar `32×32 → 16×16` en cada
  paso del `forward` ("los comentarios del código no son adorno").
- **`__init__` crea, `forward` usa** — regla de nomenclatura/uso de PyTorch.
- Diagnóstico de fallos del **Paso 6** (verificación), con nombres concretos:
  - **"Pérdida en NaN"** → learning rate demasiado alto o datos sin normalizar.
  - **"Pérdida plana en 2,30 (ln 10)"** → la red predice al azar; revisar
    etiquetas y salida. (2,30 es específico de 10 clases con
    `CrossEntropyLoss`; el valor cambia con el número de clases.)
  - **"Accuracy alta pero mala en test"** → sobreajuste: augmentation, dropout
    o menos capacidad.
- Tabla de tipos de imagen y canales: escala de grises (1 canal, 0–255),
  RGB (3, 0–255), RGBA (4, descartar alfa con `.convert("RGB")`), **imagen
  médica (DICOM)** (1 canal, unidades **Hounsfield** de −1000 a 3000, "hay que
  recortar la ventana clínica de interés antes de escalar"), multiespectral
  (4–13 bandas, cada una normalizada con su propia media/desviación).
- **Normalize** con media y desviación típica: distingue explícitamente "las de
  ImageNet" (obligatorias si se usa una red preentrenada) de "las de tu propio
  conjunto de entrenamiento" (si los datos son propios).
- Pipeline nombrado: **Carpetas → Dataset → Transforms → DataLoader → Modelo**.
- **`shuffle=True` solo en entrenamiento**; **augmentation solo en el conjunto
  de entrenamiento**.
- Categorías de data augmentation, nombradas: **Geométricas** (recorte
  aleatorio, volteo horizontal, rotaciones, escalados suaves),
  **Fotométricas** (brillo, contraste, saturación, tono), **Oclusión**
  (*Random erasing*), **Mezcla** (**MixUp**, **CutMix**).
- **`zero_grad()`**, **`model.train()` / `model.eval()`**, **`torch.no_grad()`**,
  **Scheduler** (bajar el learning rate hacia el final).
- Distinción explícita: **"una época no es una iteración"**; iteración = un
  lote, época = dataset completo (ejemplo: 50.000 imágenes, lotes de 128 → 391
  iteraciones por época).
- **Transfer learning y fine-tuning**, con dos variantes nombradas:
  - **A — Extractor congelado + cabeza nueva** ("pocos datos, <1.000 imágenes,
    o dominio parecido a ImageNet").
  - **B — Fine-tuning**: "últimas capas + lr bajo en las primeras capas +
    cabeza nueva" ("más datos o dominio distinto —imagen médica, satélite—.
    Learning rate 10× menor en las capas reutilizadas").

### Reglas prácticas / heurísticas enunciadas como norma general
- *"¿Cuántas etapas? Divide la resolución entre 2 hasta llegar a 4×4
  aproximadamente. De 32 salen tres etapas. De 224, cinco."*
- *"¿Cuántos canales? Empieza en 32 (imágenes pequeñas) o 64 (ImageNet) y
  duplica en cada etapa."*
- *"Si dudas, usa global average pooling. Es lo que hacen ResNet, Inception y
  MobileNet, y te ahorra recalcular el famoso 64×7×7 cada vez que cambias la
  entrada."*
- Comparación explícita de coste de cabeza: Flatten+Linear → 20.490 parámetros
  y "la red queda atada a entradas de 32×32 exactamente"; GAP+Linear → 1.290
  parámetros, "menos sobreajuste y funciona con imágenes de cualquier tamaño".
- *"Esta red no lleva atajos residuales, así que no conviene pasar de cinco o
  seis etapas. Si necesitas más profundidad, cambia etapa() por bloque
  residual: el resto del código no cambia."*
- Sobre por qué parar en 4×4: *"por debajo quedan tan pocas posiciones que la
  convolución ya no aporta nada espacial."*
- *"¿Por qué normalizar? Porque el optimizador avanza mejor cuando todas las
  entradas tienen escalas parecidas. Sin ello, el entrenamiento arranca lento
  y a veces no converge."*
- *"Normalize con la media y desviación del propio dataset. Sin ello, el
  entrenamiento arranca mucho peor."*
- *"La transformación [de aumentado] debe respetar la semántica: voltear una
  foto de un gato sigue siendo un gato, pero voltear un 2 manuscrito no
  produce un dígito válido. En imagen médica, además, hay orientaciones que no
  pueden alterarse."*
- Data augmentation descrita como *"la regularización más eficaz y más barata
  en visión: a menudo vale más que cambiar de arquitectura."*
- *"Scheduler: bajar el learning rate hacia el final suele valer uno o dos
  puntos de accuracy gratis."*
- Transfer learning, presentado como técnica general recomendada: *"Las
  primeras capas de una red entrenada en ImageNet detectan bordes y texturas:
  eso sirve para cualquier problema de imagen. Reutilizarlas es la diferencia
  entre necesitar un millón de ejemplos o unos cientos."* Y la regla de
  normalización asociada: *"usa la misma normalización con la que se entrenó
  la red original, o las estadísticas de entrada no coincidirán con lo que
  espera."*

### Ejemplos de código o plantillas
- Tabla completa "mapa de dimensiones" para CIFAR-10 (32×32×3, 10 clases):
  entrada 3×32×32 → etapa1 32×16×16 → etapa2 64×8×8 → etapa3 128×4×4 → cabeza
  10.
- Bloque de etapa, en el orden: `Conv 3×3 (padding 1, sin bias)` → `BatchNorm2d`
  → `ReLU` → (repetir ×2 si se quiere más capacidad) → `MaxPool 2×2`, con anotación
  de entrada/salida: *"entra Cin×H×W ... sale Cout×H/2×W/2"*.
  Comentario explícito de por qué `bias=False`: "porque BatchNorm resta la
  media justo después: el sesgo sería redundante".
  Se ofrece la alternativa "Conv stride 2 (aprende cómo reducir. Lo usan
  ResNet y las redes modernas)" frente a MaxPool ("sencillo, sin parámetros.
  Buena opción por defecto").
- Red completa montada para CIFAR-10: tres etapas (32→64→128 canales,
  resolución 32→16→8→4), GAP, `Linear` final sin softmax ("son logits, y
  `CrossEntropyLoss` se encarga del resto"), "unos 100 K parámetros".
- Explicación línea a línea del `__init__`/`forward` de esa red (sin volcar el
  código completo en el texto extraído, pero sí su glosa): herencia de
  `nn.Module`, llamada a `super().__init__()` obligatoria ("si lo olvidas, la
  red no registra ninguna capa y falla al entrenar"), uso de
  `nn.Sequential` como contenedor ordenado, primera capa con
  `in_channels=3`, `inplace=True` en ReLU para ahorrar memoria.
- Caso práctico "melanoma sí/no" resuelto a mano con números concretos: imagen
  6×6 en escala de grises (valores 0–9), kernel laplaciano 3×3
  `[[0,-1,0],[-1,4,-1],[0,-1,0]]`, cálculo explícito de una posición de
  convolución (`4×7 − 0 − 0 − 3 − 9 = 10`), ReLU eliminando un valor negativo
  (−17), max pooling 2×2 sobre el mapa 4×4 resultante, y una capa final manual
  con pesos `w = [0,25, ...]` y sesgo `b = −10` aplicando sigmoid.

---

## 3. `CNN/05_CNN3.pdf` — Redes Convolucionales III: arquitecturas comerciales

### Título y tema
Recorrido histórico y comparativo de arquitecturas: LeNet-5, AlexNet, VGG-16,
ResNet (bloques residuales, ResNet-50), convoluciones 1×1 e Inception,
convoluciones separables en profundidad (MobileNet), y una tabla comparativa
final de "cuál elegir".

### Metodología o proceso, tal cual la nombra el profesor
No enuncia un método propio en pasos numerados (a diferencia de los otros dos
PDFs). Es un recorrido cronológico/comparativo: *"De LeNet a VGG"* → ResNet →
Inception → MobileNet → tabla "Cuál elegir". El hilo conductor explícito es:
*"Las cuatro comparten el mismo esqueleto: etapas que reducen resolución y
aumentan canales. Lo único que cambia es qué ocurre dentro del bloque."*

Si se cuenta como "proceso", el más cercano a una receta es el de ResNet-50,
descrito como *"una entrada, cinco etapas y una cabeza"*, con dos tipos de
bloque dentro de cada etapa: **Conv Block** (cambia dimensiones: reduce
resolución y duplica canales, con atajo 1×1) e **ID Block** (identidad: no
cambia la forma, el atajo es la entrada tal cual).

### Vocabulario y convenciones de nomenclatura
- **LeNet-5** (1998, 60K parámetros, "establece el patrón conv → pool → conv →
  pool → densa").
- **AlexNet** (2012, 60M parámetros, "populariza ReLU, dropout, data
  augmentation y el entrenamiento en GPU").
- **VGG-16** (2014, 138M parámetros, "solo kernels 3×3, muy profunda"); familia
  **VGG-11, VGG-13, VGG-16, VGG-19** generada de una única función con un
  parámetro `n_convs`.
- **ResNet** (*residual network*, red residual), **atajo** (*shortcut*),
  **bloque residual**, notación `y = F(x) + x`, **Conv Block** vs. **ID Block**
  (identidad), **stem** (la convolución inicial 7×7 + BN + ReLU + max pooling
  de ResNet-50), **vanishing gradient**.
- **Convolución 1×1**, descrita como *"una capa densa aplicada píxel a
  píxel"*; **cuello de botella** (*bottleneck*): reducir canales con 1×1 antes
  de una convolución cara.
- **Módulo Inception**: ramas en paralelo (1×1, 1×1+3×3, 1×1+5×5, pool+1×1),
  concatenación de canales.
- **MobileNet**: **convolución separable en profundidad** (*depthwise
  separable convolution*), dos pasos nombrados **depthwise** (filtro 3×3 por
  canal, solo espacial) y **pointwise** (1×1, mezcla canales).
- Cuatro "esquemas" nombrados como una familia de un mismo lenguaje: **Clásica
  (VGG)**, **Residual (ResNet)**, **Multiescala (Inception)**, **Ligera
  (MobileNet)**.
- Nombres de arquitecturas concretas citadas en la tabla final: **ResNet-18**,
  **ResNet-50**, **Inception-v3**, **MobileNetV2**.

### Reglas prácticas / heurísticas enunciadas como norma general
- *"La lección de VGG es la que gobierna todo lo demás: bloques pequeños y
  repetidos, canales que se duplican cada vez que la resolución se reduce a la
  mitad. Todas las arquitecturas posteriores son variaciones de esa idea."*
- Sobre profundidad sin atajos: *"a partir de ~20 capas, apilar más empeora el
  error incluso en entrenamiento. La solución llega con ResNet."*
- *"Si un bloque no aporta nada, aprende a no hacer nada y deja pasar la
  información: añadir capas nunca empeora"* (en redes residuales).
- Sobre el atajo con conv 1×1: *"un detalle práctico: la suma exige que F(x) y
  x tengan la misma forma. Cuando el bloque cambia el número de canales o
  reduce la resolución, el atajo lleva una convolución 1×1 con stride 2 que
  adapta las dimensiones. Es el único caso en que el 'atajo' tiene
  parámetros."*
- Sobre 1×1 como cuello de botella: ejemplo numérico — conv 3×3 directa sobre
  256 canales cuesta 590K parámetros; pasar por 1×1 (256→64), luego 3×3
  (64→64) y luego 1×1 (64→256) cuesta 70K, "mismo campo receptivo, ocho veces
  menos parámetros".
- Tabla final "Cuál elegir", con relación arquitectura → parámetros → idea
  clave → cuándo usarla: LeNet-5 (60K, conv+pool); VGG-16 (138M, solo 3×3 muy
  profunda, "docencia y datasets pequeños tipo MNIST"); ResNet-50 (25M, atajos
  residuales, **"el punto de partida por defecto para casi cualquier
  problema"**); Inception-v3 (24M, multiescala, "referencia didáctica, hoy
  demasiado pesada para producción", "objetos de tamaños muy dispares en la
  misma imagen"); MobileNetV2 (3,4M, separables + residual, "móvil, edge o
  inferencia con latencia estricta").
- **Consejo práctico explícito, última línea del PDF**: *"no diseñes una
  arquitectura nueva. Empieza siempre por una ResNet preentrenada y dedica el
  esfuerzo a los datos y al preprocesado, que es donde está la mejora real."*

### Ejemplos de código o plantillas
Este PDF no incluye código PyTorch; son esquemas y tablas comparativas
(diagramas de bloques, tablas de parámetros, la fórmula `y = F(x) + x`, y el
cálculo numérico del cuello de botella 1×1 citado arriba).

---

## 4. Cruce con las restricciones del caso BreastDCEDL

Restricciones tomadas de `SINTESIS_EXPLORACION.md` (secciones 2 y 6).

### 4.1. Dónde la metodología de clase coincide sin problema con el caso

- El **método de seis pasos** de `04_CNN2.pdf` (qué entra/sale, cuántas etapas,
  qué bloque se repite, cómo reduzco, qué cabeza pongo, cómo lo verifico) es
  agnóstico de la arquitectura concreta: no exige ninguna red preentrenada ni
  ningún dataset en particular, así que no choca con la exigencia de "CNN
  propia" del caso.
- El **orden canónico Conv → BN → ReLU → Pooling** de `03_CNN1.pdf` y el bloque
  "Conv 3×3 → BatchNorm → ReLU" de `04_CNN2.pdf` son piezas genéricas de
  construcción, sin dependencia de pesos preentrenados.
- La cabeza con **un solo logit + `BCEWithLogitsLoss`** que exige el caso
  encaja con el vocabulario de la clase: `03_CNN1.pdf` ya describe Sigmoid /
  `BCEWithLogitsLoss` como el par estándar para "clasificación binaria o
  multietiqueta", frente a Softmax / `CrossEntropyLoss` para multiclase.
- El **Paso 6 (verificación)** de `04_CNN2.pdf` — comprobar formas con un
  tensor falso y sobreajustar un lote pequeño antes de lanzar el
  entrenamiento completo — no depende de los datos del caso y es aplicable tal
  cual.
- La **normalización según el propio dataset de entrenamiento, nunca según
  test** ("calcule la media y la desviación de su conjunto de entrenamiento,
  nunca del de test, y aplíquelas a las tres particiones", `04_CNN2.pdf`) es
  del mismo espíritu que la regla del caso de "nunca usar test para ajustar" y
  la exigencia de que "la normalización de la app sea idéntica a la del
  entrenamiento".
- La regla de que **el aumentado de datos debe respetar la semántica de la
  imagen** ("voltear un gato sigue siendo un gato... en imagen médica hay
  orientaciones que no pueden alterarse", `04_CNN2.pdf`) es compatible con la
  advertencia del caso de que cualquier transformación geométrica debe
  aplicarse al tensor de 3 canales completo, nunca canal a canal.
- El **bucle de entrenamiento de 5 pasos** (lote → forward → pérdida →
  backward → step) y las advertencias sobre `zero_grad()`, `model.train()` /
  `model.eval()` y `torch.no_grad()` son procedimiento genérico de PyTorch, sin
  relación con transfer learning ni con ninguna arquitectura prohibida.
- El diagnóstico de síntomas de fallo ("pérdida en NaN", "pérdida plana en
  ln(clases)", "accuracy alta pero mala en test") es agnóstico de si la red es
  propia o preentrenada; aplica igual a una CNN desde cero.

### 4.2. Dónde algo que los PDFs presentan como práctica estándar entra en conflicto con una restricción dura del caso

1. **Transfer learning / fine-tuning con redes preentrenadas.**
   `04_CNN2.pdf` (diapositiva 63) presenta transfer learning y fine-tuning como
   técnica recomendada en general: *"Reutilizarlas es la diferencia entre
   necesitar un millón de ejemplos o unos cientos"*, con dos variantes (extractor
   congelado / fine-tuning) y advierte de usar "la misma normalización con la
   que se entrenó la red original" (implícitamente, la normalización de
   ImageNet). El caso prohíbe explícitamente el transfer learning y los pesos
   preentrenados (`GUIA.md` D1.3 y C2; PDF del caso §7), y además prohíbe
   aplicar normalización de ImageNet a las tres fases DCE por no ser canales
   RGB (`GUIA.md` B5, D2). Conflicto directo en dos frentes: la técnica en sí
   (transfer learning) y la normalización asociada a ella.
2. **"Empieza siempre por una ResNet preentrenada" y "no diseñes una
   arquitectura nueva"** (`05_CNN3.pdf`, última diapositiva, diapositiva 42):
   presentado como consejo práctico general y como "el punto de partida por
   defecto para casi cualquier problema". El caso exige justo lo contrario: una
   arquitectura propia desde cero, y prohíbe explícitamente ResNet,
   EfficientNet, VGG y DenseNet por nombre (`GUIA.md` C2; PDF del caso §7).
3. **Uso de las medias/desviación de ImageNet para `Normalize`.**
   `04_CNN2.pdf` dedica una diapositiva completa (58) a explicar que esos
   valores "son la media y la desviación típica de ImageNet" y que "si usa una
   red preentrenada, debe emplear exactamente esos valores", presentándolo como
   parte del preprocesado estándar para imágenes RGB. El caso, para las
   imágenes DCE-MRI, prohíbe expresamente "normalización de ImageNet"
   (`GUIA.md` B5, D2) porque los tres canales no son colores.
4. **Tratamiento de "Color RGB fotografía habitual" como caso estándar con
   `in_channels=3`.** La tabla de tipos de imagen de `04_CNN2.pdf` (diapositiva
   58) solo contempla imagen médica como "un canal" (tipo radiografía/DICOM) y
   trata los "3 canales" como sinónimo de una foto en color. El caso usa
   `in_channels=3` para tres fases temporales DCE de una única modalidad en
   escala de grises, no para tres bandas de color ni para tres canales de una
   imagen médica de un solo canal repetido. No es una prohibición del PDF
   contra el caso (el PDF no dice "3 canales = siempre color"), pero es un
   supuesto implícito de la tabla que no cubre el escenario real del caso, y
   por eso el propio `GUIA.md` insiste tanto en que "los 3 canales NO son
   colores".

### 4.3. Dónde los PDFs presentan varias opciones sin que el caso obligue a una

Estas quedan abiertas; no se decide aquí ninguna.

- **Cómo reducir la resolución entre etapas:** `MaxPool 2×2` (*"sencillo, sin
  parámetros. Buena opción por defecto"*) frente a **convolución con stride 2**
  (*"aprende cómo reducir. Lo usan ResNet y las redes modernas"*) —
  `03_CNN1.pdf` diapositiva 21 y `04_CNN2.pdf` diapositiva 46. El caso no
  impone ninguna de las dos.
- **Cabeza clasificadora:** **Flatten + Linear** (ata la red a un tamaño de
  entrada fijo, más parámetros) frente a **Global Average Pooling + Linear**
  (agnóstica al tamaño de entrada, menos parámetros) — `04_CNN2.pdf`
  diapositiva 47. El caso exige una única salida (un logit) pero no impone
  cómo se llega a ella desde el volumen convolucional.
- **Repeticiones del bloque conv por etapa:** *"Conv 3×3 → BN → ReLU, una o dos
  veces por etapa"* — `04_CNN2.pdf` diapositiva 44 deja el número (una o dos)
  como variable.
- **Función de activación en capas ocultas:** ReLU por defecto, o
  LeakyReLU/GELU "si aparecen neuronas muertas o en redes muy profundas" —
  `03_CNN1.pdf` diapositiva 22.
- **Cantidad de dropout, y dónde:** *"0,5 en capas densas y 0,1–0,25 en
  convolucionales (o `Dropout2d`)... si tu red ya lleva BatchNorm y buen
  augmentation, a menudo puedes bajarlo o quitarlo"* — `03_CNN1.pdf`
  diapositiva 24. Presentado como rango orientativo, no como valor fijo.
- **Otras palancas de regularización, mencionadas como alternativas
  intercambiables:** weight decay, early stopping, data augmentation, label
  smoothing — `03_CNN1.pdf` diapositiva 24.
- **Tipo de aumentado de datos** (dentro de lo geométrico, que es lo único
  compatible con la regla de "aplicar al tensor completo, nunca canal a
  canal"): recorte aleatorio, volteo horizontal, rotaciones, escalados suaves
  — `04_CNN2.pdf` diapositiva 60. El caso no prohíbe el aumentado geométrico,
  solo el canal-a-canal y el fotométrico tipo ImageNet/color.
- **Uso de un scheduler de learning rate** al final del entrenamiento —
  `04_CNN2.pdf` diapositiva 61, presentado como algo que "suele valer uno o
  dos puntos de accuracy gratis", no como obligatorio.
- **Cuántas etapas y cuántos canales de partida**, más allá de la regla de oro
  de duplicar canales al reducir resolución a la mitad: *"empieza en 32
  (imágenes pequeñas) o 64 (ImageNet)"* — `04_CNN2.pdf` diapositiva 45. Para
  imágenes de 256×256 (el caso), ni la guía de clase ni el caso fijan un
  número concreto de etapas.
- **Bias en la convolución cuando no va seguida de BatchNorm:** la clase solo
  fija `bias=False` como regla si hay BatchNorm inmediatamente después; si no
  se usa BatchNorm en algún bloque, el propio PDF no dice qué hacer con el
  bias en ese caso.

---

## 5. Nota de alcance

Este documento no cruza los PDFs con las preguntas abiertas del enunciado
(D5/PDF §14) ni con las decisiones de diseño listadas en la sección 6 de
`SINTESIS_EXPLORACION.md` más allá de señalar coincidencias/conflictos/opciones
de arquitectura y entrenamiento. No propone qué elegir para la CNN de este
proyecto.
