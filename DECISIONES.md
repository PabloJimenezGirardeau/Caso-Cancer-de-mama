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

## Pendiente de decidir

Agregación y umbral definitivos · cuántos folds evaluar · aumentado de datos · primer cambio sobre la
base (conv+conv+pool) · inicialización (por ahora, la de PyTorch por defecto) · método de comparación
final entre pérdida normal y ponderada.
