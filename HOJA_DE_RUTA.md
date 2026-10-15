# Hoja de ruta del caso (desde el 14 de octubre de 2026)

Sustituye al plan anterior tras el correo del profesor "Pistas caso de uso" y la lectura del artículo del dataset
(Fridman et al. 2026, *Scientific Data* 13:264). Las decisiones y cifras de cada paso se anotan en `DECISIONES.md`, y la
historia, en `DIARIO.md`.

## Objetivo

Una **CNN propia, entrenada desde cero, solo con imagen** (PRE, EARLY, LATE), que prediga la pCR por **paciente**
(media de sus cortes), con:

- AUC por paciente en test dentro de lo razonable para cortes 2D y tres cohortes: **0,60-0,70**;
- **sobreajuste controlado y demostrado**: curvas de entrenamiento y validación que suben juntas, y una diferencia pequeña
  entre los dos AUC en la época elegida;
- **early stopping** documentado (criterio, época elegida de cada fold);
- **umbral elegido con validación** y con un criterio clínico explícito;
- **test una sola vez**, al final, con el informe completo;
- una **app desplegada** que reproduzca exactamente el pipeline.

Las variables de `patients.csv` sirven de **comparación**, no de modelo principal: modelo solo clínico (0,707) y combinado
CNN + clínico (0,711) en validación cruzada.

## Decisiones tomadas (2026-10-14)

1. **Criterio de parada:** se vigila el **AUC por paciente de validación** (la métrica que se evalúa). Se para tras
   **10 épocas sin mejorar**, con un **máximo de 60**. Se guarda el modelo de la época elegida y se anotan la época, el AUC de
   validación, el AUC de entrenamiento y su diferencia.
2. **Modelo final:** el **conjunto de los 5 modelos de los folds** (cada uno en su época elegida), promediando sus
   probabilidades. No hace falta reentrenar sin validación, y suele ser más estable que un solo modelo.
3. **Criterio del umbral** (lo que piden la GUIA y el profesor): el uso previsto es **ayudar a decidir si se puede desescalar
   el tratamiento**. El error grave es un **falso positivo**: decir "pCR" a una paciente que conserva tumor y que podría quedar
   infratratada. Por eso se prioriza la **especificidad y la precisión de la clase pCR**, como los autores del dataset
   (especificidad 0,95, precisión 0,70, sensibilidad 0,27). El umbral se elige con las predicciones de validación de los
   5 folds, y se compara con alternativas (máximo de F1, F1 equilibrada) para justificar la elección.

## Fases

| Fase | Contenido | GPU | Punto de la rúbrica |
|---|---|---|---|
| 0. Datos entendidos | Figuras: las tres fases y el mapa de realce, lavado, desbalance (≈30 % pCR), cohortes | No | Datos entendidos |
| 1. Early stopping | `entrenar.py`: parada con paciencia, época elegida, diferencia de AUC, aviso de modelo sobreentrenado; `comparar.py` adaptado (media ± desviación en la época elegida) | No | Entrenamiento reproducible |
| 2. Entrenamiento "BIEN" | De una en una: learning rate más bajo, dropout en las convoluciones, aumentado, realce y lavado, learning rate decreciente. Buscar curvas que suban juntas | Sí | Sobreajuste controlado, arquitectura justificada |
| 3. Pérdidas | Normal frente a ponderada con la configuración elegida; qué le pasa a la sensibilidad | Sí | Pérdida normal frente a ponderada |
| 4. Agregación y umbral | Con las predicciones de validación: media de cortes (y alternativas), umbral según el criterio clínico, F1 por clase y equilibrada | No | Umbral y agregación |
| 5. Modelo final | Los 5 modelos de los folds juntos | (fases 2-3) | Reproducible |
| 6. Test, una vez | Matriz, AUC con IC, precision/recall/F1, frente a "siempre no pCR", por cohorte, calibración, casos fallados con imágenes, frente al modelo clínico y a la bibliografía | No | Test completo, más allá de la cifra |
| 7. App | Streamlit o Gradio desplegada, solo imagen, misma normalización, validación de entradas | No | App fiel al pipeline |
| 8. Informe y 5 diapositivas | Toda la historia, también lo que no funcionó | No | Comparación y honestidad |

## Qué se reutiliza de antes

Partición y evaluación por paciente, `comparar.py`, `analizar_cohortes.py` (por cohorte, subtipo y tamaño), `combinar.py` y
`clinicas.py` (comparación con los datos clínicos), y los experimentos 1-4 como historia de la exploración.

## Notas del artículo del dataset para el informe

- Es un *data descriptor*: no trae resultados de modelos. Las cifras de AUC 0,72 que cita el profesor vienen de otro trabajo.
- Particiones originales con pCR: entrenamiento 1.099, validación 177, test 176. Nuestro entrenamiento y test son los suyos; su
  validación es probablemente el conjunto privado de la defensa.
- Heterogeneidad: más de 22 centros, tres fabricantes, sin armonizar la intensidad; 60 s entre imágenes en Duke y 120 s en I-SPY;
  en Duke, caja del tumor mayor en vez de máscara y volumen por clases de tamaño; definiciones de HR y HER2 distintas por cohorte.
- La resonancia pretratamiento predice peor que la tomada durante o después de la quimioterapia.
