"""Early stopping del proyecto: cuándo parar y qué época se elige (ver HOJA_DE_RUTA.md).

Regla (decidida el 2026-10-14):
    - se vigila el AUC por paciente de validación, SUAVIZADO con una media móvil centrada de 3 épocas
      (la época t vale la media de t-1, t y t+1). El AUC de una sola época oscila mucho (con ~219 pacientes
      por fold, ±0,04 es ruido) y su máximo suele ser un pico de suerte; la media móvil elige una zona buena;
    - la época elegida es la del máximo de esa media móvil (la primera, si hay empate);
    - se para cuando pasan 10 épocas seguidas sin superar ese máximo (paciencia), o al llegar a 60;
    - en la época elegida se anota la diferencia AUC de entrenamiento − AUC de validación; si supera 0,15
      el modelo se marca como SOBREENTRENADO (el ejemplo "bien" del profesor tiene 0,10; el "mal", 0,38).

La misma función sirve para entrenar (época a época, `final=False`) y para analizar después un registro
`*_epocas.csv` ya terminado (`comparar.py`), así que las dos cosas aplican exactamente la misma regla.
"""

from __future__ import annotations

import numpy as np

EPOCAS_MAX = 60
PACIENCIA = 10
VENTANA = 3
DIFERENCIA_MAX = 0.15


def suavizar(valores, ventana: int = VENTANA) -> np.ndarray:
    """Media móvil centrada; en los bordes se promedia lo que haya (la 1.ª época, con la 2.ª)."""
    v = np.asarray(valores, dtype=float)
    h = ventana // 2
    return np.array([v[max(0, i - h): i + h + 1].mean() for i in range(len(v))])


def elegir_epoca(auc_val, paciencia: int = PACIENCIA, ventana: int = VENTANA,
                 final: bool = False) -> tuple[int | None, bool]:
    """Con el AUC de validación de las épocas ya entrenadas (lista, épocas 1..n), devuelve
    (índice de la época elegida, empezando en 0, o None si aún no hay ninguna; si hay que parar).

    Mientras se entrena (`final=False`) la media móvil de la última época aún no está completa (le falta
    la siguiente), así que solo cuentan las épocas con la ventana completa. Al terminar (`final=True`)
    cuentan todas.
    """
    n = len(auc_val) if final else len(auc_val) - ventana // 2
    if n <= 0:
        return None, False
    s = suavizar(auc_val, ventana)[:n]
    if np.isnan(s).all():
        return 0, False
    mejor = int(np.nanargmax(s))
    return mejor, (n - 1 - mejor) >= paciencia


def recorrer(auc_val, paciencia: int = PACIENCIA, ventana: int = VENTANA,
             epocas_max: int = EPOCAS_MAX) -> dict:
    """Aplica la regla a un registro completo, como si se entrenara época a época: en qué época
    habría parado y cuál habría elegido. Sirve también para registros antiguos de 40 épocas fijas."""
    auc_val = list(auc_val)
    for e in range(1, min(len(auc_val), epocas_max) + 1):
        mejor, parar = elegir_epoca(auc_val[:e], paciencia, ventana)
        if parar:
            return {"indice": mejor, "epocas": e, "motivo": "paciencia"}
    e = min(len(auc_val), epocas_max)
    mejor, _ = elegir_epoca(auc_val[:e], paciencia, ventana, final=True)
    motivo = "maximo" if e >= epocas_max else "registro_terminado"
    return {"indice": mejor, "epocas": e, "motivo": motivo}
