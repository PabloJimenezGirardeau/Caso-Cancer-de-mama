"""Variables clinicas de patients.csv que entran al modelo, y su preparacion.

El profesor permite usar patients.csv, y en la app y en la defensa cada muestra llega con
las mismas columnas. Lo que se usa y por que (analisis en DECISIONES.md):

    HR, HER2      receptores hormonales y HER2 (de la biopsia: se conocen ANTES de la quimio).
                  Es la señal mas fuerte. pCR en entrenamiento: 14 % en HR+/HER2-, 34 % en
                  HR+/HER2+, 38 % en triple negativo (HR-/HER2-) y 57 % en HR-/HER2+.
    HRxHER2       HR por HER2: permite distinguir los 4 grupos (sin el, los HER2+ con HR+ y
                  con HR- tendrian que compartir el mismo efecto del HER2).
    edad          las pacientes mas jovenes responden algo mejor.
    log_volumen   logaritmo del volumen tumoral (tum_vol): los tumores pequeños responden
                  mejor. Las imagenes son recortes alrededor del tumor llevados a 256x256, asi
                  que la CNN no ve el tamaño real; este dato si. Logaritmo porque el volumen
                  es muy asimetrico (de menos de 1 a mas de 100).
                  Ojo (para el informe): en Duke la mascara es incompleta y el volumen sale
                  menor (mediana 6 frente a 15 en I-SPY).

Lo que NO se usa:
    pCR, split, test               la respuesta y el reparto: usarlas seria hacer trampa.
    dataset y la adquisicion       n_xy, n_z, n_times, pre, post_*, slice_thick, xy_spacing:
                                   delatan la cohorte o el escaner, no la biologia del tumor.
    race_white, race_black         la GUIA lo considera un error metodologico.
    menopause                      falta en toda la cohorte spy1 (el hueco delataria la
                                   cohorte), y la edad ya recoge casi lo mismo.
    HR_HER2_STATUS, TripleNeg,     se deducen de HR y HER2.
    HER2pos, HRposHER2neg
    mask_*, sraw, eraw, scol, ecol limites del recorte (tum_vol ya resume el tamaño).

La preparacion (rellenar huecos y estandarizar) se AJUSTA solo con pacientes de
entrenamiento (`ajustar_preparacion`) y se APLICA siempre con la misma funcion (`preparar`).
La app aplicara exactamente la misma, con los valores guardados del entrenamiento: es la
misma regla que con la normalizacion de las imagenes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNAS = ("HR", "HER2", "age", "tum_vol")                          # lo que se lee de patients.csv
VARIABLES = ("HR", "HER2", "HRxHER2", "edad", "log_volumen")        # lo que entra al modelo


def comprobar(pacientes: pd.DataFrame) -> None:
    """Error claro si falta una columna o hay un valor imposible (un hueco si se admite)."""
    faltan = [c for c in COLUMNAS if c not in pacientes.columns]
    if faltan:
        raise ValueError(f"faltan columnas clinicas: {', '.join(faltan)}")
    for c in COLUMNAS:
        v = pd.to_numeric(pacientes[c], errors="coerce")
        if (v.isna() & pacientes[c].notna()).any():
            raise ValueError(f"'{c}' tiene valores que no son numeros")
        if np.isinf(v).any():
            raise ValueError(f"'{c}' tiene valores infinitos")
    for c in ("HR", "HER2"):
        v = pacientes[c].dropna().astype(float)
        if not v.isin([0.0, 1.0]).all():
            raise ValueError(f"'{c}' debe ser 0 o 1 (o estar vacio)")
    edad = pacientes.age.dropna().astype(float)
    if ((edad <= 0) | (edad >= 120)).any():
        raise ValueError("'age' fuera de rango (debe estar entre 0 y 120)")
    if (pacientes.tum_vol.dropna().astype(float) <= 0).any():
        raise ValueError("'tum_vol' debe ser mayor que 0")


def _log_volumen(pacientes: pd.DataFrame) -> pd.Series:
    return np.log(pacientes.tum_vol.astype(float))


def ajustar_preparacion(pacientes: pd.DataFrame) -> dict:
    """Medias y desviaciones de las pacientes de ENTRENAMIENTO (se ignoran los huecos).

    La media sirve para rellenar un dato que falte (en HR y HER2 es la proporcion de
    positivas: un valor intermedio, que no se compromete con ningun subtipo). La edad y el
    log del volumen se estandarizan (media 0, desviacion 1) para que sus coeficientes sean
    comparables; HR y HER2 se quedan en 0/1.
    """
    comprobar(pacientes)
    log_vol = _log_volumen(pacientes)
    return {
        "media": {"HR": float(pacientes.HR.mean()), "HER2": float(pacientes.HER2.mean()),
                  "edad": float(pacientes.age.mean()), "log_volumen": float(log_vol.mean())},
        "desv": {"edad": float(pacientes.age.std()), "log_volumen": float(log_vol.std())},
    }


def preparar(pacientes: pd.DataFrame, prep: dict) -> np.ndarray:
    """Matriz (N, 5) con las VARIABLES en ese orden, lista para la regresion logistica."""
    comprobar(pacientes)
    media, desv = prep["media"], prep["desv"]
    hr = pacientes.HR.astype(float).fillna(media["HR"]).to_numpy()
    her2 = pacientes.HER2.astype(float).fillna(media["HER2"]).to_numpy()
    edad = pacientes.age.astype(float).fillna(media["edad"]).to_numpy()
    log_vol = _log_volumen(pacientes).fillna(media["log_volumen"]).to_numpy()
    return np.column_stack([
        hr,
        her2,
        hr * her2,
        (edad - media["edad"]) / desv["edad"],
        (log_vol - media["log_volumen"]) / desv["log_volumen"],
    ])
