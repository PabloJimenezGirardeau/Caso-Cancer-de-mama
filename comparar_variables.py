"""¿Que columnas de patients.csv merece la pena meter en el modelo clinico?

Compara, con los mismos 5 folds (solo entrenamiento; el test no se toca), la regresion
logistica con las 5 variables elegidas (clinicas.py) frente a la misma regresion con columnas
añadidas: todas las demas a la vez, o cada grupo por separado. Para cada opcion:

    AUC val       AUC por paciente en validacion (lo que importa)
    AUC entreno   AUC sobre las propias pacientes de entrenamiento. Si sube mucho mas que el
                  de validacion, las columnas nuevas sirven para memorizar, no para predecir.
    diferencia    emparejada fold a fold frente a las 5 variables, con su IC95

Las columnas añadidas entran tal cual, estandarizadas con las medias de entrenamiento; un hueco
se rellena con la media y se añade un indicador de hueco. pCR, split y test no se prueban
nunca: son la respuesta y el reparto.

Uso:
    python comparar_variables.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import clinicas
import utils_caso as uc
from comparar import resumen
from entrenar import localizar

RAIZ = Path(__file__).resolve().parent
GRUPOS = {
    "adquisicion (escaner)": ["n_xy", "n_z", "n_times", "pre", "post_early", "post_late", "slice_thick", "xy_spacing"],
    "coordenadas del recorte": ["mask_start", "mask_end", "sraw", "eraw", "scol", "ecol"],
    "dimensiones del recorte en mm": ["ext_z_mm", "ext_fil_mm", "ext_col_mm"],
    "dimensiones del recorte en pixeles": ["ext_z_px", "ext_fil_px", "ext_col_px"],
    "menopausia": ["menopause"],
    "cohorte": ["spy1", "spy2"],
    "raza": ["race_white", "race_black"],
    "indicadores de subtipo": ["TripleNeg", "HER2pos", "HRposHER2neg"],
}
# todas las columnas originales que no son pid, pCR, split ni test (edad y volumen tambien en bruto)
ORIGINALES = (GRUPOS["adquisicion (escaner)"] + GRUPOS["coordenadas del recorte"] + GRUPOS["menopausia"]
              + GRUPOS["cohorte"] + GRUPOS["raza"] + GRUPOS["indicadores de subtipo"] + ["age", "tum_vol"])


def con_derivadas(p: pd.DataFrame) -> pd.DataFrame:
    """Cohorte en 0/1 y dimensiones del recorte en cada eje, en pixeles y en mm."""
    p = p.copy()
    p["spy1"] = (p.dataset == "spy1").astype(float)
    p["spy2"] = (p.dataset == "spy2").astype(float)
    p["ext_z_px"] = p.mask_end - p.mask_start
    p["ext_fil_px"] = p.eraw - p.sraw
    p["ext_col_px"] = p.ecol - p.scol
    p["ext_z_mm"] = p.ext_z_px * p.slice_thick
    p["ext_fil_mm"] = p.ext_fil_px * p.xy_spacing
    p["ext_col_mm"] = p.ext_col_px * p.xy_spacing
    return p


def matriz(tr: pd.DataFrame, d: pd.DataFrame, extra: list[str]) -> np.ndarray:
    """Las 5 variables de clinicas.py + las columnas `extra`, preparadas SOLO con `tr`."""
    partes = [clinicas.preparar(d, clinicas.ajustar_preparacion(tr))]
    for c in extra:
        v, vt = d[c].astype(float), tr[c].astype(float)
        if not vt.std() > 0:                 # constante en entrenamiento (p. ej. pre = 0 siempre)
            continue
        partes.append(((v.fillna(vt.mean()) - vt.mean()) / vt.std()).to_numpy()[:, None])
        if vt.isna().any():
            partes.append(v.isna().astype(float).to_numpy()[:, None])
    return np.hstack(partes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raiz", type=Path, default=RAIZ, help="carpeta con metadata/ (o con breastdcedl/metadata/)")
    args = p.parse_args()

    raiz_meta = localizar("metadata", args.raiz)
    samples = uc.cargar_samples(raiz_meta)
    fold = samples[samples.fold >= 0].groupby("patient_id").fold.first()      # el test (fold -1) no se toca
    pacientes = con_derivadas(uc.cargar_patients(raiz_meta).set_index("pid").loc[fold.index])
    pacientes["fold"] = fold

    opciones = {"las 5 de clinicas.py": [], "TODAS las columnas": ORIGINALES,
                "todas menos la raza": [c for c in ORIGINALES if c not in GRUPOS["raza"]],
                **{f"+ {g}": cols for g, cols in GRUPOS.items()}}
    print(f"{len(pacientes)} pacientes de entrenamiento, 5 folds, regresion logistica\n")
    print(f"{'opcion':<36}{'variables':>10}{'AUC val':>9}{'AUC entreno':>13}   diferencia frente a las 5 (IC95)")
    referencia = None
    for nombre, extra in opciones.items():
        val, ent = [], []
        for k in range(5):
            tr, va = pacientes[pacientes.fold != k], pacientes[pacientes.fold == k]
            X_tr = matriz(tr, tr, extra)
            modelo = LogisticRegression(C=1.0, max_iter=5000).fit(X_tr, tr.pCR)
            val.append(roc_auc_score(va.pCR, modelo.predict_proba(matriz(tr, va, extra))[:, 1]))
            ent.append(roc_auc_score(tr.pCR, modelo.predict_proba(X_tr)[:, 1]))
        val = np.array(val)
        linea = f"{nombre:<36}{X_tr.shape[1]:>10}{val.mean():>9.4f}{np.mean(ent):>13.4f}"
        if referencia is None:
            referencia = val
        else:
            media, _, _, ic = resumen(list(val - referencia))
            v = "MEJORA" if media - ic > 0 else "EMPEORA" if media + ic < 0 else "no concluyente"
            linea += f"   {media:+.4f} ({media - ic:+.3f} a {media + ic:+.3f}) {v}"
        print(linea)


if __name__ == "__main__":
    main()
