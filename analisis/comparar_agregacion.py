"""¿Como pasar de los cortes a la paciente? Compara las agregaciones de la GUIA con la VALIDACION.

Agregaciones (utils_caso.agregar_por_paciente):
    media     media de las probabilidades de los cortes (la que usamos y la que pide el profesor)
    mediana   como la media, pero resistente a un corte atipico
    maximo    el corte mas alto: basta un corte sospechoso
    voto      fraccion de cortes que "votan pCR" (probabilidad del corte >= la mediana de todos los cortes,
              para que vote la mitad: un umbral neutral, no ajustado para favorecer a ningun metodo)

Con las predicciones de validacion del modelo final (3 semillas x 5 folds, en `modelo/`): por corte, media
de las semillas y calibracion (a, b de `modelo/umbral_final.json`); despues, cada agregacion. Para cada una:
AUC por fold (media +- desviacion) y diferencia emparejada frente a la media; AUC con todas las pacientes
juntas (IC95), AP, y con el criterio de desescalar (especificidad >= 0,95) la sensibilidad, la precision y
su estabilidad fold a fold. El test no se toca.

Uso (no necesita GPU; funciona en cualquier ordenador con el repositorio):
    python analisis/comparar_agregacion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # para importar los modulos de la raiz del proyecto

import numpy as np
import pandas as pd

from comparar import resumen
from umbral import (auc, barrido, cargar_config, elegir_umbrales, intervalo_auc, logit, metricas,
                    precision_recall, sigmoide)

RAIZ = Path(__file__).resolve().parent.parent   # la raiz del proyecto: este script esta en analisis/


def agregar(cortes: pd.DataFrame, metodo: str, umbral_voto: float) -> pd.DataFrame:
    g = cortes.groupby("patient_id")
    if metodo == "voto":
        prob = (cortes.p >= umbral_voto).groupby(cortes.patient_id).mean()
    else:
        prob = getattr(g.p, {"media": "mean", "mediana": "median", "maximo": "max"}[metodo])()
    return pd.DataFrame({"pCR": g.pCR.first(), "fold": g.fold.first(), "prob": prob})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--carpeta", type=Path, default=RAIZ / "modelo")
    p.add_argument("--especificidad", type=float, default=0.95)
    args = p.parse_args()

    config = json.loads((args.carpeta / "umbral_final.json").read_text(encoding="utf-8"))
    tablas = [cargar_config(args.carpeta, c) for c in config["configuraciones"]]
    cortes = tablas[0][["sample_id", "patient_id", "pCR", "fold"]].copy()
    cortes["p"] = np.mean([t.set_index("sample_id").loc[cortes.sample_id].prob.values for t in tablas], axis=0)
    cal = config["calibracion"]
    cortes["p"] = sigmoide(cal["a"] * logit(cortes.p.values) + cal["b"])
    umbral_voto = float(cortes.p.median())

    print(f"{cortes.patient_id.nunique()} pacientes de validacion, {len(cortes)} cortes; modelo final "
          f"({len(tablas)} semillas x 5 folds, calibrado); voto: un corte vota pCR si su probabilidad >= {umbral_voto:.3f}\n")
    metodos = ["media", "mediana", "maximo", "voto"]
    por_fold, filas = {}, []
    for m in metodos:
        pac = agregar(cortes, m, umbral_voto)
        y, s = pac.pCR.values, pac.prob.values
        por_fold[m] = np.array([auc(y[pac.fold.values == k], s[pac.fold.values == k]) for k in range(5)])
        lo, hi = intervalo_auc(y, s)
        elegido = list(elegir_umbrales(barrido(y, s), args.especificidad).values())[0]
        sens_fold = [metricas(y[pac.fold.values == k], s[pac.fold.values == k], elegido["umbral"])["sensibilidad"]
                     for k in range(5)]
        filas.append({"metodo": m, "auc_fold": por_fold[m].mean(), "desv": por_fold[m].std(ddof=1),
                      "auc_juntas": auc(y, s), "ic": f"{lo:.3f}-{hi:.3f}", "ap": precision_recall(y, s)[2],
                      "valores": len(np.unique(s)), "sens": elegido["sensibilidad"], "espec": elegido["especificidad"],
                      "prec": elegido["precision"], "sens_folds": " ".join(f"{x:.2f}" for x in sens_fold)})

    print(f"{'metodo':<9}{'AUC por fold':>15}{'AUC juntas':>12}{'IC95':>13}{'AP':>7}{'valores':>9}   "
          f"con especificidad >= {args.especificidad:g}: sens  espec  precision   sensibilidad por fold")
    for f in filas:
        print(f"{f['metodo']:<9}{f['auc_fold']:>8.4f} ± {f['desv']:.3f}{f['auc_juntas']:>12.4f}{f['ic']:>13}{f['ap']:>7.3f}"
              f"{f['valores']:>9}   {'':>27}{f['sens']:.2f}  {f['espec']:.2f}  {f['prec']:>9.2f}   {f['sens_folds']}")

    print("\nDiferencia emparejada del AUC por fold frente a la media:")
    for m in metodos[1:]:
        d = por_fold[m] - por_fold["media"]
        media, _, _, ic = resumen(list(d))
        v = "mejor" if media - ic > 0 else "peor" if media + ic < 0 else "no concluyente"
        print(f"  {m:<8} {media:+.4f}  IC95 {media - ic:+.3f} a {media + ic:+.3f}  -> {v}   (por fold: "
              f"{'  '.join(f'{x:+.3f}' for x in d)})")
    print("\n'valores' = cuantas probabilidades distintas da por paciente: con pocas, el umbral no se puede afinar "
          "(el voto solo da fracciones de ~10 cortes).")


if __name__ == "__main__":
    main()
