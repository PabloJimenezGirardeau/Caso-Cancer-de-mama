"""Combina la CNN con las variables clinicas de patients.csv y mide cada parte por separado.

Con los mismos 5 folds de siempre (validacion cruzada por paciente; el test NO se toca),
para cada fold k se ajusta con los otros 4 folds y se mide en el fold k:

    clinico     regresion logistica con las variables de clinicas.py
                (HR, HER2, HRxHER2, edad, log del volumen). Una fila por paciente.
    cnn         la CNN sola: la media de sus probabilidades en las ultimas 10 epocas
                (columna prob_media10 de *_probs_val_final.csv, la que deja entrenar.py).
    combinado   regresion logistica con las variables clinicas + el logit que la CNN dio a
                cada corte. Una fila por CORTE, porque en la app y en la defensa llega un
                solo corte (con los datos clinicos de su paciente).

De donde sale la prediccion de la CNN para ajustar el combinado: de las validaciones de los
otros 4 folds. Cada paciente de entrenamiento tiene una prediccion hecha por una CNN que NO
la vio al entrenar, asi que la regresion aprende cuanto fiarse de la CNN con predicciones
tan honestas como las que vera despues. (Matiz para el informe: esas CNN si vieron al fold k
al entrenar. Es la practica habitual en este tipo de combinacion, llamada "stacking", y el
efecto es pequeño, pero no es una validacion perfectamente limpia.)

Metricas: AUC por PACIENTE (media de las probabilidades de sus cortes; la principal) y por
CORTE (lo que ve la app). Diferencias emparejadas fold a fold, como en comparar.py.

Salida, en resultados/:
    combinar_<nombre>_roc.png        curvas ROC por paciente (la validacion de los 5 folds junta)
    combinar_<nombre>_oof.csv        probabilidad de cada modelo para cada corte de entrenamiento,
                                     dada por el modelo del fold que no lo vio (servira para
                                     elegir el umbral y estudiar la calibracion)
    combinar_<nombre>_resumen.json   AUC por fold y coeficientes

Uso:
    python combinar.py              # solo el modelo clinico (no necesita la CNN)
    python combinar.py bn_normal    # clinico, CNN y combinado (lee bn_normal_fold<k>_probs_val_final.csv)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import clinicas
import graficas
import utils_caso as uc
from comparar import resumen
from entrenar import localizar

RAIZ = Path(__file__).resolve().parent
REFERENCIA = RAIZ / "referencia"
NOMBRES = {"clinico": "solo clinico", "cnn": "solo CNN", "combinado": "CNN + clinico"}


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def regresion() -> LogisticRegression:
    # C=1 es la regularizacion por defecto de sklearn; con tantas filas y 5-6 variables apenas influye
    return LogisticRegression(C=1.0, max_iter=1000)


def cargar_cnn(carpeta: Path, nombre: str, columna: str) -> pd.DataFrame:
    """La probabilidad que la CNN dio a cada corte de entrenamiento cuando estaba en validacion."""
    partes = []
    for k in range(5):
        for base in (carpeta, REFERENCIA):
            ruta = base / f"{nombre}_fold{k}_probs_val_final.csv"
            if ruta.exists():
                partes.append(pd.read_csv(ruta).assign(fold_cnn=k))
                break
        else:
            raise SystemExit(f"Falta {nombre}_fold{k}_probs_val_final.csv (ni en {carpeta} ni en {REFERENCIA}).\n"
                             "Hacen falta los 5 folds: cada corte necesita la prediccion de una CNN que no lo vio.")
    d = pd.concat(partes, ignore_index=True)
    return d[["sample_id", "fold_cnn", columna]].rename(columns={columna: "prob_cnn"})


def auc_por_paciente(cortes: pd.DataFrame, prob) -> float:
    por_paciente = cortes.assign(p=prob).groupby("patient_id").agg(pCR=("pCR", "first"), p=("p", "mean"))
    return roc_auc_score(por_paciente.pCR, por_paciente.p)


def veredicto(difs, a: str, b: str) -> str:
    media, _, _, ic = resumen(list(difs))
    if media - ic > 0:
        v = f"{b} MEJORA a {a}"
    elif media + ic < 0:
        v = f"{b} EMPEORA a {a}"
    else:
        v = "NO CONCLUYENTE (el intervalo incluye el 0)"
    return f"{media:+.4f}  IC95 {media - ic:+.3f} a {media + ic:+.3f}  -> {v}"


def tabla(titulo: str, aucs: dict) -> None:
    print(f"\n{titulo}")
    print(f"{'fold':>6}" + "".join(f"{NOMBRES[m]:>15}" for m in aucs))
    for k in range(5):
        print(f"{k:>6}" + "".join(f"{v[k]:>15.4f}" for v in aucs.values()))
    print(f"{'media':>6}" + "".join(f"{np.mean(v):>15.4f}" for v in aucs.values()))
    print(f"{'IC95':>6}" + "".join(f"{'+-' + format(resumen(list(v))[3], '.3f'):>15}" for v in aucs.values()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configuracion", nargs="?",
                   help="configuracion de la CNN (nombre sin _fold<k>_probs_val_final.csv); sin ella, solo el modelo clinico")
    p.add_argument("--probs", choices=["media10", "ultima"], default="media10",
                   help="prediccion de la CNN: media de las ultimas 10 epocas (por defecto) o solo la ultima")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "resultados",
                   help="donde estan los *_probs_val_final.csv y donde se guarda la salida")
    p.add_argument("--raiz", type=Path, default=RAIZ, help="carpeta con metadata/ (o con breastdcedl/metadata/)")
    args = p.parse_args()

    raiz_meta = localizar("metadata", args.raiz)
    samples = uc.cargar_samples(raiz_meta)
    # solo entrenamiento: el test (fold -1) no se toca
    cortes = samples[samples.fold >= 0][["sample_id", "patient_id", "pCR", "fold"]].reset_index(drop=True)
    pacientes = uc.cargar_patients(raiz_meta).set_index("pid").loc[cortes.patient_id.unique()]
    pacientes["fold"] = cortes.groupby("patient_id").fold.first()
    if (pacientes.pCR != cortes.groupby("patient_id").pCR.first().loc[pacientes.index]).any():
        raise SystemExit("pCR no coincide entre samples.csv y patients.csv")

    con_cnn = args.configuracion is not None
    if con_cnn:
        columna = {"media10": "prob_media10", "ultima": "prob"}[args.probs]
        unidos = cortes.merge(cargar_cnn(args.carpeta, args.configuracion, columna), on="sample_id", how="left")
        if len(unidos) != len(cortes):
            raise SystemExit("Hay cortes repetidos en los ficheros de la CNN")
        if unidos.prob_cnn.isna().any():
            raise SystemExit(f"{unidos.prob_cnn.isna().sum()} cortes de entrenamiento sin prediccion de la CNN")
        if (unidos.fold != unidos.fold_cnn).any():
            raise SystemExit("Algun corte aparece en la validacion de un fold que no es el suyo: "
                             "los ficheros de la CNN no cuadran con samples.csv")
        cortes = unidos
        cortes["logit_cnn"] = logit(cortes.prob_cnn)
    modelos = ["clinico"] + (["cnn", "combinado"] if con_cnn else [])
    nombre = (args.configuracion + ("" if args.probs == "media10" else f"_{args.probs}")) if con_cnn else "clinico"

    print(f"{len(pacientes)} pacientes y {len(cortes)} cortes de entrenamiento, 5 folds; "
          f"variables clinicas: {', '.join(clinicas.VARIABLES)}")
    if con_cnn:
        print(f"CNN: {args.configuracion} ({columna})")

    oof = pd.DataFrame(np.nan, index=cortes.index, columns=modelos)
    auc_paciente = {m: [] for m in modelos}
    auc_corte = {m: [] for m in modelos}
    for k in range(5):
        en_tr, en_va = (cortes.fold != k).to_numpy(), (cortes.fold == k).to_numpy()
        pac_tr = pacientes[pacientes.fold != k]
        prep = clinicas.ajustar_preparacion(pac_tr)        # medias y desviaciones SOLO de entrenamiento
        X = pd.DataFrame(clinicas.preparar(pacientes, prep), index=pacientes.index)
        X_va = X.loc[cortes.patient_id[en_va]].to_numpy()

        clinico = regresion().fit(X.loc[pac_tr.index], pac_tr.pCR)
        oof.loc[en_va, "clinico"] = clinico.predict_proba(X_va)[:, 1]
        if con_cnn:
            X_tr = np.column_stack([X.loc[cortes.patient_id[en_tr]].to_numpy(), cortes.logit_cnn[en_tr]])
            combinado = regresion().fit(X_tr, cortes.pCR[en_tr])
            oof.loc[en_va, "cnn"] = cortes.prob_cnn[en_va].to_numpy()
            oof.loc[en_va, "combinado"] = combinado.predict_proba(
                np.column_stack([X_va, cortes.logit_cnn[en_va]]))[:, 1]

        va = cortes[en_va]
        for m in modelos:
            auc_paciente[m].append(auc_por_paciente(va, oof.loc[en_va, m].to_numpy()))
            auc_corte[m].append(roc_auc_score(va.pCR, oof.loc[en_va, m]))

    tabla("AUC por PACIENTE en validacion (la metrica principal)", auc_paciente)
    tabla("AUC por CORTE en validacion (lo que ve la app: un solo corte)", auc_corte)
    if con_cnn:
        print("\nDiferencias emparejadas, AUC por paciente fold a fold:")
        for a, b in (("clinico", "combinado"), ("cnn", "combinado"), ("cnn", "clinico")):
            difs = np.array(auc_paciente[b]) - np.array(auc_paciente[a])
            print(f"  {NOMBRES[b]} menos {NOMBRES[a]}: {veredicto(difs, NOMBRES[a], NOMBRES[b])}")
            print(f"      por fold: {'  '.join(f'{d:+.3f}' for d in difs)}")

    # Coeficientes con TODAS las pacientes de entrenamiento, solo para interpretarlos (no es el modelo final).
    # Positivo: sube la probabilidad de pCR. exp(coef): por cuanto se multiplican las odds de pCR al
    # pasar de 0 a 1 (HR, HER2) o al subir una desviacion (edad, log_volumen, logit_cnn).
    prep = clinicas.ajustar_preparacion(pacientes)
    X = pd.DataFrame(clinicas.preparar(pacientes, prep), index=pacientes.index)
    ajustes = {"clinico": (regresion().fit(X, pacientes.pCR), clinicas.VARIABLES)}
    if con_cnn:
        X_cortes = np.column_stack([X.loc[cortes.patient_id].to_numpy(), cortes.logit_cnn])
        ajustes["combinado"] = (regresion().fit(X_cortes, cortes.pCR), clinicas.VARIABLES + ("logit_cnn",))
    coeficientes = {}
    for m, (modelo, variables) in ajustes.items():
        coeficientes[m] = {v: float(c) for v, c in zip(variables, modelo.coef_[0])}
        coeficientes[m]["intercepto"] = float(modelo.intercept_[0])
        print(f"\nCoeficientes, {NOMBRES[m]} (ajustado con todo el entrenamiento, para interpretar):")
        for v, c in coeficientes[m].items():
            extra = "" if v == "intercepto" else f"   odds x{np.exp(c):.2f}"
            print(f"  {v:<12} {c:+.3f}{extra}")

    args.carpeta.mkdir(parents=True, exist_ok=True)
    base = args.carpeta / f"combinar_{nombre}"
    por_paciente = cortes[["patient_id", "pCR"]].join(oof).groupby("patient_id").agg(
        {"pCR": "first", **{m: "mean" for m in modelos}})
    graficas.dibujar_rocs({NOMBRES[m]: (por_paciente.pCR, por_paciente[m]) for m in modelos},
                          Path(f"{base}_roc.png"), "ROC por paciente (validacion de los 5 folds)")
    cortes[["sample_id", "patient_id", "pCR", "fold"]].join(oof).to_csv(
        f"{base}_oof.csv", index=False, float_format="%.6f")
    with open(f"{base}_resumen.json", "w", encoding="utf-8") as f:
        json.dump({"configuracion_cnn": args.configuracion, "probs_cnn": columna if con_cnn else None,
                   "variables_clinicas": list(clinicas.VARIABLES),
                   "auc_paciente": auc_paciente, "auc_corte": auc_corte,
                   "media_auc_paciente": {m: float(np.mean(v)) for m, v in auc_paciente.items()},
                   "coeficientes": coeficientes}, f, indent=2)
    print(f"\nResultados en {args.carpeta}  (prefijo combinar_{nombre})")


if __name__ == "__main__":
    main()
