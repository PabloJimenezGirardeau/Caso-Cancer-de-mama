"""Evaluacion final (fase 6): cifras y figuras del modelo final, comparado con la 2.1 y con el modelo clinico.

Dos modos, con las MISMAS cifras y figuras:

    python evaluar_test.py            ENSAYO con la validacion de los 5 folds (no toca el test). Cada paciente la
                                      predicen los modelos que no la vieron al entrenar (sus *_probs_val.csv). Es
                                      algo optimista: la calibracion y el umbral se eligieron con estos mismos datos.
                                      Salida en figuras/validacion/.
    python evaluar_test.py --test     EL TEST (176 pacientes, fold -1). UNA SOLA VEZ, cuando lo decida el usuario:
                                      si ya existe figuras/test/metricas.json, se niega a repetirlo. Cada corte pasa por
                                      predecir.ModeloFinal, lo mismo que en la app. Salida en figuras/test/.

Modelos:
    recorte 128, lr 3e-5 (principal, modelo/)       el de la app; declarado principal ANTES del test (DECISIONES.md, 2026-10-24)
    recorte 128, lr 1e-4 (anterior, modelo_c128/)   el modelo final anterior, como comparacion
    solo cohorte (referencia)          la tasa de pCR de la cohorte de cada paciente, con las de entrenamiento: si la CNN
                                       no la supera, su AUC podria venir solo de reconocer el hospital
    clinico (comparacion)              regresion logistica con HR, HER2, HRxHER2, edad y log(tum_vol); en validacion
                                       se ajusta con los otros 4 folds, en el test con todo el entrenamiento

Cada CNN se aplica tal cual: media de sus 15 modelos en cada corte, calibracion de Platt (a, b) y media de los
cortes de la paciente; pCR = 1 si la probabilidad >= su umbral (umbral_final.json: criterio EQUILIBRADO, maxima
F1 equilibrada en validacion). El umbral de DESESCALAR (especificidad >= 0,95) del recorte se muestra como
comparacion. El clinico no tiene umbral fijado: solo entra en las curvas y en el AUC.

Salida:
    0_matriz_principal.png  la matriz de confusion del modelo principal, grande y con sus metricas
    1_roc.png               curvas ROC (principal, anterior, clinico y "solo cohorte"), con AUC e IC95 y el umbral
    2_matriz_confusion.png  matrices: recorte (equilibrado y desescalar), 2.1 y "siempre pCR = 0"
    3_precision_recall.png  curvas precision-recall (clase pCR) con su AP
    4_umbral.png            sensibilidad, especificidad y precision segun el umbral (recorte)
    5_calibracion.png       calibracion por grupos e histograma de probabilidades por clase
    6_cohortes.png          AUC por cohorte (duke, spy1, spy2)
    7_casos.png             un acierto y un fallo de cada clase: sus imagenes y el cuadrado que mira la red
    metricas.json, resumen.txt, pacientes.csv
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import clinicas
import graficas
import umbral as um
import utils_caso as uc

RAIZ = Path(__file__).resolve().parent
# Claves internas: "recorte" = el modelo PRINCIPAL (modelo/: recorte 128 con lr 3e-5, decidido el 2026-10-24) y "v21" = el
# de comparacion (modelo_c128/: el recorte 128 anterior, lr 1e-4). Los nombres de las claves se conservan para no tocar el resto.
CARPETAS_CNN = {"recorte": RAIZ / "modelo", "v21": RAIZ / "modelo_c128"}
NOMBRES = {"recorte": "recorte 128, lr 3e-5 (principal)", "v21": "recorte 128, lr 1e-4 (anterior)",
           "clinico": "clínico (comparación)", "solo_cohorte": "solo cohorte (referencia)"}
COLORES = {"recorte": graficas.AZUL, "v21": graficas.NARANJA, "clinico": graficas.VERDE, "solo_cohorte": graficas.GRIS}
CURVAS = ("recorte", "v21", "clinico", "solo_cohorte")         # modelos que entran en las curvas ROC y precision-recall
COHORTES = ("duke", "spy1", "spy2")
RECORTE = 128                                   # lado del cuadrado central que mira el modelo principal


# --------------------------------------------------------------------------- #
# Datos y predicciones
# --------------------------------------------------------------------------- #

def localizar(subcarpeta: str) -> Path:
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / subcarpeta).is_dir():
            return base
    raise FileNotFoundError(f"No encuentro '{subcarpeta}/' en {RAIZ} ni en {RAIZ / 'breastdcedl'}")


def leer_config(carpeta: Path) -> dict:
    return json.loads((carpeta / "umbral_final.json").read_text(encoding="utf-8"))


def cnn_validacion(carpeta: Path, cortes: pd.DataFrame) -> np.ndarray:
    """Probabilidad calibrada de cada corte de validacion, con los probs_val guardados (fuera de fold).
    Es la misma cuenta que hace predecir.ModeloFinal: media de las semillas y Platt."""
    cfg = leer_config(carpeta)
    medias = []
    for c in cfg["configuraciones"]:
        t = pd.concat([pd.read_csv(carpeta / f"{c}_fold{k}_probs_val.csv") for k in range(5)]).set_index("sample_id")
        medias.append(t.prob.reindex(cortes.sample_id).to_numpy())
    p = np.mean(medias, axis=0)
    if np.isnan(p).any():
        raise SystemExit(f"{carpeta.name}: {int(np.isnan(p).sum())} cortes de validacion sin prediccion")
    cal = cfg.get("calibracion")
    return um.sigmoide(cal["a"] * um.logit(p) + cal["b"]) if cal else p


def cnn_test(cortes: pd.DataFrame, base_img: Path, lote: int = 128) -> tuple[dict, dict]:
    """Probabilidad calibrada de cada corte del test con predecir.ModeloFinal (lo mismo que la app).
    Las imagenes se leen una vez por lote y pasan por los dos modelos."""
    from predecir import ModeloFinal, cargar_corte
    modelos = {n: ModeloFinal(c) for n, c in CARPETAS_CNN.items()}
    salida = {n: [] for n in modelos}
    for i in range(0, len(cortes), lote):
        x = np.stack([cargar_corte(base_img / r.path_pre, base_img / r.path_early, base_img / r.path_late)
                      for r in cortes.iloc[i:i + lote].itertuples()])
        for n, m in modelos.items():
            salida[n].append(m.probabilidades(x))
        print(f"  {min(i + lote, len(cortes))}/{len(cortes)} cortes", end="\r")
    print()
    return {n: np.concatenate(v) for n, v in salida.items()}, {n: m.checksum for n, m in modelos.items()}


def clinico(pacientes: pd.DataFrame, test: bool) -> pd.Series:
    """Probabilidad del modelo clinico por paciente. La preparacion (medias, desviaciones) se ajusta solo con
    las pacientes con las que se entrena, como en clinicas.py y combinar.py."""
    from sklearn.linear_model import LogisticRegression

    def ajustar_y_predecir(tr: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
        prep = clinicas.ajustar_preparacion(tr)
        modelo = LogisticRegression(C=1.0, max_iter=1000).fit(clinicas.preparar(tr, prep), tr.pCR)
        return pd.Series(modelo.predict_proba(clinicas.preparar(ev, prep))[:, 1], index=ev.index)

    entrenamiento = pacientes[pacientes.fold >= 0]
    if test:
        return ajustar_y_predecir(entrenamiento, pacientes[pacientes.fold == -1])
    return pd.concat([ajustar_y_predecir(entrenamiento[entrenamiento.fold != k], entrenamiento[entrenamiento.fold == k])
                      for k in range(5)])


def solo_cohorte(pacientes: pd.DataFrame, test: bool) -> pd.Series:
    """Referencia "solo cohorte": a cada paciente, la tasa de pCR de su cohorte (duke, spy1, spy2) calculada con las
    pacientes de ENTRENAMIENTO (en validacion, con los otros 4 folds; en el test, con todo el entrenamiento). No mira la
    imagen: si la CNN no la supera, su AUC podria venir solo de reconocer el hospital."""
    entrenamiento = pacientes[pacientes.fold >= 0]
    if test:
        ev = pacientes[pacientes.fold == -1]
        return ev.dataset.map(entrenamiento.groupby("dataset").pCR.mean()).astype(float)
    partes = []
    for k in range(5):
        tasa = entrenamiento[entrenamiento.fold != k].groupby("dataset").pCR.mean()
        ev = entrenamiento[entrenamiento.fold == k]
        partes.append(ev.dataset.map(tasa).astype(float))
    return pd.concat(partes)


def comprobar_fuga(samples: pd.DataFrame) -> dict:
    """Comprueba, antes de evaluar, que no hay fuga entre el test y el entrenamiento: ninguna paciente ni ningun corte
    del test (fold -1) aparece en los folds de entrenamiento, y cada paciente tiene un solo fold. Si falla, se para."""
    folds_por_paciente = samples.groupby("patient_id").fold.nunique()
    test_pac = set(samples[samples.fold == -1].patient_id)
    train_pac = set(samples[samples.fold >= 0].patient_id)
    test_cortes = set(samples[samples.fold == -1].sample_id)
    train_cortes = set(samples[samples.fold >= 0].sample_id)
    r = {"pacientes_test": len(test_pac), "pacientes_entrenamiento": len(train_pac),
         "pacientes_en_ambos": len(test_pac & train_pac), "cortes_en_ambos": len(test_cortes & train_cortes),
         "pacientes_con_mas_de_un_fold": int((folds_por_paciente > 1).sum())}
    if r["pacientes_en_ambos"] or r["cortes_en_ambos"] or r["pacientes_con_mas_de_un_fold"]:
        raise SystemExit(f"FUGA DE DATOS entre el test y el entrenamiento: {r}")
    return r


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    """IC95 de Wilson para una proporcion k/n (mejor que el normal con pocos casos o proporciones extremas)."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    margen = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(centro - margen), float(centro + margen)]


def resumen_curvas(y, s) -> dict:
    lo, hi = um.intervalo_auc(y, s)
    return {"auc": um.auc(y, s), "auc_ic95": [lo, hi], "ap": um.precision_recall(y, s)[2],
            "brier": float(np.mean((np.asarray(s) - y) ** 2))}


def con_medias(m: dict) -> dict:
    """Añade la media de las dos clases (la fila "macro avg" de sklearn), que es la que pide el profesor:
    precision media = (precision de pCR + VPN) / 2; recall medio = (sensibilidad + especificidad) / 2 (= accuracy
    equilibrada); la F1 media es la F1 equilibrada."""
    m["precision_media"] = (m["precision"] + m["vpn"]) / 2
    m["recall_medio"] = (m["sensibilidad"] + m["especificidad"]) / 2
    return m


def evaluar_cnn(y, s, u: float) -> dict:
    m = con_medias(um.metricas(y, s, u))
    m["ic95"] = {"sensibilidad": wilson(m["VP"], m["VP"] + m["FN"]),
                 "especificidad": wilson(m["VN"], m["VN"] + m["FP"]),
                 "precision": wilson(m["VP"], m["VP"] + m["FP"]),
                 "vpn": wilson(m["VN"], m["VN"] + m["FN"]),
                 "accuracy": wilson(m["VP"] + m["VN"], len(y))}
    return {**resumen_curvas(y, s), **m}


def diferencia_auc(y, s1, s2, repeticiones: int = 2000, semilla: int = 0) -> dict:
    """AUC(s1) - AUC(s2) con IC95 por bootstrap emparejado (las mismas pacientes remuestreadas para los dos)."""
    rng = np.random.default_rng(semilla)
    y, s1, s2 = np.asarray(y), np.asarray(s1), np.asarray(s2)
    valores = []
    for _ in range(repeticiones):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            valores.append(um.auc(y[i], s1[i]) - um.auc(y[i], s2[i]))
    return {"diferencia": um.auc(y, s1) - um.auc(y, s2),
            "ic95": [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]}


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def guardar(fig, ruta: Path) -> None:
    fig.savefig(ruta, dpi=120, bbox_inches="tight", facecolor="white")
    _plt().close(fig)


def fig_roc(pac, res, umbrales, titulo, ruta):
    plt = _plt()
    y = pac.pCR.to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 7), constrained_layout=True)
    for n in CURVAS:
        fpr, tpr, a = graficas.calcular_roc(y, pac[n].to_numpy())
        lo, hi = res[n]["auc_ic95"]
        ax.plot(fpr, tpr, color=COLORES[n], lw=2.6 if n == "recorte" else 1.8, ls="--" if n == "solo_cohorte" else "-",
                label=f"{NOMBRES[n]}: AUC {a:.3f} (IC95 {lo:.2f}-{hi:.2f})")
        if n in umbrales:
            ax.scatter([1 - res[n]["especificidad"]], [res[n]["sensibilidad"]], color=COLORES[n], s=70, zorder=5,
                       edgecolor="black", label=f"   umbral {umbrales[n]:.3f}: sens {res[n]['sensibilidad']:.0%}, "
                                                f"espec {res[n]['especificidad']:.0%}")
    ax.plot([0, 1], [0, 1], color=graficas.GRIS, ls=":", lw=1.2, label="azar (AUC 0,5)")
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), aspect="equal",
           xlabel="1 − especificidad  (falsos positivos entre las no pCR)",
           ylabel="sensibilidad  (pCR detectadas entre las pCR)")
    ax.set_title(f"Curva ROC por paciente\n{titulo}", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8.8)
    guardar(fig, ruta)


TIPOS = [["verdaderos negativos", "falsos positivos"], ["falsos negativos", "verdaderos positivos"]]


def dibujar_matriz(ax, m, grande: bool = False) -> None:
    """Matriz 2x2 con filas = clase real (pCR = 0, pCR = 1) y columnas = prediccion; numero y % de la fila."""
    celdas = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
    filas = celdas.sum(axis=1, keepdims=True)
    ax.imshow(celdas / np.maximum(filas, 1), cmap="Blues", vmin=0, vmax=1.15)
    for i in range(2):
        for j in range(2):
            frac = celdas[i, j] / max(filas[i, 0], 1)
            color = "white" if frac > 0.6 else "#14213d"
            ax.text(j, i - 0.1, f"{celdas[i, j]}", ha="center", va="center", fontsize=40 if grande else 24,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"{frac:.0%} de las pCR = {i}\n{TIPOS[i][j]}", ha="center", va="center",
                    fontsize=12 if grande else 9, color=color)
    tam = 13 if grande else 10.5
    ax.set_xticks([0, 1], ["predicción\npCR = 0", "predicción\npCR = 1"], fontsize=tam)
    ax.set_yticks([0, 1], [f"real\npCR = 0\n({filas[0, 0]})", f"real\npCR = 1\n({filas[1, 0]})"], fontsize=tam)


def fig_matriz_principal(m, u, titulo, ruta):
    """La matriz del modelo principal sola, grande, con sus metricas debajo (para el informe y la defensa)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.6, 9.4), constrained_layout=True)
    dibujar_matriz(ax, m, grande=True)
    ax.set_title(f"Matriz de confusión · {NOMBRES['recorte']}\n{titulo}\numbral {u:.3f} (máxima F1 equilibrada en "
                 "validación) · una predicción por paciente", fontsize=13, fontweight="bold")
    prec = f"{m['precision']:.2f}" if m["VP"] + m["FP"] else "—"
    texto = (f"clase pCR = 1:  precisión {prec}   recall (sensibilidad) {m['sensibilidad']:.2f}   F1 {m['f1_pcr']:.2f}\n"
             f"clase pCR = 0:  precisión (VPN) {m['vpn']:.2f}   recall (especificidad) {m['especificidad']:.2f}   "
             f"F1 {m['f1_no_pcr']:.2f}\n"
             f"media de las dos clases:  precisión {m['precision_media']:.2f}   recall {m['recall_medio']:.2f}   "
             f"F1 {m['f1_equilibrada']:.2f}\naccuracy {m['accuracy']:.2f}")
    ax.text(0.5, -0.2, texto, transform=ax.transAxes, ha="center", va="top", fontsize=11.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f3f5", edgecolor=graficas.GRIS))
    guardar(fig, ruta)


def fig_matrices(res, siempre_no, umbrales, titulo, ruta):
    plt = _plt()
    paneles = [(NOMBRES["recorte"], res["recorte"], f"umbral equilibrado {umbrales['recorte']:.3f}"),
               ("principal · umbral de desescalar (comparación)", res["recorte_desescalar"],
                f"umbral {res['recorte_desescalar']['umbral']:.3f} (especificidad ≥ 95 % en validación)"),
               (NOMBRES["v21"], res["v21"], f"umbral equilibrado {umbrales['v21']:.3f}"),
               ("siempre pCR = 0 (referencia)", siempre_no, "no predice pCR = 1 a nadie")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), constrained_layout=True)
    fig.suptitle(f"Matrices de confusión por paciente · {titulo}", fontsize=14, fontweight="bold")
    for ax, (nombre, m, sub) in zip(axes.ravel(), paneles):
        dibujar_matriz(ax, m)
        prec = f"{m['precision']:.0%}" if m["VP"] + m["FP"] else "—"
        ax.set_title(f"{nombre}\n{sub}\nsens {m['sensibilidad']:.0%} · espec {m['especificidad']:.0%} · "
                     f"precisión {prec} · accuracy {m['accuracy']:.0%}\nmedia de las dos clases: precisión "
                     f"{m['precision_media']:.2f} · recall {m['recall_medio']:.2f} · F1 {m['f1_equilibrada']:.2f}",
                     fontsize=10.5)
    guardar(fig, ruta)


def fig_pr(pac, res, titulo, ruta):
    plt = _plt()
    y = pac.pCR.to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    for n in CURVAS:
        rec, prec, ap = um.precision_recall(y, pac[n].to_numpy())
        ax.step(rec, prec, where="post", color=COLORES[n], lw=2.6 if n == "recorte" else 1.8,
                ls="--" if n == "solo_cohorte" else "-",
                label=f"{NOMBRES[n]}: AP {ap:.3f}")
        if "sensibilidad" in res[n] and res[n]["VP"] + res[n]["FP"]:
            ax.scatter([res[n]["sensibilidad"]], [res[n]["precision"]], color=COLORES[n], s=70, zorder=5,
                       edgecolor="black")
    ax.axhline(y.mean(), color=graficas.GRIS, ls="--", label=f"azar = proporción de pCR ({y.mean():.2f})")
    ax.set(xlim=(-0.01, 1.01), ylim=(0, 1.02), xlabel="recall (sensibilidad)", ylabel="precisión (de las que dice pCR, cuántas lo son)")
    ax.set_title(f"Precisión-recall, clase pCR (puntos: umbral elegido)\n{titulo}", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    guardar(fig, ruta)


def fig_umbral(pac, u, u_desescalar, titulo, ruta):
    plt = _plt()
    tabla = um.barrido(pac.pCR.to_numpy(), pac.recorte.to_numpy())
    tabla = tabla[tabla.umbral <= pac.recorte.max() + 0.01]
    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    ax.plot(tabla.umbral, tabla.sensibilidad, color=graficas.NARANJA, lw=2.2, label="sensibilidad (pCR detectadas)")
    ax.plot(tabla.umbral, tabla.especificidad, color=graficas.AZUL, lw=2.2, label="especificidad (no pCR bien descartadas)")
    ax.plot(tabla.umbral, tabla.precision, color=graficas.VERDE, lw=2.2, label="precisión (aciertos cuando dice pCR)")
    ax.axhline(0.95, color=graficas.AZUL, ls=":", lw=1, alpha=0.7)
    ax.axvline(u, color=graficas.ROJO, ls="--", lw=1.8, label=f"umbral elegido {u:.3f} (máxima F1 equilibrada en validación)")
    ax.axvline(u_desescalar, color=graficas.GRIS, ls="--", lw=1.4,
               label=f"umbral de desescalar {u_desescalar:.3f} (especificidad ≥ 95 % en validación)")
    ax.set(ylim=(0, 1.02), xlabel="umbral sobre la probabilidad de la paciente", ylabel="proporción")
    ax.set_title(f"Qué pasa al mover el umbral · {NOMBRES['recorte']}\n{titulo}", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    guardar(fig, ruta)


def fig_calibracion(pac, u, titulo, ruta):
    plt = _plt()
    y = pac.pCR.to_numpy()
    grupos = 10 if len(y) >= 500 else 5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    fig.suptitle(f"Calibración · {titulo}", fontsize=14, fontweight="bold")
    lim = 0
    for n in ("recorte", "v21"):
        s = pac[n].to_numpy()
        cal = um.calibracion(y, s, grupos)
        a1.plot(cal.predicha, cal.real, "o-", color=COLORES[n], lw=2,
                label=f"{NOMBRES[n]} · Brier {np.mean((s - y) ** 2):.4f}")
        lim = max(lim, cal.predicha.max(), cal.real.max())
    lim *= 1.15
    a1.plot([0, lim], [0, lim], ls="--", color=graficas.GRIS, label=f"perfecta · Brier de referencia {y.mean() * (1 - y.mean()):.4f}")
    a1.set(xlim=(0, lim), ylim=(0, lim), xlabel="probabilidad media predicha (por grupo)", ylabel="fracción real de pCR")
    a1.set_title(f"{grupos} grupos de pacientes ordenadas por probabilidad", fontsize=11)
    a1.grid(alpha=0.3); a1.legend(loc="upper left", fontsize=9)

    s = pac.recorte.to_numpy()
    bins = np.linspace(s.min(), s.max(), 25)
    a2.hist(s[y == 0], bins=bins, alpha=0.6, color=graficas.GRIS, label=f"no pCR ({(y == 0).sum()})")
    a2.hist(s[y == 1], bins=bins, alpha=0.6, color=graficas.ROJO, label=f"pCR ({(y == 1).sum()})")
    a2.axvline(u, color=graficas.ROJO, ls="--", lw=1.8, label=f"umbral {u:.3f}")
    a2.set(xlabel="probabilidad de pCR (paciente)", ylabel="pacientes")
    a2.set_title(f"Probabilidades por clase · {NOMBRES['recorte']}", fontsize=11)
    a2.legend(fontsize=9)
    guardar(fig, ruta)


def fig_cohortes(cohortes, titulo, ruta):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    x = np.arange(len(COHORTES)); ancho = 0.26
    for j, n in enumerate(("recorte", "v21", "clinico")):
        vals = [cohortes[c][n]["auc"] for c in COHORTES]
        err = np.array([[v - cohortes[c][n]["auc_ic95"][0], cohortes[c][n]["auc_ic95"][1] - v]
                        for c, v in zip(COHORTES, vals)]).T
        ax.bar(x + (j - 1) * ancho, vals, ancho, yerr=err, capsize=4, color=COLORES[n], alpha=0.85, label=NOMBRES[n])
    ax.axhline(0.5, color=graficas.GRIS, ls=":", label="azar")
    ax.set_xticks(x, [f"{c}\n{cohortes[c]['pacientes']} pacientes · {cohortes[c]['pcr']} pCR "
                      f"({cohortes[c]['pcr'] / cohortes[c]['pacientes']:.0%})" for c in COHORTES])
    ax.set(ylim=(0.2, 1), ylabel="AUC por paciente (barras: IC95)")
    ax.set_title(f"AUC por cohorte\n{titulo}", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9, loc="upper right")
    guardar(fig, ruta)


def elegir_casos(pac, u) -> list[tuple[str, str]]:
    """El caso mas claro de cada celda de la matriz del modelo principal: el acierto mas seguro y el fallo
    mas seguro de cada clase."""
    pos, neg = pac[pac.pCR == 1], pac[pac.pCR == 0]
    casos = []
    for nombre, grupo, mas_alto in (("VERDADERO POSITIVO\n(real pCR = 1, predicción 1)", pos[pos.recorte >= u], True),
                                    ("FALSO POSITIVO\n(real pCR = 0, predicción 1)", neg[neg.recorte >= u], True),
                                    ("FALSO NEGATIVO\n(real pCR = 1, predicción 0)", pos[pos.recorte < u], False),
                                    ("VERDADERO NEGATIVO\n(real pCR = 0, predicción 0)", neg[neg.recorte < u], False)):
        if len(grupo):
            casos.append((nombre, grupo.recorte.idxmax() if mas_alto else grupo.recorte.idxmin()))
    return casos


def fig_casos(pac, cortes, base_img, u, titulo, ruta):
    plt = _plt()
    from matplotlib.patches import Rectangle
    casos = elegir_casos(pac, u)
    fig, axes = plt.subplots(len(casos), 4, figsize=(15, 3.9 * len(casos)), constrained_layout=True, squeeze=False)
    fig.suptitle(f"Casos del modelo principal (el más seguro de cada celda de la matriz) · {titulo}\n"
                 f"cuadrado amarillo: la zona central de {RECORTE}×{RECORTE} que mira la red; corte central de cada paciente",
                 fontsize=13, fontweight="bold")
    lado = 256; ini = (lado - RECORTE) // 2
    for fila_ax, (tipo, pid) in zip(axes, casos):
        suyos = cortes[cortes.patient_id == pid].sort_values("slice_index")
        fila = suyos.iloc[len(suyos) // 2]
        x = uc.cargar_imagen(fila, base_img)
        r = pac.loc[pid]
        imagenes = [(x[0], "PRE"), (x[1], "EARLY"), (x[2], "LATE"), (x[1] - x[0], "EARLY − PRE (realce)")]
        for ax, (img, nombre) in zip(fila_ax, imagenes):
            if nombre.startswith("EARLY −"):
                ax.imshow(img, cmap="inferno", vmin=0, vmax=max(float(img.max()), 1e-3))
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.add_patch(Rectangle((ini, ini), RECORTE, RECORTE, fill=False, edgecolor="yellow", lw=1.4, ls="--"))
            ax.set_title(nombre, fontsize=10); ax.axis("off")
        fila_ax[0].text(-0.08, 0.5, f"{tipo}\n{pid} ({r.cohorte})\nprobabilidad {r.recorte:.3f}\n(umbral {u:.3f})",
                        transform=fila_ax[0].transAxes, ha="right", va="center", fontsize=10, fontweight="bold")
    guardar(fig, ruta)


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true", help="evaluar el TEST (una sola vez)")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "figuras", help="donde se guarda la salida")
    args = p.parse_args()

    conjunto = "test" if args.test else "validacion"
    salida = args.carpeta / conjunto
    if args.test and (salida / "metricas.json").exists():
        raise SystemExit(f"El test ya se evaluó (existe {salida / 'metricas.json'}). Se usa UNA sola vez: "
                         "mira esos resultados en lugar de repetirlo.")
    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base_meta, base_img = localizar("metadata"), localizar("dataset")
    samples = uc.cargar_samples(base_meta)
    fuga = comprobar_fuga(samples)
    print(f"Comprobación de fuga: {fuga['pacientes_test']} pacientes de test y {fuga['pacientes_entrenamiento']} de "
          f"entrenamiento, 0 en común (ni pacientes ni cortes); cada paciente en un solo fold")
    pacientes = uc.cargar_patients(base_meta).set_index("pid")
    pacientes["fold"] = samples.groupby("patient_id").fold.first().reindex(pacientes.index)
    pacientes = pacientes[pacientes.fold.notna()]
    cortes = samples[samples.fold == -1] if args.test else samples[samples.fold >= 0]
    cortes = cortes.reset_index(drop=True)
    if args.test and (cortes.split != "test").any():
        raise SystemExit("Hay cortes con fold -1 que no son del split de test")
    configs = {n: leer_config(c) for n, c in CARPETAS_CNN.items()}
    umbrales = {n: float(cfg["umbral"]) for n, cfg in configs.items()}

    if args.test:
        print(f"TEST: {cortes.patient_id.nunique()} pacientes, {len(cortes)} cortes; prediciendo con los dos modelos...")
        probs, checksums = cnn_test(cortes, base_img)
    else:
        probs, checksums = {n: cnn_validacion(c, cortes) for n, c in CARPETAS_CNN.items()}, None
    for n, v in probs.items():
        cortes[n] = v

    pac = cortes.groupby("patient_id").agg(pCR=("pCR", "first"), fold=("fold", "first"), cortes=("sample_id", "size"),
                                           recorte=("recorte", "mean"), v21=("v21", "mean"))
    if (pac.pCR != pacientes.loc[pac.index, "pCR"]).any():
        raise SystemExit("pCR no coincide entre samples.csv y patients.csv")
    pac["cohorte"] = pacientes.loc[pac.index, "dataset"]
    pac["clinico"] = clinico(pacientes, args.test).reindex(pac.index)
    if pac.clinico.isna().any():
        raise SystemExit("Faltan predicciones del modelo clinico")
    pac["solo_cohorte"] = solo_cohorte(pacientes, args.test).reindex(pac.index)
    if pac.solo_cohorte.isna().any():
        raise SystemExit("Faltan predicciones de la referencia solo cohorte")
    y = pac.pCR.to_numpy()

    # ---- metricas ----
    res = {n: evaluar_cnn(y, pac[n].to_numpy(), umbrales[n]) for n in ("recorte", "v21")}
    u_desescalar = float(configs["recorte"]["umbral_desescalar"])
    res["recorte_desescalar"] = evaluar_cnn(y, pac.recorte.to_numpy(), u_desescalar)
    res["clinico"] = resumen_curvas(y, pac.clinico.to_numpy())
    res["solo_cohorte"] = resumen_curvas(y, pac.solo_cohorte.to_numpy())
    siempre_no = con_medias(um.metricas(y, np.zeros(len(y)), 0.5))
    un_corte = {n: {"auc": um.auc(cortes.pCR, cortes[n]), **um.metricas(cortes.pCR, cortes[n], umbrales[n])}
                for n in ("recorte", "v21")}
    diferencias = {"recorte - 2.1": diferencia_auc(y, pac.recorte, pac.v21),
                   "recorte - clinico": diferencia_auc(y, pac.recorte, pac.clinico),
                   "recorte - solo cohorte": diferencia_auc(y, pac.recorte, pac.solo_cohorte)}
    cohortes = {}
    for c in COHORTES:
        d = pac[pac.cohorte == c]
        cohortes[c] = {"pacientes": len(d), "pcr": int(d.pCR.sum())}
        for n in ("recorte", "v21", "clinico"):
            cohortes[c][n] = resumen_curvas(d.pCR.to_numpy(), d[n].to_numpy())

    # ---- resumen en texto ----
    titulo = (f"TEST · {len(pac)} pacientes" if args.test else
              f"VALIDACIÓN (ensayo, 5 folds fuera de fold) · {len(pac)} pacientes")
    lineas = [f"{titulo} · {len(cortes)} cortes · {int(y.sum())} pCR ({y.mean():.1%})",
              "" if args.test else "AVISO: ensayo con la validacion; la calibracion y el umbral se eligieron con estos "
                                   "mismos datos, asi que las cifras son algo optimistas. El juez es el test.", ""]
    for n in CURVAS:
        r = res[n]
        lineas.append(f"{NOMBRES[n]:<26} AUC {r['auc']:.3f} (IC95 {r['auc_ic95'][0]:.3f}-{r['auc_ic95'][1]:.3f})   "
                      f"AP {r['ap']:.3f} (azar {y.mean():.3f})   Brier {r['brier']:.4f} (referencia {y.mean() * (1 - y.mean()):.4f})")
        if args.test and n in configs:
            lineas.append(f"{'':<26} en validacion era AUC {configs[n]['auc_validacion']:.3f}")
    lineas.append("")
    for n, u, criterio in (("recorte", umbrales["recorte"], "equilibrado"),
                           ("recorte_desescalar", u_desescalar, "desescalar, comparacion"),
                           ("v21", umbrales["v21"], "equilibrado")):
        r = res[n]; ic = r["ic95"]
        f = lambda clave: f"{r[clave]:.0%} ({ic[clave][0]:.0%}-{ic[clave][1]:.0%})"
        nombre = NOMBRES[n] if n in NOMBRES else "recorte 128"
        lineas += [f"{nombre} con el umbral {u:.3f} ({criterio}):",
                   f"   matriz: VP {r['VP']}  FN {r['FN']}  VN {r['VN']}  FP {r['FP']}",
                   f"   sensibilidad {f('sensibilidad')}   especificidad {f('especificidad')}   "
                   f"precision {f('precision') if r['VP'] + r['FP'] else '— (no dice pCR a nadie)'}   VPN {f('vpn')}",
                   f"   F1 pCR {r['f1_pcr']:.2f}   F1 no pCR {r['f1_no_pcr']:.2f}   F1 equilibrada {r['f1_equilibrada']:.2f}   "
                   f"accuracy {f('accuracy')}",
                   f"   media de las dos clases (macro): precision {r['precision_media']:.2f}   recall {r['recall_medio']:.2f}   "
                   f"F1 {r['f1_equilibrada']:.2f}"]
        if n in un_corte:
            lineas.append(f"   un solo corte (como en la app): AUC {un_corte[n]['auc']:.3f}, sensibilidad "
                          f"{un_corte[n]['sensibilidad']:.0%}, especificidad {un_corte[n]['especificidad']:.0%}")
    lineas += [f"siempre 'no pCR': accuracy {siempre_no['accuracy']:.0%}; media de las dos clases: precision "
               f"{siempre_no['precision_media']:.2f}, recall {siempre_no['recall_medio']:.2f}, F1 {siempre_no['f1_equilibrada']:.2f}", ""]
    for k, d in diferencias.items():
        lineas.append(f"diferencia de AUC {k}: {d['diferencia']:+.3f} (IC95 {d['ic95'][0]:+.3f} a {d['ic95'][1]:+.3f}) -> "
                      + ("el intervalo excluye el 0" if d["ic95"][0] > 0 or d["ic95"][1] < 0 else "el intervalo incluye el 0"))
    lineas += ["", "AUC por cohorte:   " + "   ".join(f"{NOMBRES[n].split(' (')[0]:>12}" for n in ("recorte", "v21", "clinico"))]
    for c in COHORTES:
        lineas.append(f"  {c:<5} ({cohortes[c]['pacientes']:>4} pac., {cohortes[c]['pcr']:>3} pCR) "
                      + "   ".join(f"{cohortes[c][n]['auc']:>12.3f}" for n in ("recorte", "v21", "clinico")))
    print("\n".join(lineas))

    # ---- figuras ----
    print("\nDibujando figuras...")
    fig_matriz_principal(res["recorte"], umbrales["recorte"], titulo, salida / "0_matriz_principal.png")
    fig_roc(pac, res, umbrales, titulo, salida / "1_roc.png")
    fig_matrices(res, siempre_no, umbrales, titulo, salida / "2_matriz_confusion.png")
    fig_pr(pac, res, titulo, salida / "3_precision_recall.png")
    fig_umbral(pac, umbrales["recorte"], u_desescalar, titulo, salida / "4_umbral.png")
    fig_calibracion(pac, umbrales["recorte"], titulo, salida / "5_calibracion.png")
    fig_cohortes(cohortes, titulo, salida / "6_cohortes.png")
    fig_casos(pac, cortes, base_img, umbrales["recorte"], titulo, salida / "7_casos.png")

    # ---- ficheros ----
    (salida / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    pac.assign(clase_recorte=np.where(pac.recorte >= umbrales["recorte"], "pCR", "no pCR")).to_csv(
        salida / "pacientes.csv", float_format="%.6f")
    with open(salida / "metricas.json", "w", encoding="utf-8") as f:
        json.dump({"conjunto": conjunto, "fecha": datetime.now().isoformat(timespec="seconds"),
                   "pacientes": len(pac), "cortes": len(cortes), "pcr": int(y.sum()), "principal": "recorte",
                   "umbrales": umbrales, "umbral_desescalar_recorte": u_desescalar, "configuraciones": {n: cfg["configuraciones"] for n, cfg in configs.items()},
                   "checksums": checksums, "modelos": res, "siempre_no_pcr": siempre_no, "un_corte": un_corte,
                   "diferencias_auc": diferencias, "cohortes": cohortes, "comprobacion_fuga": fuga,
                   "segundos": round(time.time() - t0, 1)}, f, indent=2, ensure_ascii=False, default=float)
    print(f"Listo en {time.time() - t0:.0f} s. Resultados en {salida}")


if __name__ == "__main__":
    main()
