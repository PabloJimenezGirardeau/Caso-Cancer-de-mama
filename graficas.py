"""Graficas del proyecto. Se amplia segun se vayan necesitando.

    dibujar_curva_perdida          perdida de entrenamiento por epoca (prueba de memorizacion)
    dibujar_curvas_entrenamiento   perdida y AUC por epoca, entrenamiento frente a validacion
    calcular_roc / dibujar_roc     curva ROC (por paciente)
    dibujar_matriz_confusion       matriz de confusion (por paciente)
"""

from __future__ import annotations

import csv
from pathlib import Path

AZUL, NARANJA, ROJO, GRIS = "#1f77b4", "#e76f51", "#c1121f", "#6c757d"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def leer_columna(ruta_csv: Path, columna: str) -> list[float]:
    with open(ruta_csv, encoding="utf-8") as f:
        return [float(fila[columna]) for fila in csv.DictReader(f)]


# --------------------------------------------------------------------------- #
# Prueba de memorizacion
# --------------------------------------------------------------------------- #

def dibujar_curva_perdida(perdidas: list[float], ruta_png: Path, titulo: str,
                          perdida_azar: float | None = None) -> None:
    """Curva de perdida por epoca, en dos vistas:

    - izquierda, escala normal: se ve la forma general (meseta, descenso, picos);
    - derecha, escala logaritmica: se ve como sigue bajando cuando ya es casi 0.

    `perdida_azar`: la perdida de una red que no sabe nada (linea de referencia).
    """
    plt = _plt()
    epocas = list(range(1, len(perdidas) + 1))
    fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    fig.suptitle(titulo, fontsize=15, fontweight="bold")

    # meseta inicial: epocas seguidas, desde la primera, con perdida casi igual a la del azar
    fin_meseta = 0
    if perdida_azar:
        while fin_meseta < len(perdidas) and perdidas[fin_meseta] >= 0.97 * perdida_azar:
            fin_meseta += 1
    # el salto hacia arriba mas grande entre dos epocas seguidas (inestabilidad)
    salto, epoca_salto = 0.0, None
    for i in range(1, len(perdidas)):
        if perdidas[i] - perdidas[i - 1] > salto:
            salto, epoca_salto = perdidas[i] - perdidas[i - 1], i + 1

    for ax, escala in ((izq, "linear"), (der, "log")):
        ax.plot(epocas, perdidas, color=AZUL, lw=2.2, label="pérdida de entrenamiento")
        if perdida_azar:
            ax.axhline(perdida_azar, color=GRIS, ls="--", lw=1.4,
                       label=f"red que no sabe nada ({perdida_azar:.3f})")
        ax.set_yscale(escala)
        ax.set_xlabel("época", fontsize=11)
        ax.set_ylabel("pérdida" + (" (escala logarítmica)" if escala == "log" else ""), fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(0, len(perdidas) + 1)
    izq.set_ylim(bottom=0)
    der.set_ylim(bottom=min(perdidas) * 0.6, top=max(perdidas) * 1.6)
    izq.set_title("Escala normal: la forma general", fontsize=12)
    der.set_title("Escala logarítmica: el final de la bajada", fontsize=12)

    if fin_meseta > 1:
        izq.axvspan(0.5, fin_meseta + 0.5, color="#ffd166", alpha=0.35)
        izq.text(fin_meseta / 2 + 0.5, max(perdidas) * 0.72, "meseta:\nla red aún\nno aprende",
                 ha="center", va="center", fontsize=9.5, color="#7a5c00")
    if epoca_salto and salto > 0.25 * (perdida_azar or 1.0):
        izq.annotate("pico: salto demasiado\ngrande del optimizador",
                     xy=(epoca_salto, perdidas[epoca_salto - 1]),
                     xytext=(epoca_salto + 12, perdidas[epoca_salto - 1] * 0.92),
                     arrowprops=dict(arrowstyle="->", color=ROJO), fontsize=9.5, color=ROJO)
    der.annotate(f"{perdidas[-1]:.4f}\n(época {len(perdidas)})", xy=(len(perdidas), perdidas[-1]),
                 xytext=(len(perdidas) * 0.70, perdidas[-1] * 6), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=AZUL))
    izq.legend(loc="upper right", fontsize=10)

    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=110, bbox_inches="tight", facecolor="white")


# --------------------------------------------------------------------------- #
# Entrenamiento real: curvas, ROC y matriz de confusion
# --------------------------------------------------------------------------- #

def dibujar_curvas_entrenamiento(ruta_csv: Path, ruta_png: Path, titulo: str,
                                 perdida_ref: float | None = None, epoca_mejor: int | None = None) -> None:
    """Dos paneles con el mismo eje de epocas:

    - izquierda: perdida de entrenamiento y de validacion;
    - derecha: AUC por paciente en entrenamiento y en validacion.

    La separacion entre las curvas de entrenamiento y de validacion es lo que delata
    el sobreajuste. `perdida_ref`: la mejor perdida posible SIN mirar la imagen
    (una red que solo conociera la proporcion de pCR).
    """
    plt = _plt()
    ep = leer_columna(ruta_csv, "epoca")
    fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    fig.suptitle(titulo, fontsize=15, fontweight="bold")

    izq.plot(ep, leer_columna(ruta_csv, "perdida_train"), color=AZUL, lw=2.2, label="entrenamiento")
    izq.plot(ep, leer_columna(ruta_csv, "perdida_val"), color=NARANJA, lw=2.2, label="validación")
    if perdida_ref:
        izq.axhline(perdida_ref, color=GRIS, ls="--", lw=1.4,
                    label=f"mejor red que no mira la imagen ({perdida_ref:.3f})")
    izq.set_title("Pérdida", fontsize=12)
    izq.set_ylabel("pérdida", fontsize=11)

    der.plot(ep, leer_columna(ruta_csv, "auc_train"), color=AZUL, lw=2.2, label="entrenamiento")
    der.plot(ep, leer_columna(ruta_csv, "auc_val"), color=NARANJA, lw=2.2, label="validación")
    der.axhline(0.5, color=GRIS, ls="--", lw=1.4, label="azar (0,5)")
    der.set_ylim(0.3, 1.02)
    der.set_title("AUC por paciente", fontsize=12)
    der.set_ylabel("AUC", fontsize=11)

    for ax in (izq, der):
        ax.set_xlabel("época", fontsize=11)
        ax.grid(alpha=0.3)
        if epoca_mejor:
            ax.axvline(epoca_mejor, color=ROJO, ls=":", lw=1.6, label=f"mejor época de validación ({epoca_mejor})")
        ax.legend(fontsize=9.5)
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=110, bbox_inches="tight", facecolor="white")


def calcular_roc(y, puntuacion):
    """Curva ROC calculada a mano: devuelve (tasa de falsos positivos, sensibilidad, AUC).

    Recorre todos los umbrales posibles de mayor a menor puntuacion; en cada uno mide
    que fraccion de los positivos reales se detecta (sensibilidad) y que fraccion de
    los negativos reales se marca por error (tasa de falsos positivos).
    """
    import numpy as np
    y = np.asarray(y); s = np.asarray(puntuacion, dtype=float)
    orden = np.argsort(-s, kind="mergesort")
    y, s = y[orden], s[orden]
    vp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    ultimos = np.r_[np.where(np.diff(s))[0], len(s) - 1]       # el ultimo de cada grupo de empates
    tpr = np.r_[0.0, vp[ultimos] / vp[-1]]
    fpr = np.r_[0.0, fp[ultimos] / fp[-1]]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def dibujar_roc(y, puntuacion, ruta_png: Path, titulo: str, umbral: float = 0.5) -> None:
    import numpy as np
    plt = _plt()
    y = np.asarray(y); s = np.asarray(puntuacion, dtype=float)
    fpr, tpr, auc = calcular_roc(y, s)
    pred = s >= umbral
    sens = float((pred & (y == 1)).sum() / (y == 1).sum())
    fpr_u = float((pred & (y == 0)).sum() / (y == 0).sum())

    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    ax.plot(fpr, tpr, color=AZUL, lw=2.4, label=f"modelo (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1.4, label="azar (AUC = 0,5)")
    ax.scatter([fpr_u], [sens], color=ROJO, zorder=5, s=60, label=f"umbral {umbral:g}")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("1 − especificidad  (falsos positivos entre los no pCR)", fontsize=11)
    ax.set_ylabel("sensibilidad  (pCR detectadas entre las pCR)", fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=10)
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=110, bbox_inches="tight", facecolor="white")


def dibujar_matriz_confusion(vp: int, vn: int, fp: int, fn: int, ruta_png: Path, titulo: str) -> None:
    import numpy as np
    plt = _plt()
    total = vp + vn + fp + fn
    celdas = np.array([[vn, fp], [fn, vp]])
    fig, ax = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    ax.imshow(celdas, cmap="Blues", vmin=0, vmax=max(1, celdas.max()) * 1.15)
    nombres = [["verdadero negativo\n(no pCR, dijo no pCR)", "falso positivo\n(no pCR, dijo pCR)"],
               ["falso negativo\n(pCR, dijo no pCR)", "verdadero positivo\n(pCR, dijo pCR)"]]
    for i in range(2):
        for j in range(2):
            v = int(celdas[i, j])
            color = "white" if v > celdas.max() * 0.6 else "#14213d"
            ax.text(j, i - 0.08, f"{v}", ha="center", va="center", fontsize=26, fontweight="bold", color=color)
            ax.text(j, i + 0.22, nombres[i][j], ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks([0, 1], ["dijo no pCR", "dijo pCR"], fontsize=11)
    ax.set_yticks([0, 1], ["era no pCR", "era pCR"], fontsize=11)
    sens = vp / (vp + fn) if (vp + fn) else float("nan")
    esp = vn / (vn + fp) if (vn + fp) else float("nan")
    ax.set_title(f"{titulo}\nsensibilidad {100 * sens:.0f} %  ·  especificidad {100 * esp:.0f} %  ·  "
                 f"accuracy {100 * (vp + vn) / total:.0f} %", fontsize=11.5, fontweight="bold")
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=110, bbox_inches="tight", facecolor="white")
