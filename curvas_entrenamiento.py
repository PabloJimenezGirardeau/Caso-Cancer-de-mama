"""Curvas de entrenamiento (perdida y AUC por epoca) de los modelos finales, a partir de referencia/*_epocas.csv.

Para cada familia de modelos (3 semillas x 5 folds) dibuja:
    <familia>_folds.png    una columna por fold; arriba la perdida, abajo el AUC; las 3 semillas en colores,
                           entrenamiento en linea continua y validacion discontinua; el punto es la epoca elegida
                           por el early stopping (parada.recorrer)
y dos figuras mas:
    comparativa.png        los 15 modelos de cada familia superpuestos (2.1 frente a recorte 128)
    modelo_final.png       la figura para enseñar: modelo final por fold, media de las 3 semillas en grueso

La linea gris de puntos de la perdida es la mejor perdida posible SIN mirar la imagen: la entropia de la proporcion
de pCR de los cortes de entrenamiento (calculada desde samples.csv). Una red por encima de esa linea no ha aprendido
nada util de la imagen.

Uso (no necesita GPU ni PyTorch):
    python curvas_entrenamiento.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import graficas
import parada

RAIZ = Path(__file__).resolve().parent
REFERENCIA = RAIZ / "referencia"
SALIDA = RAIZ / "figuras" / "entrenamiento"
FAMILIAS = {"recorte": ("bn_normal_c128_lr0.0001", "recorte 128 (modelo final)"),
            "v21": ("bn_normal_lr0.0001", "2.1 (sin recorte)")}
SEMILLAS = {"42": "", "1": "_s1", "2": "_s2"}
COLORES_SEMILLA = {"42": graficas.AZUL, "1": graficas.NARANJA, "2": graficas.VERDE}


def leer(config: str) -> dict:
    with open(REFERENCIA / f"{config}_epocas.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    datos = {c: [float(r[c]) for r in filas] for c in ("epoca", "perdida_train", "perdida_val", "auc_train", "auc_val")}
    datos["elegida"] = parada.recorrer(datos["auc_val"])["indice"]
    return datos


def perdida_sin_imagen() -> float:
    """Entropia (en nats, como la BCE) de la proporcion de pCR en los cortes de entrenamiento (folds 0-4)."""
    for base in (RAIZ, RAIZ / "breastdcedl"):
        ruta = base / "metadata" / "samples.csv"
        if ruta.exists():
            with open(ruta, encoding="utf-8") as f:
                y = [int(r["pCR"]) for r in csv.DictReader(f) if int(r["fold"]) >= 0]
            p = sum(y) / len(y)
            return -(p * math.log(p) + (1 - p) * math.log(1 - p))
    raise FileNotFoundError("No encuentro metadata/samples.csv")


def figura_folds(prefijo: str, nombre: str, ref: float, ruta: Path) -> None:
    plt = graficas._plt()
    fig, ax = plt.subplots(2, 5, figsize=(24, 9), constrained_layout=True, sharey="row")
    fig.suptitle(f"Curvas de entrenamiento · {nombre} · 3 semillas × 5 folds\n"
                 "línea continua: entrenamiento · discontinua: validación · punto: época elegida (early stopping)",
                 fontsize=15, fontweight="bold")
    for k in range(5):
        for semilla, sufijo in SEMILLAS.items():
            d = leer(f"{prefijo}{sufijo}_fold{k}")
            color, i = COLORES_SEMILLA[semilla], d["elegida"]
            for fila, (tr, va) in enumerate((("perdida_train", "perdida_val"), ("auc_train", "auc_val"))):
                ax[fila, k].plot(d["epoca"], d[tr], color=color, lw=1.8, label=f"semilla {semilla}, entrenamiento")
                ax[fila, k].plot(d["epoca"], d[va], color=color, lw=1.8, ls="--", label=f"semilla {semilla}, validación")
                ax[fila, k].scatter([d["epoca"][i]], [d[va][i]], color=color, edgecolor="black", s=55, zorder=5)
        ax[0, k].axhline(ref, color=graficas.GRIS, ls=":", lw=1.4, label=f"sin mirar la imagen ({ref:.3f})")
        ax[1, k].axhline(0.5, color=graficas.GRIS, ls=":", lw=1.4, label="azar (0,5)")
        ax[0, k].set_title(f"fold {k}", fontsize=13)
        ax[1, k].set_xlabel("época")
        ax[1, k].set_ylim(0.35, 1.0)
        for fila in range(2):
            ax[fila, k].grid(alpha=0.3)
    ax[0, 0].set_ylabel("pérdida (BCE)", fontsize=12)
    ax[1, 0].set_ylabel("AUC por paciente", fontsize=12)
    manejadores, etiquetas = ax[0, 0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=7, fontsize=10)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_modelo_final(ref: float, ruta: Path) -> None:
    """La figura para enseñar: modelo final, una columna por fold, perdida arriba y AUC abajo. Lineas finas: cada
    semilla; linea gruesa: media de las 3 semillas en las epocas que entrenaron las tres; puntos: epoca elegida."""
    plt = graficas._plt()
    prefijo = FAMILIAS["recorte"][0]
    fig, ax = plt.subplots(2, 5, figsize=(24, 9.5), constrained_layout=True, sharey="row")
    fig.suptitle("Curvas de entrenamiento del modelo final (recorte central 128): 5 folds × 3 semillas\n"
                 "azul: entrenamiento · naranja: validación · línea gruesa: media de las 3 semillas · "
                 "punto: época elegida por el early stopping",
                 fontsize=15, fontweight="bold")
    for k in range(5):
        runs = [leer(f"{prefijo}{sufijo}_fold{k}") for sufijo in SEMILLAS.values()]
        n = min(len(d["epoca"]) for d in runs)
        separaciones = [d["auc_train"][d["elegida"]] - d["auc_val"][d["elegida"]] for d in runs]
        for fila, (tr, va) in enumerate((("perdida_train", "perdida_val"), ("auc_train", "auc_val"))):
            a = ax[fila, k]
            for d in runs:
                a.plot(d["epoca"], d[tr], color=graficas.AZUL, lw=0.9, alpha=0.35)
                a.plot(d["epoca"], d[va], color=graficas.NARANJA, lw=0.9, alpha=0.35)
                a.scatter([d["epoca"][d["elegida"]]], [d[va][d["elegida"]]], color=graficas.ROJO, edgecolor="black",
                          s=50, zorder=5)
            epocas = runs[0]["epoca"][:n]
            a.plot(epocas, [sum(d[tr][e] for d in runs) / 3 for e in range(n)], color=graficas.AZUL, lw=2.8,
                   label="entrenamiento (media)")
            a.plot(epocas, [sum(d[va][e] for d in runs) / 3 for e in range(n)], color=graficas.NARANJA, lw=2.8,
                   label="validación (media)")
            a.grid(alpha=0.3)
        ax[0, k].axhline(ref, color=graficas.GRIS, ls=":", lw=1.5, label=f"sin mirar la imagen ({ref:.3f})")
        ax[1, k].axhline(0.5, color=graficas.GRIS, ls=":", lw=1.5, label="azar (0,5)")
        ax[0, k].set_title(f"fold {k}\nAUC entrenamiento − validación\nen la época elegida: "
                           f"{sum(separaciones) / 3:+.3f}", fontsize=12)
        ax[1, k].set_xlabel("época", fontsize=11)
        ax[1, k].set_ylim(0.35, 1.0)
    ax[0, 0].set_ylabel("pérdida (entropía cruzada binaria)", fontsize=12)
    ax[1, 0].set_ylabel("AUC por paciente", fontsize=12)
    h0, l0 = ax[0, 0].get_legend_handles_labels()
    h1, l1 = ax[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1[-1:], l0 + l1[-1:], loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=11)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_comparativa(ref: float, ruta: Path) -> None:
    plt = graficas._plt()
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True, sharey="row")
    fig.suptitle("Las dos familias de modelos, los 15 modelos de cada una superpuestos\n"
                 "azul: entrenamiento · naranja: validación · punto: época elegida",
                 fontsize=15, fontweight="bold")
    for col, (clave, (prefijo, nombre)) in enumerate(FAMILIAS.items()):
        separaciones, epocas = [], []
        for k in range(5):
            for sufijo in SEMILLAS.values():
                d = leer(f"{prefijo}{sufijo}_fold{k}")
                i = d["elegida"]
                separaciones.append(d["auc_train"][i] - d["auc_val"][i])
                epocas.append(d["epoca"][i])
                for fila, (tr, va) in enumerate((("perdida_train", "perdida_val"), ("auc_train", "auc_val"))):
                    ax[fila, col].plot(d["epoca"], d[tr], color=graficas.AZUL, lw=1, alpha=0.55)
                    ax[fila, col].plot(d["epoca"], d[va], color=graficas.NARANJA, lw=1, alpha=0.55)
                    ax[fila, col].scatter([d["epoca"][i]], [d[va][i]], color=graficas.ROJO, s=22, zorder=5)
        sobre = sum(s > parada.DIFERENCIA_MAX for s in separaciones)
        ax[0, col].set_title(f"{nombre}\nen la época elegida: entrenamiento − validación "
                             f"{sum(separaciones) / len(separaciones):+.3f}\nsobreentrenados {sobre} de 15 · "
                             f"época elegida media {sum(epocas) / len(epocas):.1f}", fontsize=12)
        ax[0, col].axhline(ref, color=graficas.GRIS, ls=":", lw=1.4, label=f"sin mirar la imagen ({ref:.3f})")
        ax[1, col].axhline(0.5, color=graficas.GRIS, ls=":", lw=1.4, label="azar (0,5)")
        ax[1, col].set_xlabel("época"); ax[1, col].set_ylim(0.35, 1.0)
        for fila in range(2):
            ax[fila, col].grid(alpha=0.3)
            ax[fila, col].legend(fontsize=9, loc="lower left" if fila == 0 else "upper left")
    ax[0, 0].set_ylabel("pérdida (BCE)", fontsize=12)
    ax[1, 0].set_ylabel("AUC por paciente", fontsize=12)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    ref = perdida_sin_imagen()
    for clave, (prefijo, nombre) in FAMILIAS.items():
        figura_folds(prefijo, nombre, ref, SALIDA / f"{clave}_folds.png")
    figura_comparativa(ref, SALIDA / "comparativa.png")
    figura_modelo_final(ref, SALIDA / "modelo_final.png")
    print(f"Figuras en {SALIDA}")


if __name__ == "__main__":
    main()
