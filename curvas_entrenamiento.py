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
    ruta = next((c / f"{config}_epocas.csv" for c in (RAIZ / "resultados", REFERENCIA)
                 if (c / f"{config}_epocas.csv").exists()), REFERENCIA / f"{config}_epocas.csv")
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    datos = {c: [float(r[c]) for r in filas] for c in ("epoca", "perdida_train", "perdida_val", "auc_train", "auc_val")}
    # registros desde el 2026-10-23: entrenamiento medido como la validacion, con las clases pesadas como en el
    # entrenamiento (la columna antigua perdida_train_eval, sin ponderar, no es comparable y no se usa)
    if filas and filas[0].get("perdida_train_comparable"):
        datos["perdida_train_eval"] = [float(r["perdida_train_comparable"]) for r in filas]
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


def referencias_por_fold() -> dict[int, tuple[float, float]]:
    """Para cada fold k: (perdida sin mirar la imagen en su ENTRENAMIENTO, en su VALIDACION), calculadas desde samples.csv.

    Un modelo que no mira la imagen solo puede decir a todos los cortes la proporcion de pCR que ha visto: la de SU
    entrenamiento, p. En entrenamiento su perdida es la entropia de p. En validacion es la entropia cruzada de las
    etiquetas de validacion con ese mismo p (corregido el 2026-10-22: antes se usaba la proporcion de la propia
    validacion, que el modelo no puede conocer, y la linea quedaba demasiado baja en los folds 2 y 3)."""
    for base in (RAIZ, RAIZ / "breastdcedl"):
        ruta = base / "metadata" / "samples.csv"
        if ruta.exists():
            with open(ruta, encoding="utf-8") as f:
                filas = [(int(r["fold"]), int(r["pCR"])) for r in csv.DictReader(f) if int(r["fold"]) >= 0]
            break
    else:
        raise FileNotFoundError("No encuentro metadata/samples.csv")
    cruzada = lambda q, p: -(q * math.log(p) + (1 - q) * math.log(1 - p))   # etiquetas con proporcion q, prediccion p
    salida = {}
    for k in range(5):
        p_tr = sum(y for f, y in filas if f != k) / sum(1 for f, _ in filas if f != k)
        q_va = sum(y for f, y in filas if f == k) / sum(1 for f, _ in filas if f == k)
        salida[k] = (cruzada(p_tr, p_tr), cruzada(q_va, p_tr))
    return salida


def media_semillas(runs: list[dict], clave: str) -> tuple[list[float], list[float]]:
    """Epocas y media de las 3 semillas, en las epocas que entrenaron las tres."""
    n = min(len(d["epoca"]) for d in runs)
    return runs[0]["epoca"][:n], [sum(d[clave][e] for d in runs) / len(runs) for e in range(n)]


def figura_perdida(ruta: Path, prefijo: str | None = None, nombre: str = "modelo final (recorte central 128)") -> None:
    """Solo la perdida del modelo final, un panel grande por fold (rejilla 2x3; el sexto panel es la leyenda).
    Cada panel tiene SU escala, ajustada a las medias de las semillas y a sus lineas de referencia, para que se vea
    la forma de las curvas (los picos de una semilla suelta pueden salirse por arriba); epocas en numeros enteros.
    Discontinuas: la perdida sin mirar la imagen con la proporcion de pCR de ese entrenamiento (azul) y de esa
    validacion (naranja). En cada panel se anota el minimo de la validacion (media) frente a su referencia."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt = graficas._plt()
    prefijo = prefijo or FAMILIAS["recorte"][0]
    refs = referencias_por_fold()
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.5), constrained_layout=True)
    fig.suptitle(f"Pérdida · {nombre}: 5 folds × 3 semillas",
                 fontsize=16, fontweight="bold")
    for k, a in zip(range(5), axes.ravel()):
        runs = [leer(f"{prefijo}{s}_fold{k}") for s in SEMILLAS.values()]
        ep_tr, m_tr = media_semillas(runs, "perdida_train")
        ep_va, m_va = media_semillas(runs, "perdida_val")
        for d in runs:
            a.plot(d["epoca"], d["perdida_train"], color=graficas.AZUL, lw=1, alpha=0.3)
            a.plot(d["epoca"], d["perdida_val"], color=graficas.NARANJA, lw=1, alpha=0.3)
            i = d["elegida"]
            a.scatter([d["epoca"][i]], [d["perdida_val"][i]], color=graficas.ROJO, edgecolor="black", s=70, zorder=5)
        a.plot(ep_tr, m_tr, color=graficas.AZUL, lw=3.2)
        a.plot(ep_va, m_va, color=graficas.NARANJA, lw=3.2)
        if all("perdida_train_eval" in d for d in runs):   # entrenamiento medido igual que la validacion
            ep_ev, m_ev = media_semillas(runs, "perdida_train_eval")
            a.plot(ep_ev, m_ev, color=graficas.AZUL, lw=2.2, ls=":")
            m_tr = m_tr + m_ev
        a.axhline(refs[k][0], color=graficas.AZUL, ls="--", lw=1.5, alpha=0.8)
        a.axhline(refs[k][1], color=graficas.NARANJA, ls="--", lw=1.5)
        valores = m_tr + m_va + list(refs[k])
        margen = 0.08 * (max(valores) - min(valores))
        a.set_ylim(min(valores) - margen, max(valores) + margen)
        a.set_xlim(0.5, max(len(d["epoca"]) for d in runs) + 0.5)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        j = min(range(len(m_va)), key=m_va.__getitem__)
        a.set_title(f"fold {k} · mínimo de la validación {m_va[j]:.3f} (época {ep_va[j]:.0f}) "
                    f"frente a {refs[k][1]:.3f} sin imagen", fontsize=12.5)
        a.set_xlabel("época", fontsize=12)
        a.set_ylabel("pérdida (entropía cruzada binaria)", fontsize=11)
        a.tick_params(labelsize=11)
        a.grid(alpha=0.3)
    leyenda = axes.ravel()[5]
    leyenda.axis("off")
    elementos = [Line2D([], [], color=graficas.AZUL, lw=3.2, label="entrenamiento, media de las 3 semillas"),
                 Line2D([], [], color=graficas.AZUL, lw=2.2, ls=":",
                        label="entrenamiento medido como la validación\n(200 pacientes, sin dropout; si está registrado)"),
                 Line2D([], [], color=graficas.NARANJA, lw=3.2, label="validación, media de las 3 semillas"),
                 Line2D([], [], color=graficas.GRIS, lw=1, alpha=0.6, label="cada semilla (algún pico se sale por arriba)"),
                 Line2D([], [], color=graficas.AZUL, ls="--", lw=1.5, label="sin mirar la imagen (entrenamiento)"),
                 Line2D([], [], color=graficas.NARANJA, ls="--", lw=1.5, label="sin mirar la imagen (validación)"),
                 Line2D([], [], marker="o", ls="", markerfacecolor=graficas.ROJO, markeredgecolor="black",
                        markersize=10, label="época elegida por el early stopping")]
    leyenda.legend(handles=elementos, loc="center", fontsize=13, frameon=True,
                   title="Cómo leer cada panel\n(cada fold con su propia escala)", title_fontsize=13)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    refs = referencias_por_fold()
    todos = {k: [leer(f"{prefijo}{s}_fold{k}") for s in SEMILLAS.values()] for k in range(5)}
    medias = [v for k in range(5) for c in ("perdida_train", "perdida_val") for v in media_semillas(todos[k], c)[1]]
    lo, hi = min(medias) - 0.02, max(medias) + 0.03      # escala de la perdida ajustada a las medias
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
        ax[0, k].axhline(refs[k][1], color=graficas.NARANJA, ls="--", lw=1.4,
                         label="sin mirar la imagen (proporción de pCR del entrenamiento, evaluada en validación)")
        ax[1, k].axhline(0.5, color=graficas.GRIS, ls=":", lw=1.5, label="azar (0,5)")
        ax[0, k].set_title(f"fold {k}\nAUC entrenamiento − validación\nen la época elegida: "
                           f"{sum(separaciones) / 3:+.3f}", fontsize=12)
        ax[1, k].set_xlabel("época", fontsize=11)
        ax[0, k].set_ylim(lo, hi)
        ax[1, k].set_ylim(0.4, 0.95)
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
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--familia", help="solo la figura de perdida de esta familia (prefijo sin _s1/_s2 ni _fold<k>), "
                                     "p. ej. bn_normal_brillo_ls0.1_wd0.5_c128_lr0.0001; lee resultados/ y referencia/")
    args = p.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    if args.familia:
        figura_perdida(SALIDA / f"perdida_{args.familia}.png", args.familia, args.familia)
        print(f"Figura en {SALIDA / f'perdida_{args.familia}.png'}")
        return
    ref = perdida_sin_imagen()
    for clave, (prefijo, nombre) in FAMILIAS.items():
        figura_folds(prefijo, nombre, ref, SALIDA / f"{clave}_folds.png")
    figura_comparativa(ref, SALIDA / "comparativa.png")
    figura_modelo_final(ref, SALIDA / "modelo_final.png")
    figura_perdida(SALIDA / "perdida_modelo_final.png")
    print(f"Figuras en {SALIDA}")


if __name__ == "__main__":
    main()
