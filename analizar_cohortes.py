"""¿La red aprende del tumor o solo reconoce de qué cohorte viene la paciente?

El dataset junta tres cohortes (duke, spy1, spy2) con distinta proporción de pCR y con imágenes
que se ven distintas (otras máquinas, otro protocolo, otra forma de elegir los cortes). Una red
podría reconocer la cohorte por el aspecto de la imagen y sacar un AUC algo mayor que 0,5 sin
aprender nada del tumor. Este script lo comprueba calculando el AUC DENTRO de cada cohorte, donde
ese atajo desaparece.

Usa las probabilidades de validación que deja `entrenar.py` (`<config>_fold<k>_probs_val.csv`) de
los 5 folds: cada paciente de entrenamiento se valida exactamente una vez, así que se obtiene una
predicción para todas (1.097 pacientes). Por paciente se promedian las probabilidades de sus cortes.

Calcula:
  - AUC global (todas las pacientes juntas);
  - AUC dentro de cada cohorte, con su intervalo de confianza del 95 % (bootstrap por paciente);
  - AUC "estratificado": solo compara pares de pacientes de la MISMA cohorte (un número que resume
    lo que la red sabe más allá de la cohorte);
  - la referencia "solo la cohorte": predecir con la proporción de pCR de la cohorte de cada
    paciente, calculada con los folds de entrenamiento, sin mirar la imagen;
  - la probabilidad media que da la red en cada cohorte, frente a la proporción real de pCR.

Cómo leerlo: si el AUC dentro de las cohortes (y el estratificado) cae a ~0,5 mientras el global
es mayor, el AUC global venía de reconocer la cohorte, no del tumor.

Uso (no necesita GPU):
    python analizar_cohortes.py base_normal_aug
    python analizar_cohortes.py base_normal
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
COHORTES = ("duke", "spy1", "spy2")


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #

def localizar(subcarpeta: str) -> Path:
    """Carpeta que contiene `subcarpeta`: la raíz del proyecto o su subcarpeta breastdcedl/."""
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / subcarpeta).is_dir():
            return base
    raise FileNotFoundError(f"No encuentro '{subcarpeta}/' en {RAIZ} ni en {RAIZ / 'breastdcedl'}")


def leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cargar_predicciones(config: str, carpetas: list[Path]) -> dict[str, dict]:
    """Por paciente: etiqueta, probabilidad media de sus cortes y fold en el que se validó."""
    cortes = defaultdict(list)
    etiqueta, fold_de = {}, {}
    folds_encontrados = []
    for k in range(5):
        ruta = next((c / f"{config}_fold{k}_probs_val.csv" for c in carpetas
                     if (c / f"{config}_fold{k}_probs_val.csv").exists()), None)
        if ruta is None:
            continue
        folds_encontrados.append(k)
        for fila in leer_csv(ruta):
            pid = fila["patient_id"]
            cortes[pid].append(float(fila["prob"]))
            etiqueta[pid] = int(fila["pCR"])
            fold_de[pid] = k
    if not cortes:
        raise SystemExit(f"No encuentro ningún {config}_fold<k>_probs_val.csv en: " +
                         ", ".join(str(c) for c in carpetas))
    if len(folds_encontrados) < 5:
        faltan = sorted(set(range(5)) - set(folds_encontrados))
        print(f"AVISO: faltan los folds {faltan}; el análisis usa solo {folds_encontrados}.\n")
    return {pid: {"y": etiqueta[pid], "prob": float(np.mean(p)), "fold": fold_de[pid]}
            for pid, p in cortes.items()}


# --------------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------------- #

def auc(y, s) -> float:
    """AUC = probabilidad de que una paciente pCR tenga más puntuación que una no pCR (empates
    cuentan la mitad). Se calcula con rangos (test de Mann-Whitney), sin scikit-learn."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inversa, cuentas = np.unique(s, return_inverse=True, return_counts=True)
    acumulado = np.cumsum(cuentas)
    rango_medio = acumulado - (cuentas - 1) / 2.0           # rango medio de cada valor (empates)
    rangos = rango_medio[inversa]
    return float((rangos[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_estratificado(y, s, grupo) -> float:
    """AUC contando solo pares (pCR, no pCR) de la MISMA cohorte: lo que la red sabe ordenar
    cuando la cohorte no puede ayudar."""
    y, s, grupo = np.asarray(y, dtype=bool), np.asarray(s, dtype=float), np.asarray(grupo)
    aciertos, pares = 0.0, 0
    for g in np.unique(grupo):
        m = grupo == g
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if n1 and n0:
            aciertos += auc(y[m], s[m]) * n1 * n0
            pares += n1 * n0
    return aciertos / pares if pares else float("nan")


def intervalo_bootstrap(funcion, indices_por_grupo: list[np.ndarray], repeticiones: int = 2000,
                        semilla: int = 0) -> tuple[float, float]:
    """Intervalo de confianza del 95 % remuestreando pacientes con reemplazo (dentro de cada grupo,
    para que cada remuestreo conserve el tamaño de cada cohorte)."""
    rng = np.random.default_rng(semilla)
    valores = []
    for _ in range(repeticiones):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in indices_por_grupo])
        v = funcion(idx)
        if not np.isnan(v):
            valores.append(v)
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="nombre de la configuración, p. ej. base_normal_aug")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "resultados")
    p.add_argument("--repeticiones", type=int, default=2000, help="remuestreos del bootstrap")
    args = p.parse_args()

    pred = cargar_predicciones(args.config, [args.carpeta, RAIZ / "referencia"])
    meta = localizar("metadata")
    cohorte = {f["pid"]: f["dataset"] for f in leer_csv(meta / "metadata" / "patients.csv")}
    fold_train = {}                                  # fold de cada paciente de entrenamiento (samples.csv)
    for f in leer_csv(meta / "metadata" / "samples.csv"):
        if f["split"] == "train":
            fold_train[f["patient_id"]] = int(f["fold"])

    pids = sorted(pred)
    y = np.array([pred[q]["y"] for q in pids])
    s = np.array([pred[q]["prob"] for q in pids])
    g = np.array([cohorte[q] for q in pids])
    f_val = np.array([pred[q]["fold"] for q in pids])

    # referencia "solo la cohorte": proporción de pCR de la cohorte en los folds de ENTRENAMIENTO
    y_todas = {}
    for fila in leer_csv(meta / "metadata" / "patients.csv"):
        if fila["pid"] in fold_train:
            y_todas[fila["pid"]] = int(fila["pCR"])
    solo_cohorte = np.empty(len(pids))
    for k in np.unique(f_val):
        tasa = {}
        for c in COHORTES:
            otros = [y_todas[q] for q in fold_train if fold_train[q] != k and cohorte[q] == c]
            tasa[c] = float(np.mean(otros))
        solo_cohorte[f_val == k] = [tasa[c] for c in g[f_val == k]]

    grupos = [np.where(g == c)[0] for c in COHORTES]
    todos = np.arange(len(pids))

    print(f"Configuración: {args.config}   ·   {len(pids)} pacientes con predicción de validación\n")
    print(f"{'grupo':<26}{'pacientes':>10}{'pCR real':>10}{'prob. media':>13}{'AUC':>8}{'IC95':>18}")
    filas_grafica = []

    a = auc(y, s)
    lo, hi = intervalo_bootstrap(lambda i: auc(y[i], s[i]), grupos, args.repeticiones)
    print(f"{'todas (global)':<26}{len(pids):>10}{y.mean():>10.1%}{s.mean():>13.3f}{a:>8.3f}"
          f"{f'{lo:.3f} a {hi:.3f}':>18}")
    filas_grafica.append(("global", a, lo, hi))

    for c, idx in zip(COHORTES, grupos):
        a = auc(y[idx], s[idx])
        lo, hi = intervalo_bootstrap(lambda i: auc(y[i], s[i]), [idx], args.repeticiones)
        print(f"{'  dentro de ' + c:<26}{len(idx):>10}{y[idx].mean():>10.1%}{s[idx].mean():>13.3f}{a:>8.3f}"
              f"{f'{lo:.3f} a {hi:.3f}':>18}")
        filas_grafica.append((c, a, lo, hi))

    a = auc_estratificado(y, s, g)
    lo, hi = intervalo_bootstrap(lambda i: auc_estratificado(y[i], s[i], g[i]), grupos, args.repeticiones)
    print(f"{'estratificado (por cohorte)':<26}{len(pids):>10}{'':>10}{'':>13}{a:>8.3f}{f'{lo:.3f} a {hi:.3f}':>18}")
    filas_grafica.append(("estratificado", a, lo, hi))

    a = auc(y, solo_cohorte)
    lo, hi = intervalo_bootstrap(lambda i: auc(y[i], solo_cohorte[i]), grupos, args.repeticiones)
    print(f"{'referencia: solo cohorte':<26}{len(pids):>10}{'':>10}{'':>13}{a:>8.3f}{f'{lo:.3f} a {hi:.3f}':>18}")
    filas_grafica.append(("solo cohorte", a, lo, hi))

    por_fold = [auc(y[f_val == k], s[f_val == k]) for k in np.unique(f_val)]
    print(f"\n(Media del AUC global de cada fold por separado: {np.mean(por_fold):.3f}. El AUC global de arriba junta"
          f"\n los 5 folds, cada uno con su propio modelo, y por eso puede salir algo distinto.)")

    print("\nCómo leerlo:")
    print("  - Si el AUC DENTRO de las cohortes y el estratificado están cerca de 0,5 mientras el global es mayor,")
    print("    el AUC global venía de reconocer la cohorte, no del tumor.")
    print("  - Si la 'prob. media' sigue el orden de la columna 'pCR real' entre cohortes, la red ha aprendido")
    print("    las diferencias de proporción entre cohortes.")
    print("  - Con pocas pacientes por cohorte (spy1 ~100) los intervalos son anchos: no sobreinterpretar.")

    try:
        dibujar(filas_grafica, y, s, g, args.carpeta / f"cohortes_{args.config}.png", args.config)
        print(f"\nGráfica: {args.carpeta / f'cohortes_{args.config}.png'}")
    except ImportError:
        print("\n(sin matplotlib: no se dibuja la gráfica)")


def dibujar(filas, y, s, g, ruta: Path, config: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    fig.suptitle(f"¿Tumor o cohorte?  ·  {config}", fontsize=15, fontweight="bold")

    nombres = [f[0] for f in filas]
    valores = [f[1] for f in filas]
    errores = [[f[1] - f[2] for f in filas], [f[3] - f[1] for f in filas]]
    colores = ["#1f77b4", "#7fb3d5", "#7fb3d5", "#7fb3d5", "#e76f51", "#6c757d"]
    izq.bar(range(len(filas)), valores, yerr=errores, capsize=5, color=colores[:len(filas)])
    izq.axhline(0.5, color="#6c757d", ls="--", lw=1.4, label="azar (0,5)")
    izq.set_xticks(range(len(filas)), nombres, rotation=15)
    izq.set_ylim(0.3, 0.8)
    izq.set_ylabel("AUC por paciente (IC 95 %)")
    izq.set_title("AUC global, dentro de cada cohorte y referencias", fontsize=12)
    izq.legend(fontsize=10)
    izq.grid(alpha=0.3, axis="y")

    rng = np.random.default_rng(0)
    for i, c in enumerate(COHORTES):
        for etiqueta, color, dx in ((0, "#6c757d", -0.18), (1, "#c1121f", 0.18)):
            m = (g == c) & (y == etiqueta)
            der.scatter(i + dx + rng.uniform(-0.08, 0.08, m.sum()), s[m], s=10, alpha=0.5, color=color,
                        label=("pCR" if etiqueta else "no pCR") if i == 0 else None)
            if m.any():
                der.hlines(np.mean(s[m]), i + dx - 0.12, i + dx + 0.12, color="black", lw=2)
    der.set_xticks(range(len(COHORTES)), COHORTES)
    der.set_ylabel("probabilidad media de pCR (por paciente)")
    der.set_title("Lo que predice la red en cada cohorte (raya = media)", fontsize=12)
    der.legend(fontsize=10)
    der.grid(alpha=0.3, axis="y")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
