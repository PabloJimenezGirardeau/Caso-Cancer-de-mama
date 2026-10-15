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

Con --grupo se hace el mismo análisis con otros grupos de pacientes (puntos fuertes y débiles de
la red por TIPO de paciente):
  - subtipo: HR+/HER2-, HR+/HER2+, HR-/HER2+, triple negativo. El AUC estratificado por subtipo es
    lo que la CNN sabe MÁS ALLÁ del subtipo, es decir, lo que puede añadir a los datos clínicos.
  - tamano: tercios del volumen tumoral (tum_vol). Dice si la red solo funciona con tumores grandes.

Uso (no necesita GPU):
    python analizar_cohortes.py bn_normal --probs media10                  # cohorte, subtipo y tamaño
    python analizar_cohortes.py base_normal_aug --grupo cohorte
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
COHORTES = ("duke", "spy1", "spy2")
SUBTIPOS = ("HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "triple negativo")
TAMANOS = ("pequeño", "mediano", "grande")
TITULOS = {"cohorte": "cohorte", "subtipo": "subtipo del tumor", "tamano": "tamaño del tumor (tercios de tum_vol)"}


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


FUENTES = {   # --probs: de qué fichero y qué columna salen las probabilidades
    "mejor":   ("_probs_val.csv", "prob"),                 # la época con mejor AUC (optimista)
    "ultima":  ("_probs_val_final.csv", "prob"),           # la última época
    "media10": ("_probs_val_final.csv", "prob_media10"),   # media de las últimas 10 épocas
}


def cargar_predicciones(config: str, carpetas: list[Path], fuente: str = "mejor") -> dict[str, dict]:
    """Por paciente: etiqueta, probabilidad media de sus cortes y fold en el que se validó."""
    sufijo, columna = FUENTES[fuente]
    cortes = defaultdict(list)
    etiqueta, fold_de = {}, {}
    folds_encontrados = []
    for k in range(5):
        ruta = next((c / f"{config}_fold{k}{sufijo}" for c in carpetas
                     if (c / f"{config}_fold{k}{sufijo}").exists()), None)
        if ruta is None:
            continue
        folds_encontrados.append(k)
        for fila in leer_csv(ruta):
            pid = fila["patient_id"]
            cortes[pid].append(float(fila[columna]))
            etiqueta[pid] = int(fila["pCR"])
            fold_de[pid] = k
    if not cortes:
        raise SystemExit(f"No encuentro ningún {config}_fold<k>{sufijo} en: " +
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
# Grupos de pacientes
# --------------------------------------------------------------------------- #

ORDEN = {"cohorte": COHORTES, "subtipo": SUBTIPOS, "tamano": TAMANOS}
CORTO = {"cohorte": "cohorte", "subtipo": "subtipo", "tamano": "tamaño"}
LECTURA = {
    "cohorte": [
        "Si el AUC DENTRO de las cohortes y el estratificado están cerca de 0,5 mientras el global es mayor,",
        "el AUC global venía de reconocer la cohorte, no del tumor.",
        "Si la 'prob. media' sigue el orden de la columna 'pCR real' entre cohortes, la red ha aprendido",
        "las diferencias de proporción entre cohortes.",
    ],
    "subtipo": [
        "El estratificado (por subtipo) es lo que la CNN sabe MÁS ALLÁ del subtipo: lo que puede añadir",
        "a HR y HER2, que ya entran como datos clínicos.",
        "Si la 'prob. media' sigue el orden de la columna 'pCR real' entre subtipos, la CNN ha aprendido a",
        "reconocer el subtipo en la imagen (redundante: ese dato ya lo tenemos).",
        "Un AUC claramente mayor que 0,5 dentro de un subtipo es un punto fuerte real de la imagen ahí.",
    ],
    "tamano": [
        "Si el AUC solo es alto en los tumores grandes, a la red le cuesta con los pequeños. Ojo: las imágenes",
        "son recortes del tumor llevados a 256x256, así que un tumor pequeño se ve ampliado.",
        "El estratificado (por tamaño) es lo que la red sabe más allá del tamaño, que el modelo clínico ya usa.",
    ],
}


def etiquetas(grupo: str, pacientes: dict[str, dict], pids_train: list[str]) -> tuple[dict, str]:
    """Grupo de cada paciente (None si le falta el dato) y una nota sobre cómo se ha formado."""
    if grupo == "cohorte":
        return {q: f["dataset"] for q, f in pacientes.items()}, ""
    if grupo == "subtipo":
        nombres = {("1", "0"): SUBTIPOS[0], ("1", "1"): SUBTIPOS[1], ("0", "1"): SUBTIPOS[2], ("0", "0"): SUBTIPOS[3]}

        def binario(v: str):
            return str(int(float(v))) if v != "" else None

        return {q: nombres.get((binario(f["HR"]), binario(f["HER2"]))) for q, f in pacientes.items()}, ""
    # tamaño: tercios de tum_vol, con los cortes calculados sobre las pacientes de entrenamiento
    vol = {q: (float(f["tum_vol"]) if f["tum_vol"] != "" else None) for q, f in pacientes.items()}
    c1, c2 = np.quantile([vol[q] for q in pids_train if vol[q] is not None], [1 / 3, 2 / 3])

    def tercio(v):
        if v is None:
            return None
        return TAMANOS[0] if v < c1 else TAMANOS[1] if v < c2 else TAMANOS[2]

    nota = f"pequeño: tum_vol < {c1:.1f}   ·   mediano: {c1:.1f} a {c2:.1f}   ·   grande: > {c2:.1f}"
    return {q: tercio(v) for q, v in vol.items()}, nota


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def analizar(grupo: str, pred: dict, pacientes: dict, fold_train: dict, args) -> None:
    etiqueta, nota = etiquetas(grupo, pacientes, list(fold_train))
    nombres = ORDEN[grupo]
    pids = sorted(q for q in pred if etiqueta.get(q) is not None)
    y = np.array([pred[q]["y"] for q in pids])
    s = np.array([pred[q]["prob"] for q in pids])
    g = np.array([etiqueta[q] for q in pids])
    f_val = np.array([pred[q]["fold"] for q in pids])

    # referencia "solo el grupo": proporción de pCR del grupo en los folds de ENTRENAMIENTO, sin mirar la imagen
    solo_grupo = np.empty(len(pids))
    for k in np.unique(f_val):
        tasa = {}
        for c in nombres:
            otros = [int(pacientes[q]["pCR"]) for q in fold_train if fold_train[q] != k and etiqueta.get(q) == c]
            tasa[c] = float(np.mean(otros))
        solo_grupo[f_val == k] = [tasa[c] for c in g[f_val == k]]

    grupos = [np.where(g == c)[0] for c in nombres]
    print(f"\n=== Por {TITULOS[grupo]} ===")
    if nota:
        print(nota)
    if len(pids) < len(pred):
        print(f"({len(pred) - len(pids)} pacientes sin el dato se quedan fuera de este análisis)")
    print(f"{'grupo':<30}{'pacientes':>10}{'pCR real':>10}{'prob. media':>13}{'AUC':>8}{'IC95':>18}")
    filas_grafica = []

    def fila(nombre, idx, puntuacion, funcion=auc, grupos_boot=None, con_medias=True):
        a = funcion(y[idx], puntuacion[idx]) if funcion is auc else funcion(y, puntuacion, g)
        if funcion is auc:
            lo, hi = intervalo_bootstrap(lambda i: auc(y[i], puntuacion[i]), grupos_boot, args.repeticiones)
        else:
            lo, hi = intervalo_bootstrap(lambda i: funcion(y[i], puntuacion[i], g[i]), grupos_boot, args.repeticiones)
        medias = f"{y[idx].mean():>10.1%}{puntuacion[idx].mean():>13.3f}" if con_medias else f"{'':>10}{'':>13}"
        print(f"{nombre:<30}{len(idx):>10}{medias}{a:>8.3f}{f'{lo:.3f} a {hi:.3f}':>18}")
        return a, lo, hi

    todos = np.arange(len(pids))
    filas_grafica.append(("global", *fila("todas (global)", todos, s, grupos_boot=grupos)))
    for c, idx in zip(nombres, grupos):
        filas_grafica.append((c, *fila("  dentro de " + c, idx, s, grupos_boot=[idx])))
    filas_grafica.append(("estratificado", *fila(f"estratificado (por {CORTO[grupo]})", todos, s,
                                                 auc_estratificado, grupos, con_medias=False)))
    filas_grafica.append((f"solo {CORTO[grupo]}", *fila(f"referencia: solo {CORTO[grupo]}", todos, solo_grupo,
                                                        grupos_boot=grupos, con_medias=False)))

    print("\nCómo leerlo:")
    for linea in LECTURA[grupo] + ["Con grupos pequeños los intervalos son anchos: no sobreinterpretar."]:
        print("  " + linea)

    try:
        nombre_png = f"cohortes_{args.config}_{args.probs}.png" if grupo == "cohorte" else f"{grupo}_{args.config}_{args.probs}.png"
        ruta_png = args.carpeta / nombre_png
        dibujar(filas_grafica, y, s, g, nombres, ruta_png, f"{args.config} ({args.probs})", TITULOS[grupo])
        print(f"\nGráfica: {ruta_png}")
    except ImportError:
        print("\n(sin matplotlib: no se dibuja la gráfica)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="nombre de la configuración, p. ej. bn_normal")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "resultados")
    p.add_argument("--repeticiones", type=int, default=2000, help="remuestreos del bootstrap")
    p.add_argument("--probs", choices=list(FUENTES), default="mejor",
                   help="probabilidades de la mejor época (optimista), de la última o la media de las últimas 10")
    p.add_argument("--grupo", choices=["todos", *TITULOS], default="todos",
                   help="por qué grupos de pacientes analizar (por defecto, los tres)")
    args = p.parse_args()

    pred = cargar_predicciones(args.config, [args.carpeta, RAIZ / "referencia"], args.probs)
    meta = localizar("metadata")
    pacientes = {f["pid"]: f for f in leer_csv(meta / "metadata" / "patients.csv")}
    fold_train = {}                                  # fold de cada paciente de entrenamiento (samples.csv)
    for f in leer_csv(meta / "metadata" / "samples.csv"):
        if f["split"] == "train":
            fold_train[f["patient_id"]] = int(f["fold"])

    print(f"Configuración: {args.config}   ·   probabilidades: {args.probs}   ·   "
          f"{len(pred)} pacientes con predicción de validación")
    por_fold = [auc([pred[q]["y"] for q in pred if pred[q]["fold"] == k],
                    [pred[q]["prob"] for q in pred if pred[q]["fold"] == k]) for k in range(5)
                if any(pred[q]["fold"] == k for q in pred)]
    print(f"(Media del AUC de cada fold por separado: {np.mean(por_fold):.3f}. El AUC global junta los 5 folds,"
          f"\n cada uno con su propio modelo, y por eso puede salir algo distinto.)")
    for grupo in (list(TITULOS) if args.grupo == "todos" else [args.grupo]):
        analizar(grupo, pred, pacientes, fold_train, args)


def dibujar(filas, y, s, g, nombres, ruta: Path, config: str, titulo_grupo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    fig.suptitle(f"AUC de la red por {titulo_grupo}  ·  {config}", fontsize=15, fontweight="bold")

    etiquetas_x = [f[0] for f in filas]
    valores = [f[1] for f in filas]
    errores = [[f[1] - f[2] for f in filas], [f[3] - f[1] for f in filas]]
    colores = ["#1f77b4"] + ["#7fb3d5"] * len(nombres) + ["#e76f51", "#6c757d"]
    izq.bar(range(len(filas)), valores, yerr=errores, capsize=5, color=colores[:len(filas)])
    izq.axhline(0.5, color="#6c757d", ls="--", lw=1.4, label="azar (0,5)")
    izq.set_xticks(range(len(filas)), etiquetas_x, rotation=20)
    izq.set_ylim(0.3, 0.85)
    izq.set_ylabel("AUC por paciente (IC 95 %)")
    izq.set_title("AUC global, dentro de cada grupo y referencias", fontsize=12)
    izq.legend(fontsize=10)
    izq.grid(alpha=0.3, axis="y")

    rng = np.random.default_rng(0)
    for i, c in enumerate(nombres):
        for etiqueta, color, dx in ((0, "#6c757d", -0.18), (1, "#c1121f", 0.18)):
            m = (g == c) & (y == etiqueta)
            der.scatter(i + dx + rng.uniform(-0.08, 0.08, m.sum()), s[m], s=10, alpha=0.5, color=color,
                        label=("pCR" if etiqueta else "no pCR") if i == 0 else None)
            if m.any():
                der.hlines(np.mean(s[m]), i + dx - 0.12, i + dx + 0.12, color="black", lw=2)
    der.set_xticks(range(len(nombres)), nombres)
    der.set_ylabel("probabilidad media de pCR (por paciente)")
    der.set_title("Lo que predice la red en cada grupo (raya = media)", fontsize=12)
    der.legend(fontsize=10)
    der.grid(alpha=0.3, axis="y")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
