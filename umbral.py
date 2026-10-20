"""Fase 4: agregacion y umbral con las predicciones de VALIDACION (el test no se toca).

Usa las probabilidades que cada modelo dio a su fold de validacion en la epoca elegida
(`<config>_fold<k>_probs_val.csv`). Con los 5 folds juntos hay una prediccion para cada una de las
1.097 pacientes de entrenamiento, hecha por un modelo que no la vio.

1. Promedio de semillas: por defecto se juntan las 3 semillas de la configuracion elegida (2.1).
   Para cada corte se promedian las probabilidades de las 3 semillas; para cada paciente, la media de
   sus cortes (la agregacion que pide el profesor). Se compara el AUC por fold de cada semilla con el
   del promedio. El modelo final promediara 5 folds x 3 semillas, asi que el promedio de semillas se
   parece mas a lo que vera el test que una sola semilla.
   CALIBRACION (por defecto; --sin-calibrar para no hacerla): UNA sola vez, al final. Primero se
   promedian las probabilidades de las semillas sin tocar; despues, con todos los cortes de validacion,
   se ajusta un escalado de Platt: p_calibrada = sigmoide(a * logit(p_promedio) + b), a nivel de CORTE
   (en la app y en la defensa llega un solo corte). No cambia el orden de los cortes, arregla la escala
   (Brier, curva de calibracion) y el test y la app solo tienen que aplicar dos numeros (a, b).
   (Se probo antes calibrar cada modelo por separado: los modelos "sin aprendizaje", que dan casi la
   misma probabilidad a todas, recibian pendientes enormes y su ruido pesaba tanto como los demas.)
2. Curvas con las 1.097 pacientes: ROC, precision-recall (AP), sensibilidad / especificidad /
   precision segun el umbral, calibracion (con el Brier frente al de decir siempre la proporcion de pCR).
3. Umbral. Criterio clinico (HOJA_DE_RUTA.md): el uso previsto es ayudar a decidir si se puede
   DESESCALAR, y el error grave es el falso positivo (decir "pCR" a quien conserva tumor). Se elige el
   umbral mas bajo con especificidad >= 0,95 (la mayor sensibilidad posible con esa especificidad).
   Se compara con otros criterios: maximo F1 de pCR, maximo F1 equilibrada (media de las dos clases),
   Youden (sensibilidad + especificidad - 1) y el 0,5 por defecto.
4. Con cada umbral: matriz de confusion, sensibilidad, especificidad, precision (VPP), VPN, F1 de cada
   clase, F1 equilibrada y accuracy, frente a decir siempre "no pCR". Y como cambia el umbral elegido
   fold a fold (su estabilidad).

Aviso: el umbral se elige con estas mismas predicciones, asi que sus cifras son algo optimistas; el
juez independiente es el test (fase 6), donde se aplicara el umbral tal cual.

Salida, en resultados/:
    umbral_<nombre>.png             las 6 graficas
    umbral_<nombre>_pacientes.csv   probabilidad de cada paciente (promedio de semillas)
    umbral_<nombre>.json            umbral elegido y metricas (lo usaran el test y la app)

Uso (no necesita GPU):
    python umbral.py
    python umbral.py --configs bn_normal_lr0.0001                  # una sola semilla
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import graficas

RAIZ = Path(__file__).resolve().parent
REFERENCIA = RAIZ / "referencia"
POR_DEFECTO = ["bn_normal_lr0.0001", "bn_normal_lr0.0001_s1", "bn_normal_lr0.0001_s2"]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #

def cargar_config(carpeta: Path, nombre: str) -> pd.DataFrame:
    """Probabilidad de cada corte de validacion (5 folds) de una configuracion, en la epoca elegida."""
    partes = []
    for k in range(5):
        for base in (carpeta, REFERENCIA):
            ruta = base / f"{nombre}_fold{k}_probs_val.csv"
            if ruta.exists():
                partes.append(pd.read_csv(ruta).assign(fold=k))
                break
        else:
            raise SystemExit(f"Falta {nombre}_fold{k}_probs_val.csv (ni en {carpeta} ni en {REFERENCIA})")
    return pd.concat(partes, ignore_index=True)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoide(z) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, dtype=float), -50, 50)))


def platt(p, y, iteraciones: int = 100) -> tuple[float, float]:
    """Escalado de Platt: (a, b) que hacen sigmoide(a * logit(p) + b) lo mas parecido posible a las
    etiquetas (regresion logistica de una variable, por Newton). Monotono si a > 0."""
    x, y = logit(p), np.asarray(y, dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    w = np.array([1.0, 0.0])
    for _ in range(iteraciones):
        q = sigmoide(X @ w)
        g = X.T @ (q - y)
        H = (X * (q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(2)
        paso = np.linalg.solve(H, g)
        w -= paso
        if np.abs(paso).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def por_paciente(cortes: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Media de las probabilidades de los cortes de cada paciente."""
    return cortes.groupby("patient_id").agg(pCR=("pCR", "first"), fold=("fold", "first"), prob=(columna, "mean"))


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #

def auc(y, s) -> float:
    return graficas.calcular_roc(np.asarray(y), np.asarray(s, dtype=float))[2]


def precision_recall(y, s):
    """Curva precision-recall y AP (precision media ponderada por cada aumento de recall)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    orden = np.argsort(-s, kind="mergesort")
    y, s = y[orden], s[orden]
    vp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    ultimos = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    recall = vp[ultimos] / vp[-1]
    precision = vp[ultimos] / (vp[ultimos] + fp[ultimos])
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return np.r_[0.0, recall], np.r_[1.0, precision], ap


def metricas(y, s, umbral: float) -> dict:
    y = np.asarray(y); pred = np.asarray(s) >= umbral
    vp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    vn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else 0.0
    sens, esp = div(vp, vp + fn), div(vn, vn + fp)
    vpp, vpn = div(vp, vp + fp), div(vn, vn + fn)
    f1_pos, f1_neg = div(2 * vpp * sens, vpp + sens), div(2 * vpn * esp, vpn + esp)
    return {"umbral": float(umbral), "VP": vp, "FN": fn, "VN": vn, "FP": fp, "sensibilidad": sens,
            "especificidad": esp, "precision": vpp, "vpn": vpn, "f1_pcr": f1_pos, "f1_no_pcr": f1_neg,
            "f1_equilibrada": (f1_pos + f1_neg) / 2, "accuracy": div(vp + vn, len(y))}


def barrido(y, s) -> pd.DataFrame:
    """Metricas para cada umbral posible (cada probabilidad distinta, y uno por encima del maximo:
    no decir "pCR" a nadie, que siempre cumple cualquier especificidad)."""
    s = np.asarray(s, dtype=float)
    return pd.DataFrame([metricas(y, s, u) for u in np.r_[np.unique(s), s.max() + 1e-9]])


def elegir_umbrales(tabla: pd.DataFrame, especificidad: float) -> dict:
    validos = tabla[tabla.especificidad >= especificidad]
    return {
        f"desescalar (especificidad >= {especificidad:g})": validos.loc[validos.umbral.idxmin()],
        "maximo F1 de pCR": tabla.loc[tabla.f1_pcr.idxmax()],
        "maximo F1 equilibrada": tabla.loc[tabla.f1_equilibrada.idxmax()],
        "Youden (sens + espec - 1)": tabla.loc[(tabla.sensibilidad + tabla.especificidad).idxmax()],
    }


def intervalo_auc(y, s, repeticiones: int = 2000, semilla: int = 0) -> tuple[float, float]:
    """IC95 del AUC remuestreando pacientes con reemplazo."""
    rng = np.random.default_rng(semilla)
    y, s = np.asarray(y), np.asarray(s)
    valores = []
    for _ in range(repeticiones):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            valores.append(auc(y[i], s[i]))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def calibracion(y, s, grupos: int = 10):
    """Por deciles de probabilidad: probabilidad media predicha frente a fraccion real de pCR."""
    d = pd.DataFrame({"y": y, "s": s})
    d["grupo"] = pd.qcut(d.s.rank(method="first"), grupos, labels=False)
    g = d.groupby("grupo").agg(predicha=("s", "mean"), real=("y", "mean"), n=("y", "size"))
    return g


# --------------------------------------------------------------------------- #
# Grafica
# --------------------------------------------------------------------------- #

def dibujar(y, s, tabla, elegido, cal, brier, brier_ref, ap, ruta: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    AZUL, NARANJA, ROJO, GRIS, VERDE = graficas.AZUL, graficas.NARANJA, graficas.ROJO, graficas.GRIS, graficas.VERDE
    y = np.asarray(y); s = np.asarray(s)
    u = elegido["umbral"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fig.suptitle(titulo, fontsize=15, fontweight="bold")

    fpr, tpr, a = graficas.calcular_roc(y, s)
    ax[0, 0].plot(fpr, tpr, color=AZUL, lw=2.2, label=f"AUC = {a:.3f}")
    ax[0, 0].plot([0, 1], [0, 1], ls="--", color=GRIS)
    ax[0, 0].scatter([1 - elegido["especificidad"]], [elegido["sensibilidad"]], color=ROJO, s=60, zorder=5,
                     label=f"umbral {u:.3f}")
    ax[0, 0].set(title="ROC por paciente", xlabel="1 − especificidad", ylabel="sensibilidad")
    ax[0, 0].legend(loc="lower right")

    rec, prec, _ = precision_recall(y, s)
    ax[0, 1].step(rec, prec, where="post", color=AZUL, lw=2.2, label=f"AP = {ap:.3f}")
    ax[0, 1].axhline(y.mean(), ls="--", color=GRIS, label=f"azar = proporción de pCR ({y.mean():.2f})")
    ax[0, 1].scatter([elegido["sensibilidad"]], [elegido["precision"]], color=ROJO, s=60, zorder=5,
                     label=f"umbral {u:.3f}")
    ax[0, 1].set(title="Precision-recall (clase pCR)", xlabel="recall (sensibilidad)", ylabel="precisión", ylim=(0, 1.02))
    ax[0, 1].legend(loc="upper right")

    ax[0, 2].plot(tabla.umbral, tabla.sensibilidad, color=NARANJA, lw=2, label="sensibilidad")
    ax[0, 2].plot(tabla.umbral, tabla.especificidad, color=AZUL, lw=2, label="especificidad")
    ax[0, 2].plot(tabla.umbral, tabla.precision, color=VERDE, lw=2, label="precisión (VPP)")
    ax[0, 2].axvline(u, color=ROJO, ls=":", lw=1.8, label=f"umbral elegido {u:.3f}")
    ax[0, 2].set(title="Según el umbral", xlabel="umbral", ylim=(0, 1.02))
    ax[0, 2].legend(loc="center right")

    ax[1, 0].plot([0, 1], [0, 1], ls="--", color=GRIS, label="calibración perfecta")
    ax[1, 0].plot(cal.predicha, cal.real, "o-", color=AZUL, lw=2, label="modelo (deciles)")
    lim = max(cal.predicha.max(), cal.real.max()) * 1.15
    ax[1, 0].set(title=f"Calibración · Brier {brier:.4f} (referencia {brier_ref:.4f})",
                 xlabel="probabilidad media predicha", ylabel="fracción real de pCR", xlim=(0, lim), ylim=(0, lim))
    ax[1, 0].legend(loc="upper left")

    celdas = np.array([[elegido["VN"], elegido["FP"]], [elegido["FN"], elegido["VP"]]])
    ax[1, 1].imshow(celdas, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1, 1].text(j, i, int(celdas[i, j]), ha="center", va="center", fontsize=22, fontweight="bold",
                          color="white" if celdas[i, j] > celdas.max() * 0.6 else "#14213d")
    ax[1, 1].set_xticks([0, 1], ["dijo no pCR", "dijo pCR"]); ax[1, 1].set_yticks([0, 1], ["era no pCR", "era pCR"])
    ax[1, 1].set_title(f"Matriz · umbral {u:.3f}\nsens {elegido['sensibilidad']:.0%} · espec "
                       f"{elegido['especificidad']:.0%} · precisión {elegido['precision']:.0%}")

    bins = np.linspace(s.min(), s.max(), 30)
    ax[1, 2].hist(s[y == 0], bins=bins, alpha=0.6, color=GRIS, label="no pCR")
    ax[1, 2].hist(s[y == 1], bins=bins, alpha=0.6, color=ROJO, label="pCR")
    ax[1, 2].axvline(u, color=ROJO, ls=":", lw=1.8, label=f"umbral {u:.3f}")
    ax[1, 2].axvline(0.5, color=GRIS, ls="--", lw=1.2, label="0,5")
    ax[1, 2].set(title="Probabilidades por clase", xlabel="probabilidad de pCR (paciente)", ylabel="pacientes")
    ax[1, 2].legend()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=POR_DEFECTO, help="configuraciones a promediar (semillas)")
    p.add_argument("--especificidad", type=float, default=0.95, help="especificidad minima del criterio de desescalar")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "resultados")
    p.add_argument("--nombre", default="final", help="nombre de los ficheros de salida")
    p.add_argument("--sin-calibrar", action="store_true", help="no calibrar el promedio final")
    args = p.parse_args()

    # 1. promedio de semillas, corte a corte
    tablas = [cargar_config(args.carpeta, c) for c in args.configs]
    base = tablas[0][["sample_id", "patient_id", "pCR", "fold"]].copy()
    for i, (c, t) in enumerate(zip(args.configs, tablas)):
        t = t.set_index("sample_id").loc[base.sample_id]
        if (t.fold.values != base.fold.values).any() or (t.pCR.values != base.pCR.values).any():
            raise SystemExit(f"{c} no cuadra con {args.configs[0]} (folds o etiquetas distintos)")
        base[f"p{i}"] = t.prob.values
    columnas = [f"p{i}" for i in range(len(args.configs))]
    base["promedio"] = base[columnas].mean(axis=1)

    # calibracion UNICA al final: Platt sobre el promedio de semillas, con todos los cortes de validacion
    calibrar = not args.sin_calibrar
    parametros = None
    if calibrar:
        a_, b_ = platt(base["promedio"].values, base["pCR"].values)
        parametros = {"a": a_, "b": b_}
        base["calibrado"] = sigmoide(a_ * logit(base["promedio"].values) + b_)
    final = "calibrado" if calibrar else "promedio"

    print(f"{base.patient_id.nunique()} pacientes de validacion (5 folds), {len(base)} cortes; "
          f"configuraciones: {', '.join(args.configs)}\n")
    print("AUC por paciente en cada fold (media de los cortes de cada paciente):")
    nombres = [f"semilla {m.group(1)}" if (m := re.search(r"_s(\d+)$", c)) else "semilla 42" for c in args.configs]
    extra = ["promedio"] + (["calibrado"] if calibrar else [])
    print(f"{'fold':>5}" + "".join(f"{n:>14}" for n in nombres) + f"{'PROMEDIO':>14}"
          + (f"{'CALIBRADO':>14}" if calibrar else ""))
    filas = []
    for k in range(5):
        d = base[base.fold == k]
        fila = [auc(*por_paciente(d, c)[["pCR", "prob"]].values.T) for c in columnas + extra]
        filas.append(fila)
        print(f"{k:>5}" + "".join(f"{v:>14.4f}" for v in fila))
    filas = np.array(filas)
    print(f"{'media':>5}" + "".join(f"{v:>14.4f}" for v in filas.mean(axis=0)))
    print(f"{'desv':>5}" + "".join(f"{v:>14.4f}" for v in filas.std(axis=0, ddof=1)))
    ganancia = filas[:, len(columnas)] - filas[:, :len(columnas)].mean(axis=1)
    print(f"\nEl promedio de semillas frente a la media de las semillas sueltas: {ganancia.mean():+.4f} "
          f"(por fold: {'  '.join(f'{g:+.3f}' for g in ganancia)})")

    if calibrar:
        crudo = por_paciente(base, "promedio")
        yc, sc = crudo.pCR.values, crudo.prob.values
        u_crudo = elegir_umbrales(barrido(yc, sc), args.especificidad)
        u_crudo = list(u_crudo.values())[0]["umbral"]
        sens_folds = [metricas(yc[crudo.fold.values == k], sc[crudo.fold.values == k], u_crudo)["sensibilidad"]
                      for k in range(5)]
        print(f"\nSIN CALIBRAR, todas las pacientes juntas: AUC {auc(yc, sc):.4f}, Brier {np.mean((sc - yc) ** 2):.4f}; "
              f"umbral para especificidad >= {args.especificidad:g}: {u_crudo:.3f}, sensibilidad por fold "
              f"{' '.join(f'{x:.2f}' for x in sens_folds)}")
        print(f"Calibracion de Platt al final (sobre el promedio, por corte): a = {parametros['a']:.3f}, "
              f"b = {parametros['b']:.3f}  (a < 1: el promedio exageraba sus probabilidades; a > 1: las daba "
              f"demasiado juntas)")

    # 2. curvas con todas las pacientes
    pac = por_paciente(base, final)
    y, s = pac.pCR.values, pac.prob.values
    a = auc(y, s); lo, hi = intervalo_auc(y, s)
    _, _, ap = precision_recall(y, s)
    prev = y.mean()
    brier = float(np.mean((s - y) ** 2)); brier_ref = float(prev * (1 - prev))
    print(f"\n{'CALIBRADO' if calibrar else 'SIN CALIBRAR'}, todas las pacientes juntas (cada una predicha por el modelo de su fold): AUC {a:.4f} (IC95 {lo:.3f} a {hi:.3f}), "
          f"AP {ap:.4f} (azar = {prev:.3f}), Brier {brier:.4f} (siempre la proporcion de pCR: {brier_ref:.4f})")
    print(f"Probabilidades por paciente: minimo {s.min():.3f}, mediana {np.median(s):.3f}, maximo {s.max():.3f}; "
          f"pacientes por encima de 0,5: {(s >= 0.5).sum()}")

    # 3. umbrales
    tabla = barrido(y, s)
    elegidos = elegir_umbrales(tabla, args.especificidad)
    siempre_no = metricas(y, np.zeros_like(s), 0.5)
    por_defecto = metricas(y, s, 0.5)
    print("\nCriterio            umbral   sens   espec  precision   VPN   F1 pCR  F1 no pCR  F1 equil.  accuracy    VP  FN   VN  FP")
    filas_tabla = list(elegidos.items()) + [("umbral 0,5 por defecto", por_defecto), ("siempre 'no pCR'", siempre_no)]
    for nombre, m in filas_tabla:
        print(f"{nombre[:19]:<19} {m['umbral']:>6.3f} {m['sensibilidad']:>6.2f} {m['especificidad']:>7.2f} "
              f"{m['precision']:>10.2f} {m['vpn']:>5.2f} {m['f1_pcr']:>8.2f} {m['f1_no_pcr']:>10.2f} "
              f"{m['f1_equilibrada']:>10.2f} {m['accuracy']:>9.2f}  {int(m['VP']):>4} {int(m['FN']):>3} "
              f"{int(m['VN']):>4} {int(m['FP']):>3}")
    elegido = metricas(y, s, list(elegidos.values())[0]["umbral"])
    print(f"\nUMBRAL ELEGIDO (desescalar): {elegido['umbral']:.4f}  ->  de cada 100 pacientes a las que el modelo "
          f"dice 'pCR', aciertan {100 * elegido['precision']:.0f}; detecta {elegido['VP']} de {elegido['VP'] + elegido['FN']} pCR")

    print("\nEstabilidad del umbral elegido en cada fold:")
    print(f"{'fold':>5} {'pacientes':>10} {'sens':>6} {'espec':>7} {'precision':>10}")
    for k in range(5):
        m = metricas(pac.pCR[pac.fold == k].values, pac.prob[pac.fold == k].values, elegido["umbral"])
        print(f"{k:>5} {int((pac.fold == k).sum()):>10} {m['sensibilidad']:>6.2f} {m['especificidad']:>7.2f} {m['precision']:>10.2f}")

    cal = calibracion(y, s)
    print("\nCalibracion por deciles (probabilidad media predicha frente a fraccion real de pCR):")
    print("  " + "  ".join(f"{r.predicha:.2f}->{r.real:.2f}" for r in cal.itertuples()))

    args.carpeta.mkdir(parents=True, exist_ok=True)
    dibujar(y, s, tabla, elegido, cal, brier, brier_ref, ap, args.carpeta / f"umbral_{args.nombre}.png",
            f"Fase 4 · validación de los 5 folds · promedio de {len(args.configs)} semilla(s)"
            + (" · calibrado (Platt al final)" if calibrar else ""))
    pac.to_csv(args.carpeta / f"umbral_{args.nombre}_pacientes.csv", float_format="%.6f")
    with open(args.carpeta / f"umbral_{args.nombre}.json", "w", encoding="utf-8") as f:
        json.dump({"configuraciones": args.configs, "agregacion": "media de los cortes de cada paciente",
                   "criterio": f"umbral mas bajo con especificidad >= {args.especificidad}",
                   "umbral": elegido["umbral"], "metricas_validacion": elegido,
                   "auc_validacion": a, "auc_ic95": [lo, hi], "ap": ap, "brier": brier, "brier_referencia": brier_ref,
                   "auc_por_fold": {n: list(filas[:, i]) for i, n in enumerate(nombres + ["promedio"])},
                   "otros_criterios": {n: {k: float(v) for k, v in m.items()} for n, m in elegidos.items()},
                   "calibracion": ({"metodo": "Platt unico al final, a nivel de corte: p = sigmoide(a * logit(p_promedio) + b), "
                                    "donde p_promedio es la media de las probabilidades de los modelos para ese corte; "
                                    "la probabilidad de la paciente es la media de sus cortes calibrados",
                                    **parametros} if calibrar else None)},
                  f, indent=2, ensure_ascii=False)
    print(f"\nResultados en {args.carpeta}  (umbral_{args.nombre}.png / _pacientes.csv / .json)")


if __name__ == "__main__":
    main()
