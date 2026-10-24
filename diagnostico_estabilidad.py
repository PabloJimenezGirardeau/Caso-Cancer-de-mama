"""Panel de estabilidad del modelo final (recorte 128, 3 semillas x 5 folds), solo con la VALIDACION (el test no se toca).

Cuatro diagnosticos, cada uno con su figura en figuras/estabilidad/:
    1_acuerdo_semillas.png   ¿dan las 3 semillas de un fold el mismo orden a las pacientes? (correlacion de rangos de
                             Spearman entre semillas, por paciente; sano = alta)
    2_robustez.png           ¿cambia la prediccion con cambios irrelevantes? (imagen volteada, brillo x0,9 y x1,1;
                             sano = casi igual). Mide tambien el promedio con la imagen volteada (TTA) con la regla
                             fijada en DECISIONES.md
    3_coherencia_cortes.png  ¿reciben probabilidades parecidas los cortes de una misma paciente? (sano = dispersion
                             dentro de la paciente mucho menor que entre pacientes)
    4_variabilidad.png       AUC de validacion de cada semilla en cada fold y del promedio de las 3 (dispersion)
y resumen.txt con las cifras.

Cada paciente se predice con los 3 modelos de SU fold (los que no la vieron). La parte 2 necesita PyTorch y las imagenes;
con --sin-imagenes se hacen solo 1, 3 y 4 (segundos).

Uso:
    python diagnostico_estabilidad.py
    python diagnostico_estabilidad.py --sin-imagenes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import comparar as co
import graficas
import umbral as um
import utils_caso as uc

RAIZ = Path(__file__).resolve().parent
CARPETA = RAIZ / "modelo"
SALIDA = RAIZ / "figuras" / "estabilidad"
SEMILLAS = {"": "42", "_s1": "1", "_s2": "2"}


def rangos(x) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def spearman(a, b) -> float:
    return float(np.corrcoef(rangos(a), rangos(b))[0, 1])


def localizar(sub: str) -> Path:
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / sub).is_dir():
            return base
    raise FileNotFoundError(sub)


def cargar_probs(cfg: dict) -> pd.DataFrame:
    """Una fila por corte de validacion: fold, paciente, pCR y la probabilidad de cada semilla (sin calibrar)."""
    base = None
    for c in cfg["configuraciones"]:
        sem = "1" if c.endswith("_s1") else "2" if c.endswith("_s2") else "42"
        t = pd.concat([pd.read_csv(CARPETA / f"{c}_fold{k}_probs_val.csv").assign(fold=k) for k in range(5)])
        t = t.set_index("sample_id")
        if base is None:
            base = t[["patient_id", "pCR", "fold"]].copy()
        base[f"s{sem}"] = t.prob.reindex(base.index)
    return base.reset_index()


def calibrar(p, cfg) -> np.ndarray:
    cal = cfg["calibracion"]
    return um.sigmoide(cal["a"] * um.logit(np.clip(p, 1e-6, 1 - 1e-6)) + cal["b"])


def auc_paciente(d: pd.DataFrame, col: str) -> float:
    g = d.groupby("patient_id").agg(y=("pCR", "first"), p=(col, "mean"))
    return um.auc(g.y.to_numpy(), g.p.to_numpy())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sin-imagenes", action="store_true", help="solo los diagnosticos 1, 3 y 4 (sin PyTorch)")
    args = p.parse_args()
    plt = graficas._plt()
    SALIDA.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((CARPETA / "umbral_final.json").read_text(encoding="utf-8"))
    umbral = float(cfg["umbral"])
    d = cargar_probs(cfg)
    sem = ["s42", "s1", "s2"]
    d["media"] = d[sem].mean(axis=1)
    d["cal"] = calibrar(d["media"].to_numpy(), cfg)
    lineas = [f"Panel de estabilidad · modelo final ({', '.join(cfg['configuraciones'])}) · validacion de los 5 folds, "
              f"{d.patient_id.nunique()} pacientes, {len(d)} cortes", ""]

    # ---------- 1. acuerdo entre semillas (por paciente, correlacion de rangos) ----------
    pac = d.groupby("patient_id").agg(fold=("fold", "first"), pCR=("pCR", "first"), **{s: (s, "mean") for s in sem})
    pares = [("s42", "s1"), ("s42", "s2"), ("s1", "s2")]
    acuerdo = {k: [spearman(pac[pac.fold == k][a], pac[pac.fold == k][b]) for a, b in pares] for k in range(5)}
    todos = [v for vs in acuerdo.values() for v in vs]
    lineas += ["1. ACUERDO ENTRE SEMILLAS (Spearman entre las probabilidades por paciente de dos semillas del mismo fold)",
               *[f"   fold {k}: " + "  ".join(f"{a[1:]}-{b[1:]} {v:+.2f}" for (a, b), v in zip(pares, acuerdo[k]))
                 for k in range(5)],
               f"   media {np.mean(todos):+.2f} (de {min(todos):+.2f} a {max(todos):+.2f}); 1 = mismo orden, 0 = sin relacion", ""]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)
    fig.suptitle("1 · Acuerdo entre semillas: probabilidad por paciente (validación), dos semillas del mismo fold\n"
                 "cada punto una paciente; ρ de Spearman = cuánto coincide el orden (1 = idéntico)",
                 fontsize=13, fontweight="bold")
    for ax, (a, b) in zip(axes, pares):
        for k in range(5):
            q = pac[pac.fold == k]
            ax.scatter(rangos(q[a]) / len(q), rangos(q[b]) / len(q), s=9, alpha=0.5, label=f"fold {k}")
        r = np.mean([acuerdo[k][pares.index((a, b))] for k in range(5)])
        ax.plot([0, 1], [0, 1], color=graficas.GRIS, ls="--", lw=1)
        ax.set(xlabel=f"posición de la paciente según la semilla {a[1:]} (0 = menor prob.)",
               ylabel=f"posición según la semilla {b[1:]}", title=f"semillas {a[1:]} y {b[1:]} · ρ medio {r:+.2f}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, markerscale=2)
    fig.savefig(SALIDA / "1_acuerdo_semillas.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------- 3. coherencia entre cortes de una misma paciente ----------
    dentro = d.groupby("patient_id").cal.std().mean()
    entre = d.groupby("patient_id").cal.mean().std()
    lineas += ["3. COHERENCIA ENTRE CORTES DE UNA MISMA PACIENTE (modelo final calibrado)",
               f"   desviacion de las probabilidades DENTRO de una paciente (media): {dentro:.4f}",
               f"   desviacion ENTRE pacientes (de sus medias): {entre:.4f}  ->  cociente {dentro / entre:.2f} "
               "(cuanto menor, mas coherente; 1 = los cortes de una paciente varian tanto como pacientes distintas)", ""]
    rng = np.random.default_rng(0)
    medias = d.groupby("patient_id").cal.mean().sort_values()
    elegidas = medias.index[np.linspace(0, len(medias) - 1, 40).astype(int)]
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    for i, pid in enumerate(elegidas):
        v = d[d.patient_id == pid].cal.to_numpy()
        color = graficas.ROJO if d[d.patient_id == pid].pCR.iloc[0] == 1 else graficas.GRIS
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=color, alpha=0.7)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color="black", lw=1.5)
    ax.axhline(umbral, color=graficas.ROJO, ls="--", lw=1.2, label=f"umbral {umbral:.3f}")
    ax.set(xticks=[], xlabel="40 pacientes de validación, ordenadas por su probabilidad media (rojo: pCR = 1; gris: pCR = 0)",
           ylabel="probabilidad calibrada de cada corte")
    ax.set_title(f"3 · Coherencia entre cortes de una misma paciente · dispersión dentro {dentro:.3f} frente a entre "
                 f"pacientes {entre:.3f} (cociente {dentro / entre:.2f})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(SALIDA / "3_coherencia_cortes.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------- 4. variabilidad entre semillas y folds ----------
    familia = cfg["configuraciones"][0]
    por_sem = {s: co.cargar(RAIZ / "resultados", f"{familia}{suf}") for suf, s in SEMILLAS.items()}
    ens = [auc_paciente(d[d.fold == k], "media") for k in range(5)]
    lineas += ["4. VARIABILIDAD (AUC de validacion por paciente en la epoca elegida)"]
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for k in range(5):
        vals = [por_sem[s][k]["auc_val"] for s in SEMILLAS.values()]
        ax.scatter([k] * 3, vals, s=60, color=graficas.AZUL, alpha=0.7, label="cada semilla" if k == 0 else None)
        ax.plot([k - 0.15, k + 0.15], [ens[k]] * 2, color=graficas.ROJO, lw=3, label="promedio de las 3 semillas" if k == 0 else None)
        lineas.append(f"   fold {k}: semillas " + "  ".join(f"{v:.3f}" for v in vals)
                      + f"  (rango {max(vals) - min(vals):.3f})  promedio de las 3: {ens[k]:.3f}")
    rango = np.mean([max(por_sem[s][k]["auc_val"] for s in SEMILLAS.values()) - min(por_sem[s][k]["auc_val"]
                     for s in SEMILLAS.values()) for k in range(5)])
    lineas += [f"   rango medio entre semillas en un fold: {rango:.3f}; desviacion entre folds del promedio: {np.std(ens, ddof=1):.3f}", ""]
    ax.axhline(0.5, color=graficas.GRIS, ls=":", label="azar")
    ax.set(xticks=range(5), xticklabels=[f"fold {k}" for k in range(5)], ylabel="AUC de validación por paciente",
           ylim=(0.45, 0.7))
    ax.set_title(f"4 · Variabilidad entre semillas y folds · rango medio entre semillas {rango:.3f}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(SALIDA / "4_variabilidad.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------- 2. robustez a cambios irrelevantes (y TTA con volteo) ----------
    if not args.sin_imagenes:
        import torch
        from predecir import ModeloFinal, cargar_corte
        modelo = ModeloFinal(CARPETA)          # GPU si la hay (en el PC), si no CPU
        base_img = localizar("dataset")
        samples = uc.cargar_samples(localizar("metadata")).set_index("sample_id").loc[d.sample_id]
        variantes = {"original": lambda x: x, "volteada": lambda x: x[:, :, :, ::-1].copy(),
                     "brillo x0,9": lambda x: x * 0.9, "brillo x1,1": lambda x: x * 1.1}
        res = {v: np.zeros(len(d)) for v in variantes}
        for k in range(5):
            idx = np.where(d.fold.to_numpy() == k)[0]
            modelos_k = [i for i, n in enumerate(modelo.nombres) if n.endswith(f"_fold{k}")]
            for j in range(0, len(idx), 256):
                bloque = idx[j:j + 256]
                filas = samples.iloc[bloque]
                x = np.stack([cargar_corte(base_img / r.path_pre, base_img / r.path_early, base_img / r.path_late)
                              for r in filas.itertuples()])
                for v, f in variantes.items():
                    xv = f(x).astype(np.float32)
                    res[v][bloque] = np.mean([modelo.probabilidades_modelo(i, xv) for i in modelos_k], axis=0)
            print(f"  robustez: fold {k} hecho", flush=True)
        for v in variantes:
            d[f"r_{v}"] = res[v]
        verif = float(np.abs(d["r_original"] - d["media"]).max())
        lineas += ["2. ROBUSTEZ A CAMBIOS IRRELEVANTES (media de los 3 modelos del fold, por corte)",
                   f"   comprobacion: la prediccion original reproduce los probs_val guardados (dif. max {verif:.1e})"]
        cal_o = calibrar(d["r_original"].to_numpy(), cfg)
        filas_r = []
        for v in list(variantes)[1:]:
            cal_v = calibrar(d[f"r_{v}"].to_numpy(), cfg)
            dif = np.abs(cal_v - cal_o)
            po = pd.Series(cal_o).groupby(d.patient_id.values).mean()
            pv = pd.Series(cal_v).groupby(d.patient_id.values).mean()
            cambian = float(((po >= umbral) != (pv >= umbral)).mean())
            filas_r.append((v, dif.mean(), np.percentile(dif, 95), spearman(cal_o, cal_v), cambian))
            lineas.append(f"   {v:<12} cambio medio de probabilidad {dif.mean():.4f} (percentil 95 {np.percentile(dif, 95):.4f}); "
                          f"Spearman con la original {spearman(cal_o, cal_v):.3f}; pacientes que cambian de clase {cambian:.1%}")
        # TTA: promedio de original y volteada, regla de DECISIONES.md (AUC por fold, sin calibrar)
        d["tta"] = (d["r_original"] + d["r_volteada"]) / 2
        a_o = [auc_paciente(d[d.fold == k], "r_original") for k in range(5)]
        a_t = [auc_paciente(d[d.fold == k], "tta") for k in range(5)]
        difs = [t - o for o, t in zip(a_o, a_t)]
        media, desv, _, semi = co.resumen(difs)
        adopta = media > 0 and media - semi > 0
        lineas += ["   PROMEDIO CON LA IMAGEN VOLTEADA (TTA), AUC por paciente en cada fold (sin calibrar):",
                   "   sin TTA " + "  ".join(f"{v:.4f}" for v in a_o) + f"   media {np.mean(a_o):.4f}",
                   "   con TTA " + "  ".join(f"{v:.4f}" for v in a_t) + f"   media {np.mean(a_t):.4f}",
                   f"   diferencia media {media:+.4f}  IC95 {media - semi:+.4f} a {media + semi:+.4f}  ->  REGLA (positiva y el IC "
                   f"excluye el 0): {'SE ADOPTA' if adopta else 'NO se adopta'}", ""]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)
        fig.suptitle("2 · Robustez: probabilidad calibrada de cada corte con la imagen original frente a la modificada "
                     "(validación)\nsano = puntos sobre la diagonal", fontsize=13, fontweight="bold")
        for ax, (v, m, p95, r, cambian) in zip(axes, filas_r):
            ax.scatter(cal_o, calibrar(d[f"r_{v}"].to_numpy(), cfg), s=3, alpha=0.25, color=graficas.AZUL)
            lo, hi = cal_o.min(), cal_o.max()
            ax.plot([lo, hi], [lo, hi], color=graficas.ROJO, ls="--", lw=1)
            ax.set(xlabel="imagen original", ylabel=f"imagen {v}",
                   title=f"{v} · cambio medio {m:.3f} · ρ {r:.3f}\npacientes que cambian de clase: {cambian:.1%}")
            ax.grid(alpha=0.3)
        fig.savefig(SALIDA / "2_robustez.png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    (SALIDA / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    print(f"Figuras en {SALIDA}")


if __name__ == "__main__":
    main()
