"""Donde mira el modelo final (C): mapas Grad-CAM y si caen sobre el tumor.

Grad-CAM (Selvaraju et al., 2017) responde a "¿que zonas de la imagen han empujado esta decision?". Se toma el mapa del
ultimo bloque convolucional (128 canales de 16x16 sobre el cuadrado central de 128x128 que mira la red, justo antes del
ultimo pooling) y se pesa cada canal por cuanto cambia el logit si ese canal cambia (la media de su gradiente). La suma
de los canales pesados, sin la parte negativa (ReLU), es el mapa: alto donde esta la evidencia de la clase. Se calcula
hacia la clase PREDICHA (logit para pCR = 1, menos el logit para pCR = 0), se lleva a 128x128 y se normaliza a [0, 1]
en cada corte; con varias redes, se promedian sus mapas.

Dos partes:
    1. MEDIDA en las 1.097 pacientes de VALIDACION (un corte por paciente, el central, como en la app): cada una con las
       3 redes de su fold, que no la vieron al entrenar. Con la caja del tumor de patients.csv llevada al PNG
       (cajas_tumor.py), se mide que parte del mapa cae dentro de la caja, frente a la que caeria con un mapa uniforme
       (la parte del cuadrado central que ocupa la caja), y si el punto mas alto del mapa esta dentro de la caja.
    2. FIGURA de los cuatro casos del TEST que ya muestra evaluar_test.py (el acierto y el fallo mas claros de cada clase),
       con las 15 redes, como en la app. Es descriptivo: el test ya esta evaluado y aqui no se decide nada.

Uso (en el PC; necesita PyTorch):
    python mapas_gradcam.py           GPU si la hay
    python mapas_gradcam.py --cpu     en la CPU (si la GPU se cuelga; tarda un par de minutos)

Salida en figuras/mapas/:
    gradcam_casos_test.png    los cuatro casos: imagen con la caja del tumor, mapa encima y realce
    gradcam_resumen.png       mapa medio en validacion, parte del mapa dentro de la caja frente a un mapa uniforme, por cohorte
    gradcam_resumen.json      las cifras
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import graficas
import utils_caso as uc
from cajas_tumor import caja_en_png
from predecir import ModeloFinal, cargar_corte

RAIZ = Path(__file__).resolve().parent
LADO, RECORTE = 256, 128
INI = (LADO - RECORTE) // 2                 # el cuadrado central va de la fila/columna 64 a la 191
COHORTES = ("duke", "spy1", "spy2")


def localizar(subcarpeta: str) -> Path:
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / subcarpeta).is_dir():
            return base
    raise FileNotFoundError(f"No encuentro '{subcarpeta}/' en {RAIZ} ni en {RAIZ / 'breastdcedl'}")


def probabilidad_calibrada(modelo: ModeloFinal, redes: list, x: torch.Tensor) -> np.ndarray:
    """Media de las redes, recorte a [1e-6, 1 - 1e-6] y Platt: lo mismo que ModeloFinal.probabilidades."""
    with torch.no_grad():
        m = torch.stack([torch.sigmoid(r(x)) for r in redes]).mean(0).cpu().numpy()
    m = np.clip(m, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(modelo.a * np.log(m / (1 - m)) + modelo.b)))


def gradcam(modelo: ModeloFinal, redes: list, x: np.ndarray, lote: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Mapa Grad-CAM medio de `redes` para cada corte de x (N, 3, 256, 256), sobre el cuadrado central: (N, 128, 128)
    en [0, 1]; y la probabilidad calibrada de cada corte. El mapa va hacia la clase que predicen esas redes."""
    mapas, probs = [], []
    for j in range(0, len(x), lote):
        t = torch.from_numpy(np.ascontiguousarray(x[j:j + lote], dtype=np.float32)).to(modelo.dispositivo)
        p = probabilidad_calibrada(modelo, redes, t)
        signo = torch.from_numpy(np.where(p >= modelo.umbral, 1.0, -1.0).astype(np.float32)).to(modelo.dispositivo)
        t.requires_grad_(True)                 # para que se construya el grafo hasta las activaciones
        suma = torch.zeros(len(t), RECORTE, RECORTE, device=modelo.dispositivo)
        for red in redes:
            capa = red.extractor[3][2]         # ReLU del ultimo bloque: 128 @ 16x16, antes del ultimo pooling
            assert isinstance(capa, nn.ReLU), "la red no tiene la estructura esperada (conv, BN, ReLU, pool)"
            guardado = {}
            gancho = capa.register_forward_hook(lambda _m, _e, salida: guardado.__setitem__("a", salida))
            try:
                logit = red(t)
            finally:
                gancho.remove()
            a = guardado["a"]
            g, = torch.autograd.grad((logit * signo).sum(), a)       # BatchNorm en evaluacion: cortes independientes
            mapa = F.relu((g.mean(dim=(2, 3), keepdim=True) * a.detach()).sum(1))              # (n, 16, 16)
            mapa = F.interpolate(mapa[:, None], size=(RECORTE, RECORTE), mode="bilinear", align_corners=False)[:, 0]
            suma += mapa / mapa.flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
        mapas.append((suma / len(redes)).detach().cpu().numpy())
        probs.append(p)
    return np.concatenate(mapas), np.concatenate(probs)


def caja_en_recorte(caja) -> tuple[int, int, int, int] | None:
    """La caja (fila0, fila1, col0, col1 en el PNG, inclusivas) recortada al cuadrado central y en sus coordenadas
    (0-127, fin exclusivo); None si no toca el cuadrado."""
    a, b, c, e = caja
    f0, f1 = int(np.floor(max(a, INI))) - INI, int(np.floor(min(b, INI + RECORTE - 1))) - INI + 1
    c0, c1 = int(np.floor(max(c, INI))) - INI, int(np.floor(min(e, INI + RECORTE - 1))) - INI + 1
    return (f0, f1, c0, c1) if f1 > f0 and c1 > c0 else None


def medir(mapa: np.ndarray, caja) -> dict | None:
    """Parte del mapa dentro de la caja, la que tendria un mapa uniforme y si el maximo cae dentro."""
    r = caja_en_recorte(caja)
    if r is None:
        return None
    f0, f1, c0, c1 = r
    dentro = np.zeros_like(mapa, dtype=bool)
    dentro[f0:f1, c0:c1] = True
    fila, col = np.unravel_index(int(mapa.argmax()), mapa.shape)
    return {"parte_mapa": float(mapa[dentro].sum() / max(mapa.sum(), 1e-12)), "parte_uniforme": float(dentro.mean()),
            "maximo_dentro": bool(dentro[fila, col])}


def corte_central(cortes: pd.DataFrame, pid: str) -> pd.Series:
    suyos = cortes[cortes.patient_id == pid].sort_values("slice_index")
    return suyos.iloc[len(suyos) // 2]


def leer(fila: pd.Series, base_img: Path) -> np.ndarray:
    return cargar_corte(base_img / fila.path_pre, base_img / fila.path_early, base_img / fila.path_late)


def resumir(d: pd.DataFrame) -> dict:
    return {"pacientes": len(d), "parte_mapa_media": float(d.parte_mapa.mean()),
            "parte_uniforme_media": float(d.parte_uniforme.mean()),
            "cociente": float(d.parte_mapa.mean() / d.parte_uniforme.mean()),
            "pacientes_con_mas_que_uniforme": float((d.parte_mapa > d.parte_uniforme).mean()),
            "maximo_dentro": float(d.maximo_dentro.mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu", action="store_true", help="calcular en la CPU")
    args = p.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    base_meta, base_img = localizar("metadata"), localizar("dataset")
    samples = uc.cargar_samples(base_meta)
    pacientes = uc.cargar_patients(base_meta).set_index("pid")
    modelo = ModeloFinal(RAIZ / "modelo", dispositivo="cpu" if args.cpu else None)
    print(f"Modelo final: {len(modelo.redes)} redes en {modelo.dispositivo} (checksum {modelo.checksum})")
    salida = RAIZ / "figuras" / "mapas"
    salida.mkdir(parents=True, exist_ok=True)

    # ---- 1. medida en validacion: cada paciente con las 3 redes de su fold ----
    validacion = samples[samples.fold >= 0]
    filas, suma_mapas, n_mapas = [], np.zeros((RECORTE, RECORTE)), 0
    for k in range(5):
        redes = [r for r, n in zip(modelo.redes, modelo.nombres) if n.endswith(f"_fold{k}")]
        pids = sorted(validacion[validacion.fold == k].patient_id.unique())
        x = np.stack([leer(corte_central(validacion, pid), base_img) for pid in pids])
        mapas, probs = gradcam(modelo, redes, x)
        suma_mapas += mapas.sum(0); n_mapas += len(mapas)
        for pid, mapa, prob in zip(pids, mapas, probs):
            q = pacientes.loc[pid]
            caja = caja_en_png(q)
            m = medir(mapa, caja) if caja is not None else None
            if m is not None:
                filas.append({"pid": pid, "cohorte": q.dataset, "pCR": int(q.pCR), "probabilidad": float(prob), **m})
        print(f"  fold {k}: {len(pids)} pacientes ({len(redes)} redes)")
    d = pd.DataFrame(filas)
    resumen = {"todas": resumir(d), **{c: resumir(d[d.cohorte == c]) for c in COHORTES}}
    sin_caja = validacion.patient_id.nunique() - len(d)
    print(f"\nVALIDACION: {len(d)} pacientes con la caja del tumor dentro del cuadrado central ({sin_caja} sin caja utilizable)")
    for nombre, r in resumen.items():
        print(f"  {nombre:<6} parte del mapa dentro de la caja {r['parte_mapa_media']:.0%} frente a {r['parte_uniforme_media']:.0%} "
              f"con un mapa uniforme (x{r['cociente']:.2f}); mas que uniforme en el {r['pacientes_con_mas_que_uniforme']:.0%} "
              f"de las pacientes; maximo dentro de la caja {r['maximo_dentro']:.0%}")

    # ---- figura resumen ----
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.9), constrained_layout=True)
    fig.suptitle(f"¿Dónde mira el modelo final? Grad-CAM en validación ({n_mapas} pacientes, corte central, cada una con las "
                 "redes que no la vieron)", fontsize=13.5, fontweight="bold")
    im = ax[0].imshow(suma_mapas / n_mapas, cmap="inferno", vmin=0)
    ax[0].set_title("Mapa medio sobre el cuadrado central de 128×128\n(el centro de la imagen está en el centro del cuadrado)",
                    fontsize=11)
    ax[0].axis("off"); fig.colorbar(im, ax=ax[0], fraction=0.046, label="intensidad media del mapa")
    colores = {"duke": graficas.AZUL, "spy1": graficas.VERDE, "spy2": graficas.NARANJA}
    for c in COHORTES:
        s = d[d.cohorte == c]
        ax[1].scatter(s.parte_uniforme, s.parte_mapa, s=12, alpha=0.5, color=colores[c], label=c)
    ax[1].plot([0, 1], [0, 1], ls="--", color=graficas.GRIS, label="igual que un mapa uniforme")
    ax[1].set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="parte del cuadrado que ocupa la caja del tumor\n(lo que tendría un mapa uniforme)",
              ylabel="parte del mapa Grad-CAM dentro de la caja")
    r = resumen["todas"]
    ax[1].set_title(f"Por encima de la diagonal: el mapa se concentra en el tumor\n{r['pacientes_con_mas_que_uniforme']:.0%} de "
                    f"las pacientes por encima · medias {r['parte_mapa_media']:.0%} frente a {r['parte_uniforme_media']:.0%}",
                    fontsize=11)
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9, loc="lower right")
    x = np.arange(len(COHORTES) + 1); etiquetas = ["todas", *COHORTES]; ancho = 0.38
    ax[2].bar(x - ancho / 2, [resumen[e]["parte_mapa_media"] for e in etiquetas], ancho, color=graficas.AZUL,
              label="parte del mapa dentro de la caja")
    ax[2].bar(x + ancho / 2, [resumen[e]["parte_uniforme_media"] for e in etiquetas], ancho, color=graficas.GRIS,
              label="la de un mapa uniforme")
    for i, e in enumerate(etiquetas):
        ax[2].text(i, max(resumen[e]["parte_mapa_media"], resumen[e]["parte_uniforme_media"]) + 0.02,
                   f"×{resumen[e]['cociente']:.2f}\nmáx. dentro {resumen[e]['maximo_dentro']:.0%}", ha="center", fontsize=9)
    ax[2].set_xticks(x, [f"{e}\n({resumen[e]['pacientes']} pac.)" for e in etiquetas])
    ax[2].set(ylim=(0, 1), ylabel="media por paciente"); ax[2].grid(alpha=0.3, axis="y"); ax[2].legend(fontsize=9)
    ax[2].set_title("Por cohorte (×: cuántas veces más que uniforme)", fontsize=11)
    fig.savefig(salida / "gradcam_resumen.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 2. los cuatro casos del test (los de evaluar_test.py), con las 15 redes ----
    test = samples[samples.fold == -1]
    pac = pd.read_csv(RAIZ / "figuras" / "test" / "pacientes.csv", index_col=0)
    u = modelo.umbral
    pos, neg = pac[pac.pCR == 1], pac[pac.pCR == 0]
    casos = [("VERDADERO POSITIVO", pos[pos.recorte >= u].recorte.idxmax()),
             ("FALSO POSITIVO", neg[neg.recorte >= u].recorte.idxmax()),
             ("FALSO NEGATIVO", pos[pos.recorte < u].recorte.idxmin()),
             ("VERDADERO NEGATIVO", neg[neg.recorte < u].recorte.idxmin())]
    x = np.stack([leer(corte_central(test, pid), base_img) for _, pid in casos])
    mapas, probs = gradcam(modelo, modelo.redes, x)
    fig, ax = plt.subplots(len(casos), 3, figsize=(13.5, 4.5 * len(casos)), constrained_layout=True)
    fig.suptitle("Dónde mira el modelo final en los casos del test (Grad-CAM de las 15 redes, hacia la clase predicha)\n"
                 "azul: caja del tumor (patients.csv) · amarillo: cuadrado central que mira la red · corte central de cada paciente",
                 fontsize=12.5, fontweight="bold")
    casos_json = []
    for fila_ax, (tipo, pid), corte, mapa, prob in zip(ax, casos, x, mapas, probs):
        caja = caja_en_png(pacientes.loc[pid])
        m = medir(mapa, caja) if caja is not None else None
        completo = np.full((LADO, LADO), np.nan)
        completo[INI:INI + RECORTE, INI:INI + RECORTE] = mapa
        fila_ax[0].imshow(corte[1], cmap="gray", vmin=0, vmax=1)
        fila_ax[1].imshow(corte[1], cmap="gray", vmin=0, vmax=1)
        fila_ax[1].imshow(completo, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        realce = corte[1] - corte[0]
        fila_ax[2].imshow(realce, cmap="inferno", vmin=0, vmax=max(float(realce.max()), 1e-3))
        for A, titulo in zip(fila_ax, ("EARLY", "Grad-CAM sobre EARLY", "realce EARLY − PRE")):
            if caja is not None:
                a, b, c, e = caja
                A.add_patch(Rectangle((c, a), e - c, b - a, fill=False, ec="cyan", lw=1.6))
            A.add_patch(Rectangle((INI, INI), RECORTE, RECORTE, fill=False, ec="yellow", ls="--", lw=1.2))
            A.set_title(titulo, fontsize=10); A.axis("off")
        q = pacientes.loc[pid]
        texto = f"{tipo}\n{pid} ({q.dataset})\nreal pCR = {int(q.pCR)}\nprobabilidad del corte {prob:.3f}\n(umbral {u:.3f})"
        if m is not None:
            texto += f"\nmapa dentro de la caja {m['parte_mapa']:.0%}\n(uniforme {m['parte_uniforme']:.0%})"
        fila_ax[0].text(-0.08, 0.5, texto, transform=fila_ax[0].transAxes, ha="right", va="center", fontsize=10, fontweight="bold")
        casos_json.append({"tipo": tipo, "paciente": pid, "cohorte": q.dataset, "probabilidad_corte": float(prob), **(m or {})})
    fig.savefig(salida / "gradcam_casos_test.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with open(salida / "gradcam_resumen.json", "w", encoding="utf-8") as f:
        json.dump({"checksum": modelo.checksum, "capa": "ReLU del ultimo bloque (128 @ 16x16 sobre el cuadrado central)",
                   "validacion": resumen, "pacientes_sin_caja": int(sin_caja), "casos_test": casos_json,
                   "segundos": round(time.time() - t0, 1)}, f, indent=2, ensure_ascii=False)
    print(f"\nListo en {time.time() - t0:.0f} s. Figuras en {salida}")


if __name__ == "__main__":
    main()
