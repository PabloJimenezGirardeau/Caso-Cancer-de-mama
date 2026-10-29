"""Donde mira la red: mapas de atencion de un modelo entrenado con --atencion, sobre pacientes de SU validacion.

Para cada paciente dibuja el corte central (EARLY), el realce (EARLY - PRE) y, encima, la atencion de la red
(los pesos softmax del mapa de 16x16, ampliados a 256x256), con la caja del tumor de patients.csv en azul para
comparar. La caja solo se dibuja: la red no la recibe.

Uso:
    python analisis/mapas_atencion.py                                   # bn_normal_aten_lr0.0001, fold 0, 8 pacientes
    python analisis/mapas_atencion.py --config bn_normal_aten_lr0.0001_s1 --fold 3 --pacientes 12
Busca el .pt en resultados/ (y en modelo/). Salida: figuras/atencion/<config>_fold<k>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # para importar los modulos de la raiz del proyecto

import numpy as np
import torch

import modelos as mo
import utils_caso as uc
from cajas_tumor import caja_en_png, localizar

RAIZ = Path(__file__).resolve().parent.parent   # la raiz del proyecto: este script esta en analisis/


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="bn_normal_aten_lr0.0001")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--pacientes", type=int, default=8)
    p.add_argument("--semilla", type=int, default=0, help="para elegir las pacientes al azar (siempre las mismas)")
    p.add_argument("--carpeta", type=Path, default=None, help="carpeta del .pt (por defecto resultados/ y modelo/)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    nombre = f"{args.config}_fold{args.fold}.pt"
    carpetas = [args.carpeta] if args.carpeta else [RAIZ / "resultados", RAIZ / "modelo"]
    ruta = next((c / nombre for c in carpetas if (c / nombre).exists()), None)
    if ruta is None:
        raise SystemExit(f"No encuentro {nombre} en resultados/ ni en modelo/")
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    if not ck.get("atencion"):
        raise SystemExit(f"{nombre} no se entreno con --atencion")
    red = mo.RedBase(batchnorm=ck.get("batchnorm", False), atencion=True)
    red.load_state_dict(ck["state_dict"])
    red.eval()

    base_meta, base_img = localizar("metadata"), localizar("dataset")
    samples = uc.cargar_samples(base_meta)
    pacientes = uc.cargar_patients(base_meta).set_index("pid")
    validacion = samples[samples.fold == args.fold]            # pacientes que este modelo NO vio al entrenar
    rng = np.random.default_rng(args.semilla)
    elegidas = rng.choice(sorted(validacion.patient_id.unique()), args.pacientes, replace=False)

    fig, ax = plt.subplots(2, args.pacientes, figsize=(3.2 * args.pacientes, 7), constrained_layout=True, squeeze=False)
    fig.suptitle(f"Dónde mira la red · {args.config}, fold {args.fold} (pacientes de su validación)\n"
                 "arriba: imagen EARLY · abajo: realce EARLY − PRE con la atención de la red encima · "
                 "azul: caja del tumor (solo para comparar; la red no la recibe)", fontsize=12, fontweight="bold")
    for j, pid in enumerate(elegidas):
        cortes = validacion[validacion.patient_id == pid].sort_values("slice_index")
        x = uc.cargar_imagen(cortes.iloc[len(cortes) // 2], base_img)
        with torch.no_grad():
            logit, puntos = red.forward_atencion(torch.from_numpy(x)[None])
        pesos = torch.softmax(puntos.flatten(1), dim=1).view(16, 16).numpy()
        atencion = np.kron(pesos, np.ones((16, 16)))                # 16x16 -> 256x256, cada celda un bloque
        caja = caja_en_png(pacientes.loc[pid])
        ax[0, j].imshow(x[1], cmap="gray", vmin=0, vmax=1)
        ax[1, j].imshow(x[1] - x[0], cmap="gray")
        ax[1, j].imshow(atencion, cmap="jet", alpha=0.45)
        for fila in range(2):
            if caja is not None:
                a, b, c, e = caja
                ax[fila, j].add_patch(Rectangle((c, a), e - c, b - a, fill=False, ec="cyan", lw=1.6))
            ax[fila, j].axis("off")
        dentro = None
        if caja is not None:
            a, b, c, e = [int(round(v)) for v in caja]
            masc = np.zeros((256, 256), bool)
            masc[max(a, 0):b + 1, max(c, 0):e + 1] = True
            dentro = float(atencion[masc].sum()) / 256      # cada celda se repite en 16x16 = 256 pixeles
        prob = float(torch.sigmoid(logit))
        ax[0, j].set_title(f"{pid}\npCR real {int(pacientes.loc[pid].pCR)} · prob. {prob:.2f}"
                           + (f"\natención en la caja: {dentro:.0%}" if dentro is not None else ""), fontsize=9)
    salida = RAIZ / "figuras" / "atencion"
    salida.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida / f"{args.config}_fold{args.fold}.png", dpi=100, bbox_inches="tight", facecolor="white")
    print(f"Figura en {salida / f'{args.config}_fold{args.fold}.png'}")


if __name__ == "__main__":
    main()
