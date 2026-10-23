"""Prueba de memorizacion: ¿es capaz la red de aprenderse un grupo pequeño de pacientes?

Es el "paso 6B" de los apuntes de clase: si una red no consigue memorizar unos pocos
ejemplos, hay un fallo en el modelo, en la perdida o en el optimizador (nunca en los
datos). Aqui NO importa generalizar: se entrena y se mide sobre los MISMOS cortes.
Lo unico que se mira es que la perdida de entrenamiento baje hacia 0.

Ajustes provisionales (solo para esta prueba, no son decisiones del proyecto):
    optimizador Adam, learning rate 0,001, lote 32, sin aumentado, 20 pacientes
    (10 con pCR=1 y 10 con pCR=0, todos del conjunto de entrenamiento).

Uso:
    python prueba_memorizacion.py                  # 150 epocas
    python prueba_memorizacion.py --epocas 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

import modelos as mo
import utils_caso as uc

RAIZ = Path(__file__).resolve().parent


def localizar(subcarpeta: str) -> Path:
    """Carpeta que contiene `subcarpeta` (metadata o dataset): la raiz del proyecto o
    su subcarpeta breastdcedl/ (la disposicion cambia entre ordenadores)."""
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / subcarpeta).is_dir():
            return base
    raise FileNotFoundError(f"No encuentro '{subcarpeta}/' en {RAIZ} ni en {RAIZ / 'breastdcedl'}")


def elegir_pacientes(filas, por_clase: int):
    """Las filas de `por_clase` pacientes con pCR=1 y `por_clase` con pCR=0 (los primeros
    por orden de identificador, para que la prueba sea siempre la misma)."""
    etiqueta = filas.groupby("patient_id").pCR.first()
    pacientes = sorted(etiqueta[etiqueta == 1].index)[:por_clase] + \
        sorted(etiqueta[etiqueta == 0].index)[:por_clase]
    return filas[filas.patient_id.isin(pacientes)].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epocas", type=int, default=150)
    p.add_argument("--pacientes-por-clase", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", type=Path, default=RAIZ / "resultados")
    args = p.parse_args()

    torch.manual_seed(args.semilla)
    generador = torch.Generator().manual_seed(args.semilla)

    # --- datos: 20 pacientes del conjunto de entrenamiento, cargados UNA vez en memoria ---
    samples = uc.cargar_samples(localizar("metadata"))
    entrenamiento, _ = uc.particion(samples, fold_val=0)
    filas = elegir_pacientes(entrenamiento, args.pacientes_por_clase)
    raiz_img = localizar("dataset")
    X = torch.stack([torch.from_numpy(uc.cargar_imagen(f, raiz_img)) for f in filas.itertuples()])
    Y = torch.tensor(filas.pCR.values, dtype=torch.float32)
    n = len(Y)
    print(f"{filas.patient_id.nunique()} pacientes, {n} cortes ({int(Y.sum())} con pCR=1, "
          f"{n - int(Y.sum())} con pCR=0); X: {tuple(X.shape)}")
    p1 = float(Y.mean())
    perdida_azar = -(p1 * torch.log(torch.tensor(p1)) + (1 - p1) * torch.log(torch.tensor(1 - p1))).item()
    print(f"referencia: una red que diera siempre la proporcion de pCR=1 ({p1:.2f}) tendria "
          f"perdida {perdida_azar:.4f}; la meta es bajar mucho de ahi\n")

    # --- modelo, perdida y optimizador (provisionales) ---
    modelo = mo.RedBase()
    criterio = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)

    historial = []
    t0 = time.time()
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        orden = torch.randperm(n, generator=generador)          # barajar en cada epoca
        suma = 0.0
        for i in range(0, n, args.lote):
            idx = orden[i:i + args.lote]
            optimizador.zero_grad()                             # PyTorch acumula gradientes si no se limpian
            perdida = criterio(modelo(X[idx]), Y[idx])
            perdida.backward()
            optimizador.step()
            suma += perdida.item() * len(idx)
        historial.append(suma / n)
        if epoca == 1 or epoca % 10 == 0:
            print(f"epoca {epoca:4d}/{args.epocas}   perdida de entrenamiento = {historial[-1]:.4f}   "
                  f"({time.time() - t0:.0f} s)", flush=True)

    # --- resultado final: la red, ya sin dropout (modo evaluacion), sobre los mismos cortes ---
    modelo.eval()
    with torch.no_grad():
        logits = torch.cat([modelo(X[i:i + 50]) for i in range(0, n, 50)])
    prob = torch.sigmoid(logits)
    perdida_final = criterio(logits, Y).item()
    pred = (prob >= 0.5).float()
    acierto = (pred == Y).float().mean().item()
    vp = int(((pred == 1) & (Y == 1)).sum()); vn = int(((pred == 0) & (Y == 0)).sum())
    fp = int(((pred == 1) & (Y == 0)).sum()); fn = int(((pred == 0) & (Y == 1)).sum())
    print(f"\nResultado en modo evaluacion, sobre los mismos {n} cortes:")
    print(f"  perdida = {perdida_final:.4f}   accuracy = {100 * acierto:.1f} %   "
          f"(VP={vp}, VN={vn}, FP={fp}, FN={fn})")

    # --- curva de perdida: fichero CSV y, si hay matplotlib, imagen ---
    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_csv = args.salida / "prueba_memorizacion_perdida.csv"
    ruta_csv.write_text("epoca,perdida_entrenamiento\n" +
                        "\n".join(f"{e},{v:.6f}" for e, v in enumerate(historial, 1)) + "\n", encoding="utf-8")
    print(f"curva guardada en {ruta_csv}")
    try:
        import graficas
        ruta_png = args.salida / "prueba_memorizacion_curva.png"
        graficas.dibujar_curva_perdida(
            historial, ruta_png, "Prueba de memorización: 20 pacientes, red base", perdida_azar)
        print(f"imagen guardada en {ruta_png}")
    except ImportError:
        print("(sin matplotlib: no se dibuja la imagen)")


if __name__ == "__main__":
    main()
