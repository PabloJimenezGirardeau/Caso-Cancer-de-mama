"""Donde esta el tumor en los PNG: las cajas de patients.csv llevadas al sistema de coordenadas de las imagenes.

`sraw, eraw, scol, ecol` son la caja del tumor (VOI) en el volumen original (`n_xy` x `n_xy`), con filas y columnas del
array. Los PNG salen del recorte "MinCrop" del repositorio oficial (github.com/naomifridman/BreastDCEDL,
DUKE/crop_spy2_spy1.ipynb, funcion `crop_around_voi_cords`): una ventana de 256x256 centrada en el centro de la caja
y desplazada para no salirse del volumen (si `n_xy` = 256, la ventana es la imagen entera). Ademas, el PNG esta GIRADO
90 grados respecto al array: la fila r y la columna c del recorte acaban en la fila 255 - c y la columna r del PNG.
Comprobado: de las 8 orientaciones posibles, el giro de 90 grados es la unica que pone la caja sobre la zona de realce
en las tres cohortes (y se ve a ojo en la figura).

Calcula, para las pacientes de entrenamiento (el test no se mira), cuanto del tumor cae dentro del cuadrado central
de 128x128 que mira el modelo, y dibuja ejemplos.

Uso:
    python src/datos/cajas_tumor.py            -> figuras/datos/cajas_tumor.png y el resumen en pantalla
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/: para importar los paquetes del proyecto

from datos import utils_caso as uc
RAIZ = Path(__file__).resolve().parents[2]   # la raiz del proyecto: este fichero esta en src/datos/
LADO, RECORTE = 256, 128


def localizar(subcarpeta: str) -> Path:
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / subcarpeta).is_dir():
            return base
    raise FileNotFoundError(f"No encuentro '{subcarpeta}/'")


def origen_recorte(centro: int, n: int, salida: int = LADO) -> int:
    """Primera fila (o columna) de la ventana de 256 centrada en `centro` dentro de un lado de `n`
    (misma logica que crop_around_voi_cords del repositorio oficial)."""
    a, b = centro - salida // 2, centro + salida // 2
    if a < 0:
        b -= a; a = 0
    if b >= n:
        b, a = n - 1, max(0, a - (b - n + 1))
    if b - a + 1 != salida:
        b = a + salida - 1
        if b >= n:
            b, a = n - 1, n - salida
    return a


def caja_en_png(q) -> tuple[float, float, float, float] | None:
    """(fila0, fila1, col0, col1) de la caja del tumor en el PNG; None si la imagen original es menor que 256."""
    if any(np.isnan(float(v)) for v in (q.n_xy, q.sraw, q.eraw, q.scol, q.ecol)):
        return None
    n = int(q.n_xy)
    if n < LADO:
        return None
    o_f = origen_recorte(int(q.sraw + q.eraw) // 2, n)
    o_c = origen_recorte(int(q.scol + q.ecol) // 2, n)
    sr, er, sc, ec = q.sraw - o_f, q.eraw - o_f, q.scol - o_c, q.ecol - o_c
    return LADO - 1 - ec, LADO - 1 - sc, sr, er          # giro de 90 grados


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    base_meta, base_img = localizar("metadata"), localizar("dataset")
    samples = uc.cargar_samples(base_meta)
    pacientes = uc.cargar_patients(base_meta).set_index("pid")
    entrenamiento = samples[samples.fold >= 0]

    ini, fin = (LADO - RECORTE) // 2, (LADO + RECORTE) // 2 - 1          # 64 y 191
    filas = []
    for pid in entrenamiento.patient_id.unique():
        q = pacientes.loc[pid]
        caja = caja_en_png(q)
        if caja is None:
            continue
        a, b, c, e = caja
        area = (b - a + 1) * (e - c + 1)
        dentro = max(0, min(b, fin) - max(a, ini) + 1) * max(0, min(e, fin) - max(c, ini) + 1)
        cf, cc = (a + b) / 2, (c + e) / 2
        filas.append((pid, q.dataset, ini <= cf <= fin and ini <= cc <= fin, dentro / area, b - a + 1, e - c + 1, cf, cc))
    import pandas as pd
    d = pd.DataFrame(filas, columns=["pid", "cohorte", "centro_dentro", "fraccion_dentro", "alto", "ancho", "fila", "col"])
    print(f"{len(d)} pacientes de entrenamiento con n_xy >= 256 "
          f"({entrenamiento.patient_id.nunique() - len(d)} con imagen original menor, sin caja)")
    print(f"Centro del tumor: fila media {d.fila.mean():.0f}, columna media {d.col.mean():.0f} (centro de la imagen 127,5)")
    print(f"Centro del tumor dentro del cuadrado central {RECORTE}x{RECORTE}: {d.centro_dentro.mean():.0%} "
          f"(duke {d[d.cohorte == 'duke'].centro_dentro.mean():.0%}, spy1 {d[d.cohorte == 'spy1'].centro_dentro.mean():.0%}, "
          f"spy2 {d[d.cohorte == 'spy2'].centro_dentro.mean():.0%})")
    print(f"Parte de la caja dentro del cuadrado: mediana {d.fraccion_dentro.median():.0%}; menos de la mitad dentro: "
          f"{(d.fraccion_dentro < 0.5).mean():.0%}")
    print(f"Tamaño de la caja: mediana {d.alto.median():.0f} x {d.ancho.median():.0f} px; mayor de {RECORTE} en algún lado: "
          f"{((d.alto > RECORTE) | (d.ancho > RECORTE)).mean():.0%}")

    # ejemplos: 8 al azar (semilla fija) + los 4 con menos tumor dentro del cuadrado
    rng = np.random.default_rng(5)
    ejemplos = list(rng.choice(d.pid.values, 8, replace=False)) + list(d.nsmallest(4, "fraccion_dentro").pid)
    fig, ax = plt.subplots(2, 6, figsize=(21, 8.2), constrained_layout=True)
    fig.suptitle("Caja del tumor (azul, de patients.csv llevada al PNG) y cuadrado central 128×128 que mira el modelo "
                 "(amarillo)\nimagen: realce EARLY − PRE del corte central · fila de abajo, derecha: los 4 casos con "
                 "menos tumor dentro del cuadrado", fontsize=13, fontweight="bold")
    for A, pid in zip(ax.ravel(), ejemplos):
        corte = entrenamiento[entrenamiento.patient_id == pid].sort_values("slice_index")
        x = uc.cargar_imagen(corte.iloc[len(corte) // 2], base_img)
        a, b, c, e = caja_en_png(pacientes.loc[pid])
        A.imshow(x[1] - x[0], cmap="inferno")
        A.add_patch(Rectangle((c, a), e - c, b - a, fill=False, ec="cyan", lw=1.8))
        A.add_patch(Rectangle((ini, ini), RECORTE, RECORTE, fill=False, ec="yellow", ls="--", lw=1.4))
        r = d[d.pid == pid].iloc[0]
        A.set_title(f"{pid} ({r.cohorte})\n{r.fraccion_dentro:.0%} de la caja dentro", fontsize=9.5)
        A.axis("off")
    salida = RAIZ / "figuras" / "datos"
    salida.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida / "cajas_tumor.png", dpi=100, bbox_inches="tight", facecolor="white")
    print(f"Figura en {salida / 'cajas_tumor.png'}")


if __name__ == "__main__":
    main()
