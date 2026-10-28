"""Elige los cortes de la galeria de ejemplos de la app: pacientes de ENTRENAMIENTO en las que el tumor se ve bien.

Criterio (solo con las imagenes y la caja del tumor de patients.csv; NO mira lo que predice el modelo):
    1. paciente de entrenamiento (nunca del test), con su caja del tumor llevada al PNG (cajas_tumor.caja_en_png);
    2. el tumor entero dentro del cuadrado central de 128x128 que mira la red, y de al menos 24 px de lado;
    3. de sus cortes, el de mas contraste del tumor: percentil 90 del realce (EARLY - PRE) dentro de la caja menos el
       realce medio del resto del tejido. Sin los avisos de calidad de la app;
    4. imagen limpia: lo que mas realza (el 1 % de puntos con mas realce) cae al menos la mitad en la caja; brillo
       medio de PRE <= 0,18 (sin imagenes con un brillo fuera de lo normal); ruido <= 0,065 (diferencia mediana con
       la imagen suavizada, relativa al brillo del tejido); tejido >= 25 % de la imagen; saturado <= 3 %;
    5. en cada grupo cohorte x pCR, los 2 de mas contraste: 6 con pCR = 1 y 6 con pCR = 0, 4 de cada cohorte.

La pagina reconoce estos cortes como ejemplos y, al predecir, compara con la pCR real usando las 3 redes que no
vieron a la paciente (su fold). La Verificacion sigue usando los cortes de exportar_web.py, que llevan guardada la
probabilidad calculada con PyTorch.

Uso:
    python galeria_web.py      -> docs/ejemplos/<sample_id>_{PRE,EARLY,LATE}.png y docs/datos/galeria.js
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from cajas_tumor import caja_en_png, localizar

RAIZ = Path(__file__).resolve().parent
WEB = RAIZ / "docs"
INI, FIN = 64, 191                    # cuadrado central de 128x128 (filas y columnas incluidas)
POR_GRUPO = 2
COHORTES = ("spy2", "spy1", "duke")


class Caja:
    """Lo que necesita caja_en_png de una fila de patients.csv."""
    def __init__(self, fila: dict):
        for k in ("n_xy", "sraw", "eraw", "scol", "ecol"):
            setattr(self, k, float(fila[k]) if fila[k] not in ("", "nan") else float("nan"))


def leer(ruta: Path) -> np.ndarray:
    """Igual que el entrenamiento y la app: escala de grises de 8 bits ("L") entre 255."""
    return np.asarray(Image.open(ruta).convert("L"), dtype=np.float32) / 255.0


def sin_avisos(x: np.ndarray) -> bool:
    """Las comprobaciones de predecir.avisos (aqui sin importar torch)."""
    tejido = x[0] > 0.1
    if tejido.sum() < 100 or x[1][tejido].mean() <= x[0][tejido].mean():
        return False
    return all(float(c.std()) >= 1e-3 and float(c.mean()) <= 0.8 for c in x)


def medir(x: np.ndarray, caja: tuple) -> dict:
    f0, f1, c0, c1 = caja
    m = np.zeros(x.shape[1:], bool)
    m[int(f0):int(f1) + 1, int(c0):int(c1) + 1] = True
    realce = x[1] - x[0]
    tejido = x[0] > 0.1
    fuera = tejido & ~m
    alto = realce >= np.percentile(realce, 99)
    pre8 = Image.fromarray((x[0] * 255).astype(np.uint8))
    suave = np.asarray(pre8.filter(ImageFilter.MedianFilter(3)), dtype=np.float32) / 255.0
    return {
        "contraste": float(np.percentile(realce[m], 90) - (realce[fuera].mean() if fuera.sum() > 500 else 1.0)),
        "concentracion": float((alto & m).sum() / alto.sum()),
        "brillo_pre": float(x[0].mean()),
        "ruido": float(np.median(np.abs(x[0] - suave)[tejido]) / x[0][tejido].mean()),
        "tejido": float(tejido.mean()),
        "saturado": max(float((c >= 0.99).mean()) for c in x),
    }


def limpia(d: dict) -> bool:
    return (d["concentracion"] >= 0.5 and d["brillo_pre"] <= 0.18 and d["ruido"] <= 0.065
            and d["tejido"] >= 0.25 and d["saturado"] <= 0.03)


def main() -> None:
    base_img = localizar("dataset")
    pacientes = {f["pid"]: f for f in csv.DictReader(open(RAIZ / "metadata" / "patients.csv", encoding="utf-8"))}
    cortes: dict[str, list[dict]] = {}
    for f in csv.DictReader(open(RAIZ / "metadata" / "samples.csv", encoding="utf-8")):
        if f["split"] == "train":                                   # nunca el test
            cortes.setdefault(f["patient_id"], []).append(f)

    candidatas = []
    for pid, lista in cortes.items():
        caja = caja_en_png(Caja(pacientes[pid]))
        if caja is None:
            continue
        f0, f1, c0, c1 = caja
        if not (f0 >= INI and f1 <= FIN and c0 >= INI and c1 <= FIN) or min(f1 - f0, c1 - c0) < 24:
            continue
        mejor = None
        for f in lista:
            x = np.stack([leer(base_img / f[c]) for c in ("path_pre", "path_early", "path_late")])
            if not sin_avisos(x):
                continue
            d = medir(x, caja)
            if mejor is None or d["contraste"] > mejor[1]["contraste"]:
                mejor = (f, d)
        if mejor and limpia(mejor[1]):
            candidatas.append((pacientes[pid]["dataset"], *mejor))

    ejemplos = []
    for pcr in (1, 0):
        for cohorte in COHORTES:
            grupo = sorted((c for c in candidatas if c[0] == cohorte and int(c[1]["pCR"]) == pcr),
                           key=lambda c: -c[2]["contraste"])
            print(f"{cohorte} pCR={pcr}: {len(grupo)} limpias -> {[c[1]['sample_id'] for c in grupo[:POR_GRUPO]]}")
            for _, f, d in grupo[:POR_GRUPO]:
                rutas = {}
                for fase, col in (("PRE", "path_pre"), ("EARLY", "path_early"), ("LATE", "path_late")):
                    destino = WEB / "ejemplos" / f"{f['sample_id']}_{fase}.png"
                    shutil.copyfile(base_img / f[col], destino)
                    rutas[fase] = f"ejemplos/{destino.name}"
                ejemplos.append({"sample_id": f["sample_id"], "patient_id": f["patient_id"], "cohorte": cohorte,
                                 "fold": int(f["fold"]), "pCR": int(f["pCR"]), "rutas": rutas,
                                 "medidas": {k: round(v, 4) for k, v in d.items()}})

    galeria = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "criterio": "Pacientes de ENTRENAMIENTO en las que el tumor se ve bien (ver galeria_web.py). Elegidas sin "
                    "mirar lo que predice el modelo; la mayoria de las 15 redes las vieron al entrenar.",
        "candidatas": len(candidatas),
        "ejemplos": ejemplos,
    }
    (WEB / "datos" / "galeria.js").write_text("// Generado por galeria_web.py: no editar a mano.\nwindow.GALERIA = "
                                              + json.dumps(galeria, ensure_ascii=False, indent=1) + ";\n",
                                              encoding="utf-8")
    print(f"\n{len(candidatas)} pacientes limpias; {len(ejemplos)} ejemplos en {WEB / 'ejemplos'} y docs/datos/galeria.js")


if __name__ == "__main__":
    main()
