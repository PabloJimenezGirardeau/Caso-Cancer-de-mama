"""Junta los resultados de todos los entrenamientos en una tabla comparativa.

Lee los `*_resumen.json` que deja `entrenar.py` en la carpeta de resultados y
muestra dos tablas:

  1. Una fila por entrenamiento (arquitectura, perdida, fold, AUC, epocas...).
  2. Por arquitectura y perdida: media y desviacion del AUC sobre los folds
     disponibles. Es la que decide la criba y la confirmacion (DECISIONES.md,
     D2): el AUC por paciente en validacion interna, nunca el de test.

Las pruebas rapidas (--muestra-rapida) se excluyen salvo que se pida lo contrario.

    python resumen_resultados.py
    python resumen_resultados.py --carpeta resultados --incluir-pruebas
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

RAIZ_SCRIPT = Path(__file__).resolve().parent


def cargar(carpeta: Path, incluir_pruebas: bool) -> list[dict]:
    filas = []
    for ruta in sorted(carpeta.glob("*_resumen.json")):
        with open(ruta, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("muestra_rapida") and not incluir_pruebas:
            continue
        filas.append(r)
    return filas


def config(r: dict) -> str:
    """Lo que distingue a este entrenamiento de la configuracion base (los
    resultados antiguos, sin estos campos, cuentan como base)."""
    partes = []
    if r.get("entrada", "fases") != "fases":
        partes.append(r["entrada"])
    if r.get("aumentado", "basico") != "basico":
        partes.append("aug-" + r["aumentado"])
    if r.get("dropout", 0.5) != 0.5:
        partes.append(f"do{r['dropout']:g}")
    if r.get("lr", 1e-3) != 1e-3:
        partes.append(f"lr{r['lr']:g}")
    if r.get("weight_decay", 1e-4) != 1e-4:
        partes.append(f"wd{r['weight_decay']:g}")
    return "+".join(partes) or "base"


def media_desviacion(valores: list[float]) -> tuple[float, float]:
    validos = [v for v in valores if not math.isnan(v)]
    if not validos:
        return float("nan"), float("nan")
    m = sum(validos) / len(validos)
    if len(validos) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((v - m) ** 2 for v in validos) / (len(validos) - 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--carpeta", type=Path, default=RAIZ_SCRIPT / "resultados")
    p.add_argument("--incluir-pruebas", action="store_true",
                   help="incluir tambien las ejecuciones con --muestra-rapida")
    args = p.parse_args()

    filas = cargar(args.carpeta, args.incluir_pruebas)
    if not filas:
        raise SystemExit(f"No hay resultados en {args.carpeta}")

    print(f"\n1) Un entrenamiento por fila ({len(filas)})\n")
    print(f"{'arquitectura':<12} {'perdida':<10} {'config':<22} {'fold':>4} {'AUC val':>8} {'mejor ep.':>9} "
          f"{'epocas':>6} {'params':>9} {'s/epoca':>8}  dispositivo")
    for r in sorted(filas, key=lambda r: (r["arquitectura"], r["perdida"], config(r), r["fold"])):
        print(f"{r['arquitectura']:<12} {r['perdida']:<10} {config(r):<22} {r['fold']:>4} {r['auc_val']:>8.4f} "
              f"{r['mejor_epoca']:>9} {r['epocas_entrenadas']:>6} {r['parametros']:>9,} "
              f"{r['tiempo_medio_por_epoca_seg']:>8.1f}  {r['dispositivo']}")

    grupos: dict[tuple, list[dict]] = defaultdict(list)
    for r in filas:
        grupos[(r["arquitectura"], r["perdida"], config(r))].append(r)

    print("\n2) Por arquitectura, perdida y configuracion (ordenado por AUC medio)\n")
    print(f"{'arquitectura':<12} {'perdida':<10} {'config':<22} {'folds':<11} {'AUC medio':>9} {'desv.':>7}")
    resumen = []
    for (arch, perdida, cfg), rs in grupos.items():
        m, d = media_desviacion([r["auc_val"] for r in rs])
        folds = ",".join(str(r["fold"]) for r in sorted(rs, key=lambda r: r["fold"]))
        resumen.append((m, arch, perdida, cfg, folds, d))
    for m, arch, perdida, cfg, folds, d in sorted(resumen, key=lambda t: -t[0] if not math.isnan(t[0]) else 1):
        print(f"{arch:<12} {perdida:<10} {cfg:<22} {folds:<11} {m:>9.4f} {d:>7.4f}")

    print("\nRecuerda: con un solo fold (~219 pacientes) el IC95 del AUC es de unos +-0,08;"
          "\ncon los 5 folds juntos, de unos +-0,04. Diferencias menores no son concluyentes.")


if __name__ == "__main__":
    main()
