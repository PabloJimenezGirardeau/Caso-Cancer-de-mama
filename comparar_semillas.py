"""Aplica la regla prerregistrada de los experimentos complementarios: media de las 3 semillas por fold.

Para cada configuracion (A = el modelo actual, B = el candidato) y cada fold, toma el AUC de validacion en la epoca
elegida por el early stopping (como comparar.py) de las semillas 42, 1 y 2 y hace su media. Despues compara B - A fold
a fold: diferencia media e IC95 (t de Student, 4 grados de libertad). Regla (DECISIONES.md): B sustituye a A solo si la
diferencia media supera +0,02 y su IC95 excluye el 0.

Uso:
    python comparar_semillas.py bn_normal_c128_lr0.0001 bn_normal_aten_lr0.0001
Lee *_epocas.csv de resultados/ y, si no estan, de referencia/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import comparar as co

RAIZ = Path(__file__).resolve().parent
SEMILLAS = {"42": "", "1": "_s1", "2": "_s2"}


def atencion_elegida(carpeta: Path, nombre: str, epoca: int) -> float | None:
    """Fraccion de la atencion dentro de la caja del tumor en la epoca elegida (solo modelos con --atencion)."""
    for base in (carpeta, co.REFERENCIA):
        ruta = base / f"{nombre}_epocas.csv"
        if ruta.exists():
            with open(ruta, encoding="utf-8") as f:
                filas = list(csv.DictReader(f))
            valor = filas[epoca - 1].get("atencion_dentro_val", "")
            try:
                v = float(valor)
            except ValueError:
                return None
            return None if v != v else v
    return None


def cargar_familia(carpeta: Path, prefijo: str) -> dict:
    por_semilla = {s: co.cargar(carpeta, f"{prefijo}{suf}") for s, suf in SEMILLAS.items()}
    falta = [f"semilla {s} fold {k}" for s, d in por_semilla.items() for k in range(5) if k not in d]
    if falta:
        raise SystemExit(f"{prefijo}: faltan {', '.join(falta)}")
    return por_semilla


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("actual", help="prefijo del modelo actual (A), sin _s1/_s2 ni _fold<k>")
    p.add_argument("candidato", help="prefijo del candidato (B)")
    p.add_argument("--margen", type=float, default=0.02, help="mejora minima de la regla")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "resultados")
    args = p.parse_args()

    familias = {n: cargar_familia(args.carpeta, n) for n in (args.actual, args.candidato)}
    medias = {}
    for nombre, por_semilla in familias.items():
        print(f"\n{nombre}  (AUC de validacion en la epoca elegida)")
        print(f"{'fold':>5}" + "".join(f"{'semilla ' + s:>12}" for s in SEMILLAS) + f"{'MEDIA':>10}")
        medias[nombre] = []
        for k in range(5):
            valores = [por_semilla[s][k]["auc_val"] for s in SEMILLAS]
            medias[nombre].append(sum(valores) / 3)
            print(f"{k:>5}" + "".join(f"{v:>12.4f}" for v in valores) + f"{medias[nombre][-1]:>10.4f}")
        todas = [por_semilla[s][k] for s in SEMILLAS for k in range(5)]
        sobre = sum(d["sobreentrenado"] for d in todas)
        sin = sum(d["sin_aprendizaje"] for d in todas)
        print(f"{'media':>5}{'':>36}{sum(medias[nombre]) / 5:>10.4f}   (15 modelos: sobreentrenados {sobre}, "
              f"sin aprendizaje {sin}; epoca elegida media {sum(d['epoca'] for d in todas) / 15:.1f}; "
              f"diferencia train - val media {sum(d['diferencia'] for d in todas) / 15:+.3f})")
        aten = [atencion_elegida(args.carpeta, f"{nombre}{SEMILLAS[s]}_fold{k}", por_semilla[s][k]["epoca"])
                for s in SEMILLAS for k in range(5)]
        aten = [a for a in aten if a is not None]
        if aten:
            print(f"      atencion dentro de la caja del tumor en la epoca elegida (validacion): media {sum(aten) / len(aten):.1%} "
                  f"(de {min(aten):.0%} a {max(aten):.0%})")

    difs = [b - a for a, b in zip(medias[args.actual], medias[args.candidato])]
    media, desv, _, semi = co.resumen(difs)
    print(f"\nDiferencia {args.candidato} - {args.actual}, por fold (media de 3 semillas): "
          + "  ".join(f"{d:+.4f}" for d in difs))
    print(f"diferencia media {media:+.4f}  desv. {desv:.4f}  IC95 {media - semi:+.4f} a {media + semi:+.4f}  "
          f"(mejora en {sum(d > 0 for d in difs)} de 5 folds)")
    excluye = media - semi > 0
    cumple = media > args.margen and excluye
    print(f"REGLA (mejora > {args.margen:+.2f} y el IC95 excluye el 0): "
          + ("SE CUMPLE -> el candidato sustituye al modelo actual" if cumple else
             "NO se cumple -> se mantiene el modelo actual"
             + (" (el IC excluye el 0, pero la mejora no llega al margen)" if excluye else "")))


if __name__ == "__main__":
    main()
