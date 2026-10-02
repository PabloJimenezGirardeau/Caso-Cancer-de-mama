"""Compara configuraciones con la metrica oficial del proyecto (ver DECISIONES.md).

    metrica oficial = media del AUC por paciente en validacion de las ULTIMAS 10 EPOCAS
    (no la mejor epoca, que es optimista porque se elige mirando la propia validacion)

Lee los `*_epocas.csv` que deja `entrenar.py` en resultados/. El argumento es el nombre
de la configuracion SIN el `_fold<k>_epocas.csv` final: por ejemplo, para los ficheros
`base_normal_fold0_epocas.csv`, `base_normal_fold1_epocas.csv`... es `base_normal`.

Uso:
    python comparar.py base_normal                      # una configuracion: tabla por fold
    python comparar.py base_normal otra_config          # dos: ademas, la diferencia EMPAREJADA

Comparacion emparejada: se resta, fold a fold, la metrica de las dos configuraciones y se
estudia esa diferencia. Un fold dificil lo es para las dos, asi que el ruido del reparto de
pacientes se cancela en gran parte. Solo se considera que una configuracion mejora a la
otra si el intervalo de confianza del 95 % de la diferencia media NO incluye el 0.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
# valor de la t de Student para un intervalo de confianza del 95 %, segun los grados de libertad
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def leer_auc(ruta: Path) -> list[float]:
    with open(ruta, encoding="utf-8") as f:
        return [float(fila["auc_val"]) for fila in csv.DictReader(f)]


def metricas(auc: list[float], n_ultimas: int = 10) -> dict:
    n = min(n_ultimas, len(auc))
    return {"media_ult": sum(auc[-n:]) / n, "ultima": auc[-1], "mejor": max(auc), "epocas": len(auc)}


def cargar(carpeta: Path, nombre: str) -> dict[int, dict]:
    """Metricas de cada fold disponible de la configuracion `nombre`."""
    por_fold = {}
    for k in range(5):
        ruta = carpeta / f"{nombre}_fold{k}_epocas.csv"
        if ruta.exists():
            por_fold[k] = metricas(leer_auc(ruta))
    if not por_fold:
        raise SystemExit(f"No encuentro ningun {nombre}_fold<k>_epocas.csv en {carpeta}")
    return por_fold


def resumen(valores: list[float]) -> tuple[float, float, float, float]:
    """Media, desviacion, error estandar y semianchura del intervalo de confianza del 95 %."""
    n = len(valores)
    media = sum(valores) / n
    if n < 2:
        return media, float("nan"), float("nan"), float("nan")
    desv = math.sqrt(sum((v - media) ** 2 for v in valores) / (n - 1))
    ee = desv / math.sqrt(n)
    return media, desv, ee, T95.get(n - 1, 1.96) * ee


def tabla_una(nombre: str, datos: dict[int, dict]) -> None:
    print(f"\n{nombre}")
    print(f"{'fold':>5} {'media ult.10 (OFICIAL)':>23} {'ultima epoca':>13} {'mejor epoca':>12} {'epocas':>7}")
    for k, m in datos.items():
        print(f"{k:>5} {m['media_ult']:>23.4f} {m['ultima']:>13.4f} {m['mejor']:>12.4f} {m['epocas']:>7}")
    for etiqueta, clave in (("media ult.10 (OFICIAL)", "media_ult"), ("ultima epoca", "ultima"),
                            ("mejor epoca (optimista)", "mejor")):
        media, desv, ee, ic = resumen([m[clave] for m in datos.values()])
        print(f"  {etiqueta:<24} media {media:.4f}   desv. entre folds {desv:.4f}   IC95 {media - ic:.3f} a {media + ic:.3f}")
    if len(datos) < 5:
        print(f"  AVISO: solo hay {len(datos)} de 5 folds: la comparacion no es completa.")


def comparar(a: str, datos_a: dict, b: str, datos_b: dict) -> None:
    comunes = sorted(set(datos_a) & set(datos_b))
    if len(comunes) < 2:
        print("\nHacen falta al menos 2 folds en comun para la comparacion emparejada.")
        return
    print(f"\nDiferencia emparejada: {b}  menos  {a}   (metrica oficial, media de las ultimas 10 epocas)")
    print(f"{'fold':>5} {a[:18]:>20} {b[:18]:>20} {'diferencia':>11}")
    difs = []
    for k in comunes:
        d = datos_b[k]["media_ult"] - datos_a[k]["media_ult"]
        difs.append(d)
        print(f"{k:>5} {datos_a[k]['media_ult']:>20.4f} {datos_b[k]['media_ult']:>20.4f} {d:>+11.4f}")
    media, desv, ee, ic = resumen(difs)
    print(f"\n  diferencia media {media:+.4f}   desv. {desv:.4f}   IC95 {media - ic:+.3f} a {media + ic:+.3f}")
    if media - ic > 0:
        veredicto = f"{b} MEJORA a {a}: el intervalo excluye el 0."
    elif media + ic < 0:
        veredicto = f"{b} EMPEORA a {a}: el intervalo excluye el 0."
    else:
        veredicto = "NO CONCLUYENTE: el intervalo incluye el 0 (la diferencia cabe en el ruido)."
    print(f"  -> {veredicto}")
    if len(comunes) < 5:
        print(f"  AVISO: solo {len(comunes)} folds en comun.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configuracion", nargs="+", help="una o dos configuraciones (nombre sin _fold<k>_epocas.csv)")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "resultados")
    args = p.parse_args()
    if len(args.configuracion) > 2:
        raise SystemExit("Se pueden dar como maximo dos configuraciones.")
    datos = {nombre: cargar(args.carpeta, nombre) for nombre in args.configuracion}
    for nombre, d in datos.items():
        tabla_una(nombre, d)
    if len(args.configuracion) == 2:
        a, b = args.configuracion
        comparar(a, datos[a], b, datos[b])


if __name__ == "__main__":
    main()
