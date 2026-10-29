"""Compara configuraciones con la metrica del proyecto (ver HOJA_DE_RUTA.md, decidida el 2026-10-14).

    metrica = AUC por paciente de validacion en la EPOCA ELEGIDA por el early stopping (parada.py:
              maximo de la media movil de 3 epocas, paciencia 10, maximo 60), media +- desviacion de los
              5 folds. Junto a ella, la diferencia AUC de entrenamiento - AUC de validacion en esa epoca:
              si supera 0,15, el fold esta SOBREENTRENADO; si el AUC de entrenamiento no llega a 0,60, el
              fold esta SIN APRENDIZAJE (la red no distingue ni a sus pacientes de entrenamiento).

Lee los `*_epocas.csv` que deja `entrenar.py` (en resultados/ o, si no estan, en referencia/) y aplica la
regla de parada.py sobre ellos, asi que sirve tambien para los registros antiguos de 40 epocas fijas: dice
que epoca habria elegido el early stopping. El argumento es el nombre de la configuracion SIN el
`_fold<k>_epocas.csv` final: para `bn_normal_fold0_epocas.csv`... es `bn_normal`.

Uso:
    python analisis/comparar.py bn_normal                        # una configuracion: tabla por fold
    python analisis/comparar.py bn_normal bn_normal_lr0.0001     # dos: ademas, la diferencia EMPAREJADA

Comparacion emparejada: se resta, fold a fold, el AUC de validacion en la epoca elegida de las dos
configuraciones. Un fold dificil lo es para las dos, asi que el ruido del reparto de pacientes se cancela
en gran parte. Una configuracion mejora a otra solo si el IC95 de la diferencia media NO incluye el 0.
Aviso: elegir la epoca mirando la validacion es algo optimista; el juez independiente es el test.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # para importar los modulos de la raiz del proyecto

import parada as pa

RAIZ = Path(__file__).resolve().parent.parent   # la raiz del proyecto: este script esta en analisis/
REFERENCIA = RAIZ / "referencia"      # copias en git de los resultados importantes (la base, etc.)
# valor de la t de Student para un intervalo de confianza del 95 %, segun los grados de libertad
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def leer(ruta: Path) -> tuple[list[float], list[float]]:
    """AUC de validacion y de entrenamiento de cada epoca."""
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return [float(r["auc_val"]) for r in filas], [float(r["auc_train"]) for r in filas]


def metricas(auc_val: list[float], auc_train: list[float]) -> dict:
    r = pa.recorrer(auc_val)
    i = r["indice"]
    dif = auc_train[i] - auc_val[i]
    return {"epoca": i + 1, "epocas": r["epocas"], "motivo": r["motivo"], "registradas": len(auc_val),
            "auc_val": auc_val[i], "suavizado": float(pa.suavizar(auc_val)[i]), "auc_train": auc_train[i],
            "diferencia": dif, "sobreentrenado": dif > pa.DIFERENCIA_MAX,
            "sin_aprendizaje": auc_train[i] < pa.AUC_TRAIN_MIN}


def cargar(carpeta: Path, nombre: str) -> dict[int, dict]:
    """Metricas de cada fold disponible de la configuracion `nombre`."""
    por_fold = {}
    for k in range(5):
        # primero resultados/ (lo recien entrenado) y, si no esta, referencia/ (copias guardadas en git)
        for base in (carpeta, REFERENCIA):
            ruta = base / f"{nombre}_fold{k}_epocas.csv"
            if ruta.exists():
                por_fold[k] = metricas(*leer(ruta))
                break
    if not por_fold:
        raise SystemExit(f"No encuentro ningun {nombre}_fold<k>_epocas.csv en {carpeta} ni en {REFERENCIA}")
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
    print(f"{'fold':>5} {'epoca elegida':>14} {'entrenadas':>11} {'AUC val':>8} {'media movil':>12} "
          f"{'AUC train':>10} {'diferencia':>11}  estado")
    for k, m in datos.items():
        estado = ("SOBREENTRENADO" if m["sobreentrenado"] else
                  "SIN APRENDIZAJE" if m["sin_aprendizaje"] else "ok")
        if m["motivo"] == "registro_terminado":
            estado += "  (no paro en sus epocas: registro antiguo mas corto que el maximo)"
        print(f"{k:>5} {m['epoca']:>14} {m['epocas']:>11} {m['auc_val']:>8.4f} {m['suavizado']:>12.4f} "
              f"{m['auc_train']:>10.4f} {m['diferencia']:>+11.4f}  {estado}")
    media, desv, _, ic = resumen([m["auc_val"] for m in datos.values()])
    print(f"  AUC de validacion en la epoca elegida: {media:.4f} +- {desv:.4f}   IC95 {media - ic:.3f} a {media + ic:.3f}")
    media_s, desv_s, _, _ = resumen([m["suavizado"] for m in datos.values()])
    print(f"  (media movil en esa epoca, menos optimista: {media_s:.4f} +- {desv_s:.4f})")
    dif, _, _, _ = resumen([m["diferencia"] for m in datos.values()])
    n_sobre = sum(m["sobreentrenado"] for m in datos.values())
    n_sin = sum(m["sin_aprendizaje"] for m in datos.values())
    print(f"  diferencia media entrenamiento - validacion: {dif:+.4f}   folds sobreentrenados: {n_sobre}, "
          f"sin aprendizaje: {n_sin}, ok: {len(datos) - n_sobre - n_sin} (de {len(datos)})"
          f"   epocas entrenadas en total: {sum(m['epocas'] for m in datos.values())}")
    if len(datos) < 5:
        print(f"  AVISO: solo hay {len(datos)} de 5 folds: la comparacion no es completa.")


def comparar(a: str, datos_a: dict, b: str, datos_b: dict) -> None:
    comunes = sorted(set(datos_a) & set(datos_b))
    if len(comunes) < 2:
        print("\nHacen falta al menos 2 folds en comun para la comparacion emparejada.")
        return
    print(f"\nDiferencia emparejada: {b}  menos  {a}   (AUC de validacion en la epoca elegida)")
    print(f"{'fold':>5} {a[:18]:>20} {b[:18]:>20} {'diferencia':>11}")
    difs = []
    for k in comunes:
        d = datos_b[k]["auc_val"] - datos_a[k]["auc_val"]
        difs.append(d)
        print(f"{k:>5} {datos_a[k]['auc_val']:>20.4f} {datos_b[k]['auc_val']:>20.4f} {d:>+11.4f}")
    media, desv, ee, ic = resumen(difs)
    print(f"\n  diferencia media {media:+.4f}   desv. {desv:.4f}   IC95 {media - ic:+.3f} a {media + ic:+.3f}")
    if media - ic > 0:
        veredicto = f"{b} MEJORA a {a}: el intervalo excluye el 0."
    elif media + ic < 0:
        veredicto = f"{b} EMPEORA a {a}: el intervalo excluye el 0."
    else:
        veredicto = "NO CONCLUYENTE: el intervalo incluye el 0 (la diferencia cabe en el ruido)."
    print(f"  -> {veredicto}")
    n_a = sum(datos_a[k]["sobreentrenado"] for k in comunes)
    n_b = sum(datos_b[k]["sobreentrenado"] for k in comunes)
    s_a = sum(datos_a[k]["sin_aprendizaje"] for k in comunes)
    s_b = sum(datos_b[k]["sin_aprendizaje"] for k in comunes)
    print(f"  folds sobreentrenados: {a} {n_a}, {b} {n_b};  sin aprendizaje: {a} {s_a}, {b} {s_b} (de {len(comunes)})")
    if len(comunes) < 5:
        print(f"  AVISO: solo {len(comunes)} folds en comun.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configuracion", nargs="+", help="una o dos configuraciones (nombre sin _fold<k>_epocas.csv)")
    p.add_argument("--carpeta", type=Path, default=RAIZ / "resultados")
    args = p.parse_args()
    if len(args.configuracion) > 2:
        raise SystemExit("Se pueden dar como maximo dos configuraciones.")
    print(f"Early stopping: maximo de la media movil de {pa.VENTANA} epocas del AUC de validacion, "
          f"paciencia {pa.PACIENCIA}, maximo {pa.EPOCAS_MAX}; sobreentrenado si la diferencia supera {pa.DIFERENCIA_MAX:g}; "
          f"sin aprendizaje si el AUC de entrenamiento no llega a {pa.AUC_TRAIN_MIN:g}")
    datos = {nombre: cargar(args.carpeta, nombre) for nombre in args.configuracion}
    for nombre, d in datos.items():
        tabla_una(nombre, d)
    if len(args.configuracion) == 2:
        a, b = args.configuracion
        comparar(a, datos[a], b, datos[b])


if __name__ == "__main__":
    main()
