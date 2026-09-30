"""Entrena y evalua una de las arquitecturas candidatas (ver modelos.py).

Protocolo comun a las cuatro arquitecturas (DECISIONES.md, D4), para que la
comparacion entre ellas sea justa:

    Entrada         tal cual llega de utils_caso, en [0,1], sin normalizar mas
    Aumentado       volteo horizontal (p=0.5) al tensor de 3 canales completo,
                    solo en entrenamiento
    Optimizador     AdamW, lr=1e-3, weight_decay=1e-4
    Scheduler       ReduceLROnPlateau sobre el AUC de validacion (factor 0.5,
                    paciencia 3 epocas)
    Lote            32
    Epocas          maximo 40, parada si el AUC por paciente no mejora en 8
    Metrica         AUC por paciente (agregacion "mean", provisional: el
                    metodo definitivo se decide mas adelante)
    Semilla         42 (Python, NumPy, PyTorch, cuDNN determinista)

Rutas: `metadata/` y `dataset/` se buscan por separado, primero en --raiz y
luego en --raiz/breastdcedl/ (por defecto --raiz es la carpeta de este script).
Asi funciona tanto con la disposicion de GUIA.md (todo dentro de breastdcedl/,
lo que crea `descargar_datos.py` en una maquina nueva) como con la de este
repositorio (metadata/ en la raiz, imagenes en breastdcedl/dataset/).

Requisitos: torch (con CUDA para usar la GPU), numpy, pandas, pillow y
scikit-learn. Sin scikit-learn, utils_caso devuelve AUC = nan sin avisar, asi
que el script se niega a arrancar si no esta instalado.

Ejemplos:

    # comprobacion rapida (2-3 min), antes de lanzar nada largo
    python entrenar.py --arch R_BN --fold 0 --muestra-rapida 20 --epocas 2

    # criba: las 4 arquitecturas en el fold 0, perdida ponderada
    python entrenar.py --arch R      --fold 0
    python entrenar.py --arch R_BN   --fold 0
    python entrenar.py --arch R5     --fold 0
    python entrenar.py --arch RX2    --fold 0

    # confirmacion de las 2 finalistas en los folds 1-4
    python entrenar.py --arch R_BN --fold 1
    ...

    # comparacion obligatoria de perdidas, solo sobre la ganadora
    python entrenar.py --arch R_BN --fold 0 --loss normal
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Necesario para que cuBLAS (capas lineales en GPU) sea determinista con
# torch.use_deterministic_algorithms; tiene que fijarse antes de usar CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import modelos as mo
import utils_caso as uc

RAIZ_SCRIPT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #

def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Rutas y seleccion de pacientes
# --------------------------------------------------------------------------- #

def localizar(raiz: Path, subcarpeta: str) -> Path:
    """Devuelve la carpeta que contiene `subcarpeta` (metadata o dataset):
    `raiz` o `raiz/breastdcedl`."""
    for base in (raiz, raiz / "breastdcedl"):
        if (base / subcarpeta).is_dir():
            return base
    raise FileNotFoundError(
        f"No encuentro '{subcarpeta}/' ni en {raiz} ni en {raiz / 'breastdcedl'}. "
        "Descarga los datos con `python descargar_datos.py` o indica --raiz.")


def muestra_estratificada(filas, n: int):
    """Las filas de n pacientes (mitad pCR=1, mitad pCR=0), para --muestra-rapida.
    Con pacientes de una sola clase el AUC no se puede calcular."""
    por_paciente = filas.groupby("patient_id").pCR.first()
    positivas = sorted(por_paciente[por_paciente == 1].index)[: n // 2]
    negativas = sorted(por_paciente[por_paciente == 0].index)[: n - len(positivas)]
    return filas[filas.patient_id.isin(positivas + negativas)]


# --------------------------------------------------------------------------- #
# Aumentado: al tensor de 3 canales completo, nunca canal a canal
# --------------------------------------------------------------------------- #

class VolteoHorizontal:
    """Voltea horizontalmente el tensor (3, H, W) completo, con probabilidad p.

    Se aplica a las tres fases a la vez. Voltear una fase por separado
    desalinearia PRE/EARLY/LATE y destruiria el realce, que es la señal del
    problema (GUIA.md, B5 y D2).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            x = torch.flip(x, dims=[-1])
        return x


# --------------------------------------------------------------------------- #
# Un epoca de entrenamiento / validacion
# --------------------------------------------------------------------------- #

def entrenar_una_epoca(modelo, dl, criterio, optimizador, device) -> float:
    modelo.train()
    perdida_total = 0.0
    n = 0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizador.zero_grad()          # PyTorch acumula gradientes si no se limpian
        logits = modelo(x)
        perdida = criterio(logits, y)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * x.size(0)
        n += x.size(0)
    return perdida_total / n


@torch.no_grad()
def validar(modelo, dl, filas_val, device, umbral: float, metodo: str) -> dict:
    """Evalua por paciente. `dl` debe iterar con shuffle=False sobre `filas_val`
    en el mismo orden, o las probabilidades dejan de casar con las filas
    (GUIA.md, D2 y D3)."""
    modelo.eval()
    probs = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        logits = modelo(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    return uc.evaluar_por_paciente(probs, filas_val, umbral=umbral, metodo=metodo)


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=list(mo.ARQUITECTURAS),
                   help="arquitectura candidata (ver modelos.py)")
    p.add_argument("--fold", type=int, default=0, help="fold_val para uc.particion (0-4)")
    p.add_argument("--loss", choices=["ponderada", "normal"], default="ponderada",
                   help="ponderada: BCEWithLogitsLoss(pos_weight=uc.pos_weight(...))")
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--paciencia", type=int, default=8, help="epocas sin mejorar el AUC antes de parar")
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--factor-lr", type=float, default=0.5)
    p.add_argument("--paciencia-lr", type=int, default=3)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--umbral", type=float, default=0.5, help="solo afecta a la matriz de confusion, no al AUC")
    p.add_argument("--metodo-agregacion", default="mean", choices=uc.METODOS_AGREGACION,
                   help="provisional para elegir arquitectura; la definitiva se decide aparte")
    p.add_argument("--raiz", type=Path, default=RAIZ_SCRIPT,
                   help="donde buscar metadata/ y dataset/ (tambien en su subcarpeta breastdcedl/)")
    p.add_argument("--salida", type=Path, default=RAIZ_SCRIPT / "resultados")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--muestra-rapida", type=int, default=0,
                   help="si > 0, limita a N pacientes por lado para una prueba rapida del pipeline")
    return p


def main() -> None:
    args = construir_parser().parse_args()

    try:
        import sklearn  # noqa: F401  (utils_caso lo usa para el AUC)
    except ImportError:
        raise SystemExit("Falta scikit-learn: sin el, el AUC sale nan y la comparacion "
                         "no vale. Instalalo con:  pip install scikit-learn")

    fijar_semilla(args.semilla)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("*** AVISO: no se detecta GPU CUDA. Se entrena en CPU, muchisimo mas lento. ***\n"
              "*** Si esperabas usar la GPU, revisa que torch este instalado con CUDA.     ***")

    raiz_meta = localizar(args.raiz, "metadata")
    raiz_img = localizar(args.raiz, "dataset")
    print(f"metadata/ en {raiz_meta}   |   dataset/ en {raiz_img}")

    samples = uc.cargar_samples(raiz_meta)
    entrenamiento, validacion = uc.particion(samples, fold_val=args.fold)

    if args.muestra_rapida:
        entrenamiento = muestra_estratificada(entrenamiento, args.muestra_rapida)
        validacion = muestra_estratificada(validacion, args.muestra_rapida)
        print(f"--muestra-rapida: {entrenamiento.patient_id.nunique()} pacientes train, "
              f"{validacion.patient_id.nunique()} val (mitad pCR=1)")

    print(f"fold {args.fold}: entrena {len(entrenamiento)} cortes / "
          f"{entrenamiento.patient_id.nunique()} pacientes, valida {len(validacion)} / "
          f"{validacion.patient_id.nunique()}")

    ds_tr = uc.BreastDCEDataset(entrenamiento, raiz=raiz_img, transform=VolteoHorizontal(0.5))
    ds_va = uc.BreastDCEDataset(validacion, raiz=raiz_img, transform=None)  # sin aumentado en validacion

    # persistent_workers: en Windows cada proceso lector tarda segundos en
    # arrancar; sin esto se relanzarian en cada epoca.
    opciones_loader = dict(num_workers=args.num_workers,
                           pin_memory=(device.type == "cuda"),
                           persistent_workers=args.num_workers > 0)
    generador = torch.Generator().manual_seed(args.semilla)
    dl_tr = DataLoader(ds_tr, batch_size=args.lote, shuffle=True,
                       generator=generador, **opciones_loader)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False,   # shuffle=False: obligatorio, ver GUIA.md D2
                       **opciones_loader)

    modelo = mo.crear_modelo(args.arch).to(device)
    n_params = mo.contar_parametros(modelo)

    valor_pos_weight = None
    if args.loss == "ponderada":
        valor_pos_weight = uc.pos_weight(entrenamiento)   # N0/N1 de ESTE fold, desde el CSV
        criterio = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(valor_pos_weight, dtype=torch.float32, device=device))
        print(f"pos_weight calculado de este fold de entrenamiento: {valor_pos_weight:.4f}")
    else:
        criterio = nn.BCEWithLogitsLoss()

    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="max", factor=args.factor_lr, patience=args.paciencia_lr)

    args.salida.mkdir(parents=True, exist_ok=True)
    prefijo = f"{args.arch}_fold{args.fold}_{args.loss}"
    ruta_checkpoint = args.salida / f"{prefijo}.pt"
    ruta_log = args.salida / f"{prefijo}_epocas.csv"
    ruta_resumen = args.salida / f"{prefijo}_resumen.json"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mejor_auc = -1.0
    mejor_epoca = -1
    epocas_sin_mejora = 0
    tiempos_epoca = []
    t_inicio = time.time()

    with open(ruta_log, "w", encoding="utf-8") as f:
        f.write("epoca,perdida_train,auc_val,accuracy_val,sensibilidad_val,especificidad_val,lr,tiempo_seg\n")

        for epoca in range(1, args.epocas + 1):
            t0 = time.time()
            perdida = entrenar_una_epoca(modelo, dl_tr, criterio, optimizador, device)
            resultado = validar(modelo, dl_va, validacion, device, args.umbral, args.metodo_agregacion)
            duracion = time.time() - t0
            tiempos_epoca.append(duracion)

            lr_actual = optimizador.param_groups[0]["lr"]
            auc = resultado["auc"]
            scheduler.step(auc if not np.isnan(auc) else 0.0)

            f.write(f"{epoca},{perdida:.6f},{auc:.6f},{resultado['accuracy']:.6f},"
                    f"{resultado['sensibilidad']:.6f},{resultado['especificidad']:.6f},"
                    f"{lr_actual:.8f},{duracion:.2f}\n")
            f.flush()

            print(f"epoca {epoca:3d}/{args.epocas}  perdida={perdida:.4f}  "
                  f"AUC={auc:.4f}  acc={resultado['accuracy']:.4f}  "
                  f"sens={resultado['sensibilidad']:.4f}  espec={resultado['especificidad']:.4f}  "
                  f"lr={lr_actual:.2e}  {duracion:.1f}s")

            # la primera epoca se guarda siempre: si el AUC saliera nan, al menos
            # queda un checkpoint (nan > x siempre es False)
            if mejor_epoca == -1 or auc > mejor_auc:
                mejor_auc = auc
                mejor_epoca = epoca
                epocas_sin_mejora = 0
                torch.save({
                    "arch": args.arch, "fold": args.fold, "loss": args.loss,
                    "epoca": epoca, "auc_val": auc, "umbral": args.umbral,
                    "metodo_agregacion": args.metodo_agregacion,
                    "state_dict": modelo.state_dict(),
                }, ruta_checkpoint)
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= args.paciencia:
                    print(f"Parada temprana: sin mejora en {args.paciencia} epocas "
                          f"(mejor: epoca {mejor_epoca}, AUC={mejor_auc:.4f})")
                    break

    t_total = time.time() - t_inicio

    resumen = {
        "arquitectura": args.arch,
        "parametros": n_params,
        "fold": args.fold,
        "perdida": args.loss,
        "pos_weight": valor_pos_weight,
        "pacientes_train": int(entrenamiento.patient_id.nunique()),
        "pacientes_val": int(validacion.patient_id.nunique()),
        "muestra_rapida": args.muestra_rapida,
        "semilla": args.semilla,
        "umbral": args.umbral,
        "metodo_agregacion": args.metodo_agregacion,
        "mejor_epoca": mejor_epoca,
        "auc_val": mejor_auc,
        "epocas_entrenadas": epoca,
        "tiempo_total_seg": round(t_total, 1),
        "tiempo_medio_por_epoca_seg": round(sum(tiempos_epoca) / len(tiempos_epoca), 2),
        "dispositivo": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "memoria_gpu_max_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                               if device.type == "cuda" else None,
        "version_torch": torch.__version__,
        "version_cuda": torch.version.cuda,
        "lote": args.lote, "lr": args.lr, "weight_decay": args.wd,
        "checkpoint": str(ruta_checkpoint),
        "log_por_epoca": str(ruta_log),
    }
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\nMejor epoca: {mejor_epoca}  AUC(val)={mejor_auc:.4f}")
    print(f"Guardado: {ruta_checkpoint}")
    print(f"Resumen:  {ruta_resumen}")


if __name__ == "__main__":
    main()
