"""Entrena la red base y deja el diagnostico completo: curvas, ROC y matriz de confusion.

Ajustes acordados (ver DECISIONES.md):
    red            RedBase (modelos.py): la arquitectura de referencia del profesor;
                   --batchnorm la entrena con BatchNorm tras cada convolucion;
                   --realce añade dentro de la red los canales EARLY-PRE (realce) y EARLY-LATE (lavado);
                   --filtros 8 es la red estrecha (8-16-32-64 en vez de 16-32-64-128);
                   --doble es conv+conv (dos convoluciones por bloque);
                   --normalizar reescala cada corte con su media y desviacion (dentro de la red);
                   --recorte 128 hace que la red solo mire el cuadrado central de 128x128 (dentro de la red);
                   --atencion: atencion guiada; la red pondera las zonas del mapa final y, al entrenar, esa
                   atencion se ajusta a la caja del tumor (patients.csv, cajas_tumor.py); la caja NO es entrada;
                   --aux añade la tarea auxiliar: predecir tambien HR y HER2 (patients.csv)
                   solo durante el entrenamiento; la red final sigue dando un unico logit
    perdida        --perdida normal | ponderada     (se entrenan las dos y se comparan)
    optimizador    Adam, learning rate 0,0003 (--lr)
    lote           128
    epocas         EARLY STOPPING (parada.py, decidido el 2026-10-14): se vigila el AUC por paciente
                   de validacion suavizado con una media movil de 3 epocas; se para tras 10 epocas
                   sin mejorar (--paciencia) o al llegar a 60 (--epocas). Se guarda el modelo de la
                   EPOCA ELEGIDA y se anota la diferencia AUC entrenamiento - validacion en ella
                   (si supera 0,15, el modelo se marca como sobreentrenado)
    datos          entrada tal cual (PRE, EARLY, LATE en [0, 1]);
                   --aumentado ninguno (por defecto) | geometrico (ver AumentadoAfin)
    validacion     un fold (--fold, por defecto 0); el test NO se toca
    metricas       por PACIENTE: se promedian las probabilidades de sus cortes, umbral 0,5

Cada epoca se mide en validacion (sobre todos los cortes del fold) y tambien, para
poder ver el sobreajuste, sobre un grupo fijo de pacientes de ENTRENAMIENTO
(--pacientes-vigilados) con el modelo en modo evaluacion.

Salida, en resultados/ (los nombres incluyen la perdida y el fold):
    *_epocas.csv        una fila por epoca (los datos en bruto)
    *_curvas.png        perdida y AUC por epoca, entrenamiento frente a validacion
    *_roc.png           curva ROC por paciente, en la epoca elegida
    *_confusion.png     matriz de confusion por paciente, en la epoca elegida
    *_probs_val.csv     probabilidad de cada corte de validacion en la epoca elegida
    *_probs_val_final.csv  probabilidad de cada corte en la ULTIMA epoca (prob) y la media de
                        sus probabilidades en las ultimas 10 epocas (prob_media10)
    *.pt                pesos de la epoca elegida
    *_resumen.json      configuracion, resultado, dispositivo y tiempos

Ejemplos:
    python entrenar.py --perdida normal                      # entrenamiento real
    python entrenar.py --perdida ponderada
    python entrenar.py --perdida normal --muestra-rapida 20 --epocas 3 --num-workers 0   # prueba rapida

En el PC con GPU AMD (ROCm) hay que exportar antes:
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export HSA_ENABLE_SDMA=0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

# Para que las capas lineales en GPU sean deterministas; hay que fijarlo antes de usar CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import graficas
import modelos as mo
import parada as pa
import utils_caso as uc

RAIZ = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

class AumentadoAfin:
    """Aumentado geometrico: volteo horizontal + rotacion + desplazamiento + escala.

    Cada corte de entrenamiento se transforma de una forma ligeramente distinta en cada
    epoca, para que la red no pueda memorizarlo tal cual (sobreajuste).

    - volteo horizontal con probabilidad 0,5;
    - rotacion aleatoria de hasta +-15 grados;
    - desplazamiento aleatorio de hasta +-10 % del lado (unos 25 pixeles);
    - escala aleatoria entre 0,9 y 1,1.

    Se sortea UNA sola transformacion por corte y se aplica con la misma malla de
    muestreo a los tres canales: PRE, EARLY y LATE no pueden desalinearse (desalinearlas
    destruiria el realce, que es la señal del problema). Lo que queda fuera de la imagen
    se rellena con 0, el fondo negro. Sin cambios de brillo ni de color: las fases no son
    colores. Solo se aplica al entrenamiento; validacion y test no se tocan.
    """

    def __init__(self, p_volteo: float = 0.5, grados: float = 15.0,
                 desplazamiento: float = 0.10, escala: tuple = (0.9, 1.1)):
        self.p_volteo, self.grados = p_volteo, grados
        self.desplazamiento, self.escala = desplazamiento, escala

    def __call__(self, x: torch.Tensor) -> torch.Tensor:        # x: (3, H, W)
        u = torch.rand(5).tolist()                              # cinco sorteos en [0, 1)
        volteo = -1.0 if u[0] < self.p_volteo else 1.0
        ang = (2 * u[1] - 1) * self.grados * math.pi / 180
        tx = (2 * u[2] - 1) * self.desplazamiento * 2           # el lado mide 2 en coordenadas [-1, 1]
        ty = (2 * u[3] - 1) * self.desplazamiento * 2
        s = self.escala[0] + u[4] * (self.escala[1] - self.escala[0])
        c, sn = math.cos(ang), math.sin(ang)
        theta = torch.tensor([[[volteo * c / s, -sn / s, tx],
                               [volteo * sn / s, c / s, ty]]], dtype=x.dtype)
        malla = F.affine_grid(theta, (1, *x.shape), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), malla, mode="bilinear",
                             padding_mode="zeros", align_corners=False).squeeze(0)


class AumentadoBrillo:
    """Multiplica el corte entero (las TRES fases por el MISMO factor) por un numero al azar entre 1 - variacion y
    1 + variacion. Cambia el brillo global, como pasa entre escaneres y hospitales, pero conserva la relacion entre
    fases (el realce EARLY - PRE solo se escala). Asi el brillo absoluto deja de servir como atajo. Solo entrenamiento."""

    def __init__(self, variacion: float = 0.2):
        self.variacion = variacion

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1 + (2 * torch.rand(1).item() - 1) * self.variacion)


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def localizar(subcarpeta: str, raiz: Path) -> Path:
    """Carpeta que contiene `subcarpeta` (metadata o dataset): `raiz` o `raiz/breastdcedl`
    (la disposicion cambia entre ordenadores)."""
    for base in (raiz, raiz / "breastdcedl"):
        if (base / subcarpeta).is_dir():
            return base
    raise FileNotFoundError(f"No encuentro '{subcarpeta}/' en {raiz} ni en {raiz / 'breastdcedl'}. "
                            "Descarga los datos con `python descargar_datos.py` o usa --raiz.")


def muestra_estratificada(filas, n_pacientes: int):
    """Las filas de `n_pacientes` pacientes, mitad pCR=1 y mitad pCR=0, siempre los
    mismos (orden por identificador). Con pacientes de una sola clase no hay AUC."""
    etiqueta = filas.groupby("patient_id").pCR.first()
    positivas = sorted(etiqueta[etiqueta == 1].index)[: n_pacientes // 2]
    negativas = sorted(etiqueta[etiqueta == 0].index)[: n_pacientes - len(positivas)]
    return filas[filas.patient_id.isin(positivas + negativas)]


def perdida_de_referencia(proporcion_pcr: float, peso_positivos: float) -> float:
    """La menor perdida que se puede conseguir SIN mirar la imagen: la de una red que
    diera a todos los cortes la mejor probabilidad constante. Si la perdida de validacion
    se queda cerca de este valor, la red no esta aportando nada."""
    a = peso_positivos * proporcion_pcr          # peso total de los positivos
    b = 1.0 - proporcion_pcr                     # peso total de los negativos
    p = a / (a + b)                              # probabilidad constante optima
    return -(a * math.log(p) + b * math.log(1.0 - p))


def perdida_de_referencia_validacion(proporcion_train: float, proporcion_val: float, peso_positivos: float) -> float:
    """La perdida en VALIDACION del modelo sin imagen: la probabilidad constante optima del ENTRENAMIENTO (la unica que
    puede conocer) evaluada con las etiquetas de validacion. Es la referencia justa para la curva de validacion."""
    a = peso_positivos * proporcion_train
    p = a / (a + 1.0 - proporcion_train)
    return -(peso_positivos * proporcion_val * math.log(p) + (1.0 - proporcion_val) * math.log(1.0 - p))


def perdida_ponderada(y, probs, proporcion_pcr: float) -> float:
    """Entropia cruzada media con cada clase pesada como en el entrenamiento: proporcion_pcr x (perdida media de los
    positivos) + (1 - proporcion_pcr) x (perdida media de los negativos). Para medir la perdida de los vigilados (mitad y
    mitad) en la misma escala que la de validacion."""
    y = np.asarray(y)
    p = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    pos = -np.log(p[y == 1]).mean() if (y == 1).any() else 0.0
    neg = -np.log(1 - p[y == 0]).mean() if (y == 0).any() else 0.0
    return float(proporcion_pcr * pos + (1 - proporcion_pcr) * neg)


def por_paciente(filas, probs):
    """(etiqueta, probabilidad media) de cada paciente, en el mismo orden."""
    agregado = uc.agregar_por_paciente(filas.patient_id.values, probs, "mean")
    etiqueta = filas.groupby("patient_id").pCR.first().loc[agregado.index]
    return etiqueta.values, agregado.prob.values


# --------------------------------------------------------------------------- #
# Tarea auxiliar: el subtipo del tumor (HR y HER2) como objetivo extra de entrenamiento
# --------------------------------------------------------------------------- #

AUXILIARES = ("HR", "HER2")


def cargar_etiquetas_aux(raiz_meta: Path) -> dict[str, tuple[float, float]]:
    """HR y HER2 de cada paciente (1/0), sacados de patients.csv; nan si falta el dato."""
    pacientes = uc.cargar_patients(raiz_meta)
    return {fila.pid: tuple(float(getattr(fila, c)) for c in AUXILIARES)       # NaN se conserva como nan
            for fila in pacientes.itertuples()}


class ConAuxiliares(torch.utils.data.Dataset):
    """Envuelve un BreastDCEDataset y añade, a cada corte, las etiquetas auxiliares de su
    paciente y una máscara (1 si el dato existe, 0 si falta: esas no cuentan en la pérdida).
    Solo se usa con los datos de ENTRENAMIENTO."""

    def __init__(self, base, filas, etiquetas_aux: dict):
        self.base = base
        pids = filas.reset_index(drop=True).patient_id.values      # mismo orden que el BreastDCEDataset
        aux = np.array([etiquetas_aux.get(p, (np.nan,) * len(AUXILIARES)) for p in pids], dtype=np.float32)
        self.mascara = torch.from_numpy((~np.isnan(aux)).astype(np.float32))
        self.aux = torch.from_numpy(np.nan_to_num(aux, nan=0.0))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, self.aux[i], self.mascara[i]


# --------------------------------------------------------------------------- #
# Atencion guiada: la caja del tumor como objetivo extra de entrenamiento (--atencion)
# --------------------------------------------------------------------------- #

CELDAS = 16          # el mapa final de la red (sin recorte) es de 16x16: cada celda cubre 16x16 pixeles


def objetivos_caja(filas, raiz_meta: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Para cada corte de `filas`: (objetivo (N, 16, 16), valida (N,)).

    objetivo[i, f, c] = fraccion de la celda (f, c) del mapa que cubre la caja del tumor de su paciente, con la
    caja de patients.csv llevada al PNG (cajas_tumor.caja_en_png: ventana del MinCrop y giro de 90 grados).
    valida = 0 si la paciente no tiene caja utilizable (imagen original menor de 256): ese corte no cuenta en la
    perdida de atencion. Las cajas solo se usan para ENTRENAR; la red no las recibe como entrada."""
    from cajas_tumor import caja_en_png
    pacientes = uc.cargar_patients(raiz_meta).set_index("pid")
    bordes = np.arange(CELDAS + 1) * (256 // CELDAS)                 # 0, 16, ..., 256
    por_paciente = {}
    for pid in filas.patient_id.unique():
        caja = caja_en_png(pacientes.loc[pid])
        if caja is None:
            por_paciente[pid] = (np.zeros((CELDAS, CELDAS), np.float32), 0.0)
            continue
        a, b, c, e = caja                                            # filas a..b y columnas c..e, inclusive
        sol_f = np.clip(np.minimum(bordes[1:], b + 1) - np.maximum(bordes[:-1], a), 0, None) / (256 // CELDAS)
        sol_c = np.clip(np.minimum(bordes[1:], e + 1) - np.maximum(bordes[:-1], c), 0, None) / (256 // CELDAS)
        por_paciente[pid] = (np.outer(sol_f, sol_c).astype(np.float32), 1.0)
    pids = filas.reset_index(drop=True).patient_id.values
    objetivo = torch.from_numpy(np.stack([por_paciente[p][0] for p in pids]))
    valida = torch.tensor([por_paciente[p][1] for p in pids], dtype=torch.float32)
    return objetivo, valida


class ConCaja(torch.utils.data.Dataset):
    """Envuelve un BreastDCEDataset y añade, a cada corte, el objetivo de atencion (la caja del tumor en la
    rejilla de 16x16) y si es valido. Solo para ENTRENAR."""

    def __init__(self, base, objetivo: torch.Tensor, valida: torch.Tensor):
        self.base, self.objetivo, self.valida = base, objetivo, valida

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, self.objetivo[i], self.valida[i]


def perdida_atencion(puntos: torch.Tensor, objetivo: torch.Tensor, valida: torch.Tensor) -> torch.Tensor:
    """Entropia cruzada binaria entre las puntuaciones de atencion (logits por celda) y la caja del tumor,
    media de las celdas, solo en los cortes con caja valida."""
    por_corte = F.binary_cross_entropy_with_logits(puntos, objetivo, reduction="none").mean(dim=(1, 2))
    return (por_corte * valida).sum() / valida.sum().clamp(min=1.0)


def auc_aux_por_paciente(filas, probs_aux, etiquetas_aux: dict) -> list[float]:
    """AUC por paciente de cada etiqueta auxiliar (media de las probabilidades de sus cortes)."""
    from sklearn.metrics import roc_auc_score
    pids = filas.patient_id.values
    resultados = []
    for j in range(len(AUXILIARES)):
        media = {}
        for p, prob in zip(pids, probs_aux[:, j]):
            media.setdefault(p, []).append(prob)
        y, s = [], []
        for p, ps in media.items():
            etiqueta = etiquetas_aux.get(p, (np.nan,) * len(AUXILIARES))[j]
            if not np.isnan(etiqueta):
                y.append(etiqueta)
                s.append(float(np.mean(ps)))
        resultados.append(float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan"))
    return resultados


# --------------------------------------------------------------------------- #
# Una epoca de entrenamiento y la medicion
# --------------------------------------------------------------------------- #

def entrenar_una_epoca(modelo, dl, criterio, optimizador, device, peso_aux: float = 0.0, suavizado: float = 0.0):
    """Pasa una vez por todos los datos de entrenamiento, lote a lote (con el dropout activo,
    tal como se entrena). Devuelve (perdida media de pCR, perdida media auxiliar o None).

    Con tarea auxiliar, la perdida que se minimiza es:
        perdida_pCR + peso_aux * perdida_auxiliar   (media de HR y HER2, sin los datos que faltan)
    Con atencion guiada (el modelo tiene `atencion`), la auxiliar es la de atencion frente a la caja del tumor.
    Con `suavizado` > 0 (label smoothing) la red aprende con objetivos suavizados (0 -> s/2, 1 -> 1 - s/2) para que no
    se vuelva demasiado segura; la perdida que se DEVUELVE se calcula con las etiquetas reales (comparable con la base
    y con la referencia sin imagen).
    La perdida de pCR que se devuelve es solo la de pCR, comparable con la de la base."""
    modelo.train()
    suma, suma_aux, n = 0.0, 0.0, 0
    for lote in dl:
        x, y_real = lote[0].to(device, non_blocking=True), lote[1].to(device, non_blocking=True)
        y = y_real * (1 - suavizado) + suavizado / 2 if suavizado else y_real
        optimizador.zero_grad()              # PyTorch acumula gradientes si no se limpian
        if modelo.atencion:                  # con la caja del tumor como objetivo de la atencion
            objetivo, valida = lote[2].to(device, non_blocking=True), lote[3].to(device, non_blocking=True)
            logit, puntos = modelo.forward_atencion(x)
            perdida_pcr = criterio(logit, y)
            perdida_aux = perdida_atencion(puntos, objetivo, valida)
            perdida = perdida_pcr + peso_aux * perdida_aux
            suma_aux += perdida_aux.item() * x.size(0)
        elif len(lote) == 4:                 # con etiquetas auxiliares
            aux, mascara = lote[2].to(device, non_blocking=True), lote[3].to(device, non_blocking=True)
            logit, logit_aux = modelo.forward_multitarea(x)
            perdida_pcr = criterio(logit, y)
            por_elemento = F.binary_cross_entropy_with_logits(logit_aux, aux, reduction="none")
            perdida_aux = (por_elemento * mascara).sum() / mascara.sum().clamp(min=1.0)
            perdida = perdida_pcr + peso_aux * perdida_aux
            suma_aux += perdida_aux.item() * x.size(0)
        else:
            logit = modelo(x)
            perdida_pcr = perdida = criterio(logit, y)
        perdida.backward()
        optimizador.step()
        if suavizado:                        # se informa con las etiquetas reales, no con las suavizadas
            with torch.no_grad():
                perdida_pcr = criterio(logit, y_real)
        suma += perdida_pcr.item() * x.size(0)
        n += x.size(0)
    return suma / n, (suma_aux / n if (modelo.n_auxiliares or modelo.atencion) else None)


@torch.no_grad()
def atencion_dentro(modelo, dl, objetivo: torch.Tensor, valida: torch.Tensor, device) -> float:
    """Media, en los cortes con caja, de la fraccion de la atencion (pesos softmax, suman 1) que cae en celdas
    tocadas por la caja del tumor. Con atencion uniforme saldria la fraccion de celdas de la caja (~0,1)."""
    modelo.eval()
    fracciones = []
    for x, _ in dl:
        _, puntos = modelo.forward_atencion(x.to(device, non_blocking=True))
        fracciones.append(torch.softmax(puntos.flatten(1), dim=1).cpu())
    pesos = torch.cat(fracciones)                                         # (N, 256)
    dentro = (pesos * (objetivo.flatten(1) > 0)).sum(dim=1)
    return float((dentro * valida).sum() / valida.sum().clamp(min=1.0))


@torch.no_grad()
def medir(modelo, dl, filas, criterio, device, umbral: float):
    """Modo evaluacion (sin dropout) sobre `filas`. `dl` debe iterar con shuffle=False
    sobre esas mismas filas, o las probabilidades dejan de casar con ellas.
    Devuelve (perdida por corte, metricas por paciente, probabilidad de cada corte,
    probabilidades auxiliares de cada corte o None)."""
    modelo.eval()
    salidas = [modelo.forward_multitarea(x.to(device, non_blocking=True)) for x, _ in dl]
    logits = torch.cat([s[0] for s in salidas])
    probs_aux = (torch.sigmoid(torch.cat([s[1] for s in salidas])).cpu().numpy()
                 if salidas[0][1] is not None else None)
    y = torch.as_tensor(np.array(filas.pCR.values), dtype=torch.float32, device=device)   # np.array copia: sin aviso
    perdida = criterio(logits, y).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    return perdida, uc.evaluar_por_paciente(probs, filas, umbral=umbral, metodo="mean"), probs, probs_aux


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perdida", choices=["normal", "ponderada"], default="normal",
                   help="ponderada: BCEWithLogitsLoss(pos_weight = N0/N1 de las filas de entrenamiento)")
    p.add_argument("--fold", type=int, default=0, help="fold de validacion (0-4)")
    p.add_argument("--epocas", type=int, default=pa.EPOCAS_MAX, help="maximo de epocas")
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--pacientes-vigilados", type=int, default=200,
                   help="pacientes de entrenamiento sobre los que tambien se mide cada epoca")
    p.add_argument("--raiz", type=Path, default=RAIZ,
                   help="donde buscar metadata/ y dataset/ (tambien en su subcarpeta breastdcedl/)")
    p.add_argument("--salida", type=Path, default=RAIZ / "resultados")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--muestra-rapida", type=int, default=0,
                   help="si > 0, usa solo N pacientes por lado: prueba rapida del flujo completo")
    p.add_argument("--suavizado", type=float, default=0.0,
                   help="label smoothing: objetivos 0 -> s/2 y 1 -> 1 - s/2 al entrenar (p. ej. 0.1)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="si > 0, AdamW con este weight decay (penaliza los pesos grandes); 0 = Adam como siempre")
    p.add_argument("--aumentado", choices=["ninguno", "geometrico", "brillo"], default="ninguno",
                   help="geometrico: volteo + rotacion + desplazamiento + escala, solo en entrenamiento")
    p.add_argument("--batchnorm", action="store_true",
                   help="BatchNorm tras cada convolucion (Conv -> BN -> ReLU -> pool)")
    p.add_argument("--realce", action="store_true",
                   help="añadir dentro de la red los canales EARLY-PRE (realce) y EARLY-LATE (lavado)")
    p.add_argument("--filtros", type=int, default=16,
                   help="filtros del primer bloque (cada bloque dobla): 16 = red del profesor, 8 = red estrecha")
    p.add_argument("--doble", action="store_true",
                   help="conv+conv: dos convoluciones 3x3 por bloque (idea del profesor)")
    p.add_argument("--normalizar", action="store_true",
                   help="normalizar cada corte con su propia media y desviacion (las mismas para las 3 fases)")
    p.add_argument("--recorte", type=int, default=0,
                   help="si > 0, la red solo mira el cuadrado central de ese lado (p. ej. 128)")
    p.add_argument("--aux", action="store_true",
                   help="tarea auxiliar: predecir tambien HR y HER2 (de patients.csv) durante el entrenamiento")
    p.add_argument("--peso-aux", type=float, default=0.5,
                   help="peso de la perdida auxiliar frente a la de pCR (solo con --aux)")
    p.add_argument("--atencion", action="store_true",
                   help="atencion guiada: media ponderada del mapa final, entrenada para mirar la caja del tumor "
                        "(cajas de patients.csv solo como objetivo de entrenamiento; imagen entera, sin recorte)")
    p.add_argument("--peso-atencion", type=float, default=1.0,
                   help="peso de la perdida de atencion frente a la de pCR (solo con --atencion)")
    p.add_argument("--paciencia", type=int, default=pa.PACIENCIA,
                   help="early stopping: epocas sin mejorar la media movil del AUC de validacion (0 = sin parada)")
    p.add_argument("--sobrescribir", action="store_true",
                   help="permitir sobrescribir resultados existentes con el mismo nombre")
    return p


def main() -> None:
    args = construir_parser().parse_args()
    if args.atencion and (args.recorte or args.aux):
        raise SystemExit("--atencion trabaja con la imagen entera (mapa de 16x16) y no se combina con --recorte ni --aux")

    try:
        import sklearn  # noqa: F401   (utils_caso lo usa para el AUC)
    except ImportError:
        raise SystemExit("Falta scikit-learn: sin el, el AUC sale nan. Instalalo con: pip install scikit-learn")

    # El nombre incluye todo lo que se aparte de la base, para que configuraciones distintas
    # nunca se sobrescriban entre si y comparar.py las distinga:
    #   base_normal_fold0, base_normal_aug_fold0, bn_normal_fold0, ...
    red = "bn" if args.batchnorm else "base"
    prefijo = (f"{red}_{args.perdida}" + ("_aug" if args.aumentado == "geometrico" else "") +
               ("_brillo" if args.aumentado == "brillo" else "") +
               (f"_ls{args.suavizado:g}" if args.suavizado else "") +
               (f"_wd{args.weight_decay:g}" if args.weight_decay else "") +
               (f"_f{args.filtros}" if args.filtros != 16 else "") +
               ("_doble" if args.doble else "") + ("_norm" if args.normalizar else "") +
               (f"_c{args.recorte}" if args.recorte else "") +
               ("_realce" if args.realce else "") + ("_aux" if args.aux else "") +
               ("_aten" + (f"{args.peso_atencion:g}" if args.peso_atencion != 1.0 else "") if args.atencion else "") +
               (f"_lr{args.lr:g}" if args.lr != 3e-4 else "") +
               (f"_s{args.semilla}" if args.semilla != 42 else "") +
               f"_fold{args.fold}" + (f"_prueba{args.muestra_rapida}" if args.muestra_rapida else ""))
    args.salida.mkdir(parents=True, exist_ok=True)
    ruta = {k: args.salida / f"{prefijo}{s}" for k, s in
            dict(csv="_epocas.csv", curvas="_curvas.png", roc="_roc.png", conf="_confusion.png",
                 probs="_probs_val.csv", probs_final="_probs_val_final.csv",
                 pesos=".pt", resumen="_resumen.json").items()}
    if ruta["resumen"].exists() and not args.sobrescribir:
        raise SystemExit(f"Ya existe {ruta['resumen']}.\nNo lo sobrescribo para no perder un resultado: "
                         "usa --sobrescribir si es lo que quieres.")

    fijar_semilla(args.semilla)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("*** AVISO: no se detecta GPU. Se entrena en CPU, mucho mas lento. ***")

    # --- datos: entrenamiento y validacion por PACIENTE (columna `fold`), test intacto ---
    raiz_meta = localizar("metadata", args.raiz)
    samples = uc.cargar_samples(raiz_meta)
    raiz_img = localizar("dataset", args.raiz)
    entrenamiento, validacion = uc.particion(samples, fold_val=args.fold)
    if args.muestra_rapida:
        entrenamiento = muestra_estratificada(entrenamiento, args.muestra_rapida)
        validacion = muestra_estratificada(validacion, args.muestra_rapida)
    vigilados = muestra_estratificada(entrenamiento, args.pacientes_vigilados)
    print(f"fold {args.fold}: entrena {len(entrenamiento)} cortes / {entrenamiento.patient_id.nunique()} pacientes; "
          f"valida {len(validacion)} / {validacion.patient_id.nunique()}; "
          f"vigilados de entrenamiento: {vigilados.patient_id.nunique()} pacientes")

    opciones = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                    persistent_workers=args.num_workers > 0)
    transformacion = {"geometrico": AumentadoAfin(), "brillo": AumentadoBrillo()}.get(args.aumentado)   # solo entrenamiento
    ds_train = uc.BreastDCEDataset(entrenamiento, raiz=raiz_img, transform=transformacion)
    etiquetas_aux = None
    if args.aux:   # HR y HER2 de patients.csv: solo como OBJETIVO de entrenamiento, nunca como entrada
        etiquetas_aux = cargar_etiquetas_aux(raiz_meta)
        ds_train = ConAuxiliares(ds_train, entrenamiento, etiquetas_aux)
        print(f"tarea auxiliar: {' y '.join(AUXILIARES)} (peso {args.peso_aux:g}); con los dos datos en "
              f"{int(ds_train.mascara.min(dim=1).values.sum())} de {len(ds_train)} cortes de entrenamiento")
    cajas_val = None
    if args.atencion:   # cajas del tumor: solo como OBJETIVO de la atencion al entrenar, nunca como entrada
        objetivo, valida = objetivos_caja(entrenamiento, raiz_meta)
        ds_train = ConCaja(ds_train, objetivo, valida)
        cajas_val = objetivos_caja(validacion, raiz_meta)     # solo para MEDIR si la atencion cae en el tumor
        print(f"atencion guiada (peso {args.peso_atencion:g}): caja del tumor en {int(valida.sum())} de "
              f"{len(valida)} cortes de entrenamiento; la caja ocupa de media el "
              f"{float((objetivo[valida > 0] > 0).float().mean()):.0%} de las celdas del mapa")
    dl_train = DataLoader(ds_train, batch_size=args.lote, shuffle=True,
                          generator=torch.Generator().manual_seed(args.semilla), **opciones)
    # shuffle=False en validacion y vigilados: obligatorio para que las probabilidades casen con las filas
    dl_val = DataLoader(uc.BreastDCEDataset(validacion, raiz=raiz_img), batch_size=args.lote, shuffle=False, **opciones)
    dl_vig = DataLoader(uc.BreastDCEDataset(vigilados, raiz=raiz_img), batch_size=args.lote, shuffle=False, **opciones)

    # --- modelo, perdida y optimizador ---
    modelo = mo.RedBase(batchnorm=args.batchnorm, n_auxiliares=len(AUXILIARES) if args.aux else 0,
                        realce=args.realce, filtros=args.filtros, doble=args.doble,
                        normalizar=args.normalizar, recorte=args.recorte, atencion=args.atencion).to(device)
    print(f"red: {'RedBase con BatchNorm' if args.batchnorm else 'RedBase (la de la diapositiva)'}"
          f"{f' estrecha ({args.filtros}-{2 * args.filtros}-{4 * args.filtros}-{8 * args.filtros})' if args.filtros != 16 else ''}"
          f"{' conv+conv' if args.doble else ''}{' + normalizacion por corte' if args.normalizar else ''}"
          f"{f' + recorte central {args.recorte}x{args.recorte}' if args.recorte else ''}"
          f"{' + canales de realce y lavado' if args.realce else ''}"
          f"{' + atencion guiada por la caja del tumor' if args.atencion else ''}, "
          f"{mo.contar_parametros(modelo):,} parametros")
    if args.perdida == "ponderada":
        peso = uc.pos_weight(entrenamiento)          # N0/N1 de ESTAS filas de entrenamiento, calculado del CSV
        criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso, dtype=torch.float32, device=device))
    else:
        peso = 1.0
        criterio = nn.BCEWithLogitsLoss()
    proporcion = float((entrenamiento.pCR == 1).mean())
    referencia = perdida_de_referencia(proporcion, peso)
    # validacion: la constante del entrenamiento evaluada con las etiquetas de validacion (corregido el 2026-10-22)
    referencia_val = perdida_de_referencia_validacion(proporcion, float((validacion.pCR == 1).mean()), peso)
    print(f"perdida {args.perdida}" + (f" (pos_weight = {peso:.4f})" if args.perdida == "ponderada" else "") +
          f"; proporcion de cortes con pCR=1: {proporcion:.3f}; "
          f"mejor perdida sin mirar la imagen: {referencia:.4f} (entrenamiento), {referencia_val:.4f} (validacion)")
    optimizador = (torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                   if args.weight_decay else torch.optim.Adam(modelo.parameters(), lr=args.lr))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- bucle de entrenamiento con EARLY STOPPING (parada.py): se vigila el AUC de validacion suavizado
    #     (media movil de 3 epocas); se para tras `paciencia` epocas sin mejorar o al llegar a --epocas ---
    auc_historial, auc_train_historial = [], []      # AUC de validacion y de entrenamiento de cada epoca
    recientes = deque(maxlen=pa.VENTANA // 2 + 1)    # (indice, pesos, probs) de las ultimas epocas: la elegida
                                                     # puede ser la anterior (su media movil necesita la siguiente)
    ult_probs = deque(maxlen=10)                     # probabilidades de las ultimas 10 epocas
    elegida, elegida_probs, motivo = None, None, "maximo"
    paciencia = args.paciencia if args.paciencia > 0 else 10**9   # 0 = sin parada: todas las epocas

    def guardar(indice: int) -> None:
        """Guarda los pesos y las probabilidades de validacion de la epoca `indice` (empieza en 0)."""
        nonlocal elegida, elegida_probs
        _, pesos, probs = next(r for r in recientes if r[0] == indice)
        elegida, elegida_probs = indice, probs
        torch.save({"red": "RedBase", "batchnorm": args.batchnorm, "realce": args.realce, "aux": args.aux,
                    "filtros": args.filtros, "doble": args.doble, "normalizar": args.normalizar,
                    "recorte": args.recorte, "atencion": args.atencion,
                    "perdida": args.perdida, "fold": args.fold, "epoca": indice + 1,
                    "auc_val": auc_historial[indice], "lr": args.lr, "lote": args.lote,
                    "state_dict": pesos}, ruta["pesos"])
        with open(ruta["probs"], "w", encoding="utf-8") as g:
            g.write("sample_id,patient_id,pCR,prob\n")
            for fila, q in zip(validacion.itertuples(), probs):
                g.write(f"{fila.sample_id},{fila.patient_id},{fila.pCR},{q:.6f}\n")

    tiempos, t_inicio = [], time.time()
    with open(ruta["csv"], "w", encoding="utf-8") as f:
        f.write("epoca,perdida_train,perdida_val,auc_train,auc_val,accuracy_val,sensibilidad_val,"
                "especificidad_val,lr,tiempo_seg,perdida_aux_train,auc_val_hr,auc_val_her2,atencion_dentro_val,"
                "perdida_train_comparable\n")
        atencion_historial = []
        for epoca in range(1, args.epocas + 1):
            t0 = time.time()
            perdida_train, perdida_aux = entrenar_una_epoca(modelo, dl_train, criterio, optimizador, device,
                                                            args.peso_atencion if args.atencion else args.peso_aux,
                                                            args.suavizado)
            perdida_val, res_val, probs_val, probs_aux = medir(modelo, dl_val, validacion, criterio, device, args.umbral)
            auc_aux = (auc_aux_por_paciente(validacion, probs_aux, etiquetas_aux)
                       if probs_aux is not None else [float("nan")] * len(AUXILIARES))
            # perdida de los pacientes vigilados de ENTRENAMIENTO medida como la de validacion (modo evaluacion, sin
            # dropout ni aumentado ni suavizado). Los vigilados son mitad pCR y mitad no pCR, asi que se pondera cada
            # clase para que pese como en el entrenamiento (29 % de pCR): asi es comparable con la de validacion.
            _, res_vig, probs_vig, _ = medir(modelo, dl_vig, vigilados, criterio, device, args.umbral)
            perdida_train_eval = perdida_ponderada(vigilados.pCR.to_numpy(), probs_vig, proporcion)
            aten_val = atencion_dentro(modelo, dl_val, *cajas_val, device) if args.atencion else float("nan")
            atencion_historial.append(aten_val)
            auc_historial.append(res_val["auc"])
            auc_train_historial.append(res_vig["auc"])
            ult_probs.append(probs_val)
            recientes.append((epoca - 1, {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
                              probs_val))
            dur = time.time() - t0
            tiempos.append(dur)
            f.write(f"{epoca},{perdida_train:.6f},{perdida_val:.6f},{res_vig['auc']:.6f},{res_val['auc']:.6f},"
                    f"{res_val['accuracy']:.6f},{res_val['sensibilidad']:.6f},{res_val['especificidad']:.6f},"
                    f"{args.lr:.8f},{dur:.2f},"
                    f"{'' if perdida_aux is None else f'{perdida_aux:.6f}'},{auc_aux[0]:.6f},{auc_aux[1]:.6f},"
                    f"{aten_val:.6f},{perdida_train_eval:.6f}\n")
            f.flush()

            indice, parar = pa.elegir_epoca(auc_historial, paciencia)
            if indice is not None and indice != elegida:
                guardar(indice)
            marca = (f"  [elegida {elegida + 1}: media movil {pa.suavizar(auc_historial)[elegida]:.3f}]"
                     if elegida is not None else "")
            print(f"epoca {epoca:3d}/{args.epocas}  perdida train {perdida_train:.4f} (medida como val: "
                  f"{perdida_train_eval:.4f}) val {perdida_val:.4f}  "
                  f"AUC train {res_vig['auc']:.3f} val {res_val['auc']:.3f}  "
                  f"(val: acc {res_val['accuracy']:.3f} sens {res_val['sensibilidad']:.3f} "
                  f"espec {res_val['especificidad']:.3f})"
                  + (f"  aux val HR {auc_aux[0]:.3f} HER2 {auc_aux[1]:.3f}" if args.aux else "")
                  + (f"  atencion en el tumor (val) {aten_val:.0%}" if args.atencion else "")
                  + f"  {dur:.1f}s{marca}", flush=True)
            if parar:
                motivo = "paciencia"
                print(f"PARADA: {args.paciencia} epocas seguidas sin mejorar la media movil del AUC de validacion.")
                break
    if motivo == "maximo":     # sin parada por paciencia: al terminar ya cuenta tambien la ultima epoca
        indice, _ = pa.elegir_epoca(auc_historial, paciencia, final=True)
        if indice != elegida:
            guardar(indice)
    # --- epoca elegida: AUC de validacion y de entrenamiento, y su diferencia (control del sobreajuste) ---
    epoca_elegida = elegida + 1
    auc_val_elegida, auc_train_elegida = auc_historial[elegida], auc_train_historial[elegida]
    auc_val_suav = float(pa.suavizar(auc_historial)[elegida])
    diferencia = auc_train_elegida - auc_val_elegida
    sobreentrenado = bool(diferencia > pa.DIFERENCIA_MAX)
    sin_aprendizaje = bool(auc_train_elegida < pa.AUC_TRAIN_MIN)

    # --- graficas del diagnostico, en la epoca elegida ---
    titulo = (f"Red {'con BatchNorm' if args.batchnorm else 'base'} · pérdida {args.perdida}"
              + (" · aumentado" if args.aumentado == "geometrico" else "")
              + (" · brillo aleatorio" if args.aumentado == "brillo" else "")
              + (f" · suavizado {args.suavizado:g}" if args.suavizado else "")
              + (f" · weight decay {args.weight_decay:g}" if args.weight_decay else "")
              + (f" · filtros {args.filtros}" if args.filtros != 16 else "")
              + (" · conv+conv" if args.doble else "") + (" · normalización por corte" if args.normalizar else "")
              + (f" · recorte central {args.recorte}" if args.recorte else "")
              + (" · atención guiada" if args.atencion else "")
              + (" · realce y lavado" if args.realce else "")
              + (f" · lr {args.lr:g}" if args.lr != 3e-4 else "")
              + (" · tarea auxiliar HR/HER2" if args.aux else "") + f" · fold {args.fold}")
    graficas.dibujar_curvas_entrenamiento(ruta["csv"], ruta["curvas"], titulo, referencia, epoca_elegida, referencia_val)

    # probabilidades de la ULTIMA epoca y media de las ultimas (hasta 10) epocas, por si se quieren comparar
    probs_media_ult = np.mean(np.stack(list(ult_probs)), axis=0)
    with open(ruta["probs_final"], "w", encoding="utf-8") as g:
        g.write("sample_id,patient_id,pCR,prob,prob_media10\n")
        for fila, q, qm in zip(validacion.itertuples(), probs_val, probs_media_ult):
            g.write(f"{fila.sample_id},{fila.patient_id},{fila.pCR},{q:.6f},{qm:.6f}\n")
    y_pac, p_pac = por_paciente(validacion, elegida_probs)
    graficas.dibujar_roc(y_pac, p_pac, ruta["roc"], f"ROC por paciente · época elegida {epoca_elegida}", args.umbral)
    res_elegida = uc.evaluar_por_paciente(elegida_probs, validacion, umbral=args.umbral, metodo="mean")
    mc = res_elegida["matriz_confusion"]
    graficas.dibujar_matriz_confusion(mc["VP"], mc["VN"], mc["FP"], mc["FN"], ruta["conf"],
                                      f"Matriz de confusión por paciente · época {epoca_elegida} · umbral {args.umbral:g}")
    t_total = time.time() - t_inicio

    resumen = {
        "red": "RedBase", "batchnorm": args.batchnorm, "realce": args.realce, "filtros": args.filtros,
        "doble": args.doble, "normalizar": args.normalizar, "recorte": args.recorte,
        "atencion": args.atencion, "peso_atencion": args.peso_atencion if args.atencion else None,
        "atencion_dentro_val_elegida": atencion_historial[elegida] if args.atencion else None,
        "parametros": mo.contar_parametros(modelo), "perdida": args.perdida,
        "aumentado": args.aumentado, "suavizado": args.suavizado, "weight_decay": args.weight_decay,
        "optimizador_clase": "AdamW" if args.weight_decay else "Adam",
        "aux": args.aux, "peso_aux": args.peso_aux if args.aux else None,
        "auc_val_hr_ultima": auc_aux[0] if args.aux else None,
        "auc_val_her2_ultima": auc_aux[1] if args.aux else None,
        "pos_weight": peso if args.perdida == "ponderada" else None, "fold": args.fold,
        "optimizador": "Adam", "lr": args.lr, "lote": args.lote, "epocas": args.epocas, "semilla": args.semilla,
        "umbral": args.umbral, "agregacion": "mean", "muestra_rapida": args.muestra_rapida,
        "pacientes_train": int(entrenamiento.patient_id.nunique()), "pacientes_val": int(validacion.patient_id.nunique()),
        "perdida_de_referencia": referencia, "perdida_de_referencia_val": referencia_val,
        "parada": {"criterio": "AUC de validacion, media movil centrada", "ventana": pa.VENTANA,
                   "paciencia": args.paciencia, "epocas_max": args.epocas, "motivo": motivo},
        "epocas_entrenadas": len(auc_historial), "epoca_elegida": epoca_elegida,
        "auc_val": auc_val_elegida, "auc_val_suavizado": auc_val_suav, "auc_train": auc_train_elegida,
        "diferencia_train_val": diferencia, "sobreentrenado": sobreentrenado, "sin_aprendizaje": sin_aprendizaje,
        "diferencia_max": pa.DIFERENCIA_MAX, "auc_val_ultima": auc_historial[-1],
        "matriz_confusion_val": mc, "sensibilidad_val": res_elegida["sensibilidad"],
        "especificidad_val": res_elegida["especificidad"], "accuracy_val": res_elegida["accuracy"],
        "tiempo_total_seg": round(t_total, 1), "tiempo_medio_por_epoca_seg": round(sum(tiempos) / len(tiempos), 2),
        "dispositivo": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "memoria_gpu_max_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None,
        "version_torch": torch.__version__, "version_hip": getattr(torch.version, "hip", None),
    }
    with open(ruta["resumen"], "w", encoding="utf-8") as g:
        json.dump(resumen, g, indent=2, ensure_ascii=False)

    print(f"\nEPOCA ELEGIDA: {epoca_elegida} de {len(auc_historial)} entrenadas "
          f"({'parada por paciencia' if motivo == 'paciencia' else 'se llego al maximo de epocas'})")
    print(f"AUC por paciente en la epoca elegida: validacion {auc_val_elegida:.4f} "
          f"(media movil {auc_val_suav:.4f}), entrenamiento {auc_train_elegida:.4f}")
    print(f"Diferencia entrenamiento - validacion = {diferencia:.4f} -> "
          + (f"SOBREENTRENADO (supera {pa.DIFERENCIA_MAX:g})" if sobreentrenado else f"OK (no supera {pa.DIFERENCIA_MAX:g})"))
    if sin_aprendizaje:
        print(f"AVISO: SIN APRENDIZAJE: en la epoca elegida el AUC de entrenamiento es {auc_train_elegida:.3f} "
              f"(< {pa.AUC_TRAIN_MIN:g}): la red aun no distingue ni a las pacientes con las que entrena.")
    if args.atencion:
        print(f"Atencion en el tumor (validacion, epoca elegida): {atencion_historial[elegida]:.1%} de la atencion "
              f"cae en las celdas de la caja")
    print(f"Matriz de confusion (umbral {args.umbral:g}): {mc}")
    if args.aux:
        print(f"Tarea auxiliar, AUC por paciente en validacion (ultima epoca): "
              f"HR {auc_aux[0]:.4f}  HER2 {auc_aux[1]:.4f}")
    print(f"Resultados en {args.salida}  (prefijo {prefijo})")


if __name__ == "__main__":
    main()
