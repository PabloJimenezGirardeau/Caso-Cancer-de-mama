"""Entrena la red base y deja el diagnostico completo: curvas, ROC y matriz de confusion.

Ajustes acordados (ver DECISIONES.md):
    red            RedBase (modelos.py): la arquitectura de referencia del profesor
    perdida        --perdida normal | ponderada     (se entrenan las dos y se comparan)
    optimizador    Adam, learning rate 0,0003
    lote           128
    epocas         40 fijas, SIN parada temprana (se ve la curva entera);
                   se guarda el modelo de la epoca con mejor AUC por paciente en validacion
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
    *_roc.png           curva ROC por paciente, en la mejor epoca
    *_confusion.png     matriz de confusion por paciente, en la mejor epoca
    *_probs_val.csv     probabilidad de cada corte de validacion en la mejor epoca
    *.pt                pesos de la mejor epoca
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


def por_paciente(filas, probs):
    """(etiqueta, probabilidad media) de cada paciente, en el mismo orden."""
    agregado = uc.agregar_por_paciente(filas.patient_id.values, probs, "mean")
    etiqueta = filas.groupby("patient_id").pCR.first().loc[agregado.index]
    return etiqueta.values, agregado.prob.values


# --------------------------------------------------------------------------- #
# Una epoca de entrenamiento y la medicion
# --------------------------------------------------------------------------- #

def entrenar_una_epoca(modelo, dl, criterio, optimizador, device) -> float:
    """Pasa una vez por todos los datos de entrenamiento, lote a lote. Devuelve la
    perdida media de la epoca (con el dropout activo, tal como se entrena)."""
    modelo.train()
    suma, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizador.zero_grad()              # PyTorch acumula gradientes si no se limpian
        perdida = criterio(modelo(x), y)
        perdida.backward()
        optimizador.step()
        suma += perdida.item() * x.size(0)
        n += x.size(0)
    return suma / n


@torch.no_grad()
def medir(modelo, dl, filas, criterio, device, umbral: float):
    """Modo evaluacion (sin dropout) sobre `filas`. `dl` debe iterar con shuffle=False
    sobre esas mismas filas, o las probabilidades dejan de casar con ellas.
    Devuelve (perdida por corte, metricas por paciente, probabilidad de cada corte)."""
    modelo.eval()
    logits = torch.cat([modelo(x.to(device, non_blocking=True)) for x, _ in dl])
    y = torch.as_tensor(np.array(filas.pCR.values), dtype=torch.float32, device=device)   # np.array copia: sin aviso
    perdida = criterio(logits, y).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    return perdida, uc.evaluar_por_paciente(probs, filas, umbral=umbral, metodo="mean"), probs


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perdida", choices=["normal", "ponderada"], default="normal",
                   help="ponderada: BCEWithLogitsLoss(pos_weight = N0/N1 de las filas de entrenamiento)")
    p.add_argument("--fold", type=int, default=0, help="fold de validacion (0-4)")
    p.add_argument("--epocas", type=int, default=40)
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
    p.add_argument("--aumentado", choices=["ninguno", "geometrico"], default="ninguno",
                   help="geometrico: volteo + rotacion + desplazamiento + escala, solo en entrenamiento")
    p.add_argument("--sobrescribir", action="store_true",
                   help="permitir sobrescribir resultados existentes con el mismo nombre")
    return p


def main() -> None:
    args = construir_parser().parse_args()

    try:
        import sklearn  # noqa: F401   (utils_caso lo usa para el AUC)
    except ImportError:
        raise SystemExit("Falta scikit-learn: sin el, el AUC sale nan. Instalalo con: pip install scikit-learn")

    # El nombre incluye todo lo que se aparte de la base, para que configuraciones distintas
    # nunca se sobrescriban entre si y comparar.py las distinga: base_normal_fold0, base_normal_aug_fold0, ...
    prefijo = (f"base_{args.perdida}" + ("_aug" if args.aumentado == "geometrico" else "") +
               f"_fold{args.fold}" + (f"_prueba{args.muestra_rapida}" if args.muestra_rapida else ""))
    args.salida.mkdir(parents=True, exist_ok=True)
    ruta = {k: args.salida / f"{prefijo}{s}" for k, s in
            dict(csv="_epocas.csv", curvas="_curvas.png", roc="_roc.png", conf="_confusion.png",
                 probs="_probs_val.csv", pesos=".pt", resumen="_resumen.json").items()}
    if ruta["resumen"].exists() and not args.sobrescribir:
        raise SystemExit(f"Ya existe {ruta['resumen']}.\nNo lo sobrescribo para no perder un resultado: "
                         "usa --sobrescribir si es lo que quieres.")

    fijar_semilla(args.semilla)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("*** AVISO: no se detecta GPU. Se entrena en CPU, mucho mas lento. ***")

    # --- datos: entrenamiento y validacion por PACIENTE (columna `fold`), test intacto ---
    samples = uc.cargar_samples(localizar("metadata", args.raiz))
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
    transformacion = AumentadoAfin() if args.aumentado == "geometrico" else None   # solo en entrenamiento
    dl_train = DataLoader(uc.BreastDCEDataset(entrenamiento, raiz=raiz_img, transform=transformacion),
                          batch_size=args.lote, shuffle=True,
                          generator=torch.Generator().manual_seed(args.semilla), **opciones)
    # shuffle=False en validacion y vigilados: obligatorio para que las probabilidades casen con las filas
    dl_val = DataLoader(uc.BreastDCEDataset(validacion, raiz=raiz_img), batch_size=args.lote, shuffle=False, **opciones)
    dl_vig = DataLoader(uc.BreastDCEDataset(vigilados, raiz=raiz_img), batch_size=args.lote, shuffle=False, **opciones)

    # --- modelo, perdida y optimizador ---
    modelo = mo.RedBase().to(device)
    if args.perdida == "ponderada":
        peso = uc.pos_weight(entrenamiento)          # N0/N1 de ESTAS filas de entrenamiento, calculado del CSV
        criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso, dtype=torch.float32, device=device))
    else:
        peso = 1.0
        criterio = nn.BCEWithLogitsLoss()
    proporcion = float((entrenamiento.pCR == 1).mean())
    referencia = perdida_de_referencia(proporcion, peso)
    referencia_val = perdida_de_referencia(float((validacion.pCR == 1).mean()), peso)   # con la proporcion de la VALIDACION
    print(f"perdida {args.perdida}" + (f" (pos_weight = {peso:.4f})" if args.perdida == "ponderada" else "") +
          f"; proporcion de cortes con pCR=1: {proporcion:.3f}; "
          f"mejor perdida sin mirar la imagen: {referencia:.4f} (entrenamiento), {referencia_val:.4f} (validacion)")
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- bucle de entrenamiento: epocas fijas, nos quedamos con la mejor segun el AUC de validacion ---
    mejor_auc, mejor_epoca, mejor_probs = -1.0, 0, None
    auc_historial = []                       # AUC de validacion de cada epoca (para la metrica oficial)
    tiempos, t_inicio = [], time.time()
    with open(ruta["csv"], "w", encoding="utf-8") as f:
        f.write("epoca,perdida_train,perdida_val,auc_train,auc_val,accuracy_val,sensibilidad_val,"
                "especificidad_val,lr,tiempo_seg\n")
        for epoca in range(1, args.epocas + 1):
            t0 = time.time()
            perdida_train = entrenar_una_epoca(modelo, dl_train, criterio, optimizador, device)
            perdida_val, res_val, probs_val = medir(modelo, dl_val, validacion, criterio, device, args.umbral)
            auc_historial.append(res_val["auc"])
            _, res_vig, _ = medir(modelo, dl_vig, vigilados, criterio, device, args.umbral)
            dur = time.time() - t0
            tiempos.append(dur)
            f.write(f"{epoca},{perdida_train:.6f},{perdida_val:.6f},{res_vig['auc']:.6f},{res_val['auc']:.6f},"
                    f"{res_val['accuracy']:.6f},{res_val['sensibilidad']:.6f},{res_val['especificidad']:.6f},"
                    f"{args.lr:.8f},{dur:.2f}\n")
            f.flush()
            print(f"epoca {epoca:3d}/{args.epocas}  perdida train {perdida_train:.4f} val {perdida_val:.4f}  "
                  f"AUC train {res_vig['auc']:.3f} val {res_val['auc']:.3f}  "
                  f"(val: acc {res_val['accuracy']:.3f} sens {res_val['sensibilidad']:.3f} "
                  f"espec {res_val['especificidad']:.3f})  {dur:.1f}s", flush=True)

            if epoca == 1 or res_val["auc"] > mejor_auc:     # la 1.a se guarda siempre (por si el AUC fuera nan)
                mejor_auc, mejor_epoca, mejor_probs = res_val["auc"], epoca, probs_val
                torch.save({"red": "RedBase", "perdida": args.perdida, "fold": args.fold, "epoca": epoca,
                            "auc_val": res_val["auc"], "lr": args.lr, "lote": args.lote,
                            "state_dict": modelo.state_dict()}, ruta["pesos"])
                with open(ruta["probs"], "w", encoding="utf-8") as g:
                    g.write("sample_id,patient_id,pCR,prob\n")
                    for fila, p in zip(validacion.itertuples(), probs_val):
                        g.write(f"{fila.sample_id},{fila.patient_id},{fila.pCR},{p:.6f}\n")
    t_total = time.time() - t_inicio

    # --- gráficas del diagnostico, en la mejor epoca ---
    titulo = f"Red base · pérdida {args.perdida} · fold {args.fold}"
    graficas.dibujar_curvas_entrenamiento(ruta["csv"], ruta["curvas"], titulo, referencia, mejor_epoca, referencia_val)
    # metrica OFICIAL (DECISIONES.md): media del AUC de validacion de las ultimas 10 epocas
    n_ult = min(10, len(auc_historial))
    auc_media_ult = sum(auc_historial[-n_ult:]) / n_ult
    y_pac, p_pac = por_paciente(validacion, mejor_probs)
    graficas.dibujar_roc(y_pac, p_pac, ruta["roc"], f"ROC por paciente · época {mejor_epoca}", args.umbral)
    mejor = uc.evaluar_por_paciente(mejor_probs, validacion, umbral=args.umbral, metodo="mean")
    mc = mejor["matriz_confusion"]
    graficas.dibujar_matriz_confusion(mc["VP"], mc["VN"], mc["FP"], mc["FN"], ruta["conf"],
                                      f"Matriz de confusión por paciente · época {mejor_epoca} · umbral {args.umbral:g}")

    resumen = {
        "red": "RedBase", "parametros": mo.contar_parametros(modelo), "perdida": args.perdida,
        "aumentado": args.aumentado,
        "pos_weight": peso if args.perdida == "ponderada" else None, "fold": args.fold,
        "optimizador": "Adam", "lr": args.lr, "lote": args.lote, "epocas": args.epocas, "semilla": args.semilla,
        "umbral": args.umbral, "agregacion": "mean", "muestra_rapida": args.muestra_rapida,
        "pacientes_train": int(entrenamiento.patient_id.nunique()), "pacientes_val": int(validacion.patient_id.nunique()),
        "perdida_de_referencia": referencia, "perdida_de_referencia_val": referencia_val,
        "mejor_epoca": mejor_epoca, "auc_val": mejor_auc,
        "auc_val_media_ultimas10": auc_media_ult, "auc_val_ultima": auc_historial[-1],
        "matriz_confusion_val_ultima": res_val["matriz_confusion"],
        "matriz_confusion_val": mc, "sensibilidad_val": mejor["sensibilidad"],
        "especificidad_val": mejor["especificidad"], "accuracy_val": mejor["accuracy"],
        "tiempo_total_seg": round(t_total, 1), "tiempo_medio_por_epoca_seg": round(sum(tiempos) / len(tiempos), 2),
        "dispositivo": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "memoria_gpu_max_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if device.type == "cuda" else None,
        "version_torch": torch.__version__, "version_hip": getattr(torch.version, "hip", None),
    }
    with open(ruta["resumen"], "w", encoding="utf-8") as g:
        json.dump(resumen, g, indent=2, ensure_ascii=False)

    print(f"\nMETRICA OFICIAL: media del AUC de validacion de las ultimas {n_ult} epocas = {auc_media_ult:.4f}")
    print(f"Mejor epoca: {mejor_epoca}  AUC por paciente en validacion = {mejor_auc:.4f}  "
          f"(optimista: se elige mirando la validacion)")
    print(f"Matriz de confusion (umbral {args.umbral:g}): {mc}")
    print(f"Resultados en {args.salida}  (prefijo {prefijo})")


if __name__ == "__main__":
    main()
