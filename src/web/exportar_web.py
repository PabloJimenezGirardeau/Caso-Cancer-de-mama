"""Prepara la app web (GitHub Pages, carpeta docs/) a partir del modelo final de `modelo/`.

La pagina ejecuta el modelo dentro del navegador, asi que necesita los 15 modelos en formato ONNX (el que entiende el
navegador) y los datos de la calibracion y el umbral. Este script:
    1. convierte cada modelo de PyTorch a ONNX (en modo evaluacion; el recorte central va dentro del modelo), con dos
       salidas: el logit y el MAPA de donde mira la red (Grad-CAM, ver ConMapa);
    2. comprueba con onnxruntime que cada ONNX da EXACTAMENTE lo mismo que PyTorch sobre cortes reales, y que el mapa
       coincide con el Grad-CAM calculado con autograd (el de src/analisis/mapas_gradcam.py);
    3. elige cortes de ejemplo (pacientes de ENTRENAMIENTO: nunca del test) y guarda, para cada uno, la probabilidad que
       da `predecir.py`: la pagina la recalcula y la compara (verificacion de que es el mismo pipeline);
    4. guarda en docs/datos/web.js la lista de modelos, el checksum web y los ejemplos con su probabilidad de Python.
       Estos cortes los usa la Verificacion de la app; la galeria de ejemplos usa los de galeria_web.py.
       (El umbral, la calibracion, las metricas y las curvas los escribe datos_web.py en docs/datos/datos.js.)

Uso (en el PC, que tiene PyTorch):
    pip install onnxruntime
    python src/web/exportar_web.py

Salida, en docs/:
    modelos/<nombre>.onnx                         entrada "imagen" float32 [N, 3, 256, 256] en [0, 1] (PRE, EARLY, LATE);
                                                  salidas "logit" [N] y "mapa" [N, 16, 16] (Grad-CAM con signo)
    ejemplos/<sample_id>_{PRE,EARLY,LATE}.png     los cortes de ejemplo, copiados tal cual
    datos/web.js                                  modelos, checksum web y ejemplos con su probabilidad de Python
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/: para importar los paquetes del proyecto

from datos import utils_caso as uc
from evaluacion.predecir import ModeloFinal, cargar_corte

RAIZ = Path(__file__).resolve().parents[2]   # la raiz del proyecto: este fichero esta en src/web/
WEB = RAIZ / "docs"
N_EJEMPLOS = 12           # 6 con pCR y 6 sin pCR, de pacientes de entrenamiento
LADO = 256
OPSET = 17


def localizar(sub: str) -> Path:
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / sub).is_dir():
            return base
    raise FileNotFoundError(sub)


class ConMapa(torch.nn.Module):
    """La red del modelo final con una segunda salida: el mapa Grad-CAM, con signo, del ultimo bloque.

    Grad-CAM pesa cada canal k del mapa A (128 @ 16x16: la ReLU del ultimo bloque, sobre el cuadrado central) por la media
    espacial del gradiente del logit respecto a ese canal. En esta red (maxpool 2x2 -> media global de 8x8 -> lineal ->
    ReLU -> lineal) ese gradiente medio vale exactamente g_k / 256, con g = d logit / d caracteristicas = W1^T (w2 * [h > 0]):
    el maxpool pasa el gradiente a un solo punto de cada ventana (64 ventanas x g_k / 64 = g_k, repartido entre 256 puntos).
    Asi el mapa se calcula sin autograd y va dentro del ONNX: mapa = sum_k g_k A_k (Grad-CAM x 256; la escala se quita al
    normalizar). Positivo = evidencia de pCR = 1; la app usa la parte positiva si predice pCR = 1 y la negativa si no."""

    def __init__(self, red: torch.nn.Module):
        super().__init__()
        assert red.recorte and not red.realce and not red.normalizar and red.puntuacion is None, "red no prevista"
        assert red.tipo_cabeza == "normal" and len(red.extractor[3]) == 4, "red no prevista"
        self.red = red

    def forward(self, x: torch.Tensor):
        r = self.red
        ini = (LADO - r.recorte) // 2
        x = x[:, :, ini:ini + r.recorte, ini:ini + r.recorte]                 # el mismo cuadrado central que RedBase.mapa
        h = r.extractor[2](r.extractor[1](r.extractor[0](x)))
        b = r.extractor[3]
        a = b[2](b[1](b[0](h)))                                               # (N, 128, 16, 16)
        f = b[3](a).mean(dim=(2, 3))                                          # maxpool y media global -> (N, 128)
        lin1, lin2 = r.cabeza[0], r.cabeza[3]                                 # el dropout no hace nada al evaluar
        oculta = lin1(f)
        logit = lin2(torch.relu(oculta)).squeeze(1)
        g = (lin2.weight[0] * (oculta > 0).to(oculta.dtype)) @ lin1.weight    # d logit / d f -> (N, 128)
        return logit, (g[:, :, None, None] * a).sum(1)                        # (N,), (N, 16, 16)


def gradcam_autograd(red: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Grad-CAM con signo calculado con autograd (como src/analisis/mapas_gradcam.py, antes de la ReLU): para comprobar ConMapa."""
    guardado = {}
    gancho = red.extractor[3][2].register_forward_hook(lambda _m, _e, salida: guardado.__setitem__("a", salida))
    t = x.clone().requires_grad_(True)
    try:
        logit = red(t)
    finally:
        gancho.remove()
    g, = torch.autograd.grad(logit.sum(), guardado["a"])
    return (g.mean(dim=(2, 3), keepdim=True) * guardado["a"]).sum(1).detach()


def exportar(red: torch.nn.Module, ruta: Path) -> None:
    ejemplo = torch.rand(1, 3, 256, 256)
    argumentos = dict(input_names=["imagen"], output_names=["logit", "mapa"],
                      dynamic_axes={"imagen": {0: "n"}, "logit": {0: "n"}, "mapa": {0: "n"}},
                      opset_version=OPSET, do_constant_folding=True)
    try:
        torch.onnx.export(red, ejemplo, str(ruta), dynamo=False, **argumentos)   # exportador clasico (TorchScript)
    except TypeError:                                                         # versiones sin el argumento dynamo
        torch.onnx.export(red, ejemplo, str(ruta), **argumentos)


def main() -> None:
    import onnxruntime as ort

    for sub in ("modelos", "ejemplos", "datos"):
        (WEB / sub).mkdir(parents=True, exist_ok=True)
    modelo = ModeloFinal(RAIZ / "modelo", dispositivo="cpu")
    base_meta, base_img = localizar("metadata"), localizar("dataset")
    samples = uc.cargar_samples(base_meta)

    # --- 1 y 2: exportar y comprobar contra PyTorch con cortes reales de validacion ---
    rng = np.random.default_rng(0)
    entrenamiento = samples[samples.fold >= 0]
    prueba = entrenamiento.iloc[rng.choice(len(entrenamiento), 64, replace=False)]
    x = np.stack([cargar_corte(base_img / r.path_pre, base_img / r.path_early, base_img / r.path_late)
                  for r in prueba.itertuples()]).astype(np.float32)
    archivos, peor, peor_mapa, peor_grad = [], 0.0, 0.0, 0.0
    xt = torch.from_numpy(x)
    for nombre, red in zip(modelo.nombres, modelo.redes):
        ruta = WEB / "modelos" / f"{nombre}.onnx"
        envuelta = ConMapa(red.eval()).eval()
        exportar(envuelta, ruta)
        with torch.no_grad():
            esperado = red(xt).numpy()
            logit_env, mapa_env = (v.numpy() for v in envuelta(xt))
        referencia = gradcam_autograd(red, xt).numpy() * 256           # Grad-CAM de autograd en la escala de ConMapa
        sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        logit_onnx, mapa_onnx = sesion.run(["logit", "mapa"], {"imagen": x})
        escala = float(np.abs(referencia).max()) or 1.0
        dif = max(float(np.abs(logit_onnx.reshape(-1) - esperado).max()), float(np.abs(logit_env - esperado).max()))
        dif_mapa = float(np.abs(mapa_onnx - mapa_env).max()) / escala      # ONNX frente a PyTorch (relativa)
        dif_grad = float(np.abs(mapa_env - referencia).max()) / escala     # formula cerrada frente a autograd (relativa)
        peor, peor_mapa, peor_grad = max(peor, dif), max(peor_mapa, dif_mapa), max(peor_grad, dif_grad)
        archivos.append(ruta.name)
        print(f"  {nombre:<34} logit {dif:.1e} · mapa ONNX/PyTorch {dif_mapa:.1e} · mapa/Grad-CAM autograd {dif_grad:.1e}")
    bien = peor < 1e-4 and peor_mapa < 1e-4 and peor_grad < 1e-4
    print(f"Diferencias maximas en los 15 modelos: logit {peor:.2e}, mapa {peor_mapa:.2e}, frente a Grad-CAM {peor_grad:.2e}  ->  "
          + ("OK" if bien else "ATENCION: no coincide"))

    # --- 3: cortes de ejemplo (entrenamiento), su probabilidad con predecir.py ---
    etiqueta = entrenamiento.groupby("patient_id").pCR.first()
    elegidas = list(rng.choice(sorted(etiqueta[etiqueta == 1].index), N_EJEMPLOS // 2, replace=False)) + \
               list(rng.choice(sorted(etiqueta[etiqueta == 0].index), N_EJEMPLOS // 2, replace=False))
    ejemplos = []
    for pid in elegidas:
        cortes = entrenamiento[entrenamiento.patient_id == pid].sort_values("slice_index")
        r = cortes.iloc[len(cortes) // 2]
        rutas = {}
        for fase, col in (("PRE", "path_pre"), ("EARLY", "path_early"), ("LATE", "path_late")):
            destino = WEB / "ejemplos" / f"{r.sample_id}_{fase}.png"
            shutil.copyfile(base_img / getattr(r, col), destino)
            rutas[fase] = f"ejemplos/{destino.name}"
        res = modelo.predecir_corte(base_img / r.path_pre, base_img / r.path_early, base_img / r.path_late)
        ejemplos.append({"sample_id": r.sample_id, "patient_id": pid, "fold": int(r.fold), "pCR": int(r.pCR),
                         "rutas": rutas, "probabilidad_python": res["probabilidad"], "clase_python": res["clase"],
                         "avisos_python": res["avisos"]})
    # --- 4: datos/web.js (lo que depende de los ONNX; el resto lo escribe datos_web.py en datos/datos.js) ---
    cfg = json.loads((RAIZ / "modelo" / "umbral_final.json").read_text(encoding="utf-8"))
    huella = hashlib.sha256()
    for a in archivos:
        huella.update((WEB / "modelos" / a).read_bytes())
    huella.update(f"{cfg['umbral']!r}|{cfg['calibracion']['a']!r}|{cfg['calibracion']['b']!r}".encode())
    web = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelos": archivos, "opset_onnx": OPSET, "onnx_frente_a_pytorch": peor,
        "mapa": "salida 'mapa' [N, 16, 16]: Grad-CAM con signo del ultimo bloque (x 256) sobre el cuadrado central; "
                "positivo = evidencia de pCR = 1",
        "mapa_frente_a_gradcam": peor_grad,
        "checksum_web": huella.hexdigest()[:16],
        "checksum_web_como": "SHA-256 de los 15 .onnx (en este orden) y de 'umbral|a|b'; primeros 16 caracteres",
        "checksum_pytorch": modelo.checksum,
        "nota_ejemplos": "Pacientes de ENTRENAMIENTO (nunca del test). La mayoria de los 15 modelos las vieron al entrenar, "
                         "asi que sus predicciones no miden el rendimiento: sirven para comprobar que la pagina reproduce "
                         "a Python.",
        "ejemplos": ejemplos,
    }
    (WEB / "datos" / "web.js").write_text("// Generado por exportar_web.py: no editar a mano.\nwindow.WEB = "
                                          + json.dumps(web, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    print(f"\nEjemplos: {len(ejemplos)} cortes en {WEB / 'ejemplos'}; checksum web {web['checksum_web']}; "
          f"checksum PyTorch {modelo.checksum}")
    print(f"Listo: {WEB}  (recuerda ejecutar tambien python src/web/datos_web.py)")

if __name__ == "__main__":
    main()
