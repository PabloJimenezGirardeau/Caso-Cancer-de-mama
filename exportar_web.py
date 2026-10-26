"""Prepara la app web (GitHub Pages, carpeta docs/) a partir del modelo final de `modelo/`.

La pagina ejecuta el modelo dentro del navegador, asi que necesita los 15 modelos en formato ONNX (el que entiende el
navegador) y los datos de la calibracion y el umbral. Este script:
    1. convierte cada modelo de PyTorch a ONNX (en modo evaluacion; el recorte central va dentro del modelo);
    2. comprueba con onnxruntime que cada ONNX da EXACTAMENTE lo mismo que PyTorch sobre cortes reales;
    3. elige cortes de ejemplo (pacientes de ENTRENAMIENTO: nunca del test) y guarda, para cada uno, la probabilidad que
       da `predecir.py`: la pagina la recalcula y la compara (verificacion de que es el mismo pipeline);
    4. guarda en docs/datos/web.js la lista de modelos, el checksum web y los ejemplos con su probabilidad de Python.
       (El umbral, la calibracion, las metricas y las curvas los escribe datos_web.py en docs/datos/datos.js.)

Uso (en el PC, que tiene PyTorch):
    pip install onnxruntime
    python exportar_web.py

Salida, en docs/:
    modelos/<nombre>.onnx                         entrada "imagen" float32 [N, 3, 256, 256] en [0, 1] (PRE, EARLY, LATE);
                                                  salida "logit" [N]
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

import utils_caso as uc
from predecir import ModeloFinal, cargar_corte

RAIZ = Path(__file__).resolve().parent
WEB = RAIZ / "docs"
N_EJEMPLOS = 12           # 6 con pCR y 6 sin pCR, de pacientes de entrenamiento
OPSET = 17


def localizar(sub: str) -> Path:
    for base in (RAIZ, RAIZ / "breastdcedl"):
        if (base / sub).is_dir():
            return base
    raise FileNotFoundError(sub)


def exportar(red: torch.nn.Module, ruta: Path) -> None:
    ejemplo = torch.rand(1, 3, 256, 256)
    argumentos = dict(input_names=["imagen"], output_names=["logit"],
                      dynamic_axes={"imagen": {0: "n"}, "logit": {0: "n"}}, opset_version=OPSET, do_constant_folding=True)
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
    archivos, peor = [], 0.0
    for nombre, red in zip(modelo.nombres, modelo.redes):
        ruta = WEB / "modelos" / f"{nombre}.onnx"
        exportar(red.eval(), ruta)
        with torch.no_grad():
            esperado = red(torch.from_numpy(x)).numpy()
        sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        obtenido = sesion.run(["logit"], {"imagen": x})[0].reshape(-1)
        dif = float(np.abs(obtenido - esperado).max())
        peor = max(peor, dif)
        archivos.append(ruta.name)
        print(f"  {nombre:<34} ONNX frente a PyTorch: diferencia maxima de logit {dif:.2e}")
    print(f"Diferencia maxima en los 15 modelos: {peor:.2e}  ->  " + ("OK" if peor < 1e-4 else "ATENCION: no coincide"))

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
    print(f"Listo: {WEB}  (recuerda ejecutar tambien python datos_web.py)")

if __name__ == "__main__":
    main()
