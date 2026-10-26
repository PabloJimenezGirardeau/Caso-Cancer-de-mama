"""Datos de la app web que NO necesitan PyTorch: se generan en cualquier ordenador a partir de modelo/ y referencia/.

Escribe docs/datos/datos.js (un script que define window.DATOS, para que la pagina lo pueda leer tambien abierta como
fichero, sin servidor) con:
    modelo       umbral, criterio, calibracion, checksum de PyTorch (el mismo que calcula predecir.py), parametros del
                 entrenamiento y metricas internas de validacion
    validacion   las 1.097 pacientes de validacion: probabilidad calibrada (fuera de fold), pCR real, fold y cohorte
                 (para la curva ROC, el deslizador del umbral, la calibracion y el contexto del resultado)
    curvas       curvas de entrenamiento del modelo final (semilla 42, los 5 folds) con su referencia sin imagen
    comparacion  validacion y TEST de los cuatro modelos de evaluar_test.py (CNN solo imagen, solo datos clinicos,
                 imagen + datos clinicos, solo cohorte): metricas y la probabilidad de cada paciente (sin su
                 identificador), para las curvas ROC y la tabla de la vista Modelo
    gradcam      el resumen de mapas_gradcam.py (parte del mapa dentro de la caja del tumor en validacion)

Lo que SI necesita PyTorch (los modelos ONNX, los ejemplos y la verificacion) lo genera exportar_web.py en el PC.

Uso:
    python datos_web.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import curvas_entrenamiento as ce

RAIZ = Path(__file__).resolve().parent
MODELO = RAIZ / "modelo"
SALIDA = RAIZ / "docs" / "datos" / "datos.js"


def checksum_pytorch(cfg: dict) -> str:
    """El mismo que predecir.ModeloFinal.checksum: SHA-256 del json y de los 15 .pt en orden; 16 caracteres."""
    huella = hashlib.sha256((MODELO / "umbral_final.json").read_bytes().replace(b"\r\n", b"\n"))   # LF, como predecir.py
    for nombre in cfg["configuraciones"]:
        for k in range(5):
            huella.update((MODELO / f"{nombre}_fold{k}.pt").read_bytes())
    return huella.hexdigest()[:16]


MODELOS_COMPARADOS = ("recorte", "clinico", "combinado", "solo_cohorte")


def comparacion(conjunto: str) -> dict | None:
    """Metricas y probabilidades por paciente de los cuatro modelos (figuras/<conjunto>/, de evaluar_test.py)."""
    carpeta = RAIZ / "figuras" / conjunto
    if not (carpeta / "metricas.json").exists():
        return None
    m = json.loads((carpeta / "metricas.json").read_text(encoding="utf-8"))
    with open(carpeta / "pacientes.csv", encoding="utf-8") as f:
        filas = [[round(float(r[n]), 6) for n in MODELOS_COMPARADOS] + [int(r["pCR"]), r["cohorte"]]
                 for r in csv.DictReader(f)]
    claves = ("VP", "FN", "VN", "FP", "sensibilidad", "especificidad", "precision", "vpn", "f1_pcr", "f1_no_pcr",
              "f1_equilibrada", "accuracy", "precision_media", "recall_medio", "ic95")
    principal = m["modelos"]["recorte"]
    return {
        "pacientes_n": m["pacientes"], "cortes": m["cortes"], "pcr": m["pcr"],
        "modelos": {n: {k: m["modelos"][n][k] for k in ("auc", "auc_ic95", "ap", "brier")} for n in MODELOS_COMPARADOS},
        "con_el_umbral": {k: principal[k] for k in claves}, "umbral": m["umbrales"]["recorte"],
        "un_corte": {k: m["un_corte"]["recorte"][k] for k in ("auc", "sensibilidad", "especificidad")},
        "diferencias": m["diferencias_auc"], "coeficientes_combinado": m.get("combinado_coeficientes"),
        "cohortes": {c: {"pacientes": d["pacientes"], "pcr": d["pcr"],
                         **{n: {"auc": d[n]["auc"], "auc_ic95": d[n]["auc_ic95"]} for n in ("recorte", "clinico", "combinado")}}
                     for c, d in m["cohortes"].items()},
        "columnas": [*MODELOS_COMPARADOS, "pCR", "cohorte"], "filas": filas,
    }


def main() -> None:
    cfg = json.loads((MODELO / "umbral_final.json").read_text(encoding="utf-8"))
    nombres = [f"{c}_fold{k}" for c in cfg["configuraciones"] for k in range(5)]
    resumenes = [json.loads((MODELO / f"{n}_resumen.json").read_text(encoding="utf-8")) for n in nombres]
    r0 = resumenes[0]

    with open(ce.RAIZ / "metadata" / "patients.csv", encoding="utf-8") as f:
        cohorte = {r["pid"]: r["dataset"] for r in csv.DictReader(f)}
    with open(MODELO / "umbral_final_pacientes.csv", encoding="utf-8") as f:
        pacientes = [[float(r["prob"]), int(r["pCR"]), int(r["fold"]), cohorte[r["patient_id"]]]
                     for r in csv.DictReader(f)]

    refs = ce.referencias_por_fold()
    curvas = []
    for k in range(5):
        d = ce.leer(f"{cfg['configuraciones'][0]}_fold{k}")
        curvas.append({"fold": k, "epocas": [int(e) for e in d["epoca"]],
                       "perdida_val": [round(v, 5) for v in d["perdida_val"]],
                       "perdida_train": [round(v, 5) for v in d.get("perdida_train_eval", d["perdida_train"])],
                       "auc_val": [round(v, 4) for v in d["auc_val"]], "auc_train": [round(v, 4) for v in d["auc_train"]],
                       "elegida": int(d["elegida"]) + 1, "referencia_val": round(refs[k][1], 5),
                       "referencia_train": round(refs[k][0], 5)})

    datos = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelo": {
            "nombre": "Modelo final: recorte central 128, tasa de aprendizaje 3e-5 (3 semillas × 5 folds)",
            "redes": nombres,
            "umbral": cfg["umbral"], "criterio_umbral": cfg["criterio"], "umbral_desescalar": cfg.get("umbral_desescalar"),
            "calibracion": {"a": cfg["calibracion"]["a"], "b": cfg["calibracion"]["b"]},
            "checksum_pytorch": checksum_pytorch(cfg),
            "entrenamiento": {
                "parametros": r0["parametros"], "optimizador": r0["optimizador_clase"], "lr": r0["lr"], "lote": r0["lote"],
                "epocas_max": r0["epocas"], "recorte": r0["recorte"], "semillas": sorted({r["semilla"] for r in resumenes}),
                "epocas_elegidas": [r["epoca_elegida"] for r in resumenes],
                "auc_val_por_red": [round(r["auc_val"], 4) for r in resumenes],
                "version_torch": r0["version_torch"], "dispositivo_entrenamiento": r0["dispositivo"],
                "segundos_por_epoca": r0["tiempo_medio_por_epoca_seg"],
            },
            "metricas_validacion": {
                "auc": cfg["auc_validacion"], "auc_ic95": cfg["auc_ic95"], "ap": cfg["ap"], "brier": cfg["brier"],
                "brier_referencia": cfg["brier_referencia"], "con_el_umbral": cfg["metricas_validacion"],
                "auc_por_fold": cfg["auc_por_fold"].get("promedio"),
            },
        },
        "validacion": {"columnas": ["probabilidad", "pCR", "fold", "cohorte"], "pacientes": pacientes},
        "curvas": curvas,
        "comparacion": {"validacion": comparacion("validacion"), "test": comparacion("test")},
    }
    ruta_gradcam = RAIZ / "figuras" / "mapas" / "gradcam_resumen.json"
    if ruta_gradcam.exists():
        datos["gradcam"] = json.loads(ruta_gradcam.read_text(encoding="utf-8"))["validacion"]
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text("// Generado por datos_web.py: no editar a mano.\nwindow.DATOS = "
                      + json.dumps(datos, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"{SALIDA}: {len(pacientes)} pacientes de validacion, {len(curvas)} curvas, "
          f"checksum PyTorch {datos['modelo']['checksum_pytorch']} ({SALIDA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
