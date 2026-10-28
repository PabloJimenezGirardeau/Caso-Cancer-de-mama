"""Control positivo: la MISMA red, el MISMO codigo y los MISMOS datos, con un objetivo que SI se ve en la imagen.

Si la red llega a un AUC alto en validacion prediciendo la cohorte (Duke frente a I-SPY: escaneres y protocolos distintos),
el entrenamiento, la lectura de imagenes y la evaluacion funcionan, y el AUC bajo de la pCR se debe a que la pCR casi no se
ve en un corte 2D, no a un error del codigo. Se reutilizan entrenar.py (una epoca, medir) y modelos.py (RedBase): solo
cambia la etiqueta, que pasa a ser 1 si la paciente es de Duke.

Uso (CPU o GPU):
    python analisis/control_positivo.py                     # fold 0, 4 epocas, recorte 128 como el modelo final
    python analisis/control_positivo.py --epocas 6 --muestra-rapida 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # para importar los modulos de la raiz del proyecto

import torch
from torch import nn
from torch.utils.data import DataLoader

import entrenar as en
import modelos as mo
import utils_caso as uc

RAIZ = Path(__file__).resolve().parent.parent   # la raiz del proyecto: este script esta en analisis/


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epocas", type=int, default=4)
    p.add_argument("--muestra-rapida", type=int, default=0, help="si > 0, N pacientes por lado (para CPU)")
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    en.fijar_semilla(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raiz_meta, raiz_img = en.localizar("metadata", RAIZ), en.localizar("dataset", RAIZ)
    samples = uc.cargar_samples(raiz_meta)
    cohorte = uc.cargar_patients(raiz_meta).set_index("pid").dataset
    samples = samples[samples.fold >= 0].copy()
    samples["pCR"] = (samples.patient_id.map(cohorte) == "duke").astype(int)   # la "etiqueta" pasa a ser la cohorte
    entrenamiento, validacion = uc.particion(samples, fold_val=args.fold)
    if args.muestra_rapida:
        entrenamiento = en.muestra_estratificada(entrenamiento, args.muestra_rapida)
        validacion = en.muestra_estratificada(validacion, args.muestra_rapida)
    print(f"CONTROL POSITIVO: etiqueta = paciente de Duke. Fold {args.fold}: entrena "
          f"{entrenamiento.patient_id.nunique()} pacientes ({entrenamiento.groupby('patient_id').pCR.first().mean():.0%} Duke), "
          f"valida {validacion.patient_id.nunique()} ({validacion.groupby('patient_id').pCR.first().mean():.0%} Duke)")

    dl_train = DataLoader(uc.BreastDCEDataset(entrenamiento, raiz=raiz_img), batch_size=128, shuffle=True,
                          generator=torch.Generator().manual_seed(42), num_workers=args.num_workers)
    dl_val = DataLoader(uc.BreastDCEDataset(validacion, raiz=raiz_img), batch_size=128, shuffle=False,
                        num_workers=args.num_workers)
    modelo = mo.RedBase(batchnorm=True, recorte=128).to(device)        # la misma red que el modelo final
    criterio = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-4)
    for epoca in range(1, args.epocas + 1):
        perdida, _ = en.entrenar_una_epoca(modelo, dl_train, criterio, optimizador, device)
        perdida_val, res, _, _ = en.medir(modelo, dl_val, validacion, criterio, device, 0.5)
        print(f"epoca {epoca}: perdida train {perdida:.4f} val {perdida_val:.4f}   "
              f"AUC por paciente en VALIDACION (Duke frente a I-SPY): {res['auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
