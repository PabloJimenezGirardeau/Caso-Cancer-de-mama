"""Prediccion con el modelo final. La MISMA funcion la usan el test (evaluar_test.py) y la app.

Pipeline fijado en la fase 4 (DECISIONES.md, `modelo/umbral_final.json`):
    1. cada fase (PRE, EARLY, LATE) es un PNG en escala de grises de 256x256; se lee y se divide
       entre 255, exactamente como `utils_caso.cargar_fase` en el entrenamiento;
    2. los 15 modelos (configuracion 2.1: 3 semillas x 5 folds), en modo evaluacion y sin gradientes,
       dan cada uno su probabilidad para el corte (sigmoide de su logit);
    3. se promedian las 15 probabilidades;
    4. calibracion de Platt: p = sigmoide(a * logit(p_media) + b), con los (a, b) del json;
    5. paciente con varios cortes: media de sus cortes calibrados (la agregacion que pide el profesor);
    6. clase: "pCR" si p >= umbral (0,348; criterio de desescalar, especificidad >= 0,95 en validacion).

Uso como modulo:
    from predecir import ModeloFinal
    modelo = ModeloFinal()                                  # carga modelo/ una sola vez
    r = modelo.predecir_corte("pre.png", "early.png", "late.png")   # rutas, bytes o ficheros abiertos
    print(r["probabilidad"], r["clase"])

Uso desde la terminal:
    python predecir.py --muestra ISPY1_1001_z012            # un corte del dataset, por su sample_id
    python predecir.py --pre a.png --early b.png --late c.png
    python predecir.py --verificar 20                       # comprueba que reproduce las probabilidades de
                                                            # validacion guardadas por cada uno de los 15 modelos
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import modelos as mo

RAIZ = Path(__file__).resolve().parent
CARPETA_MODELO = RAIZ / "modelo"
LADO = 256
TAMANO_MAX_BYTES = 5 * 2**20          # 5 MB por PNG: muy por encima de lo normal (~30-80 KB)


# --------------------------------------------------------------------------- #
# Lectura y validacion de las imagenes (igual que en el entrenamiento)
# --------------------------------------------------------------------------- #

class EntradaNoValida(ValueError):
    """La entrada no se puede usar (formato, tamaño, valores). El mensaje es para la persona usuaria."""


def cargar_fase(fuente, nombre: str = "fase") -> np.ndarray:
    """Lee UNA fase y devuelve un array (256, 256) float32 en [0, 1].

    `fuente` puede ser una ruta, los bytes del fichero o un fichero abierto. Se hace lo mismo que en el
    entrenamiento: convertir a escala de grises de 8 bits ("L") y dividir entre 255. Sin redimensionar:
    si no mide 256x256 se rechaza, porque el modelo nunca ha visto imagenes de otro tamaño.
    """
    if isinstance(fuente, (str, Path)):
        datos = Path(fuente).read_bytes()
    elif isinstance(fuente, (bytes, bytearray)):
        datos = bytes(fuente)
    else:
        datos = fuente.read()
    if len(datos) == 0:
        raise EntradaNoValida(f"{nombre}: el fichero está vacío.")
    if len(datos) > TAMANO_MAX_BYTES:
        raise EntradaNoValida(f"{nombre}: el fichero ocupa demasiado ({len(datos) / 2**20:.1f} MB; máximo 5 MB).")
    try:
        with Image.open(io.BytesIO(datos)) as png:
            formato = png.format
            png.load()
            if formato != "PNG":
                raise EntradaNoValida(f"{nombre}: debe ser un PNG (se ha recibido {formato or 'un formato desconocido'}).")
            if png.size != (LADO, LADO):
                raise EntradaNoValida(f"{nombre}: debe medir {LADO}x{LADO} píxeles (mide {png.size[0]}x{png.size[1]}).")
            x = np.asarray(png.convert("L"), dtype=np.float32) / 255.0
    except EntradaNoValida:
        raise
    except Exception as error:                      # fichero corrupto o que no es una imagen
        raise EntradaNoValida(f"{nombre}: no se puede leer como imagen ({type(error).__name__}).") from None
    if not np.isfinite(x).all():
        raise EntradaNoValida(f"{nombre}: contiene valores no finitos.")
    return x


def cargar_corte(pre, early, late) -> np.ndarray:
    """Las tres fases apiladas en el orden del entrenamiento: (3, 256, 256) = PRE, EARLY, LATE."""
    return np.stack([cargar_fase(pre, "PRE"), cargar_fase(early, "EARLY"), cargar_fase(late, "LATE")])


def avisos(corte: np.ndarray) -> list[str]:
    """Comprobaciones que no impiden predecir pero conviene mostrar."""
    lista = []
    tejido = corte[0] > 0.1
    if tejido.sum() < 100:
        lista.append("La imagen PRE es casi toda negra: ¿es un corte de resonancia de mama?")
    elif corte[1][tejido].mean() <= corte[0][tejido].mean():
        lista.append("En el tejido, EARLY no es más brillante que PRE (en el dataset lo es en el 100 % de las "
                     "pacientes): ¿están las fases en el orden correcto?")
    if any(float(c.std()) < 1e-3 for c in corte):
        lista.append("Alguna fase es una imagen prácticamente uniforme.")
    return lista


# --------------------------------------------------------------------------- #
# Modelo final
# --------------------------------------------------------------------------- #

class ModeloFinal:
    """Los 15 modelos + la calibracion + el umbral, cargados una sola vez."""

    def __init__(self, carpeta: Path | str = CARPETA_MODELO, dispositivo: str | None = None):
        self.carpeta = Path(carpeta)
        ruta_json = self.carpeta / "umbral_final.json"
        if not ruta_json.exists():
            raise FileNotFoundError(f"No encuentro {ruta_json}")
        self.config = json.loads(ruta_json.read_text(encoding="utf-8"))
        self.umbral = float(self.config["umbral"])
        cal = self.config.get("calibracion")
        self.a, self.b = (float(cal["a"]), float(cal["b"])) if cal else (1.0, 0.0)
        self.dispositivo = torch.device(dispositivo or ("cuda" if torch.cuda.is_available() else "cpu"))

        huella = hashlib.sha256(ruta_json.read_bytes())
        self.nombres, self.redes = [], []
        for nombre in self.config["configuraciones"]:
            for k in range(5):
                ruta = self.carpeta / f"{nombre}_fold{k}.pt"
                if not ruta.exists():
                    raise FileNotFoundError(f"Falta el modelo {ruta}")
                huella.update(ruta.read_bytes())
                ck = torch.load(ruta, map_location="cpu", weights_only=False)
                red = mo.RedBase(batchnorm=ck.get("batchnorm", False), realce=ck.get("realce", False),
                                 filtros=ck.get("filtros", 16), doble=ck.get("doble", False),
                                 normalizar=ck.get("normalizar", False))
                red.load_state_dict(ck["state_dict"])
                red.to(self.dispositivo).eval()
                self.redes.append(red)
                self.nombres.append(f"{nombre}_fold{k}")
        self.checksum = huella.hexdigest()[:16]     # cambia si cambia cualquier modelo o el json

    @torch.no_grad()
    def probabilidades_modelo(self, i: int, x: np.ndarray, lote: int = 64) -> np.ndarray:
        """Probabilidad SIN calibrar del modelo i para cada corte de x (N, 3, 256, 256)."""
        salida = []
        for j in range(0, len(x), lote):
            t = torch.from_numpy(np.ascontiguousarray(x[j:j + lote])).to(self.dispositivo)
            salida.append(torch.sigmoid(self.redes[i](t)).cpu().numpy())
        return np.concatenate(salida)

    def probabilidades(self, x: np.ndarray) -> np.ndarray:
        """Probabilidad calibrada de cada corte: media de los 15 modelos y despues Platt."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None]
        if x.shape[1:] != (3, LADO, LADO):
            raise EntradaNoValida(f"Se esperaba un corte de forma (3, {LADO}, {LADO}) y llegó {x.shape[1:]}.")
        media = np.mean([self.probabilidades_modelo(i, x) for i in range(len(self.redes))], axis=0)
        media = np.clip(media, 1e-6, 1 - 1e-6)
        z = self.a * np.log(media / (1 - media)) + self.b
        return 1 / (1 + np.exp(-z))

    def _resultado(self, p: float, t0: float, n_cortes: int) -> dict:
        return {"probabilidad": float(p), "clase": "pCR" if p >= self.umbral else "no pCR",
                "umbral": self.umbral, "cortes": n_cortes, "modelos": len(self.redes),
                "dispositivo": str(self.dispositivo), "segundos": round(time.time() - t0, 3),
                "checksum": self.checksum}

    def predecir_corte(self, pre, early, late) -> dict:
        """Un corte (lo que llega en la app y en la defensa)."""
        t0 = time.time()
        corte = cargar_corte(pre, early, late)
        resultado = self._resultado(self.probabilidades(corte)[0], t0, 1)
        resultado["avisos"] = avisos(corte)
        return resultado

    def predecir_paciente(self, cortes: np.ndarray) -> dict:
        """Varios cortes de UNA paciente (N, 3, 256, 256): media de sus probabilidades calibradas."""
        t0 = time.time()
        return self._resultado(float(np.mean(self.probabilidades(cortes))), t0, len(cortes))


# --------------------------------------------------------------------------- #
# Terminal: predecir un corte o verificar que se reproduce la validacion
# --------------------------------------------------------------------------- #

def localizar_dataset(raiz: Path) -> Path:
    for base in (raiz, raiz / "breastdcedl"):
        if (base / "dataset").is_dir():
            return base
    raise FileNotFoundError(f"No encuentro dataset/ en {raiz} ni en {raiz / 'breastdcedl'}")


def localizar_metadata(raiz: Path) -> Path:
    for base in (raiz, raiz / "breastdcedl"):
        if (base / "metadata").is_dir():
            return base
    raise FileNotFoundError(f"No encuentro metadata/ en {raiz} ni en {raiz / 'breastdcedl'}")


def verificar(modelo: ModeloFinal, raiz: Path, pacientes: int) -> None:
    """Para cada uno de los 15 modelos, recalcula con este modulo las probabilidades de validacion de
    `pacientes` pacientes de su fold y las compara con las que guardo el entrenamiento. Si coinciden, la
    lectura de imagenes y la red son las mismas que en el entrenamiento (la app no puede diferir)."""
    import pandas as pd
    base_img = localizar_dataset(raiz)
    print(f"Verificacion con {pacientes} pacientes por modelo (dispositivo: {modelo.dispositivo})")
    peor = 0.0
    for i, nombre in enumerate(modelo.nombres):
        guardado = pd.read_csv(modelo.carpeta / f"{nombre}_probs_val.csv")
        elegidas = sorted(guardado.patient_id.unique())[:pacientes]
        filas = guardado[guardado.patient_id.isin(elegidas)]
        samples = pd.read_csv(localizar_metadata(raiz) / "metadata" / "samples.csv").set_index("sample_id")
        rutas = samples.loc[filas.sample_id]
        x = np.stack([cargar_corte(base_img / r.path_pre, base_img / r.path_early, base_img / r.path_late)
                      for r in rutas.itertuples()])
        p = modelo.probabilidades_modelo(i, x)
        diferencia = float(np.abs(p - filas.prob.values).max())
        peor = max(peor, diferencia)
        print(f"  {nombre:<32} {len(filas):>4} cortes   diferencia maxima {diferencia:.2e}")
    print(f"\nDiferencia maxima en los 15 modelos: {peor:.2e}  ->  "
          + ("OK: el modulo reproduce el entrenamiento" if peor < 1e-3 else "ATENCION: no coincide"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--muestra", help="sample_id de samples.csv (p. ej. ISPY1_1001_z012)")
    p.add_argument("--pre"); p.add_argument("--early"); p.add_argument("--late")
    p.add_argument("--verificar", type=int, metavar="N", help="comprobar con N pacientes por modelo")
    p.add_argument("--raiz", type=Path, default=RAIZ)
    p.add_argument("--carpeta", type=Path, default=CARPETA_MODELO)
    args = p.parse_args()

    t0 = time.time()
    modelo = ModeloFinal(args.carpeta)
    print(f"Modelo final: {len(modelo.redes)} modelos, umbral {modelo.umbral:.4f}, calibracion a={modelo.a:.3f} "
          f"b={modelo.b:.3f}, checksum {modelo.checksum}, cargado en {time.time() - t0:.1f} s ({modelo.dispositivo})")
    if args.verificar:
        verificar(modelo, args.raiz, args.verificar)
        return
    if args.muestra:
        import pandas as pd
        fila = pd.read_csv(localizar_metadata(args.raiz) / "metadata" / "samples.csv").set_index("sample_id").loc[args.muestra]
        base = localizar_dataset(args.raiz)
        args.pre, args.early, args.late = base / fila.path_pre, base / fila.path_early, base / fila.path_late
        print(f"Muestra {args.muestra} (paciente {fila.patient_id}, pCR real {fila.pCR})")
    if not (args.pre and args.early and args.late):
        raise SystemExit("Indica --muestra o las tres fases (--pre, --early, --late), o --verificar N.")
    try:
        r = modelo.predecir_corte(args.pre, args.early, args.late)
    except EntradaNoValida as error:
        raise SystemExit(f"Entrada no valida: {error}")
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
