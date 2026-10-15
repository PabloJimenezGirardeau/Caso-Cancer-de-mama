"""Red base del caso BreastDCEDL: la arquitectura de referencia del profesor.

Sale de `presentacion_caso.pdf` (diapositiva 8, "el embudo de la CNN"):

    entrada          3 @ 256x256   las 3 fases (PRE, EARLY, LATE), NO son colores
    bloque 1         conv 3x3 + ReLU -> 16 @ 256x256,  maxpool 2x2 -> 16 @ 128x128
    bloque 2         conv 3x3 + ReLU -> 32 @ 128x128,  maxpool 2x2 -> 32 @  64x64
    bloque 3         conv 3x3 + ReLU -> 64 @  64x64,   maxpool 2x2 -> 64 @  32x32
    bloque 4         conv 3x3 + ReLU -> 128 @ 32x32,   maxpool 2x2 -> 128 @ 16x16
    media global     128 numeros (la media de cada mapa)
    densa + dropout  64
    salida           1 logit  (la sigmoide se aplica fuera, solo al evaluar)

Lo que la diapositiva NO dice y hemos decidido nosotros:
    - hay una ReLU despues de la capa densa de 64
    - el dropout apaga el 50 % de esas 64 neuronas (solo durante el entrenamiento)

Lo que se deduce de los numeros de la diapositiva (105.761 parametros):
    - padding=1 y stride=1 en las convoluciones (el mapa no se encoge al convolucionar)
    - las convoluciones llevan sesgo (bias) y no hay BatchNorm

Uso:

    import modelos as mo
    modelo = mo.RedBase()
    logits = modelo(x)        # x: (N, 3, 256, 256)  ->  logits: (N,)

Ejecutar este fichero (`python modelos.py`) imprime el tamaño y los parametros
de cada capa y comprueba que el total es 105.761.
"""

from __future__ import annotations

import torch
from torch import nn


def bloque(canales_entrada: int, canales_salida: int, batchnorm: bool = False,
           doble: bool = False) -> nn.Sequential:
    """Un bloque: convolucion 3x3 -> [BatchNorm] -> ReLU -> maxpool 2x2.

    doble=True (conv+conv, la idea del profesor): DOS convoluciones 3x3 (cada una con su
    BatchNorm y su ReLU) antes del maxpool. Ve patrones mas complejos a cada escala y amplia
    el campo receptivo (76 pixeles al final, frente a 46).

    - Conv2d: `canales_salida` filtros de 3x3. Cada filtro mira los `canales_entrada`
      canales a la vez (3x3xcanales_entrada numeros + un sesgo). padding=1 anade un
      borde de ceros para que el mapa conserve su tamaño.
    - BatchNorm2d (opcional): reescala cada canal para que, dentro del lote, tenga media 0
      y varianza 1, y lo reajusta con dos parametros aprendidos (gamma y beta). Estabiliza
      el entrenamiento y ayuda a que el gradiente llegue a las primeras capas. Con BatchNorm
      la convolucion va SIN sesgo: BatchNorm resta la media justo despues y su beta hace de
      sesgo ("orden canonico" de los apuntes: Conv -> BN -> ReLU).
    - ReLU: los valores negativos pasan a 0.
    - MaxPool2d(2): se queda con el maximo de cada cuadradito de 2x2 -> el mapa
      pasa a tener la mitad de alto y de ancho.
    """
    capas = []
    for entrada in ([canales_entrada, canales_salida] if doble else [canales_entrada]):
        capas.append(nn.Conv2d(entrada, canales_salida, kernel_size=3, padding=1, bias=not batchnorm))
        if batchnorm:
            capas.append(nn.BatchNorm2d(canales_salida))
        capas.append(nn.ReLU())
    capas.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*capas)


class RedBase(nn.Module):
    """La red base: 4 bloques, media global, capa densa y un logit.

    batchnorm=False -> la arquitectura de la diapositiva tal cual (105.761 parametros).
    batchnorm=True  -> igual, con BatchNorm tras cada convolucion (106.001 parametros:
                       se quitan los 240 sesgos de las convoluciones y se anaden 480
                       parametros de BatchNorm, gamma y beta por canal).

    realce=True     -> antes del primer bloque se añaden dos canales calculados dentro de la red:
                       EARLY-PRE (realce: cuanto se ilumina el tejido con el contraste) y EARLY-LATE
                       (lavado: si el contraste se va). La entrada sigue siendo (N, 3, 256, 256); la
                       primera convolucion ve 5 canales (+288 parametros). Al calcularlos aqui, la app
                       los obtiene exactamente igual que el entrenamiento.

    filtros=16      -> filtros del primer bloque; cada bloque dobla: 16-32-64-128 (la red del profesor).
                       filtros=8 es la RED ESTRECHA: 8-16-32-64, la mitad de filtros por bloque y
                       ~1/4 de parametros (28.873 con BatchNorm): menos capacidad para memorizar.
                       La media global da entonces 64 numeros (8 x filtros) en vez de 128.

    doble=True      -> conv+conv: dos convoluciones por bloque (302.321 parametros con BatchNorm).

    normalizar=True -> cada corte se reescala con SU PROPIA media y desviacion, las mismas para las
                       tres fases (para no romper el realce): x -> (x - media) / desviacion. Quita las
                       diferencias de brillo y contraste entre escaneres y hospitales (los datos no
                       vienen armonizados). Sin parametros; al hacerse dentro de la red, la app lo
                       aplica exactamente igual.

    n_auxiliares > 0 -> anade una cabeza auxiliar (una capa lineal 8*filtros -> n_auxiliares) que,
                       a partir de las mismas caracteristicas, predice otras etiquetas
                       (el subtipo del tumor: HR y HER2). SOLO se usa durante el entrenamiento
                       (`forward_multitarea`), para que la red aprenda rasgos biologicos del
                       tumor. `forward` sigue devolviendo un unico logit de pCR, y para la app
                       se puede cargar el modelo sin esa cabeza (load_state_dict(strict=False)).
    """

    def __init__(self, dropout: float = 0.5, batchnorm: bool = False, n_auxiliares: int = 0,
                 realce: bool = False, filtros: int = 16, doble: bool = False, normalizar: bool = False):
        super().__init__()
        self.n_auxiliares = n_auxiliares
        self.realce = realce
        self.normalizar = normalizar
        f, d = filtros, doble              # con f = 16 (por defecto), los tamaños de los comentarios
        self.extractor = nn.Sequential(
            bloque(5 if realce else 3, f, batchnorm, d),    # 3 (o 5) @ 256x256  ->  16 @ 128x128
            bloque(f, 2 * f, batchnorm, d),       # 16 @ 128x128 ->  32 @  64x64
            bloque(2 * f, 4 * f, batchnorm, d),   # 32 @ 64x64   ->  64 @  32x32
            bloque(4 * f, 8 * f, batchnorm, d),   # 64 @ 32x32   -> 128 @  16x16
        )
        self.cabeza = nn.Sequential(
            nn.Linear(8 * f, 64),   # cada una de las 64 neuronas mira los 128 numeros (8 x filtros)
            nn.ReLU(),              # decision nuestra: la diapositiva no la indica
            nn.Dropout(dropout),    # durante entrenar apaga al azar el 50 % de las 64
            nn.Linear(64, 1),       # un unico logit
        )
        # cabeza auxiliar (solo entrenamiento): mismas 128 caracteristicas -> HR, HER2
        self.cabeza_aux = nn.Linear(8 * f, n_auxiliares) if n_auxiliares else None

    def caracteristicas(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalizar:          # media y desviacion de CADA corte (las tres fases juntas)
            media = x.mean(dim=(1, 2, 3), keepdim=True)
            desv = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            x = (x - media) / desv
        if self.realce:              # canales 0, 1, 2 = PRE, EARLY, LATE
            x = torch.cat([x, x[:, 1:2] - x[:, 0:1], x[:, 1:2] - x[:, 2:3]], dim=1)   # (N, 5, 256, 256)
        x = self.extractor(x)        # (N, 128, 16, 16)
        return x.mean(dim=(2, 3))    # media global (GAP): cada mapa -> un numero  -> (N, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cabeza(self.caracteristicas(x))   # (N, 1)
        # Se devuelve (N,) y no (N, 1): BCEWithLogitsLoss no avisa si le das logits (N, 1)
        # contra etiquetas (N,), hace un broadcasting a (N, N) y entrena mal sin error.
        return x.squeeze(1)

    def forward_multitarea(self, x: torch.Tensor):
        """Devuelve (logit de pCR (N,), logits auxiliares (N, n_auxiliares) o None)."""
        f = self.caracteristicas(x)
        logit = self.cabeza(f).squeeze(1)
        return logit, (self.cabeza_aux(f) if self.cabeza_aux is not None else None)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())


def imprimir_resumen(modelo: nn.Module, forma_entrada=(1, 3, 256, 256)) -> int:
    """Pasa un tensor de prueba por la red e imprime, capa a capa, el tamaño de la
    salida y los parametros. Devuelve el total de parametros."""
    filas = []

    def anotar(nombre):
        def gancho(modulo, entrada, salida):
            propios = sum(p.numel() for p in modulo.parameters(recurse=False))
            filas.append((nombre, tuple(salida.shape[1:]), propios))
        return gancho

    ganchos = []
    for nombre, modulo in modelo.named_modules():
        if isinstance(modulo, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Linear, nn.Dropout)):
            ganchos.append(modulo.register_forward_hook(anotar(nombre)))
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.zeros(*forma_entrada))
    for g in ganchos:
        g.remove()

    print(f"{'capa':<24}{'tipo':<13}{'forma de salida':<20}{'parametros':>11}")
    print(f"{'(entrada)':<24}{'':<13}{str(tuple(forma_entrada[1:])):<20}{'':>11}")
    tipos = {nombre: type(m).__name__ for nombre, m in modelo.named_modules()}
    ultima_del_extractor = [n for n, _, _ in filas if n.startswith("extractor")][-1]
    for nombre, forma, p in filas:
        print(f"{nombre:<24}{tipos[nombre]:<13}{str(forma):<20}{p:>11,}")
        if nombre == ultima_del_extractor:
            print(f"{'(media global)':<24}{'mean':<13}{'(128,)':<20}{'0':>11}")
    total = contar_parametros(modelo)
    print(f"{'TOTAL':<57}{total:>11,}")
    print(f"salida de la red para 1 corte: {tuple(salida.shape)}")
    return total


if __name__ == "__main__":
    total = imprimir_resumen(RedBase())
    assert total == 105_761, f"se esperaban 105.761 parametros y hay {total:,}"
    print("\nOK: 105.761 parametros, igual que la diapositiva del profesor.\n")
    total_bn = imprimir_resumen(RedBase(batchnorm=True))
    assert total_bn == 106_001, f"con BatchNorm se esperaban 106.001 parametros y hay {total_bn:,}"
    print("\nOK: con BatchNorm, 106.001 parametros (-240 sesgos de las convoluciones, +480 de BatchNorm).\n")
    total_realce = imprimir_resumen(RedBase(batchnorm=True, realce=True))
    assert total_realce == 106_289, f"con realce se esperaban 106.289 parametros y hay {total_realce:,}"
    print("\nOK: con BatchNorm y realce, 106.289 parametros (+288: la primera convolucion ve 5 canales).\n")
    total_estrecha = imprimir_resumen(RedBase(batchnorm=True, filtros=8))
    assert total_estrecha == 28_873, f"la red estrecha deberia tener 28.873 parametros y tiene {total_estrecha:,}"
    print("\nOK: red estrecha (8-16-32-64) con BatchNorm, 28.873 parametros (~1/4 de la del profesor).\n")
    total_doble = imprimir_resumen(RedBase(batchnorm=True, doble=True))
    assert total_doble == 302_321, f"conv+conv deberia tener 302.321 parametros y tiene {total_doble:,}"
    print("\nOK: conv+conv con BatchNorm, 302.321 parametros (~3 veces la del profesor).")
    total_norm = contar_parametros(RedBase(batchnorm=True, normalizar=True))
    assert total_norm == 106_001, f"la normalizacion no deberia anadir parametros ({total_norm:,})"
    print("OK: con normalizacion por corte, 106.001 parametros (no anade ninguno).")
