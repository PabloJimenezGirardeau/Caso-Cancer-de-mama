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


def bloque(canales_entrada: int, canales_salida: int) -> nn.Sequential:
    """Un bloque: convolucion 3x3 -> ReLU -> maxpool 2x2.

    - Conv2d: `canales_salida` filtros de 3x3. Cada filtro mira los `canales_entrada`
      canales a la vez (3x3xcanales_entrada numeros + un sesgo). padding=1 anade un
      borde de ceros para que el mapa conserve su tamaño.
    - ReLU: los valores negativos pasan a 0.
    - MaxPool2d(2): se queda con el maximo de cada cuadradito de 2x2 -> el mapa
      pasa a tener la mitad de alto y de ancho.
    """
    return nn.Sequential(
        nn.Conv2d(canales_entrada, canales_salida, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class RedBase(nn.Module):
    """La red base: 4 bloques, media global, capa densa y un logit."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.extractor = nn.Sequential(
            bloque(3, 16),      # 3 @ 256x256  ->  16 @ 128x128
            bloque(16, 32),     # 16 @ 128x128 ->  32 @  64x64
            bloque(32, 64),     # 32 @ 64x64   ->  64 @  32x32
            bloque(64, 128),    # 64 @ 32x32   -> 128 @  16x16
        )
        self.cabeza = nn.Sequential(
            nn.Linear(128, 64),     # cada una de las 64 neuronas mira los 128 numeros
            nn.ReLU(),              # decision nuestra: la diapositiva no la indica
            nn.Dropout(dropout),    # durante entrenar apaga al azar el 50 % de las 64
            nn.Linear(64, 1),       # un unico logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)        # (N, 128, 16, 16)
        x = x.mean(dim=(2, 3))       # media global (GAP): cada mapa -> un numero  -> (N, 128)
        x = self.cabeza(x)           # (N, 1)
        # Se devuelve (N,) y no (N, 1): BCEWithLogitsLoss no avisa si le das logits (N, 1)
        # contra etiquetas (N,), hace un broadcasting a (N, N) y entrena mal sin error.
        return x.squeeze(1)


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
        if isinstance(modulo, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear, nn.Dropout)):
            ganchos.append(modulo.register_forward_hook(anotar(nombre)))
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.zeros(*forma_entrada))
    for g in ganchos:
        g.remove()

    print(f"{'capa':<24}{'tipo':<10}{'forma de salida':<20}{'parametros':>11}")
    print(f"{'(entrada)':<24}{'':<10}{str(tuple(forma_entrada[1:])):<20}{'':>11}")
    tipos = {nombre: type(m).__name__ for nombre, m in modelo.named_modules()}
    for nombre, forma, p in filas:
        print(f"{nombre:<24}{tipos[nombre]:<10}{str(forma):<20}{p:>11,}")
        if nombre == "extractor.3.2":
            print(f"{'(media global)':<24}{'mean':<10}{'(128,)':<20}{'0':>11}")
    total = contar_parametros(modelo)
    print(f"{'TOTAL':<54}{total:>11,}")
    print(f"salida de la red para 1 corte: {tuple(salida.shape)}")
    return total


if __name__ == "__main__":
    total = imprimir_resumen(RedBase())
    assert total == 105_761, f"se esperaban 105.761 parametros y hay {total:,}"
    print("\nOK: 105.761 parametros, igual que la diapositiva del profesor.")
