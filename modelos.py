"""Arquitecturas candidatas para el caso BreastDCEDL.

Las cuatro derivan de la arquitectura de referencia de `presentacion_caso.pdf`
(diapositiva 8): bloques Conv 3x3 + ReLU + MaxPool 2x2, terminando en Global
Average Pooling + una capa densa + 1 logit. Cada variante cambia una sola cosa
respecto a la referencia, para poder atribuir el efecto a ese cambio concreto
(ver DECISIONES.md, D1):

    R     referencia tal cual: canales 16-32-64-128, sin BatchNorm.
    R_BN  + BatchNorm2d tras cada convolucion (bloque canonico de clase).
    R5    + BatchNorm + una 5a etapa de 256 canales (mas campo receptivo).
    RX2   + BatchNorm + el doble de canales en cada etapa (mas capacidad).

Parametros verificados en codigo (deben coincidir exactamente al ejecutar
este fichero, ver el bloque de auto-comprobacion al final):

    R      105.761   (coincide con la cifra de presentacion_caso.pdf)
    R_BN   106.001
    R5     409.617
    RX2    405.409

in_channels=3 siempre: son las tres fases DCE (PRE, EARLY, LATE), nunca color.
Salida: un unico logit por corte, sin sigmoid (BCEWithLogitsLoss ya la aplica
internamente; la sigmoid solo se calcula al evaluar).

Uso:

    import modelos as mo
    modelo = mo.crear_modelo("R_BN")
    logits = modelo(x)   # x: (N, 3, 256, 256) -> logits: (N,)
"""

from __future__ import annotations

import torch
from torch import nn

DROPOUT = 0.5   # capa densa antes del logit (regla de los apuntes de clase)
OCULTA = 64     # neuronas de la capa densa oculta (igual que la referencia)

# canales de salida de cada etapa, y si la etapa lleva BatchNorm2d
ARQUITECTURAS: dict[str, dict] = {
    "R":    {"canales": [16, 32, 64, 128],      "bn": False},
    "R_BN": {"canales": [16, 32, 64, 128],      "bn": True},
    "R5":   {"canales": [16, 32, 64, 128, 256], "bn": True},
    # RX2 DESCARTADA (DECISIONES.md, D6): 1,7x mas lenta y ~4x mas parametros
    # que R_BN, sin ventaja observable. Se conserva solo para poder cargar su
    # checkpoint y reproducir la criba; no se usa en los experimentos nuevos.
    "RX2":  {"canales": [32, 64, 128, 256],     "bn": True},
}


ENTRADAS = ("fases", "realce")


class EntradaRealce(nn.Module):
    """Convierte (PRE, EARLY, LATE) en (PRE, EARLY-PRE, LATE-EARLY).

    Canal 0: la anatomia (la fase sin contraste). Canal 1: cuanto capta
    contraste cada pixel (el realce, que es la señal del problema). Canal 2:
    cuanto cambia despues (lavado o *washout*). Es una combinacion lineal de
    los mismos tres canales: no se pierde ni se añade informacion, solo se le
    da a la red ya calculado lo que tendria que descubrir sola. Sin parametros.

    Va DENTRO del modelo, como primera operacion, y no en el Dataset: asi quien
    use el modelo (el entrenamiento, la evaluacion o la app) le entrega siempre
    PRE, EARLY y LATE tal cual, y el calculo no puede diferir entre
    entrenamiento e inferencia.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (N, 3, H, W)
        pre, early, late = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([pre, early - pre, late - early], dim=1)


class BloqueConv(nn.Module):
    """Conv 3x3 (padding 1) [+ BatchNorm2d] + ReLU + MaxPool 2x2.

    Sin BatchNorm la convolucion lleva sesgo (bias=True): no hay nada que lo
    sustituya. Con BatchNorm, bias=False, porque BatchNorm ya recentra la
    salida (apuntes de clase, 03_CNN1.pdf: "la convolucion no lleva sesgo,
    porque BatchNorm lo sustituye").
    """

    def __init__(self, in_ch: int, out_ch: int, bn: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not bn)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class EmbudoCNN(nn.Module):
    """CNN "embudo": N bloques (Conv+[BN]+ReLU+MaxPool), GAP, densa, 1 logit.

    Nota sobre la forma de salida: a diferencia del fragmento de GUIA.md C4
    (`modelo(x).squeeze(1)`, que asume una salida (N,1)), este modelo ya
    devuelve el logit aplanado a (N,). Hacerlo al reves -- devolver (N,1) y
    olvidar el squeeze en el bucle de entrenamiento -- es el error de forma
    mas peligroso posible aqui: BCEWithLogitsLoss no avisa si le pasas
    logits (N,1) contra etiquetas (N,), simplemente hace broadcasting a
    (N,N) y entrena con una perdida incorrecta sin lanzar ningun error.
    Por eso se aplana aqui, una sola vez, dentro del propio modelo.
    """

    def __init__(self, canales: list[int], bn: bool, in_channels: int = 3,
                 oculta: int = OCULTA, dropout: float = DROPOUT, entrada: str = "fases"):
        super().__init__()
        if entrada not in ENTRADAS:
            raise ValueError(f"entrada debe ser una de {ENTRADAS}, recibido {entrada!r}")
        self.entrada = EntradaRealce() if entrada == "realce" else nn.Identity()
        capas = []
        c_in = in_channels
        for c_out in canales:
            capas.append(BloqueConv(c_in, c_out, bn))
            c_in = c_out
        self.extractor = nn.Sequential(*capas)
        # Global Average Pooling = media de cada canal sobre alto y ancho. Se
        # calcula con .mean() en forward en vez de nn.AdaptiveAvgPool2d(1): el
        # resultado es identico, pero el gradiente de AdaptiveAvgPool2d en CUDA
        # no es determinista y romperia la reproducibilidad con semilla fija.
        self.clasificador = nn.Sequential(
            nn.Linear(c_in, oculta),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(oculta, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entrada(x)            # (N, 3, H, W): fases tal cual, o realce
        x = self.extractor(x)          # (N, C, h, w)
        x = x.mean(dim=(2, 3))         # GAP -> (N, C)
        x = self.clasificador(x)       # (N, 1)
        return x.squeeze(1)            # (N,)


def crear_modelo(nombre: str, entrada: str = "fases", dropout: float = DROPOUT) -> EmbudoCNN:
    """Crea una de las arquitecturas candidatas por su nombre corto.

    `entrada`: "fases" (PRE, EARLY, LATE tal cual) o "realce" (ver EntradaRealce).
    `dropout`: probabilidad de apagado en la capa densa.
    """
    if nombre not in ARQUITECTURAS:
        raise ValueError(f"arquitectura desconocida {nombre!r}; opciones: {list(ARQUITECTURAS)}")
    cfg = ARQUITECTURAS[nombre]
    return EmbudoCNN(canales=cfg["canales"], bn=cfg["bn"], entrada=entrada, dropout=dropout)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())


if __name__ == "__main__":
    # Paso 6 del metodo de clase ("Construir una CNN desde cero"): comprobar
    # formas y parametros con un tensor falso ANTES de entrenar nada de verdad.
    x = torch.randn(2, 3, 256, 256)
    print(f"{'arquitectura':6s}  {'entrada':>8s}  {'parametros':>11s}  {'salida':>10s}")
    for nombre in ARQUITECTURAS:
        for entrada in ENTRADAS:
            m = crear_modelo(nombre, entrada=entrada)
            y = m(x)
            assert y.shape == (2,), f"forma de salida inesperada: {tuple(y.shape)}"
            print(f"{nombre:6s}  {entrada:>8s}  {contar_parametros(m):>11,}  {tuple(y.shape)!s:>10s}")
    # el realce debe ser exactamente (PRE, EARLY-PRE, LATE-EARLY)
    r = EntradaRealce()(x)
    assert torch.equal(r[:, 0], x[:, 0]) and torch.equal(r[:, 1], x[:, 1] - x[:, 0]) \
        and torch.equal(r[:, 2], x[:, 2] - x[:, 1]), "EntradaRealce no calcula lo esperado"
    print("EntradaRealce: (PRE, EARLY-PRE, LATE-EARLY) correcto")
    print("\nSi ves un AssertionError o una excepcion arriba, NO entrenes todavia:")
    print("el fallo es de dimensiones, no de aprendizaje (ver 04_CNN2.pdf, Paso 6).")
