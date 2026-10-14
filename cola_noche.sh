#!/bin/bash
# Cola de experimentos para dejar el PC entrenando solo (fase 2 de HOJA_DE_RUTA.md).
#
# Uso, desde la carpeta del proyecto y con el entorno sin activar:
#     nohup bash cola_noche.sh > cola_noche.log 2>&1 &
# Para ver por donde va:
#     tail -n 5 cola_noche.log
#
# Todos los experimentos parten de la mejor configuracion (2.1: BatchNorm, lr 1e-4, perdida normal)
# y cambian UNA sola cosa. Si un experimento falla, la cola sigue con el siguiente. Si un experimento
# ya tiene resultados, entrenar.py se niega a sobrescribirlos y la cola pasa al siguiente.

cd "$(dirname "$0")"
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0

correr() {      # correr <nombre> <opciones de entrenar.py...>: los 5 folds de una configuracion
    local nombre="$1"; shift
    echo "=== $(date '+%d/%m %H:%M')  EMPIEZA $nombre: $*"
    for k in 0 1 2 3 4; do
        python entrenar.py --batchnorm --lr 1e-4 --fold "$k" "$@"
    done
    echo "=== $(date '+%d/%m %H:%M')  TERMINA $nombre"
}

# --- Parte 1: codigo ya probado ---
correr "2.3 red estrecha"            --perdida normal --filtros 8
correr "2.4 aumentado geometrico"    --perdida normal --aumentado geometrico
correr "3.1 perdida ponderada"       --perdida ponderada

# --- Parte 2: experimentos que se suban mas tarde (si no hay, no pasa nada) ---
echo "=== $(date '+%d/%m %H:%M')  git pull para la parte 2"
git pull
if [ -f cola_noche_2.sh ]; then
    source cola_noche_2.sh
else
    echo "=== no hay cola_noche_2.sh: fin de la cola"
fi
echo "=== $(date '+%d/%m %H:%M')  COLA TERMINADA"
