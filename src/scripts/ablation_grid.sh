#!/bin/bash
# Grille d'ablation du preentrainement MAE sur le catalogue LEO.
#
# Un seul facteur bouge a la fois autour du point de reference W=256 / P=8 / masking 0,65 /
# 12 blocs, sous la configuration actuelle : 12 canaux, increments a poids nul, scaler global
# + RevIN par fenetre sur sma_local (plancher 0,0013 = 40 m pour un sigma global de 29,8 km).
#
# 200 epoques pour tous les points, reference comprise : une ablation compare des
# configurations a budget EGAL, elle ne cherche pas la meilleure loss absolue. Les mesures du
# 25/08 montrent que l'essentiel est acquis avant l'epoque 150.
#
# Le run de reference a 500 epoques (2026-08-26_12-28-30) est conserve a part : il repond a la
# question distincte de savoir ce qu'apportent les 300 epoques suivantes.
#
# Usage : tmux new-session -d -s ablation 'bash src/scripts/ablation_grid.sh'

set -u
cd ~/Documents/ssa-analysis || exit 1
export PYTHONPATH=src

EPOCHS=200

COMMON="task=pretrain model=MAEv2 data=spacetrack data.dataset=leo data.stride=31 \
data.revin_norm=False data.revin_per_window_norm=True data.revin_sigma_floor=0.0013 \
train.epochs=$EPOCHS wandb.enabled=true"

run () {
    local tag="$1"; shift
    echo "=== [$(date +%H:%M:%S)] $tag : $* ==="
    .venv/bin/python src/ml/train.py $COMMON "$@" > "/tmp/abl_${tag}.log" 2>&1
    echo "=== [$(date +%H:%M:%S)] $tag termine (code $?) ==="
}

## patch_size doit diviser window_size : 4, 8 et 16 divisent 256 ; 8 divise 192.
run reference data.window_size=256 model.patch_size=8  model.masking_ratio=0.65
run window192 data.window_size=192 model.patch_size=8  model.masking_ratio=0.65
run patch4    data.window_size=256 model.patch_size=4  model.masking_ratio=0.65
run patch16   data.window_size=256 model.patch_size=16 model.masking_ratio=0.65
run mask045   data.window_size=256 model.patch_size=8  model.masking_ratio=0.45
run mask030   data.window_size=256 model.patch_size=8  model.masking_ratio=0.30
run depth6    data.window_size=256 model.patch_size=8  model.masking_ratio=0.65 \
              model.encoder_n_blocks=6 model.decoder_n_blocks=2

echo "GRID_DONE" > /tmp/abl_done
