#!/bin/bash
# Points de la grille d'ablation interrompus par l'arret du serveur tmux.
# reference, window192, patch4 et patch16 sont deja produits ; il reste l'axe masquage et
# l'axe profondeur. Memes reglages que src/scripts/ablation_grid.sh, 200 epoques.

set -u
cd ~/Documents/ssa-analysis || exit 1
export PYTHONPATH=src

COMMON="task=pretrain model=MAEv2 data=spacetrack data.dataset=leo data.stride=31 \
data.revin_norm=False data.revin_per_window_norm=True data.revin_sigma_floor=0.0013 \
train.epochs=200 wandb.enabled=true"

run () {
    local tag="$1"; shift
    echo "=== [$(date +%H:%M:%S)] $tag : $* ==="
    .venv/bin/python src/ml/train.py $COMMON "$@" > "/tmp/abl_${tag}.log" 2>&1
    echo "=== [$(date +%H:%M:%S)] $tag termine (code $?) ==="
}

run mask045 data.window_size=256 model.patch_size=8 model.masking_ratio=0.45
run mask030 data.window_size=256 model.patch_size=8 model.masking_ratio=0.30
run depth6  data.window_size=256 model.patch_size=8 model.masking_ratio=0.65 \
            model.encoder_n_blocks=6 model.decoder_n_blocks=2

echo "RESTE_DONE" > /tmp/abl_reste_done
