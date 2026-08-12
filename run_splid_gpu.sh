#!/bin/bash
cd ~/Documents/ssa-analysis
export PYTHONPATH=src
for m in cnn_lstm1 small_cnn small_lstm naive_baseline; do
  echo "=== localizer $m ==="
  .venv/bin/python src/ml/train.py task=localizer model=$m train.epochs=15
done
echo "=== classifier cnn_lstm1 ==="
.venv/bin/python src/ml/train.py task=classifier model=cnn_lstm1 train.epochs=150
echo "=== ALL DONE ==="
