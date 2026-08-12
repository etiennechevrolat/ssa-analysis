#!/bin/bash
cd ~/Documents/ssa-analysis
export PYTHONPATH=src
for m in cnn_lstm1 small_cnn small_lstm naive_baseline; do
  echo "=== localizer $m ==="
  .venv/bin/python src/ml/train.py task=localizer model=$m train.epochs=30
done
echo "=== ALL DONE ==="
