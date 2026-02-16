#!/bin/bash
# Small-scale test with tiny model on CPU
# For verifying the pipeline works before full training
# Completes in ~1 minute, uses <2GB memory
set -e

echo "NanoChat MLX - Small Test"
echo "========================="
echo ""

# Train a tiny model with synthetic-like settings
python -m scripts.base_train \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=512 \
    --num-iterations=20 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run=dummy

echo ""
echo "Small test complete!"
