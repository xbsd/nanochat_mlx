#!/bin/bash
# Full NanoChat MLX training run
# Trains a depth=12 model (286M params) on fineweb + smoltalk
# Expected: ~32GB memory on M4 Mac with 128GB
set -e

echo "NanoChat MLX - Full Speedrun"
echo "============================"
echo ""

# Step 1: Download dataset (if not already present)
echo "Step 1: Downloading dataset..."
python -m nanochat_mlx.dataset -n 4  # Start with 4 shards for testing
echo ""

# Step 2: Pretrain base model
echo "Step 2: Pretraining base model (depth=12)..."
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=16 \
    --eval-every=250 \
    --save-every=500 \
    --run=dummy
echo ""

# Step 3: Supervised fine-tuning
echo "Step 3: SFT training..."
python -m scripts.chat_sft \
    --device-batch-size=16 \
    --run=dummy
echo ""

# Step 4: Evaluate
echo "Step 4: Evaluation..."
python -m scripts.chat_cli --prompt "The capital of France is"
echo ""

echo "Speedrun complete!"
