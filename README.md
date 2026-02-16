# NanoChat MLX

A complete GPT language model training, fine-tuning, evaluation, and inference system optimized for **Apple Silicon** using [MLX](https://github.com/ml-explore/mlx). Ported from [Andrej Karpathy's NanoChat](https://github.com/karpathy/nanochat) (PyTorch) to run natively on M-series Macs with unified memory.

Train a 286M-parameter GPT from scratch, fine-tune it on conversations, evaluate on standard benchmarks, and chat with it interactively — all on a single Mac.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Verify Your Setup](#1-verify-your-setup)
  - [Download Data](#2-download-data)
  - [Pretrain a Base Model](#3-pretrain-a-base-model)
  - [Supervised Fine-Tuning](#4-supervised-fine-tuning-sft)
  - [Evaluate](#5-evaluate)
  - [Chat](#6-chat)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
  - [Base Pretraining](#base-pretraining)
  - [Supervised Fine-Tuning](#supervised-fine-tuning)
- [Evaluation](#evaluation)
  - [Base Model Evaluation](#base-model-evaluation)
  - [Chat Model Evaluation](#chat-model-evaluation)
- [Interactive Chat](#interactive-chat)
- [Task System](#task-system)
- [Tokenizer](#tokenizer)
- [Optimizer](#optimizer)
- [Configuration Reference](#configuration-reference)
- [Run Scripts](#run-scripts)
- [Testing](#testing)
- [Key Differences from PyTorch Version](#key-differences-from-pytorch-version)
- [License](#license)

## Features

- **Full training pipeline** — pretrain from scratch on FineWeb-Edu, fine-tune on conversational data, evaluate on benchmarks, chat interactively
- **Apple Silicon native** — built on MLX for M1/M2/M3/M4 Macs with unified memory
- **Modern architecture** — Grouped Query Attention, Sliding Window Attention, RoPE, Squared ReLU, Logit Softcapping, Value Embeddings (ResFormer), per-layer residual scaling
- **Muon optimizer** — hybrid MuonAdamW with 2nd-order Newton-Schulz updates on transformer matrices
- **Efficient data packing** — BOS-aligned best-fit packing achieves 100% token utilization
- **Comprehensive evaluation** — HellaSwag, MMLU, ARC, GSM8K, HumanEval, SpellingBee with CORE metric aggregation
- **Interactive chat** — streaming multi-turn chat with tool use (calculator)
- **Weights & Biases integration** — optional experiment tracking
- **Checkpoint management** — safetensors format with full training state resumption

## Requirements

| Requirement | Version |
|-------------|---------|
| macOS | Apple Silicon (M1/M2/M3/M4) |
| Python | >= 3.10 |
| MLX | >= 0.22.0 |
| Memory | 8GB+ (dry run), 16GB+ (small model), 32GB+ (full 286M) |

> **Note**: This project targets Apple Silicon exclusively. It uses MLX's unified memory model — there are no `.to(device)` calls, no CUDA, no multi-GPU.

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/xbsd/nanochat_mlx.git
cd nanochat_mlx
```

**2. Create a virtual environment (recommended):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the package:**

```bash
# Standard install
pip install -e .

# With development dependencies (pytest)
pip install -e ".[dev]"

# With optional fast tokenizer training
pip install -e ".[rustbpe]"
```

**4. Verify:**

```bash
python -c "import mlx.core as mx; print(f'MLX {mx.__version__} on {mx.default_device()}')"
```

## Quick Start

### 1. Verify Your Setup

Run the dry run to confirm everything works. This creates a tiny model and tests the full pipeline in under 30 seconds using less than 1GB of memory:

```bash
bash runs/dry_run.sh
```

This runs 7 verification stages:
1. Model creation and weight initialization
2. Forward pass shape validation
3. Loss computation
4. Gradient flow verification
5. Optimizer step and loss decrease confirmation
6. Checkpoint save/load roundtrip
7. KV cache inference

### 2. Download Data

Download FineWeb-Edu parquet shards for pretraining:

```bash
# Download 4 shards for a quick test (~400MB each)
python -m nanochat_mlx.dataset -n 4

# Download 100 shards for a real training run
python -m nanochat_mlx.dataset -n 100 -w 16
```

Data is stored in `~/.cache/nanochat/base_data/` by default. Set `NANOCHAT_BASE_DIR` to change the location.

### 3. Pretrain a Base Model

```bash
# Quick test (4 layers, 20 steps, ~1 minute)
python -m scripts.base_train \
    --depth 4 \
    --device-batch-size 1 \
    --num_iterations 20 \
    --eval-every -1 \
    --run my_test

# Full training (12 layers, 286M params)
python -m scripts.base_train \
    --depth 12 \
    --device-batch-size 16 \
    --num_iterations 4578 \
    --warmdown_iters 1450 \
    --eval-every 250 \
    --save-every 500 \
    --run my_base_run
```

Checkpoints are saved to `log/<run_name>/` as `model.safetensors` + `metadata.json`.

### 4. Supervised Fine-Tuning (SFT)

Fine-tune the base model on a conversational data mixture (SmolTalk, MMLU, GSM8K, SpellingBee, and more):

```bash
python -m scripts.chat_sft \
    --checkpoint log/my_base_run/ckpt_final \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --eval-every 100 \
    --save-every 500
```

Training runs for one full epoch through the data mixture, then stops automatically.

### 5. Evaluate

**Base model evaluation:**

```bash
python -m scripts.base_eval \
    --checkpoint log/my_base_run/ckpt_final \
    --tasks bpb hellaswag mmlu arc_easy arc_challenge
```

**Chat model evaluation:**

```bash
python -m scripts.chat_eval \
    --checkpoint log/my_sft_run/ckpt_final \
    --tasks arc_easy arc_challenge mmlu gsm8k humaneval spellingbee
```

### 6. Chat

```bash
# Interactive multi-turn chat
python -m scripts.chat_cli \
    --checkpoint log/my_sft_run/ckpt_final

# Single prompt
python -m scripts.chat_cli \
    --checkpoint log/my_sft_run/ckpt_final \
    --prompt "Explain the theory of relativity in simple terms."

# With custom generation parameters
python -m scripts.chat_cli \
    --checkpoint log/my_sft_run/ckpt_final \
    --temperature 0.7 \
    --top-k 50 \
    --max-tokens 1024
```

## Project Structure

```
nanochat_mlx/
├── nanochat_mlx/                # Core library
│   ├── gpt.py                   # GPT model architecture
│   ├── tokenizer.py             # BPE tokenizer with conversation rendering
│   ├── engine.py                # Inference engine with KV cache
│   ├── optim.py                 # MuonAdamW hybrid optimizer
│   ├── checkpoint_manager.py    # Save/load checkpoints (safetensors)
│   ├── dataloader.py            # BOS-aligned bestfit packing dataloader
│   ├── dataset.py               # FineWeb-Edu dataset downloader
│   ├── common.py                # Utilities, logging, device FLOPS table
│   ├── core_eval.py             # CORE metric evaluation
│   ├── loss_eval.py             # Bits-per-byte evaluation
│   ├── execution.py             # Sandboxed code execution
│   └── report.py                # JSON reporting utility
│
├── scripts/                     # CLI entry points
│   ├── base_train.py            # Base model pretraining
│   ├── base_eval.py             # Base model evaluation
│   ├── chat_sft.py              # Supervised fine-tuning
│   ├── chat_eval.py             # Chat model evaluation
│   ├── chat_cli.py              # Interactive chat interface
│   └── dry_run.py               # Full pipeline verification
│
├── tasks/                       # Training & evaluation tasks
│   ├── common.py                # Task base class, TaskMixture, TaskSequence
│   ├── gsm8k.py                 # GSM8K math reasoning
│   ├── mmlu.py                  # MMLU knowledge benchmark
│   ├── smoltalk.py              # SmolTalk conversational data
│   ├── spellingbee.py           # Spelling & letter counting
│   ├── customjson.py            # Custom JSONL task loader
│   └── ...                      # ARC, HumanEval (inline in eval scripts)
│
├── tests/                       # Unit tests (pytest)
│   ├── test_model.py            # GPT architecture tests
│   ├── test_optimizer.py        # Optimizer tests
│   ├── test_training.py         # Training loop tests
│   └── test_engine.py           # Inference & KV cache tests
│
├── runs/                        # Shell run scripts
│   ├── dry_run.sh               # Quick pipeline verification (<30s)
│   ├── small_test.sh            # Small-scale training test (~1 min)
│   └── speedrun.sh              # Full training pipeline
│
└── pyproject.toml               # Build config & dependencies
```

## Model Architecture

NanoChat MLX implements a modern GPT with several architectural innovations:

### Configuration (Default 286M Model)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_embd` | 768 | Hidden dimension |
| `n_layer` | 12 | Transformer blocks |
| `n_head` | 6 | Query attention heads |
| `n_kv_head` | 6 | Key-value heads (supports GQA when < n_head) |
| `sequence_len` | 2048 | Maximum context length |
| `vocab_size` | 32768 | BPE vocabulary (padded to 64-boundary) |
| `window_pattern` | `"SSSL"` | Per-layer attention window pattern |

### Architectural Components

| Component | Description |
|-----------|-------------|
| **Functional RMSNorm** | Parameter-free normalization: `x * rsqrt(mean(x^2) + eps)` |
| **Rotary Position Embeddings (RoPE)** | Precomputed for 10x buffer; applied to Q/K after projection |
| **Grouped Query Attention (GQA)** | Configurable KV heads; reduces cache memory when `n_kv_head < n_head` |
| **Sliding Window Attention** | `"SSSL"` = layers 0-2 attend to half context (1024), layer 3 attends to full (2048); pattern repeats |
| **Squared ReLU** | `max(x, 0)^2` activation in MLP blocks |
| **Logit Softcapping** | `15 * tanh(logits / 15)` bounds pre-softmax logits to [-15, 15] |
| **Value Embeddings (ResFormer)** | Alternating layers add gated learnable per-token embeddings to V |
| **Per-layer Residual Scaling** | Learned `resid_lambda` and `x0_lambda` scalars per block |
| **Untied Embeddings** | Separate input embedding (`wte`) and output projection (`lm_head`) |

### Weight Initialization

| Layer | Method |
|-------|--------|
| Token embedding | Normal(0, 1) |
| LM head | Normal(0, 0.001) |
| Attention Q/K/V | Uniform(-sqrt(3/d), sqrt(3/d)) |
| Attention output projection | Zeros |
| MLP up-projection | Uniform(-sqrt(3/d), sqrt(3/d)) |
| MLP down-projection | Zeros |
| Value embedding gates | Zeros |
| `resid_lambda` | Ones |
| `x0_lambda` | 0.1 |

## Training Pipeline

### Base Pretraining

Trains a GPT from scratch on the [FineWeb-Edu](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) dataset.

```bash
python -m scripts.base_train [OPTIONS]
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` / `--n_layer` | 12 | Number of transformer layers |
| `--n_embd` | 768 | Model dimension |
| `--batch_size` / `--device-batch-size` | 64 | Micro-batch size |
| `--sequence_length` | 2048 | Context length |
| `--num_iterations` | 4578 | Total training steps |
| `--warmup_iters` | 0 | Linear warmup steps |
| `--warmdown_iters` | 1450 | Cosine warmdown steps |
| `--weight_decay` | 0.0 | Weight decay (ramps during warmdown) |
| `--lr_scale` | 1.0 | Global learning rate multiplier |
| `--grad_accum_steps` | 1 | Gradient accumulation steps |
| `--target_tokens` | None | Token budget (overrides `num_iterations`) |
| `--eval-every` | 250 | Validate every N steps (-1 to disable) |
| `--save-every` | 0 | Checkpoint every N steps (0 to disable) |
| `--save_best` | False | Save on best validation loss |
| `--resume` | None | Resume from checkpoint directory |
| `--wandb_project` | "" | W&B project (empty = disabled) |
| `--seed` | 42 | Random seed |

**Learning rate schedule:**

```
LR
 |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 |  warmup │     constant      │ cosine  │
 |─────────┼────────────────────┼─────────┤→ steps
 0    warmup_iters        total - warmdown   total
```

**Metrics tracked:** training loss, validation BPB (bits per byte), MFU (model FLOPS utilization), memory usage, tokens/sec.

### Supervised Fine-Tuning

Fine-tunes a pretrained base model on a conversation mixture for one epoch:

```bash
python -m scripts.chat_sft [OPTIONS]
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to base model checkpoint |
| `--output-dir` | auto | Output directory |
| `--batch-size` | 4 | Micro-batch size |
| `--grad-accum-steps` | 4 | Gradient accumulation |
| `--sequence-length` | 2048 | Context length |
| `--lr` | None | Override base LR |
| `--weight-decay` | 0.01 | Weight decay |
| `--eval-every` | 100 | Evaluation frequency |
| `--save-every` | 500 | Checkpoint frequency |
| `--wandb` | False | Enable W&B logging |
| `--seed` | 42 | Random seed |

**Data mixture (6 tasks):**

| Task | Source | Purpose |
|------|--------|---------|
| SmolTalk | HuggingFace `smoltalk` | General conversation |
| MMLU | `cais/mmlu` auxiliary train | Knowledge questions |
| GSM8K | `openai/gsm8k` train | Math reasoning |
| CustomJSON | Local JSONL files | Custom data |
| SpellingBee | Synthetic | Letter counting |
| SimpleSpelling | Synthetic | Word spelling |

**Data packing:** Conversations are tokenized with role-specific special tokens, then packed into fixed-length rows using a best-fit algorithm. Each row starts with `<|bos|>`. Padding positions are masked with -1 in targets (ignored by the loss function). This achieves 100% token utilization.

**LR schedule:** Constant for the first 80% of steps, then linear decay to 0 over the final 20%.

## Evaluation

### Base Model Evaluation

Evaluates pretrained models using completion-based scoring:

```bash
python -m scripts.base_eval --checkpoint <path> [--tasks TASK...]
```

| Task | Type | Metric | Method |
|------|------|--------|--------|
| `bpb` | Compression | Bits per byte | CE loss / ln(2) / bytes_per_token |
| `hellaswag` | 4-choice | Accuracy | Log-prob of each completion |
| `mmlu` | 4-choice | Accuracy | Log-prob of A/B/C/D |
| `arc_easy` | Multi-choice | Accuracy | Log-prob of choices |
| `arc_challenge` | Multi-choice | Accuracy | Log-prob of choices |

**BaseCORE** = mean(HellaSwag, MMLU, ARC-Easy, ARC-Challenge)

### Chat Model Evaluation

Evaluates fine-tuned models using both categorical and generative approaches:

```bash
python -m scripts.chat_eval --checkpoint <path> [--tasks TASK...]
```

| Task | Type | Examples | Metric |
|------|------|----------|--------|
| `arc_easy` | Categorical | ~1,400 | Accuracy |
| `arc_challenge` | Categorical | ~1,170 | Accuracy |
| `mmlu` | Categorical | ~14,000 | Accuracy |
| `gsm8k` | Generative | ~1,300 | Accuracy (answer extraction) |
| `humaneval` | Generative | 164 | Heuristic correctness |
| `spellingbee` | Generative | 1,000 | Accuracy |

**ChatCORE** = mean accuracy across all evaluated tasks.

**Additional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-tokens` | 512 | Maximum generation tokens |
| `--temperature` | 0.0 | Sampling temperature |
| `--top-k` | 0 | Top-k sampling (0 = disabled) |
| `--max-examples` | None | Limit examples per task |
| `--verbose` | False | Print per-example results |
| `--output` | None | Save results as JSON |

## Interactive Chat

The chat CLI provides a multi-turn conversational interface with streaming output:

```bash
python -m scripts.chat_cli --checkpoint <path>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Model checkpoint path |
| `--prompt` | None | Single prompt (non-interactive mode) |
| `--system` | None | System prompt |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-k` | 50 | Top-k sampling |
| `--max-tokens` | 1024 | Maximum generation tokens |
| `--no-stream` | False | Disable streaming output |

**Interactive commands:**

| Command | Effect |
|---------|--------|
| `/quit` or `/exit` | Exit the chat |
| `/reset` or `/new` | Clear conversation history |
| `/system <message>` | Set a new system prompt |
| `/temp <value>` | Change sampling temperature |
| `/tokens <value>` | Change max generation tokens |

**Tool use:** The engine can detect `<|python_start|>...<|python_end|>` patterns in generated text and safely evaluate math expressions, injecting results back into the generation stream.

## Task System

All training and evaluation tasks inherit from the `Task` base class in `tasks/common.py`:

```python
class Task(ABC):
    def num_examples(self) -> int: ...      # Total examples
    def get_example(self, index) -> Dict: ...  # Returns {'conversation': [...], 'ideal': ...}
    def evaluate(self, index, output) -> Dict: ... # Returns {'correct': bool, 'score': float}
```

**Composition patterns:**

- **`TaskMixture`** — Shuffles examples from multiple tasks together (used in SFT)
- **`TaskSequence`** — Presents tasks sequentially (for curriculum learning)

**Loading custom data:**

```python
from tasks.customjson import CustomJSON

# JSONL with messages format
task = CustomJSON("path/to/data.jsonl")

# Supported formats per line:
# {"messages": [{"role": "user", "content": "..."}, ...], "ideal": "..."}
# {"question": "...", "answer": "..."}
# {"prompt": "...", "completion": "..."}
```

## Tokenizer

BPE tokenizer with 9 special tokens for conversation formatting:

| Token | Purpose |
|-------|---------|
| `<\|bos\|>` | Beginning of sequence |
| `<\|user_start\|>` / `<\|user_end\|>` | User message boundaries |
| `<\|assistant_start\|>` / `<\|assistant_end\|>` | Assistant message boundaries |
| `<\|python_start\|>` / `<\|python_end\|>` | Code block boundaries |
| `<\|output_start\|>` / `<\|output_end\|>` | Code output boundaries |

**Key functions:**

- `get_tokenizer()` — Load the default tokenizer from `~/.cache/nanochat/tokenizer/`
- `get_token_bytes()` — Per-token byte lengths (for BPB metric)
- `render_conversation(conv)` — Tokenize a conversation with training masks
- `render_for_completion(conv)` — Tokenize for generation (primes with `<|assistant_start|>`)

## Optimizer

**MuonAdamW** is a hybrid optimizer that assigns different algorithms to different parameter types:

| Parameter Type | Algorithm | Default LR |
|----------------|-----------|------------|
| Token embeddings | AdamW | 0.2 * scale |
| LM head | AdamW | 0.004 * scale |
| Value embeddings | AdamW | 0.2 * scale |
| `resid_lambda` scalars | AdamW (beta1=0.8) | 0.005 |
| `x0_lambda` scalars | AdamW (beta1=0.96) | 0.5 |
| 2D weight matrices | Muon | 0.02 |

Where `scale = (n_embd / 768)^-0.5` for dimension-dependent LR scaling.

**Muon features:**
- Newton-Schulz orthogonal projection (5 iterations) for polar decomposition
- Nesterov momentum with configurable ramp-down
- NorMuon per-neuron variance reduction
- Cautious weight decay (only in directions agreeing with gradient)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOCHAT_BASE_DIR` | `~/.cache/nanochat` | Root data/cache directory |

### Directory Layout

```
$NANOCHAT_BASE_DIR/
├── base_data/             # FineWeb-Edu parquet shards
├── tokenizer/             # Tokenizer artifacts (tokenizer.pkl, token_bytes.npy)
├── base_checkpoints/      # Pretrained model checkpoints
├── chatsft_checkpoints/   # SFT model checkpoints
└── report.json            # Evaluation reports
```

### Checkpoint Format

Each checkpoint directory contains:

| File | Format | Contents |
|------|--------|----------|
| `model.safetensors` | Safetensors | Model weights |
| `metadata.json` | JSON | GPTConfig, training step, best loss, dataloader state |
| `optimizer.npz` | NumPy NPZ | Optimizer state (optional, for resuming) |

### Apple Silicon FLOPS Reference

The MFU (Model FLOPS Utilization) metric uses these peak BF16 TFLOPS estimates:

| Chip | TFLOPS |
|------|--------|
| M4 Ultra | 28.0 |
| M4 Max | 6.8 |
| M4 Pro | 4.0 |
| M3 Ultra | 22.0 |
| M3 Max | 6.0 |
| M2 Ultra | 27.2 |
| M1 Ultra | 21.0 |

## Run Scripts

Pre-configured shell scripts in `runs/`:

### `dry_run.sh` — Pipeline Verification

```bash
bash runs/dry_run.sh
```

Runs the full dry run test suite + pytest unit tests. Completes in <30 seconds, uses <1GB memory.

### `small_test.sh` — Small Training Test

```bash
bash runs/small_test.sh
```

Trains a tiny 4-layer model for 20 steps. Verifies the training loop works end-to-end. Completes in ~1 minute, uses <2GB memory.

### `speedrun.sh` — Full Training Pipeline

```bash
bash runs/speedrun.sh
```

Runs the complete pipeline: dataset download, base pretraining (12 layers), SFT, and a chat demo.

## Testing

Run the unit test suite:

```bash
# All tests
python -m pytest tests/ -v --timeout=60

# Individual test modules
python -m pytest tests/test_model.py -v      # Architecture tests
python -m pytest tests/test_optimizer.py -v   # Optimizer tests
python -m pytest tests/test_training.py -v    # Training loop tests
python -m pytest tests/test_engine.py -v      # Inference & KV cache tests
```

**Test coverage:**

| Module | Tests |
|--------|-------|
| `test_model.py` | Config creation, RMSNorm, RoPE, attention, MLP, forward pass shapes, gradients, FLOPS, sliding window |
| `test_optimizer.py` | Optimizer creation, update step, loss decrease, LR scheduling |
| `test_training.py` | Single step, loss decrease over 10 steps, gradient accumulation equivalence |
| `test_engine.py` | KV cache CRUD, position tracking, greedy/top-k sampling, prefill + autoregressive consistency |

## Key Differences from PyTorch Version

| Aspect | PyTorch (NanoChat) | MLX (NanoChat MLX) |
|--------|--------------------|--------------------|
| Compute | CUDA / multi-GPU | Apple Silicon unified memory |
| Autograd | `loss.backward()` | `nn.value_and_grad()` + tree map |
| Compilation | `torch.compile` | Not applicable |
| Distributed | DDP | Single device only |
| FP8 / Autocast | Supported | Not available |
| KV Cache | In-place updates | Concatenation (immutable arrays) |
| Weight Format | PyTorch pickle (`.pt`) | Safetensors (`.safetensors`) |
| Memory Tracking | `torch.cuda.memory_allocated` | `psutil` RSS |
| Device Management | `.to(device)` | Automatic (unified memory) |

## License

MIT
