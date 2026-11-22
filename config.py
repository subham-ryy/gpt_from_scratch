"""Central configuration for the GPT pretraining project.

Contains settings for data streaming, train/validation splitting,
tokenizer training, experiment tracking, monitoring, and model hyperparameters.
"""
import json
from pathlib import Path
import torch

# Base directory for gpt_pretraining
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 1. Dataset Configuration (FineWeb-Edu)
# -----------------------------------------------------------------------------
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT = "train"

# Controlled sample size for exploration and initial pretraining
NUM_DOCUMENTS = 5000
TRAIN_SPLIT_RATIO = 0.90  # 90% train (4,500 docs), 10% validation (500 docs)
RANDOM_SEED = 1337

# Data directory paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TOKENIZED_DATA_DIR = DATA_DIR / "tokenized"

# Document storage files
RAW_DOCUMENTS_FILE = RAW_DATA_DIR / "documents.jsonl"
TRAIN_DOCUMENTS_FILE = PROCESSED_DATA_DIR / "train.jsonl"
VAL_DOCUMENTS_FILE = PROCESSED_DATA_DIR / "val.jsonl"

# -----------------------------------------------------------------------------
# 2. Tokenizer Configuration (Byte-Level BPE)
# -----------------------------------------------------------------------------
TOKENIZER_DIR = BASE_DIR / "tokenizer"
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"
TOKENIZER_CONFIG_FILE = TOKENIZER_DIR / "tokenizer_config.json"

TOKENIZER_VOCAB_SIZE = 16384
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"

# -----------------------------------------------------------------------------
# 3. Experiment Tracking & Monitoring Configuration
# -----------------------------------------------------------------------------
EXPERIMENTS_DIR = BASE_DIR / "experiments"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

TENSORBOARD_LOGGING = True
LOG_SYSTEM_METRICS = True
METRICS_LOG_INTERVAL = 10     # log hardware/metrics every N training steps
EVAL_INTERVAL = 50            # run evaluation and log validation metrics every N steps

# -----------------------------------------------------------------------------
# 4. Model & Training Hyperparameters (Phase 4 Architecture)
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_vocab_size() -> int:
    """Read vocabulary size dynamically from the trained tokenizer configuration."""
    if TOKENIZER_CONFIG_FILE.exists():
        try:
            with open(TOKENIZER_CONFIG_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                return int(meta.get("vocab_size", TOKENIZER_VOCAB_SIZE))
        except Exception:
            pass
    return TOKENIZER_VOCAB_SIZE

# Architectural parameters for Phase 4 GPT (~98.4M parameters)
VOCAB_SIZE = load_vocab_size()   # 16,384 tokens
BLOCK_SIZE = 1024                # Context window length
N_LAYER = 12                     # Number of Transformer layers
N_HEAD = 12                      # Number of attention heads
N_EMBD = 768                     # Embedding dimension
HEAD_DIM = N_EMBD // N_HEAD      # 64 dimension per head
FFN_DIM = 4 * N_EMBD             # 3,072 feed-forward dimension
DROPOUT = 0.1                    # Configurable dropout probability
BIAS = True                      # Include bias in Linear and LayerNorm layers
TIE_WEIGHTS = True               # Tie token embeddings and lm_head weights

# ── Phase 6: Training Engine Configuration ────────────────────────────────
# Micro-batch: sequences per forward pass on one GPU
MICRO_BATCH_SIZE = 4
# Gradient accumulation: how many micro-batches before one optimizer step
GRAD_ACCUM_STEPS = 8
# Effective batch size = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS = 32 sequences

# Optimizer (AdamW) — Track A Continued Pretraining Defaults
LEARNING_RATE = 1.5e-4       # peak learning rate
MIN_LR        = 1.5e-5       # cosine decay floor (1/10 of peak)
WEIGHT_DECAY  = 0.1
BETA1         = 0.9
BETA2         = 0.95
GRAD_CLIP     = 1.0          # max gradient norm before clipping

# LR schedule — Track A 9,000 steps (~2.5 passes over 114M tokens)
WARMUP_STEPS  = 300          # linear warmup from 0 → LEARNING_RATE
MAX_STEPS     = 9000         # total optimizer steps

# Precision
USE_AMP       = True         # BF16 mixed precision via torch.amp.autocast
AMP_DTYPE     = "bfloat16"   # "bfloat16" or "float16"

# Evaluation and checkpointing
EVAL_INTERVAL  = 500         # run validation every N optimizer steps
SAVE_INTERVAL  = 500         # save/update last.pt and best.pt every N optimizer steps
LOG_INTERVAL   = 10          # log metrics every N optimizer steps

# DataLoader workers (0 = main process only, safer on Windows)
NUM_WORKERS    = 0

# Tokenized binary shards
TRAIN_BIN = DATA_DIR / "tokenized" / "train.bin"
VAL_BIN   = DATA_DIR / "tokenized" / "val.bin"

# Legacy alias
BATCH_SIZE = MICRO_BATCH_SIZE

# Text Generation sampling defaults (Track A Upgraded)
GEN_TEMPERATURE = 0.8
GEN_TOP_K = None               # Disabled in favour of Top-P and Min-P
GEN_TOP_P = 0.90
GEN_MIN_P = 0.05
GEN_REPETITION_PENALTY = 1.15
GEN_NO_REPEAT_NGRAM_SIZE = 3
