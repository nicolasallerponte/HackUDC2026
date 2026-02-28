"""
config.py — Hiperparámetros optimizados para GTX 1650 Ti (4GB VRAM)
Modelo elegido: FashionCLIP (openai/clip-vit-base-patch16 fine-tuned en moda)
  → ViT-B/16 es el sweet-spot: potente pero cabe en 4GB con AMP + acumulación de gradientes.
  → Alternativa más ligera: openai/clip-vit-base-patch32 si la VRAM da problemas.
Reranker: BLIP-2 está descartado (demasiado grande). Se usa un MLP ligero sobre
  embeddings concatenados (bundle + product) para rescoring.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CACHE_DIR       = DATA_DIR / "image_cache"
EMBED_DIR       = DATA_DIR / "embeddings"
INDEX_DIR       = DATA_DIR / "faiss_index"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
SUBMISSION_DIR  = BASE_DIR / "submissions"

for d in [DATA_DIR, CACHE_DIR, EMBED_DIR, INDEX_DIR, CHECKPOINT_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Raw data CSVs ────────────────────────────────────────────────────────────
PRODUCT_CSV     = DATA_DIR / "product_dataset.csv"
BUNDLES_CSV     = DATA_DIR / "bundles_dataset.csv"
TRAIN_CSV       = DATA_DIR / "bundles_product_match_train.csv"
TEST_CSV        = DATA_DIR / "bundles_product_match_test.csv"

# ─── Model Selection ──────────────────────────────────────────────────────────
# FashionCLIP: CLIP ViT-B/16 pre-trained on fashion data (Hugging Face Hub)
# Paper: https://arxiv.org/abs/2204.03972
# ~150MB weights, embedding dim = 512
CLIP_MODEL_ID   = "patrickjohncyh/fashion-clip"   # Fashion-specific CLIP
CLIP_EMBED_DIM  = 512

# Fallback si fashion-clip no está disponible:
# CLIP_MODEL_ID = "openai/clip-vit-base-patch16"

# ─── Training / Fine-Tuning ───────────────────────────────────────────────────
# Estrategia: Frozen CLIP backbone + trainable Projection Head (MLP ligero)
# Esto evita saturar VRAM entrenando los 86M parámetros del ViT completo.
# Si quieres LoRA sobre CLIP, instala peft y cambia USE_LORA = True.
USE_LORA            = False   # True → LoRA sobre los attention layers del ViT
LORA_RANK           = 8
LORA_ALPHA          = 16
FREEZE_BACKBONE     = True    # True = solo entrena projection head + neck

TRAIN_BATCH_SIZE    = 16      # 16 pares (anchor, positive) → ~2.5GB VRAM con AMP
GRAD_ACCUM_STEPS    = 4       # Effective batch = 16 * 4 = 64
VAL_SPLIT           = 0.1
EPOCHS              = 15
LR_PROJECTION       = 3e-4    # Head MLP
LR_BACKBONE         = 1e-5    # Solo si FREEZE_BACKBONE = False
WEIGHT_DECAY        = 1e-4
WARMUP_RATIO        = 0.1
MIXED_PRECISION     = True    # torch.cuda.amp — reduce VRAM ~40%
TEMPERATURE         = 0.07    # InfoNCE / SupCon temperature
MARGIN              = 0.3     # Para TripletLoss (modo alternativo)

# ─── Projection Head ─────────────────────────────────────────────────────────
# Neck: CLIP_EMBED_DIM → PROJ_DIM (espacio métrico unificado bundle-product)
PROJ_DIM            = 256
PROJ_HIDDEN         = 512
PROJ_DROPOUT        = 0.1

# ─── Augmentación de datos (moda) ─────────────────────────────────────────────
IMAGE_SIZE          = 224
AUGMENT_TRAIN       = True
# Augmentaciones conservadoras para moda (no flip agresivo en texturas/logos)
COLOR_JITTER        = 0.2
RANDOM_CROP_SCALE   = (0.85, 1.0)
RANDOM_HFLIP_P      = 0.3   # Bajo: simetría importa en bolsos/zapatos

# ─── Indexación FAISS ─────────────────────────────────────────────────────────
FAISS_INDEX_TYPE    = "IVFFlat"   # IVFFlat: buen balance velocidad/precisión
FAISS_NLIST         = 128         # Número de clusters Voronoi (√27600 ≈ 166, usamos 128)
FAISS_NPROBE        = 32          # Celdas a explorar en búsqueda (precisión vs velocidad)
FAISS_METRIC        = "cosine"    # "cosine" o "l2"
INDEX_BATCH_SIZE    = 64          # Batch para generar embeddings del catálogo

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE      = 50   # Recuperar top-50 candidatos en stage 1
TOP_K_RERANK        = 15   # Output final (métrica del reto: Recall@15)
USE_SECTION_FILTER  = True # Filtrar por bundle_id_section antes de buscar

# ─── Descarga paralela de imágenes ────────────────────────────────────────────
DOWNLOAD_WORKERS    = 16
DOWNLOAD_TIMEOUT    = 10
IMAGE_FORMAT        = "jpg"

# ─── Reproducibilidad ─────────────────────────────────────────────────────────
SEED                = 42

# ─── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"