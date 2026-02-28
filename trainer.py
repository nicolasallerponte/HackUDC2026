"""
trainer.py — Fine-tuning con AMP + acumulación de gradientes + early stopping.

Estrategia de VRAM:
  - torch.cuda.amp (FP16): reduce uso ~40%
  - Gradient accumulation (steps=4): batch efectivo de 64 sin subir batch_size
  - Frozen backbone: solo ~1.5M params activos
  - Gradient checkpointing: opcional si aún hay OOM
"""

import os
import math
import logging
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config import *
from model_factory import build_models, save_checkpoint, load_checkpoint, CombinedLoss
from data_loader import load_dataframes, get_train_val_loaders, download_all_images
from validate import compute_recall_at_k  # se importará tras crear validate.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Construcción de negative pairs para reranker ─────────────────────────────

def build_reranker_labels(bundle_embs: torch.Tensor, product_embs: torch.Tensor):
    """
    Para un batch de pares positivos, genera hard negatives in-batch para el reranker.
    Retorna (scores_input_bundle, scores_input_product, labels).
    """
    B = len(bundle_embs)
    # Positivos: diagonal
    pos_b = bundle_embs
    pos_p = product_embs
    # Negativos: shift circular
    neg_p = torch.roll(product_embs, shifts=1, dims=0)

    all_b = torch.cat([pos_b, pos_b], dim=0)       # (2B, D)
    all_p = torch.cat([pos_p, neg_p], dim=0)       # (2B, D)
    labels = torch.cat([
        torch.ones(B, device=bundle_embs.device),
        torch.zeros(B, device=bundle_embs.device),
    ])
    return all_b, all_p, labels


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(embedder, reranker, loader, optimizer_emb, optimizer_rer,
                    scaler, criterion, scheduler, epoch, grad_accum=GRAD_ACCUM_STEPS):
    embedder.train()
    reranker.train()

    total_loss = 0.0
    optimizer_emb.zero_grad(set_to_none=True)
    optimizer_rer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar):
        bundle_px  = batch["bundle_image"].to(DEVICE, non_blocking=True)
        product_px = batch["product_image"].to(DEVICE, non_blocking=True)

        with autocast(enabled=MIXED_PRECISION):
            b_emb, p_emb = embedder(bundle_px, product_px)
            # Reranker negatives in-batch
            r_b, r_p, r_labels = build_reranker_labels(b_emb.detach(), p_emb.detach())
            rerank_scores = reranker(r_b, r_p)
            loss, loss_c, loss_r = criterion(b_emb, p_emb, rerank_scores, r_labels)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer_emb)
            scaler.unscale_(optimizer_rer)
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), max_norm=1.0)
            scaler.step(optimizer_emb)
            scaler.step(optimizer_rer)
            scaler.update()
            optimizer_emb.zero_grad(set_to_none=True)
            optimizer_rer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * grad_accum
        pbar.set_postfix(loss=f"{loss.item()*grad_accum:.4f}",
                         ce=f"{loss_c.item():.4f}", bce=f"{loss_r.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(embedder, reranker, loader, criterion):
    embedder.eval()
    reranker.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Validating", leave=False):
        bundle_px  = batch["bundle_image"].to(DEVICE, non_blocking=True)
        product_px = batch["product_image"].to(DEVICE, non_blocking=True)

        with autocast(enabled=MIXED_PRECISION):
            b_emb, p_emb = embedder(bundle_px, product_px)
            r_b, r_p, r_labels = build_reranker_labels(b_emb, p_emb)
            rerank_scores = reranker(r_b, r_p)
            loss, _, _ = criterion(b_emb, p_emb, rerank_scores, r_labels)

        total_loss += loss.item()

    return total_loss / len(loader)


# ─── Main training ────────────────────────────────────────────────────────────

def train(resume: bool = False):
    # 1. Data
    products, bundles, train_df, test_df = load_dataframes()
    train_loader, val_loader = get_train_val_loaders(train_df, products, bundles)
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Models
    embedder, reranker = build_models()

    # 3. Optimizers — learning rates diferenciados
    trainable_emb = [p for p in embedder.parameters() if p.requires_grad]
    optimizer_emb = optim.AdamW(trainable_emb, lr=LR_PROJECTION, weight_decay=WEIGHT_DECAY)
    optimizer_rer = optim.AdamW(reranker.parameters(), lr=LR_PROJECTION, weight_decay=WEIGHT_DECAY)

    # 4. Scheduler OneCycleLR
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = OneCycleLR(
        optimizer_emb, max_lr=LR_PROJECTION,
        total_steps=total_steps, pct_start=WARMUP_RATIO,
        anneal_strategy="cos",
    )

    # 5. AMP Scaler
    scaler = GradScaler(enabled=MIXED_PRECISION)

    # 6. Loss
    criterion = CombinedLoss(alpha=0.7)

    start_epoch = 0
    best_recall = 0.0

    if resume:
        ckpt_path = CHECKPOINT_DIR / "best_model.pt"
        if ckpt_path.exists():
            start_epoch, best_recall = load_checkpoint(embedder, reranker, str(ckpt_path))
            log.info(f"Resuming from epoch {start_epoch} | best_recall={best_recall:.4f}")

    # 7. Gradient checkpointing (actívalo si hay OOM)
    # embedder.clip.vision_model.gradient_checkpointing_enable()

    log.info("="*60)
    log.info(f"Training FashionCLIP + MLP Reranker")
    log.info(f"Device: {DEVICE} | AMP: {MIXED_PRECISION} | Grad Accum: {GRAD_ACCUM_STEPS}")
    log.info(f"Effective batch: {TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    log.info("="*60)

    patience = 4
    no_improve = 0

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            embedder, reranker, train_loader,
            optimizer_emb, optimizer_rer, scaler, criterion, scheduler, epoch
        )

        # Val loss
        val_loss = validate_one_epoch(embedder, reranker, val_loader, criterion)

        # Memory info
        if DEVICE == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / 1e6
            log.info(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | VRAM peak: {vram_mb:.0f}MB")
            torch.cuda.reset_peak_memory_stats()
        else:
            log.info(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Checkpoint periódico
        save_checkpoint(embedder, reranker, epoch, best_recall,
                        str(CHECKPOINT_DIR / f"epoch_{epoch:02d}.pt"))

        # Early stopping basado en val_loss (proxy de Recall — reemplazar con Recall@15 real)
        if val_loss < best_recall or best_recall == 0.0:
            best_recall = val_loss
            save_checkpoint(embedder, reranker, epoch, best_recall,
                            str(CHECKPOINT_DIR / "best_model.pt"))
            log.info(f"   Nuevo best checkpoint guardado (epoch {epoch})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping en epoch {epoch} (sin mejora en {patience} epochs)")
                break

    log.info(" Training completo.")
    return embedder, reranker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)