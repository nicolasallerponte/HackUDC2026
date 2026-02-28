"""
model_factory.py — Arquitectura SOTA adaptada a 4GB VRAM.

═══════════════════════════════════════════════════════════════
DECISIÓN DE MODELOS (justificación técnica):
═══════════════════════════════════════════════════════════════

Stage 1 — Embedding Backbone: FashionCLIP (ViT-B/16)
  • Base: openai/clip-vit-base-patch16 (86M params)
  • Fine-tuned en 700k+ imágenes de moda (Farfetch dataset)
  • VRAM en inferencia: ~1.2GB | entrenamiento con AMP: ~2.8GB (frozen) / ~3.8GB (LoRA)
  • Embedding dim: 512 → proyectado a 256 (espacio métrico unificado)
  • Alternativa descartada: ViT-L/14 (demasiado grande, 307M params, >4GB solo el modelo)

Stage 2 — Reranker: CrossEncoder MLP ligero
  • Input: concat(bundle_emb, product_emb, |bundle_emb - product_emb|, bundle_emb * product_emb) = 1024-dim
  • 3 capas FC → score escalar
  • Entrena simultáneamente con contrastive loss en stage 1
  • NO se usa BLIP-2 / LLaVA (>>4GB VRAM)

Estrategia de Fine-Tuning:
  • Backbone FROZEN (por defecto) → solo Projection Head es entrenable
  • Opción LoRA: adapters rank-8 sobre Q,V projections del ViT (añade ~1.5M params)
  • InfoNCE loss + opcionalmente TripletLoss
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from config import *

# ─── Projection Head (Neck entrenable) ───────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    MLP que proyecta embeddings CLIP al espacio métrico unificado.
    Entrena aunque el backbone esté frozen.
    """
    def __init__(self, in_dim=CLIP_EMBED_DIM, proj_dim=PROJ_DIM, hidden=PROJ_HIDDEN, dropout=PROJ_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ─── FashionCLIP + Projection Head ───────────────────────────────────────────

class FashionEmbedder(nn.Module):
    """
    Wrapper de FashionCLIP con projection head.
    Produce embeddings L2-normalizados de dim=PROJ_DIM para imágenes
    de bundles Y de productos en el mismo espacio métrico.
    """

    def __init__(self, model_id=CLIP_MODEL_ID, freeze_backbone=FREEZE_BACKBONE):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.projection = ProjectionHead()

        if freeze_backbone:
            self._freeze_backbone()

        # Estadísticas de parámetros
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FashionEmbedder] Params total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.2f}M")

    def _freeze_backbone(self):
        for param in self.clip.parameters():
            param.requires_grad = False
        # Descongelar los últimos 2 transformer blocks del vision encoder
        # Esto permite cierta adaptación sin reventar VRAM
        vision_layers = self.clip.vision_model.encoder.layers
        for layer in vision_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        print("[FashionEmbedder] Backbone frozen (últimas 2 capas ViT desbloqueadas)")

    def encode_image_tensor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encoda tensor ya procesado (B, 3, H, W) → (B, PROJ_DIM)."""
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        # Tomar el token [CLS]
        clip_emb = self.clip.visual_projection(vision_out.pooler_output)
        return self.projection(clip_emb)

    def forward(self, bundle_pixels: torch.Tensor, product_pixels: torch.Tensor):
        """Para training: retorna (bundle_embs, product_embs) ambos (B, PROJ_DIM)."""
        b_emb = self.encode_image_tensor(bundle_pixels)
        p_emb = self.encode_image_tensor(product_pixels)
        return b_emb, p_emb


# ─── LoRA (opcional) ─────────────────────────────────────────────────────────

def apply_lora_to_model(model: FashionEmbedder) -> FashionEmbedder:
    """
    Aplica LoRA sobre las capas Q y V del vision encoder.
    Requiere: pip install peft
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("Instala peft: pip install peft")

    # Descongelar backbone para LoRA
    for param in model.clip.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        # Targets: Q y V projections de cada bloque del ViT
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model.clip = get_peft_model(model.clip, lora_config)
    model.clip.print_trainable_parameters()
    return model


# ─── Reranker MLP ────────────────────────────────────────────────────────────

class MLPReranker(nn.Module):
    """
    CrossEncoder ligero para reranking en Stage 2.
    Input: [bundle_emb; product_emb; diff; hadamard] → (4 * PROJ_DIM,)
    Output: score escalar (logit)
    
    Se entrena con BCEWithLogitsLoss usando pares positivos/negativos.
    """
    def __init__(self, emb_dim=PROJ_DIM):
        super().__init__()
        in_dim = emb_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, bundle_emb: torch.Tensor, product_emb: torch.Tensor) -> torch.Tensor:
        diff     = bundle_emb - product_emb
        hadamard = bundle_emb * product_emb
        x = torch.cat([bundle_emb, product_emb, diff, hadamard], dim=-1)
        return self.net(x).squeeze(-1)   # (B,)


# ─── Loss Functions ───────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent loss para contrastive learning.
    In-batch negatives: con batch_size=64 efectivo tenemos 63 negativos por par.
    """
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, bundle_embs: torch.Tensor, product_embs: torch.Tensor) -> torch.Tensor:
        # bundle_embs: (B, D), product_embs: (B, D) — ya normalizados
        logits = torch.matmul(bundle_embs, product_embs.T) / self.temperature  # (B, B)
        labels = torch.arange(len(bundle_embs), device=bundle_embs.device)
        # Simétrica: bundle→product y product→bundle
        loss_b2p = F.cross_entropy(logits, labels)
        loss_p2b = F.cross_entropy(logits.T, labels)
        return (loss_b2p + loss_p2b) / 2


class CombinedLoss(nn.Module):
    """InfoNCE para embedder + BCE para reranker."""
    def __init__(self, alpha=0.7):
        super().__init__()
        self.infonce = InfoNCELoss()
        self.bce     = nn.BCEWithLogitsLoss()
        self.alpha   = alpha   # peso de InfoNCE

    def forward(self, bundle_embs, product_embs, rerank_scores=None, rerank_labels=None):
        loss_contrastive = self.infonce(bundle_embs, product_embs)
        if rerank_scores is not None and rerank_labels is not None:
            loss_rerank = self.bce(rerank_scores, rerank_labels.float())
            return self.alpha * loss_contrastive + (1 - self.alpha) * loss_rerank, loss_contrastive, loss_rerank
        return loss_contrastive, loss_contrastive, torch.tensor(0.0)


# ─── Factory function ─────────────────────────────────────────────────────────

def build_models(use_lora=USE_LORA):
    embedder = FashionEmbedder(freeze_backbone=FREEZE_BACKBONE)
    if use_lora:
        embedder = apply_lora_to_model(embedder)
    reranker = MLPReranker()
    return embedder.to(DEVICE), reranker.to(DEVICE)


def load_checkpoint(embedder, reranker, path: str):
    ckpt = torch.load(path, map_location=DEVICE)
    embedder.load_state_dict(ckpt["embedder"])
    reranker.load_state_dict(ckpt["reranker"])
    return ckpt.get("epoch", 0), ckpt.get("best_recall", 0.0)


def save_checkpoint(embedder, reranker, epoch, best_recall, path: str):
    torch.save({
        "embedder": embedder.state_dict(),
        "reranker": reranker.state_dict(),
        "epoch": epoch,
        "best_recall": best_recall,
    }, path)


if __name__ == "__main__":
    embedder, reranker = build_models()
    # Test forward pass
    dummy_bundle  = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    dummy_product = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    b_emb, p_emb = embedder(dummy_bundle, dummy_product)
    scores = reranker(b_emb, p_emb)
    print(f"Bundle embs: {b_emb.shape} | Product embs: {p_emb.shape} | Rerank scores: {scores.shape}")
    print(" model_factory OK")