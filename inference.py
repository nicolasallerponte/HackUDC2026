"""
inference.py — Pipeline completo de inferencia.

Flujo para cada bundle:
  1. Codificar bundle image → embedding (FashionCLIP + Projection Head)
  2. Stage 1 — Retrieval: buscar top-50 en índice FAISS de su sección
     (fallback a índice global si sección desconocida)
  3. Stage 2 — Reranking: MLP reranker sobre los 50 candidatos
  4. Output top-15 → submission.csv (formato del reto)
"""

import numpy as np
import pandas as pd
import torch
import faiss
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast

from config import *
from model_factory import build_models, load_checkpoint
from data_loader import load_dataframes, get_bundle_loader
from indexer import load_indexes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Stage 1: FAISS Retrieval ─────────────────────────────────────────────────

def retrieve_candidates(
    bundle_emb: np.ndarray,      # (1, PROJ_DIM) float32, ya normalizado
    section: int,
    index_dict: dict,
    top_k: int = TOP_K_RETRIEVE,
) -> list[str]:
    """Busca top_k candidatos para un bundle dado su embedding y sección."""

    # Filtrado jerárquico: usar índice de sección si existe
    if USE_SECTION_FILTER and section in index_dict:
        idx_data = index_dict[section]
    else:
        idx_data = index_dict["global"]

    index  = idx_data["index"]
    ids    = idx_data["ids"]

    actual_k = min(top_k, index.ntotal)
    D, I = index.search(bundle_emb, actual_k)

    candidates = [ids[i] for i in I[0] if i >= 0]
    return candidates


# ─── Stage 2: MLP Reranking ───────────────────────────────────────────────────

@torch.no_grad()
def rerank_candidates(
    bundle_emb_tensor: torch.Tensor,      # (1, PROJ_DIM)
    candidate_ids: list[str],
    product_emb_map: dict[str, np.ndarray],
    reranker,
    top_k: int = TOP_K_RERANK,
) -> list[str]:
    """Reordena candidatos usando MLP reranker."""
    if not candidate_ids:
        return []

    # Stack embeddings de candidatos
    cand_embs = np.stack([
        product_emb_map.get(cid, np.zeros(PROJ_DIM, dtype=np.float32))
        for cid in candidate_ids
    ])  # (N_cand, PROJ_DIM)

    cand_tensor = torch.from_numpy(cand_embs).to(DEVICE)
    bundle_rep  = bundle_emb_tensor.expand(len(candidate_ids), -1)  # (N_cand, PROJ_DIM)

    with autocast(enabled=MIXED_PRECISION):
        scores = reranker(bundle_rep, cand_tensor)  # (N_cand,)

    scores = scores.cpu().float().numpy()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [candidate_ids[i] for i in top_indices]


# ─── Pipeline completo ────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(split: str = "test") -> pd.DataFrame:
    """
    split: "test" → inferencia en bundles del test set
           "val"  → inferencia en bundles de validación (para calcular Recall@15)
    """
    products_df, bundles_df, train_df, test_df = load_dataframes()

    # Seleccionar bundles objetivo
    if split == "test":
        target_ids = test_df["bundle_asset_id"].tolist()
    else:
        # Usar los últimos 10% de train como val local
        val_bundles = train_df["bundle_asset_id"].unique()
        n_val = max(1, int(len(val_bundles) * 0.1))
        target_ids = val_bundles[-n_val:].tolist()

    target_bundles = bundles_df[bundles_df["bundle_asset_id"].isin(target_ids)].copy()
    log.info(f"Procesando {len(target_bundles)} bundles ({split})")

    # 1. Cargar modelos
    embedder, reranker = build_models()
    best_ckpt = CHECKPOINT_DIR / "best_model.pt"
    if best_ckpt.exists():
        epoch, recall = load_checkpoint(embedder, reranker, str(best_ckpt))
        log.info(f"Checkpoint: epoch={epoch}, best_recall={recall:.4f}")
    embedder.eval()
    reranker.eval()

    # 2. Cargar índices FAISS y embeddings del catálogo
    index_dict, all_prod_ids, all_prod_embs = load_indexes()

    # Normalizar embeddings del catálogo para dot-product = cosine sim
    faiss.normalize_L2(all_prod_embs)
    product_emb_map = {pid: all_prod_embs[i] for i, pid in enumerate(all_prod_ids)}

    # 3. Cargar bundles y generar embeddings
    bundle_loader = get_bundle_loader(target_bundles)
    results = []

    for batch in tqdm(bundle_loader, desc="Inferencia bundles"):
        pixel_values = batch["image"].to(DEVICE, non_blocking=True)
        asset_ids    = batch["asset_id"]
        sections     = batch["section"].tolist()

        with autocast(enabled=MIXED_PRECISION):
            bundle_embs = embedder.encode_image_tensor(pixel_values)  # (B, PROJ_DIM)

        bundle_embs_np = bundle_embs.cpu().float().numpy()
        faiss.normalize_L2(bundle_embs_np)

        for i, (bid, section) in enumerate(zip(asset_ids, sections)):
            b_emb_np = bundle_embs_np[i:i+1]    # (1, PROJ_DIM)
            b_emb_t  = bundle_embs[i:i+1]        # (1, PROJ_DIM) tensor

            # Stage 1: FAISS retrieval
            candidates = retrieve_candidates(b_emb_np, int(section), index_dict)

            # Stage 2: MLP reranking
            top_products = rerank_candidates(b_emb_t, candidates, product_emb_map, reranker)

            results.append({
                "bundle_asset_id": bid,
                "product_asset_ids": " ".join(top_products),  # espacio-separado
            })

    results_df = pd.DataFrame(results)
    return results_df


# ─── Submission ───────────────────────────────────────────────────────────────

def generate_submission():
    results_df = run_inference(split="test")

    submission_path = SUBMISSION_DIR / "submission.csv"
    # Formato: una fila por (bundle, product) — ajusta según spec del reto
    rows = []
    for _, row in results_df.iterrows():
        for prod_id in row["product_asset_ids"].split():
            rows.append({"bundle_asset_id": row["bundle_asset_id"], "product_asset_id": prod_id})

    submission = pd.DataFrame(rows)
    submission.to_csv(submission_path, index=False)
    log.info(f" Submission guardada: {submission_path} ({len(submission)} filas)")
    return submission


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()

    if args.submission or args.split == "test":
        generate_submission()
    else:
        results = run_inference(split=args.split)
        print(results.head())