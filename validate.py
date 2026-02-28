"""
validate.py — Validación local con métrica Recall@15.

Uso:
  python validate.py                    # valida con 10% de train como val set
  python validate.py --k 15             # Recall@K personalizado
  python validate.py --split full       # valida en todo train (overfit check)
"""

import numpy as np
import pandas as pd
import torch
import faiss
import logging
from tqdm import tqdm
from pathlib import Path

from config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Métrica ──────────────────────────────────────────────────────────────────

def compute_recall_at_k(predictions: dict[str, list[str]],
                         ground_truth: dict[str, list[str]],
                         k: int = TOP_K_RERANK) -> float:
    """
    Recall@K: fracción de bundles donde al menos 1 producto correcto
    aparece en los top-K predichos.

    predictions: {bundle_id: [product_id_1, ..., product_id_K]}
    ground_truth: {bundle_id: [correct_product_id_1, ...]}
    """
    hits = 0
    total = 0

    for bundle_id, true_products in ground_truth.items():
        if bundle_id not in predictions:
            total += 1
            continue
        pred_set = set(predictions[bundle_id][:k])
        true_set = set(true_products)
        if pred_set & true_set:
            hits += 1
        total += 1

    recall = hits / total if total > 0 else 0.0
    return recall


def compute_map_at_k(predictions: dict[str, list[str]],
                      ground_truth: dict[str, list[str]],
                      k: int = TOP_K_RERANK) -> float:
    """
    mAP@K: más estricta, penaliza predicciones desordenadas.
    """
    aps = []
    for bundle_id, true_products in ground_truth.items():
        if bundle_id not in predictions:
            aps.append(0.0)
            continue
        preds = predictions[bundle_id][:k]
        true_set = set(true_products)
        hits = 0
        prec_sum = 0.0
        for i, pid in enumerate(preds):
            if pid in true_set:
                hits += 1
                prec_sum += hits / (i + 1)
        ap = prec_sum / min(len(true_set), k) if true_set else 0.0
        aps.append(ap)
    return np.mean(aps)


# ─── Evaluación completa ──────────────────────────────────────────────────────

def evaluate(k: int = TOP_K_RERANK, val_fraction: float = 0.1):
    """
    Evaluación local completa: genera predicciones para val set y calcula métricas.
    """
    from inference import run_inference
    from data_loader import load_dataframes

    products_df, bundles_df, train_df, test_df = load_dataframes()

    # Construir ground truth desde train
    gt = train_df.groupby("bundle_asset_id")["product_asset_id"].apply(list).to_dict()

    # Usar val split
    val_bundle_ids = list(gt.keys())
    n_val = max(10, int(len(val_bundle_ids) * val_fraction))
    val_bundle_ids = val_bundle_ids[-n_val:]
    gt_val = {bid: gt[bid] for bid in val_bundle_ids if bid in gt}

    log.info(f"Evaluando en {len(gt_val)} bundles de validación...")

    # Generar predicciones
    results_df = run_inference(split="val")

    predictions = {}
    for _, row in results_df.iterrows():
        bid = row["bundle_asset_id"]
        pids = row["product_asset_ids"].split()
        predictions[bid] = pids

    # Métricas
    recall = compute_recall_at_k(predictions, gt_val, k=k)
    map_k  = compute_map_at_k(predictions, gt_val, k=k)

    log.info("="*50)
    log.info(f"  Recall@{k}  : {recall:.4f} ({recall*100:.1f}%)")
    log.info(f"  mAP@{k}     : {map_k:.4f} ({map_k*100:.1f}%)")
    log.info(f"  Val bundles: {len(gt_val)}")
    log.info("="*50)

    return {"recall": recall, "map": map_k}


# ─── Quick sanity check (sin inferencia real) ─────────────────────────────────

def sanity_check():
    """Verifica que la métrica funciona con predicciones dummy."""
    predictions = {
        "B_001": ["P_A", "P_B", "P_C"],
        "B_002": ["P_X", "P_Y", "P_Z"],
        "B_003": ["P_1", "P_2"],
    }
    ground_truth = {
        "B_001": ["P_B"],    # hit en pos 2
        "B_002": ["P_W"],    # miss
        "B_003": ["P_2"],    # hit en pos 2
    }
    recall = compute_recall_at_k(predictions, ground_truth, k=3)
    expected = 2/3
    assert abs(recall - expected) < 1e-6, f"Expected {expected}, got {recall}"
    log.info(f" Sanity check OK: Recall@3 = {recall:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=int, default=15)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--sanity",   action="store_true")
    args = parser.parse_args()

    if args.sanity:
        sanity_check()
    else:
        evaluate(k=args.k, val_fraction=args.val_frac)