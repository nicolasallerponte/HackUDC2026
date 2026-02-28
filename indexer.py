"""
indexer.py — Genera embeddings del catálogo completo (27k productos) y construye índice FAISS.

Flujo:
  1. Carga modelo desde checkpoint (o base si no hay fine-tuning aún)
  2. Genera embeddings en batches con AMP
  3. Construye índice FAISS IVFFlat particionado por bundle_id_section
  4. Guarda índice + mapeo id→embedding a disco

Índice por sección: permite filtrado jerárquico en inferencia.
"""

import os
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
from data_loader import load_dataframes, get_catalog_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Utilidades FAISS ─────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, metric: str = FAISS_METRIC) -> faiss.Index:
    """
    Construye un índice IVFFlat.
    Para 27k vectores de dim 256:
      - IVFFlat con nlist=128 es óptimo (training ~50k vectors)
      - Búsqueda: nprobe=32 → ~25% celdas exploradas
    """
    d = embeddings.shape[1]

    if metric == "cosine":
        # Normalizar para que producto interno = similitud coseno
        faiss.normalize_L2(embeddings)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)
    else:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, FAISS_NLIST, faiss.METRIC_L2)

    log.info(f"Entrenando índice FAISS ({len(embeddings)} vectores, nlist={FAISS_NLIST})...")
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = FAISS_NPROBE
    log.info(f"Índice FAISS construido: {index.ntotal} vectores")
    return index


# ─── Generación de embeddings ─────────────────────────────────────────────────

@torch.no_grad()
def generate_embeddings(embedder, loader) -> tuple[np.ndarray, list[str]]:
    """
    Genera embeddings para todo el catálogo.
    Retorna (embeddings_array: [N, PROJ_DIM], asset_ids: [N])
    """
    embedder.eval()
    all_embs = []
    all_ids  = []

    for batch in tqdm(loader, desc="Generando embeddings del catálogo"):
        pixel_values = batch["image"].to(DEVICE, non_blocking=True)

        with autocast(enabled=MIXED_PRECISION):
            embs = embedder.encode_image_tensor(pixel_values)

        all_embs.append(embs.cpu().float().numpy())
        all_ids.extend(batch["asset_id"])

    embeddings = np.vstack(all_embs).astype(np.float32)
    log.info(f"Embeddings generados: {embeddings.shape}")
    return embeddings, all_ids


# ─── Indexación por secciones ─────────────────────────────────────────────────

def build_section_indexes(
    embeddings: np.ndarray,
    asset_ids: list[str],
    products_df: pd.DataFrame
) -> dict:
    """
    Construye un índice FAISS SEPARADO por cada bundle_id_section.
    Esto permite el filtrado jerárquico en inferencia:
      buscar solo entre productos de la misma sección que el bundle.

    También construye un índice global como fallback.

    Retorna un dict con estructura:
    {
        "global": {"index": faiss.Index, "ids": [...]},
        1: {"index": faiss.Index, "ids": [...]},
        2: {"index": faiss.Index, "ids": [...]},
        ...
    }
    
    NOTA: products_df no tiene bundle_id_section directamente.
    La sección de un producto se infiere de los pares de entrenamiento.
    Para productos sin sección asignada → van al índice global.
    """
    # Crear mapping product_id → section desde train
    # (Se carga desde train_df en el main)
    pass  # implementado en main


def save_index(index_dict: dict, asset_ids: list[str], embeddings: np.ndarray):
    """Guarda índices y metadatos a disco."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Índice global
    faiss.write_index(index_dict["global"]["index"], str(INDEX_DIR / "global.index"))

    # Índices por sección
    for section, data in index_dict.items():
        if section == "global":
            continue
        faiss.write_index(data["index"], str(INDEX_DIR / f"section_{section}.index"))
        with open(INDEX_DIR / f"section_{section}_ids.pkl", "wb") as f:
            pickle.dump(data["ids"], f)

    # IDs globales y embeddings
    with open(INDEX_DIR / "global_ids.pkl", "wb") as f:
        pickle.dump(index_dict["global"]["ids"], f)

    np.save(str(EMBED_DIR / "product_embeddings.npy"), embeddings)
    with open(EMBED_DIR / "product_ids.pkl", "wb") as f:
        pickle.dump(asset_ids, f)

    log.info(f" Índices guardados en {INDEX_DIR}")


def load_indexes() -> tuple[dict, list[str], np.ndarray]:
    """Carga índices desde disco."""
    index_dict = {}

    # Global
    global_index = faiss.read_index(str(INDEX_DIR / "global.index"))
    global_index.nprobe = FAISS_NPROBE
    with open(INDEX_DIR / "global_ids.pkl", "rb") as f:
        global_ids = pickle.load(f)
    index_dict["global"] = {"index": global_index, "ids": global_ids}

    # Secciones (1, 2, 3, ...)
    for path in INDEX_DIR.glob("section_*.index"):
        section = int(path.stem.replace("section_", ""))
        idx = faiss.read_index(str(path))
        idx.nprobe = FAISS_NPROBE
        ids_path = INDEX_DIR / f"section_{section}_ids.pkl"
        with open(ids_path, "rb") as f:
            ids = pickle.load(f)
        index_dict[section] = {"index": idx, "ids": ids}

    embeddings = np.load(str(EMBED_DIR / "product_embeddings.npy"))
    with open(EMBED_DIR / "product_ids.pkl", "rb") as f:
        all_ids = pickle.load(f)

    log.info(f"Índices cargados: {list(index_dict.keys())}")
    return index_dict, all_ids, embeddings


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    products_df, bundles_df, train_df, test_df = load_dataframes()

    # Construir mapping producto → sección desde pares de entrenamiento
    product_to_section = dict(
        train_df[["product_asset_id", "bundle_id_section"]].dropna()
        .drop_duplicates("product_asset_id")
        .values
    )
    products_df["section"] = products_df["product_asset_id"].map(product_to_section).fillna(-1).astype(int)

    # Cargar modelo
    embedder, reranker = build_models()
    best_ckpt = CHECKPOINT_DIR / "best_model.pt"
    if best_ckpt.exists():
        epoch, recall = load_checkpoint(embedder, reranker, str(best_ckpt))
        log.info(f"Checkpoint cargado: epoch={epoch}, best_recall={recall:.4f}")
    else:
        log.warning("No hay checkpoint — usando pesos base de FashionCLIP")

    embedder.eval()

    # Generar embeddings
    catalog_loader = get_catalog_loader(products_df)
    embeddings, asset_ids = generate_embeddings(embedder, catalog_loader)

    # ID → index mapping
    id_to_idx = {aid: i for i, aid in enumerate(asset_ids)}

    # Construir índices
    index_dict = {}

    # Índice global
    log.info("Construyendo índice GLOBAL...")
    global_embs = embeddings.copy()
    faiss.normalize_L2(global_embs)
    global_index = build_faiss_index(global_embs)
    index_dict["global"] = {"index": global_index, "ids": asset_ids}

    # Índices por sección
    sections = products_df["section"].unique()
    for section in sorted(sections):
        if section == -1:
            continue
        mask = (products_df["section"] == section).values
        sec_ids  = [aid for aid, m in zip(asset_ids, mask) if m]
        sec_idxs = [id_to_idx[aid] for aid in sec_ids]
        sec_embs = embeddings[sec_idxs].copy()

        if len(sec_embs) < FAISS_NLIST:
            # Muy pocos vectores: usar índice plano
            faiss.normalize_L2(sec_embs)
            sec_idx = faiss.IndexFlatIP(sec_embs.shape[1])
            sec_idx.add(sec_embs)
        else:
            sec_idx = build_faiss_index(sec_embs)

        index_dict[int(section)] = {"index": sec_idx, "ids": sec_ids}
        log.info(f"  Sección {section}: {len(sec_ids)} productos")

    # Guardar
    save_index(index_dict, asset_ids, embeddings)
    log.info(" Indexación completa.")


if __name__ == "__main__":
    main()