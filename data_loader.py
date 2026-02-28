"""
data_loader.py — Pipeline de datos con descarga paralela y aumentación para moda.

Flujo:
  1. Lee los 4 CSVs y construye un DataFrame maestro.
  2. Descarga imágenes en paralelo con ThreadPoolExecutor (caché local).
  3. Expone PyTorch Datasets para:
     - Catálogo de productos (inferencia/indexación)
     - Pares Bundle-Product para entrenamiento contrastivo
     - Bundles de test para inferencia final
"""

import os
import csv
import time
import hashlib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFile
import requests
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Augmentaciones ───────────────────────────────────────────────────────────

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=RANDOM_CROP_SCALE),
        transforms.RandomHorizontalFlip(p=RANDOM_HFLIP_P),
        transforms.ColorJitter(
            brightness=COLOR_JITTER,
            contrast=COLOR_JITTER,
            saturation=COLOR_JITTER * 0.5,
            hue=0.05
        ),
        transforms.RandomGrayscale(p=0.05),  # raro: colores son clave en moda
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP stats
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 16),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

# ─── Utilidades de descarga ───────────────────────────────────────────────────

# Headers que imitan un navegador real para evitar bloqueos de Zara/CDN
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.zara.com/",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "cross-site",
}

_MAX_RETRIES   = 4
_BACKOFF_BASE  = 1.5   # segundos: 1.5, 2.25, 3.37, 5.06

def _url_to_path(asset_id: str, suffix: str = ".jpg") -> Path:
    return CACHE_DIR / f"{asset_id}{suffix}"

def _download_image(row: tuple) -> tuple[str, bool]:
    """(asset_id, url) → (asset_id, success). Reintentos con backoff exponencial."""
    asset_id, url = row
    dest = _url_to_path(asset_id)
    if dest.exists() and dest.stat().st_size > 0:
        return asset_id, True

    session = requests.Session()
    session.headers.update(_HEADERS)

    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            resp.raise_for_status()

            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            # Verificar que es imagen válida antes de mover
            img = Image.open(tmp)
            img.verify()
            tmp.rename(dest)
            return asset_id, True

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code in (403, 404, 410):
                # No reintentar: recurso no existe o prohibido
                log.debug(f"HTTP {code} permanente para {asset_id}")
                break
            wait = _BACKOFF_BASE ** attempt
            log.debug(f"HTTP {code} en {asset_id}, reintento {attempt+1} en {wait:.1f}s")
            time.sleep(wait)

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = _BACKOFF_BASE ** attempt
            log.debug(f"Conexión fallida {asset_id} ({e}), reintento {attempt+1} en {wait:.1f}s")
            time.sleep(wait)

        except Exception as e:
            log.debug(f"Error inesperado {asset_id}: {e}")
            break

        finally:
            tmp_path = dest.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()

    if dest.exists():
        dest.unlink()
    return asset_id, False


def download_all_images(df: pd.DataFrame, id_col: str, url_col: str,
                         desc: str = "Downloading") -> list[str]:
    """
    Descarga paralela de imágenes con caché y reintentos.
    Workers reducidos a 8 para no saturar el CDN y evitar rate-limiting.
    """
    # Filtrar las que ya existen
    rows = [
        (aid, url) for aid, url in zip(df[id_col], df[url_col])
        if not _url_to_path(aid).exists()
    ]
    already = len(df) - len(rows)
    if already:
        log.info(f"[{desc}] {already} imágenes ya en caché, descargando {len(rows)} restantes")

    failed = []
    # Usamos 8 workers (no 16) — el CDN de Zara detecta y bloquea ráfagas muy altas
    workers = min(8, DOWNLOAD_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_image, r): r[0] for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            asset_id, ok = fut.result()
            if not ok:
                failed.append(asset_id)

    total = len(df)
    success = total - len(failed)
    log.info(f"[{desc}]  {success}/{total} descargadas | ❌ Fallidas: {len(failed)}")
    if failed:
        # Guardar lista de fallidas para diagnóstico
        fail_path = DATA_DIR / f"failed_{desc.lower().replace(' ', '_')}.txt"
        fail_path.write_text("\n".join(failed))
        log.info(f"  IDs fallidas guardadas en: {fail_path}")
    return failed

def load_image(asset_id: str) -> Image.Image | None:
    path = _url_to_path(asset_id)
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

# ─── Carga de DataFrames maestros ─────────────────────────────────────────────

def load_dataframes():
    """Carga y une todos los CSVs. Retorna (products_df, bundles_df, train_df, test_df)."""
    products = pd.read_csv(PRODUCT_CSV)
    bundles  = pd.read_csv(BUNDLES_CSV)
    train    = pd.read_csv(TRAIN_CSV)
    test     = pd.read_csv(TEST_CSV)

    # Merge section info into train
    train = train.merge(bundles[["bundle_asset_id", "bundle_id_section"]], on="bundle_asset_id", how="left")

    log.info(f"Products: {len(products)} | Bundles: {len(bundles)} | Train pairs: {len(train)} | Test: {len(test)}")
    return products, bundles, train, test

# ─── Datasets PyTorch ─────────────────────────────────────────────────────────

class ProductCatalogDataset(Dataset):
    """Dataset del catálogo completo para generar embeddings (inferencia)."""

    def __init__(self, products_df: pd.DataFrame, transform=None):
        self.df = products_df.reset_index(drop=True)
        self.transform = transform or get_val_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row["product_asset_id"])
        if img is None:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (200, 200, 200))
        img = self.transform(img)
        return {
            "image": img,
            "asset_id": row["product_asset_id"],
            "description": row.get("product_description", ""),
            "section": int(row.get("product_id_section", -1)) if "product_id_section" in row else -1,
        }


class BundleDataset(Dataset):
    """Dataset de bundles (train o test) para inferencia."""

    def __init__(self, bundles_df: pd.DataFrame, transform=None):
        self.df = bundles_df.reset_index(drop=True)
        self.transform = transform or get_val_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row["bundle_asset_id"])
        if img is None:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (200, 200, 200))
        img = self.transform(img)
        return {
            "image": img,
            "asset_id": row["bundle_asset_id"],
            "section": int(row["bundle_id_section"]),
        }


class ContrastivePairsDataset(Dataset):
    """
    Dataset para entrenamiento contrastivo (InfoNCE / SupCon).
    Cada ítem es un par (bundle_image, product_image).
    Negative mining: dentro del mismo batch (in-batch negatives).
    """

    def __init__(self, train_df: pd.DataFrame, products_df: pd.DataFrame,
                 bundles_df: pd.DataFrame, transform=None, augment: bool = True):
        self.pairs = train_df[["bundle_asset_id", "product_asset_id"]].drop_duplicates()
        self.pairs = self.pairs.reset_index(drop=True)
        self.transform = transform or (get_train_transforms() if augment else get_val_transforms())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        bundle_img  = load_image(row["bundle_asset_id"])
        product_img = load_image(row["product_asset_id"])

        if bundle_img is None:
            bundle_img  = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (180, 180, 180))
        if product_img is None:
            product_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (200, 200, 200))

        return {
            "bundle_image":  self.transform(bundle_img),
            "product_image": self.transform(product_img),
            "bundle_id":     row["bundle_asset_id"],
            "product_id":    row["product_asset_id"],
        }


# ─── DataLoaders helper ───────────────────────────────────────────────────────

def get_train_val_loaders(train_df, products_df, bundles_df, seed=SEED):
    torch.manual_seed(seed)
    dataset = ContrastivePairsDataset(train_df, products_df, bundles_df, augment=True)

    val_size  = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Val no augmenta
    val_ds.dataset = ContrastivePairsDataset(train_df, products_df, bundles_df, augment=False)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN_BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


def get_catalog_loader(products_df):
    ds = ProductCatalogDataset(products_df)
    return DataLoader(ds, batch_size=INDEX_BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)


def get_bundle_loader(bundles_df):
    ds = BundleDataset(bundles_df)
    return DataLoader(ds, batch_size=INDEX_BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)


# ─── CLI: descarga de imágenes ────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil

    log.info("Copiando CSVs a data/...")
    for src_name in ["product_dataset.csv", "bundles_dataset.csv",
                     "bundles_product_match_train.csv", "bundles_product_match_test.csv"]:
        src = Path("../") / src_name  # ajusta según tu estructura
        if src.exists():
            shutil.copy(src, DATA_DIR / src_name)

    products, bundles, train, test = load_dataframes()

    log.info("=== Descargando imágenes de productos ===")
    download_all_images(products, "product_asset_id", "product_image_url", "Productos")

    log.info("=== Descargando imágenes de bundles (train + test) ===")
    all_bundles = pd.concat([
        bundles[bundles["bundle_asset_id"].isin(train["bundle_asset_id"])],
        bundles[bundles["bundle_asset_id"].isin(test["bundle_asset_id"])],
    ]).drop_duplicates()
    download_all_images(all_bundles, "bundle_asset_id", "bundle_image_url", "Bundles")

    log.info(" Descarga completa.")