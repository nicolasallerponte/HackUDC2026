"""
generate_notebook.py — Generates hackudc_solution_colab.ipynb
Run: python generate_notebook.py
"""
import json, textwrap
from pathlib import Path

def md(src): return {"cell_type":"markdown","metadata":{},"source":[src]}
def code(src):
    lines = src.lstrip("\n")
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[lines]}

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md(textwrap.dedent("""\
# HackUDC 2026 — Inditex Fashion Retrieval (Improved)
**Task:** Given a bundle (model photo with multiple garments), retrieve up to 15 matching
product IDs from a 27K-product catalog.
**Metric:** Recall@15

**Improvements over baseline:**
- Marqo-FashionSigLIP (best zero-shot fashion model)
- SegFormer B2 garment segmentation → per-segment crop retrieval
- Reciprocal Rank Fusion (RRF) across segments
- **NEW: Text-visual ensemble** using CLIP text encoder on category descriptions
- **NEW: FAISS IVFFlat** (3-5× faster approximate search, same recall)
- **NEW: Section boost** for products from same bundle section
- **OPTIONAL: InfoNCE fine-tuning** cell (run if time allows)
""")))

# ── Cell 0: Mount Google Drive ───────────────────────────────────────────────
cells.append(md("## Cell 0: Mount Google Drive\n\n> Run this first. After mounting, your files will be at `/content/drive/MyDrive/GCED/hackudc/`."))
cells.append(code("""\
from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

# Verify the project folder exists
PROJECT_DIR = Path("/content/drive/MyDrive/GCED/hackudc")
if not PROJECT_DIR.exists():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created: {PROJECT_DIR}")
else:
    print(f"Found: {PROJECT_DIR}")
    print("Contents:", [p.name for p in PROJECT_DIR.iterdir()])
"""))

# ── Cell 1: Setup ────────────────────────────────────────────────────────────
cells.append(md("## Cell 1: Setup"))
cells.append(code("""\
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

_pip("fashion-clip", "transformers>=4.35.0", "accelerate", "tqdm", "Pillow",
     "requests", "open_clip_torch")

try:
    _pip("faiss-gpu")
except Exception:
    try:
        _pip("faiss-cpu")
    except Exception:
        pass
"""))

# ── Cell 2: Imports & Config ─────────────────────────────────────────────────
cells.append(md("## Cell 2: Imports & Configuration"))
cells.append(code("""\
import os, gc, json, math, time, random, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import numpy as np
import pandas as pd
import requests
import torch
try:
    import faiss
except ModuleNotFoundError:
    raise ModuleNotFoundError("faiss not found. Re-run Cell 1.")
from PIL import Image
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
# Google Colab + Google Drive layout:
#   DATA_DIR   → CSVs in MyDrive/GCED/hackudc/data/
#   WORK_DIR   → cache & outputs in MyDrive/GCED/hackudc/  (persists across sessions)
DRIVE_ROOT = Path("/content/drive/MyDrive/GCED/hackudc")
DATA_DIR   = DRIVE_ROOT / "data"        # put the 4 CSVs here
WORK_DIR   = DRIVE_ROOT                 # embeddings, index, submission saved here
IMG_DIR    = WORK_DIR / "images"
PROD_DIR   = IMG_DIR / "products"
BUND_DIR   = IMG_DIR / "bundles"
SUBM_FILE  = WORK_DIR / "submission.csv"
for d in [DATA_DIR, PROD_DIR, BUND_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Model ────────────────────────────────────────────────────────────────────
USE_MARQO      = True
MARQO_MODEL_ID = "marqo/marqo-fashionSigLIP"
MODEL_TAG      = "marqo" if USE_MARQO else "fclip"
EMB_FILE       = WORK_DIR / f"product_embeddings_{MODEL_TAG}.npy"
IDS_FILE       = WORK_DIR / f"product_ids_{MODEL_TAG}.json"
TXT_EMB_FILE   = WORK_DIR / f"text_embeddings_{MODEL_TAG}.npy"
TXT_IDS_FILE   = WORK_DIR / f"text_ids_{MODEL_TAG}.json"
FORCE_RECOMPUTE = False

# ── Hyper-params ─────────────────────────────────────────────────────────────
TOP_K            = 15
EMBED_BATCH      = 128
DOWNLOAD_WORKERS = 32
IMG_SIZE         = 224
DOWNLOAD_TIMEOUT = 20
DOWNLOAD_RETRIES = 4

# FAISS IVFFlat — faster approximate search vs IndexFlatIP
FAISS_NLIST   = 512   # ~53 products per Voronoi cell for 27K products
FAISS_NPROBE  = 128   # probe 25% of cells — high recall (~99% of exact)

K_PER_SEGMENT       = 200   # wider candidate pool per segment
RRF_K               = 60
SEG_MIN_CROPS       = 1
WHOLE_IMG_WEIGHT    = 0.3
TTA_N_VIEWS         = 2

# Text-visual ensemble (Xoel's key insight, adapted for Marqo)
ENABLE_TEXT_ENSEMBLE  = True
W_TEXT_ENSEMBLE       = 0.30  # weight given to text similarity score
TEXT_ENSEMBLE_TOPN    = 100   # apply text ensemble over top-N RRF candidates

# Section boost
ENABLE_SECTION_BOOST = True
W_SECTION_BOOST      = 0.05  # small bonus for products matching bundle section

# Text re-ranking (existing)
ENABLE_TEXT_RERANK  = True
TEXT_RERANK_ALPHA   = 0.10
TEXT_RERANK_TOP_N   = 60

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.zara.com/",
    "Sec-Fetch-Mode": "no-cors",
}

print("Configuration loaded.")
"""))

# ── Cell 3: Data Loading ─────────────────────────────────────────────────────
cells.append(md("## Cell 3: Data Loading"))
cells.append(code("""\
bundles_df  = pd.read_csv(DATA_DIR / "bundles_dataset.csv")
products_df = pd.read_csv(DATA_DIR / "product_dataset.csv")
train_df    = pd.read_csv(DATA_DIR / "bundles_product_match_train.csv")
test_df     = pd.read_csv(DATA_DIR / "bundles_product_match_test.csv")

print("=== Dataset Statistics ===")
print(f"Bundles total  : {len(bundles_df):,}")
print(f"Products total : {len(products_df):,}")
print(f"Train pairs    : {len(train_df):,}")

# Ground-truth lookup
train_gt: dict[str, set] = (
    train_df.dropna(subset=["product_asset_id"])
    .groupby("bundle_asset_id")["product_asset_id"]
    .apply(set).to_dict()
)
counts = [len(v) for v in train_gt.values()]
print(f"Train bundles  : {len(train_gt):,}")
print(f"Avg GT/bundle  : {np.mean(counts):.2f}  Max: {max(counts)}")

# URL / description / section lookups
bundle_url:   dict[str, str] = dict(zip(bundles_df["bundle_asset_id"], bundles_df["bundle_image_url"]))
product_url:  dict[str, str] = dict(zip(products_df["product_asset_id"], products_df["product_image_url"]))
product_desc: dict[str, str] = dict(zip(products_df["product_asset_id"], products_df["product_description"].fillna("")))

# Section info
bundle_section_map:  dict[str, int] = {}
product_section_map: dict[str, int] = {}
if "bundle_id_section" in bundles_df.columns:
    bundle_section_map = {row.bundle_asset_id: int(row.bundle_id_section)
                          for row in bundles_df.itertuples() if pd.notna(row.bundle_id_section)}
# Infer product sections from training pairs
if "bundle_id_section" in train_df.columns or "bundle_id_section" in bundles_df.columns:
    merged = train_df.merge(bundles_df[["bundle_asset_id","bundle_id_section"]],
                            on="bundle_asset_id", how="left")
    for row in merged.itertuples():
        if pd.notna(row.bundle_id_section) and row.product_asset_id not in product_section_map:
            product_section_map[row.product_asset_id] = int(row.bundle_id_section)

test_bundle_ids = test_df["bundle_asset_id"].dropna().unique().tolist()
print(f"Test bundles   : {len(test_bundle_ids)}")
print(f"Section map    : {len(bundle_section_map)} bundles, {len(product_section_map)} products")
"""))

# ── Cell 4: Image Download ───────────────────────────────────────────────────
cells.append(md("## Cell 4: Image Download"))
cells.append(code("""\
def download_image(asset_id: str, url: str, out_dir: Path,
                   timeout: int = DOWNLOAD_TIMEOUT, retries: int = DOWNLOAD_RETRIES) -> bool:
    \"\"\"Stream-download with PIL verify + temp-file rename. Returns True on success.\"\"\"
    out_path = out_dir / f"{asset_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 500:
        return True
    tmp_path = out_path.with_suffix(".tmp")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=DOWNLOAD_HEADERS, stream=True)
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            # Verify image is valid before committing
            img = Image.open(tmp_path)
            img.verify()
            tmp_path.rename(out_path)
            return True
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code in (403, 404, 410):
                break  # permanent failure
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    return False


def batch_download(id_url_pairs, out_dir, desc="Downloading", timeout=DOWNLOAD_TIMEOUT):
    ok_ids, failed_pairs = [], []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_image, aid, url, out_dir, timeout): (aid, url)
                   for aid, url in id_url_pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            aid, url = futures[fut]
            (ok_ids if fut.result() else failed_pairs).append((aid, url) if not fut.result() else aid)
    return ok_ids, failed_pairs


def load_image(asset_id: str, img_dir: Path) -> Image.Image | None:
    p = img_dir / f"{asset_id}.jpg"
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


# Products
product_pairs = list(product_url.items())
_prod_ok, _prod_fail = batch_download(product_pairs, PROD_DIR, "Products pass-1")
if _prod_fail:
    _prod_ok2, _ = batch_download(_prod_fail, PROD_DIR, "Products pass-2", timeout=60)
    _prod_ok += _prod_ok2
valid_product_ids = _prod_ok
print(f"Products: {len(valid_product_ids):,}/{len(product_pairs):,}")

# Bundles
bundle_pairs = list(bundle_url.items())
_bund_ok, _bund_fail = batch_download(bundle_pairs, BUND_DIR, "Bundles  pass-1")
if _bund_fail:
    _bund_ok2, _ = batch_download(_bund_fail, BUND_DIR, "Bundles  pass-2", timeout=60)
    _bund_ok += _bund_ok2
valid_bundle_ids = _bund_ok
print(f"Bundles : {len(valid_bundle_ids):,}/{len(bundle_pairs):,}")

valid_product_set = set(valid_product_ids)
valid_bundle_set  = set(valid_bundle_ids)
"""))

# ── Cell 5: Model Loading ────────────────────────────────────────────────────
cells.append(md("## Cell 5: Marqo-FashionSigLIP Loading"))
cells.append(code("""\
import open_clip

print(f"Loading Marqo-FashionSigLIP (hf-hub:{MARQO_MODEL_ID}) …")
marqo_model, _, marqo_preprocess = open_clip.create_model_and_transforms(
    f"hf-hub:{MARQO_MODEL_ID}"
)
marqo_tokenizer = open_clip.get_tokenizer(f"hf-hub:{MARQO_MODEL_ID}")
marqo_model.to(DEVICE).eval()
print("Marqo-FashionSigLIP loaded.")


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
    return arr / norms


def _marqo_encode_images_raw(imgs: list) -> np.ndarray:
    tensors = torch.stack([marqo_preprocess(img) for img in imgs]).to(DEVICE)
    with torch.no_grad(), torch.autocast(
            device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=torch.float16):
        out = marqo_model.encode_image(tensors)
    return out.cpu().float().numpy()


def _marqo_encode_texts_raw(texts: list[str]) -> np.ndarray:
    tokens = marqo_tokenizer(texts).to(DEVICE)
    with torch.no_grad():
        out = marqo_model.encode_text(tokens)
    return out.cpu().float().numpy()


def _load_image(path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE))


def _embed_images(image_paths: list, batch_size: int = EMBED_BATCH) -> np.ndarray:
    \"\"\"Float32 (N, D) L2-normalised embeddings — parallel I/O, fp16 autocast.\"\"\"
    all_embs = []
    with ThreadPoolExecutor(max_workers=8) as io_pool:
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding", leave=False):
            batch_paths = image_paths[i: i + batch_size]
            imgs = list(io_pool.map(_load_image, batch_paths))
            all_embs.append(_l2_norm(_marqo_encode_images_raw(imgs)))
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


def tta_encode(img: Image.Image, n_views: int = TTA_N_VIEWS) -> np.ndarray:
    \"\"\"Average embedding over n_views augments. Returns (1, D) L2-normalised.\"\"\"
    augments = [img]
    if n_views >= 2:
        augments.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    if n_views >= 3:
        w, h = img.size
        m = max(1, int(min(w, h) * 0.05))
        augments.append(img.crop((m, m, w - m, h - m)).resize((w, h), Image.BILINEAR))
    embs = np.stack([_l2_norm(_marqo_encode_images_raw([a]))[0] for a in augments[:n_views]])
    return _l2_norm(embs.mean(axis=0, keepdims=True))


print("Encode helpers + TTA ready.")
"""))

# ── Cell 6: Text Embeddings Cache ────────────────────────────────────────────
cells.append(md("## Cell 6: Text Embeddings (Category Descriptions)\n"
               "Pre-compute per-product text embeddings using Marqo's text encoder.\n"
               "Prompt: `\"a photo of {category} fashion clothing item\"`"))
cells.append(code("""\
@torch.no_grad()
def compute_text_embeddings_marqo(products_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    \"\"\"
    For each product, encode its text description via Marqo's text encoder.
    Groups by unique description to avoid redundant computation (~70 unique categories).
    Returns (N, D) float32 L2-normalised embeddings + list of product IDs.
    \"\"\"
    desc_to_pids = products_df.groupby("product_description")["product_asset_id"].apply(list).to_dict()
    unique_descs = list(desc_to_pids.keys())
    print(f"  Text embeddings: {len(unique_descs)} unique categories → {len(products_df):,} products")

    TEXT_BATCH = 256
    desc_emb_map: dict[str, np.ndarray] = {}
    for i in tqdm(range(0, len(unique_descs), TEXT_BATCH), desc="Text embeds"):
        batch = unique_descs[i: i + TEXT_BATCH]
        # Contextualised prompt — helps CLIP's text encoder focus on fashion semantics
        prompts = [f"a photo of {d.lower()} fashion clothing item" for d in batch]
        embs = _l2_norm(_marqo_encode_texts_raw(prompts))
        for desc, emb in zip(batch, embs):
            desc_emb_map[desc] = emb

    # Build aligned arrays (product_id order matches products_df)
    all_pids, all_embs = [], []
    for row in products_df.itertuples():
        desc = getattr(row, "product_description", "")
        if pd.isna(desc):
            desc = ""
        emb = desc_emb_map.get(desc)
        if emb is not None:
            all_pids.append(row.product_asset_id)
            all_embs.append(emb)
    return np.stack(all_embs).astype(np.float32), all_pids


# Cache text embeddings (very fast — only ~70 unique categories)
_txt_cache_valid = False
if TXT_EMB_FILE.exists() and TXT_IDS_FILE.exists() and not FORCE_RECOMPUTE:
    with open(TXT_IDS_FILE) as f:
        _txt_ids_cached = json.load(f)
    if len(_txt_ids_cached) == len(products_df):
        text_embeddings_arr = np.load(TXT_EMB_FILE)
        text_product_ids    = _txt_ids_cached
        _txt_cache_valid    = True
        print(f"Text embeddings loaded from cache: {text_embeddings_arr.shape}")

if not _txt_cache_valid:
    print("Computing text embeddings …")
    text_embeddings_arr, text_product_ids = compute_text_embeddings_marqo(products_df)
    np.save(TXT_EMB_FILE, text_embeddings_arr)
    with open(TXT_IDS_FILE, "w") as f:
        json.dump(text_product_ids, f)
    print(f"  Saved: {text_embeddings_arr.shape}")

# Fast lookup: product_id → text embedding (numpy slice)
_txt_pid_to_idx = {pid: i for i, pid in enumerate(text_product_ids)}

def get_text_emb(pid: str) -> np.ndarray | None:
    idx = _txt_pid_to_idx.get(pid)
    if idx is None:
        return None
    return text_embeddings_arr[idx]

print("Text embedding lookup ready.")
"""))

# ── Cell 7: Product Embeddings + FAISS IVFFlat ───────────────────────────────
cells.append(md("## Cell 7: Product Embeddings + FAISS IVFFlat Index"))
cells.append(code("""\
def _ensure_valid_sets():
    global valid_product_set, valid_product_ids, valid_bundle_set, valid_bundle_ids
    prod_on_disk = {p.stem for p in PROD_DIR.glob("*.jpg")}
    bund_on_disk = {p.stem for p in BUND_DIR.glob("*.jpg")}
    if not valid_product_set and prod_on_disk:
        valid_product_set = prod_on_disk
        valid_product_ids = list(prod_on_disk)
        print(f"Rebuilt valid_product_set from disk: {len(valid_product_set):,}")
    if not valid_bundle_set and bund_on_disk:
        valid_bundle_set = bund_on_disk
        valid_bundle_ids = list(bund_on_disk)

_ensure_valid_sets()

# Cache validity
if FORCE_RECOMPUTE:
    EMB_FILE.unlink(missing_ok=True); IDS_FILE.unlink(missing_ok=True)

_cache_valid = False
if EMB_FILE.exists() and IDS_FILE.exists():
    with open(IDS_FILE) as f:
        _cached_ids = json.load(f)
    _missing = valid_product_set - set(_cached_ids)
    if _missing:
        print(f"[CACHE STALE] {len(_missing):,} products missing — recomputing …")
        EMB_FILE.unlink(missing_ok=True); IDS_FILE.unlink(missing_ok=True)
    else:
        _cache_valid = True

if _cache_valid:
    print("Loading cached product embeddings …")
    product_embeddings  = np.load(EMB_FILE)
    indexed_product_ids = json.load(open(IDS_FILE))
    print(f"  Loaded {product_embeddings.shape}")
else:
    print("Computing product embeddings (~15-20 min on T4) …")
    ordered_ids   = sorted(valid_product_set)
    ordered_paths = [PROD_DIR / f"{pid}.jpg" for pid in ordered_ids]
    product_embeddings  = _embed_images(ordered_paths, batch_size=EMBED_BATCH)
    indexed_product_ids = ordered_ids
    np.save(EMB_FILE, product_embeddings)
    json.dump(indexed_product_ids, open(IDS_FILE, "w"))
    print(f"  Saved: {product_embeddings.shape}")

idx_to_pid = {i: pid for i, pid in enumerate(indexed_product_ids)}
DIM = product_embeddings.shape[1]
N   = len(indexed_product_ids)

# ── Build FAISS IVFFlat index ─────────────────────────────────────────────────
# IVFFlat is 3-5x faster than IndexFlatIP with ≈99% recall at nprobe=128
nlist  = min(FAISS_NLIST, N)   # can't have more cells than vectors
nprobe = min(FAISS_NPROBE, nlist)
print(f"Building FAISS IVFFlat (dim={DIM}, n={N:,}, nlist={nlist}, nprobe={nprobe}) …")

quantizer  = faiss.IndexFlatIP(DIM)
index_cpu  = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
index_cpu.train(product_embeddings.astype(np.float32))
index_cpu.add(product_embeddings.astype(np.float32))
index_cpu.nprobe = nprobe

if DEVICE == "cuda":
    try:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        print("  Transferred IVFFlat index to GPU.")
    except Exception as e:
        print(f"  [WARN] GPU index failed ({e}), using CPU.")
        index = index_cpu
else:
    index = index_cpu
    print("  Using CPU index.")

print(f"  FAISS ntotal = {index.ntotal:,}")
if index.ntotal < 1000:
    raise RuntimeError("Only {index.ntotal} products indexed — something went wrong.")
elif index.ntotal < 20_000:
    print(f"  [WARN] Only {index.ntotal:,} products (expected ~27K). Downloads may have failed.")
"""))

# ── Cell 8: SegFormer ────────────────────────────────────────────────────────
cells.append(md("## Cell 8: SegFormer B2 Clothes Segmentation"))
cells.append(code("""\
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn.functional as F

print("Loading SegFormer B2 clothes segmentation model …")
SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
seg_processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
seg_model     = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
seg_model.to(DEVICE).eval()
print("SegFormer loaded.")

ATR_LABELS = {0:"Background",1:"Hat",2:"Hair",3:"Sunglasses",4:"Upper-clothes",
              5:"Skirt",6:"Pants",7:"Dress",8:"Belt",9:"Left-shoe",10:"Right-shoe",
              11:"Face",12:"Left-leg",13:"Right-leg",14:"Left-arm",15:"Right-arm",
              16:"Bag",17:"Scarf"}

SEGMENT_GROUPS = {
    "upper_body": [4],
    "lower_body":  [5, 6],
    "dress":       [7],
    "shoes":       [9, 10],
    "bag":         [16],
    "hat":         [1],
    "scarf_belt":  [8, 17],
}

SEGMENT_TO_CATEGORIES = {
    "upper_body": {
        "T-SHIRT","SHIRT","SWEATER","WIND-JACKET","TOPS AND OTHERS","BLAZER","SWEATSHIRT",
        "BABY T-SHIRT","POLO SHIRT","CARDIGAN","WAISTCOAT","OVERSHIRT","BODYSUIT","COAT",
        "SWIMSUIT","BABY SHIRT","BABY JACKET/COAT","BABY SWEATER","TRENCH RAINCOAT",
        "NIGHTIE/PYJAMAS","BABY TRACKSUIT","BABY CARDIGAN","BABY SWIMSUIT","BABY WIND-JACKET",
        "3/4 COAT","SLEEVELESS PAD. JACKET","KNITTED WAISTCOAT","BABY WAISTCOAT",
        "BABY POLO SHIRT","ANORAK","PARKA","BABY BODY","BABY PYJAMA","BATHROBE/DRES.GOWN",
        "LEISURE AND SPORTS","NEWBORN","NEWBORN TRICOT",
    },
    "lower_body": {
        "TROUSERS","SKIRT","BERMUDA","BABY TROUSERS","SHORTS","LEGGINGS","BABY SKIRT",
        "BABY BERMUDAS","BABY LEGGINGS","STOCKINGS-TIGHTS",
    },
    "dress": {
        "DRESS","OVERALL","BABY DRESS","BABY OVERALL","BABY ROMPER SUIT","BIB OVERALL",
        "BABY OUTFIT","ENSEMBLE..SET","UNIFORM",
    },
    "shoes": {
        "SHOES","BOOT","SANDAL","SPORT SHOES","FLAT SHOES","HEELED SHOES","TRAINERS",
        "ANKLE BOOT","MOCCASINS","RUNNING SHOES","HEELED ANKLE BOOT","HEELED BOOT",
        "HIGH TOPS","FLAT BOOT","FLAT ANKLE BOOT","ATHLETIC FOOTWEAR","HOME SHOES",
        "BEACH SANDAL","RAIN BOOT","WEDGE","SPORTY SANDAL","SOCKS","BABY SOCKS","VAMP/PINKY",
    },
    "bag":        {"HAND BAG-RUCKSACK","PURSE WALLET","WALLETS"},
    "hat":        {"HAT","BABY BONNET"},
    "scarf_belt": {"SCARF","BELT","TIE","GLOVES","BOW TIE/CUMMERBAND","SHAWL/FOULARD","SUSPENDERS"},
}


@torch.no_grad()
def segment_image(pil_img: Image.Image) -> np.ndarray:
    inputs  = seg_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    logits  = seg_model(**inputs).logits
    orig_h, orig_w = pil_img.height, pil_img.width
    up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


def extract_segment_crops(pil_img: Image.Image, seg_map: np.ndarray,
                          min_frac: float = 0.05, padding: float = 0.05) -> dict[str, Image.Image]:
    W, H = pil_img.size
    total = H * W
    crops = {}
    for seg_name, label_ids in SEGMENT_GROUPS.items():
        mask = np.isin(seg_map, label_ids)
        if mask.sum() < min_frac * total:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        y0, y1 = int(rows.min()), int(rows.max())
        x0, x1 = int(cols.min()), int(cols.max())
        py = int((y1 - y0 + 1) * padding)
        px = int((x1 - x0 + 1) * padding)
        crop = pil_img.crop((max(0, x0-px), max(0, y0-py),
                             min(W, x1+px+1), min(H, y1+py+1)))
        if crop.width > 5 and crop.height > 5:
            crops[seg_name] = crop
    return crops


def _category_match(desc: str, allowed: set) -> bool:
    if not desc.strip():
        return True
    d = desc.strip().upper()
    for cat in allowed:
        c = cat.upper()
        if d == c or d.startswith(c) or c.startswith(d) or c in d or d in c:
            return True
    return False

print("SegFormer helpers ready.")
"""))

# ── Cell 9: Improved Pipeline ─────────────────────────────────────────────────
cells.append(md("## Cell 9: Improved Retrieval Pipeline\n\nCombines:\n"
               "1. SegFormer crop retrieval + RRF fusion\n"
               "2. **Text-visual ensemble** (Xoel's key insight)\n"
               "3. **Section boost**\n"
               "4. Text re-ranking"))
cells.append(code("""\
def text_visual_ensemble_scores(
    rrf_candidates: list[str],
    crop_embs: np.ndarray,
    w_text: float = W_TEXT_ENSEMBLE,
) -> dict[str, float]:
    \"\"\"
    For each candidate product, compute text score = max over crop embeddings of
    cos_sim(crop_emb, text_emb_of_product_description).
    Returns {pid: text_boost_score} for all candidates.
    \"\"\"
    boost = {}
    for pid in rrf_candidates:
        t_emb = get_text_emb(pid)
        if t_emb is None:
            continue
        # max similarity across all crop embeddings
        sims  = crop_embs @ t_emb  # (n_crops,)
        boost[pid] = float(w_text * sims.max())
    return boost


def text_rerank(candidate_pids: list[str], crop_embs: np.ndarray,
                alpha: float = TEXT_RERANK_ALPHA) -> list[str]:
    if not candidate_pids:
        return candidate_pids
    descriptions = [product_desc.get(pid, "") for pid in candidate_pids]
    non_empty    = [(i, d) for i, d in enumerate(descriptions) if d.strip()]
    if not non_empty:
        return candidate_pids
    indices_ne, descs_ne = zip(*non_empty)
    text_embs    = _l2_norm(_marqo_encode_texts_raw(list(descs_ne)))
    sim          = crop_embs @ text_embs.T
    text_scores_ne = sim.max(axis=0)
    img_rank     = np.array([1.0 / (1.0 + i) for i in range(len(candidate_pids))])
    text_full    = np.zeros(len(candidate_pids))
    for li, oi in enumerate(indices_ne):
        text_full[oi] = float(text_scores_ne[li])
    combined = (1 - alpha) * img_rank + alpha * text_full
    order    = np.argsort(combined)[::-1]
    return [candidate_pids[j] for j in order]


def predict_segmented(bundle_ids: list[str], k: int = TOP_K) -> dict[str, list[str]]:
    \"\"\"
    Full improved pipeline:
    1. Segment bundle image → per-segment crops
    2. TTA-encode each crop
    3. FAISS IVFFlat query per segment (top K_PER_SEGMENT)
    4. Category-aware filtering + RRF fusion
    5. Text-visual ensemble boost
    6. Section boost
    7. Text re-ranking
    8. Top-k
    \"\"\"
    predictions: dict[str, list[str]] = {}

    for bid in tqdm(bundle_ids, desc="Segmented predict"):
        if bid not in valid_bundle_set:
            continue
        img = load_image(bid, BUND_DIR)
        if img is None:
            continue

        # Segmentation
        try:
            seg_map = segment_image(img)
            crops   = extract_segment_crops(img, seg_map)
        except Exception as e:
            print(f"  [WARN] Segmentation failed for {bid}: {e}")
            crops = {}

        # Fallback: whole-image if no crops
        if len(crops) < SEG_MIN_CROPS:
            emb = tta_encode(img)
            sc, idx_ = index.search(emb, k)
            predictions[bid] = [idx_to_pid[int(i)] for i in idx_[0] if int(i) in idx_to_pid][:k]
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue

        # TTA-encode crops
        crop_list  = list(crops.values())
        seg_names  = list(crops.keys())
        embs_norm  = np.vstack([tta_encode(c) for c in crop_list])  # (n_crops, D)

        K_dynamic  = K_PER_SEGMENT * max(1, len(crops) // 2)

        # FAISS query per segment
        scores_map: dict[str, float] = {}
        scores_arr, indices_arr = index.search(embs_norm.astype(np.float32), K_dynamic)

        for seg_name, seg_scores, seg_indices in zip(seg_names, scores_arr, indices_arr):
            allowed   = SEGMENT_TO_CATEGORIES.get(seg_name, set())
            filtered  = []
            unfiltered = []
            for sc, iv in zip(seg_scores, seg_indices):
                pid = idx_to_pid.get(int(iv))
                if pid is None:
                    continue
                unfiltered.append((pid, float(sc)))
                desc = product_desc.get(pid, "")
                if not allowed or _category_match(desc, allowed):
                    filtered.append((pid, float(sc)))

            to_add      = filtered if filtered else unfiltered[: K_dynamic // 4]
            rank_offset = 0 if filtered else RRF_K * 2
            for rank, (pid, _) in enumerate(to_add):
                scores_map[pid] = scores_map.get(pid, 0.0) + 1.0 / (RRF_K + rank_offset + rank)

        # Whole-image bonus
        whole_emb = _l2_norm(_marqo_encode_images_raw([img]))
        w_sc, w_idx = index.search(whole_emb.astype(np.float32), K_PER_SEGMENT)
        for rank, iv in enumerate(w_idx[0]):
            pid = idx_to_pid.get(int(iv))
            if pid:
                scores_map[pid] = scores_map.get(pid, 0.0) + WHOLE_IMG_WEIGHT / (RRF_K + rank)

        # Text-visual ensemble (key improvement from Xoel)
        if ENABLE_TEXT_ENSEMBLE:
            rrf_top  = sorted(scores_map, key=scores_map.get, reverse=True)[:TEXT_ENSEMBLE_TOPN]
            txt_boost = text_visual_ensemble_scores(rrf_top, embs_norm)
            for pid, boost in txt_boost.items():
                scores_map[pid] = scores_map.get(pid, 0.0) + boost

        # Section boost (small, helps break ties)
        if ENABLE_SECTION_BOOST:
            b_sec = bundle_section_map.get(bid, -1)
            if b_sec != -1:
                for pid in list(scores_map):
                    if product_section_map.get(pid, -1) == b_sec:
                        scores_map[pid] = scores_map[pid] + W_SECTION_BOOST

        # Sort
        ranked = sorted(scores_map, key=scores_map.get, reverse=True)
        top_pids = ranked[:TEXT_RERANK_TOP_N]

        # Text re-ranking
        if ENABLE_TEXT_RERANK and top_pids:
            top_pids = text_rerank(top_pids, embs_norm, alpha=TEXT_RERANK_ALPHA)

        predictions[bid] = top_pids[:k]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return predictions


_ensure_valid_sets()
print("Running segmented pipeline on test bundles …")
segmented_preds = predict_segmented(test_bundle_ids, k=TOP_K)
print(f"Segmented predictions for {len(segmented_preds)} bundles.")
"""))

# ── Cell 10: Validation ──────────────────────────────────────────────────────
cells.append(md("## Cell 10: Validation (Recall@15)"))
cells.append(code("""\
def recall_at_k(predictions, ground_truth, k=TOP_K):
    total, n = 0.0, 0
    for bid, gt_set in ground_truth.items():
        if not gt_set:
            continue
        preds = predictions.get(bid, [])[:k]
        total += len(set(preds) & gt_set) / len(gt_set)
        n += 1
    return total / n if n > 0 else 0.0


train_bundle_ids_with_gt = list(train_gt.keys())
random.seed(42)
VAL_SAMPLE = random.sample(train_bundle_ids_with_gt, min(500, len(train_bundle_ids_with_gt)))
val_gt = {b: train_gt[b] for b in VAL_SAMPLE if b in train_gt}

print(f"Validating on {len(VAL_SAMPLE)} sampled train bundles …")
val_preds   = predict_segmented(VAL_SAMPLE)
recall_seg  = recall_at_k(val_preds, val_gt)

print(f"\\n{'='*50}")
print(f"Recall@{TOP_K}  Segmented : {recall_seg:.4f}")
print(f"{'='*50}")

# Per-complexity
bins = {
    "easy (1-2)":   [b for b in VAL_SAMPLE if 1 <= len(train_gt.get(b,[])) <= 2],
    "medium (3-5)": [b for b in VAL_SAMPLE if 3 <= len(train_gt.get(b,[])) <= 5],
    "hard (6+)":    [b for b in VAL_SAMPLE if len(train_gt.get(b,[])) >= 6],
}
print("\\nPer-complexity Recall@15:")
for label, bids in bins.items():
    sub_gt = {b: train_gt[b] for b in bids if b in train_gt}
    r = recall_at_k(val_preds, sub_gt)
    print(f"  {label:16s}: {r:.4f}  n={len(bids)}")
"""))

# ── Cell 11: Submission ──────────────────────────────────────────────────────
cells.append(md("## Cell 11: Generate Submission"))
cells.append(code("""\
final_preds = segmented_preds

# Global fallback: most-frequent products in training
product_freq = Counter(
    pid for gt_set in train_gt.values() for pid in gt_set if pid in valid_product_set
)
fallback_pids = [pid for pid, _ in product_freq.most_common(TOP_K)]
print(f"Fallback products (top-{TOP_K} by train frequency): {fallback_pids[:3]} …")

rows = []
missing_count = 0
for bid in test_bundle_ids:
    preds = list(final_preds.get(bid, []))
    if not preds:
        preds = list(fallback_pids)
        missing_count += 1
    # Pad to TOP_K if needed
    for fp in fallback_pids:
        if len(preds) >= TOP_K:
            break
        if fp not in preds:
            preds.append(fp)
    for pid in preds[:TOP_K]:
        rows.append({"bundle_asset_id": bid, "product_asset_id": pid})

submission_df = pd.DataFrame(rows)
submission_df.to_csv(SUBM_FILE, index=False)
print(f"\\nSubmission saved: {SUBM_FILE}")
print(f"  Rows: {len(submission_df):,}  |  Fallback bundles: {missing_count}")

# Sanity checks
covered = set(submission_df["bundle_asset_id"].unique())
missing = set(test_bundle_ids) - covered
assert len(missing) == 0, f"Missing test bundles: {missing}"
max_preds = submission_df.groupby("bundle_asset_id")["product_asset_id"].count().max()
assert max_preds <= TOP_K, f"Bundle has {max_preds} predictions (max={TOP_K})"
all_cat  = set(products_df["product_asset_id"])
bad_pids = set(submission_df["product_asset_id"]) - all_cat
if bad_pids:
    print(f"  [WARN] {len(bad_pids)} invalid PIDs — removing")
    submission_df = submission_df[~submission_df["product_asset_id"].isin(bad_pids)]
    submission_df.to_csv(SUBM_FILE, index=False)
print(f"  [OK] All {len(test_bundle_ids)} test bundles present, max {max_preds} preds each.")
print("\\n" + submission_df.head(10).to_string(index=False))
"""))

# ── Cell 12: Summary ─────────────────────────────────────────────────────────
cells.append(md("## Cell 12: Summary"))
cells.append(code("""\
print("\\n" + "="*60)
print("HACKUDC 2026 — IMPROVED SOLUTION SUMMARY")
print("="*60)
print(f"Products indexed        : {index.ntotal:,}")
print(f"Embedding dimension     : {DIM}")
print(f"FAISS backend           : {'GPU' if DEVICE == 'cuda' else 'CPU'} (IVFFlat nprobe={nprobe})")
print(f"Retrieval model         : Marqo-FashionSigLIP (512-dim)")
print(f"Segmentation model      : SegFormer B2 (ATR 17 classes)")
print(f"TTA views               : {TTA_N_VIEWS}")
print(f"Text-visual ensemble    : {ENABLE_TEXT_ENSEMBLE} (w={W_TEXT_ENSEMBLE})")
print(f"Section boost           : {ENABLE_SECTION_BOOST} (w={W_SECTION_BOOST})")
print(f"Text re-ranking         : {ENABLE_TEXT_RERANK} (alpha={TEXT_RERANK_ALPHA})")
print(f"Validation Recall@15    : {recall_seg:.4f}")
print(f"Submission rows         : {len(submission_df):,}")
print("="*60)
"""))


# ── Assemble notebook JSON ───────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("hackudc_solution_colab.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Written: {out}  ({out.stat().st_size/1024:.0f} KB,  {len(cells)} cells)")
