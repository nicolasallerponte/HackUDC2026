# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # HackUDC 2026 — Inditex Fashion Retrieval
# **Task:** Given a bundle (model photo with multiple garments), retrieve up to 15 matching
# product IDs from a 27K-product catalog.
# **Metric:** Recall@15

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 1: Setup

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:04:49.254018Z","iopub.execute_input":"2026-02-28T10:04:49.254740Z","iopub.status.idle":"2026-02-28T10:04:57.487881Z","shell.execute_reply.started":"2026-02-28T10:04:49.254684Z","shell.execute_reply":"2026-02-28T10:04:57.487304Z"}}
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# fashion-clip and supporting libs
_pip("fashion-clip", "transformers>=4.35.0", "accelerate", "tqdm", "Pillow", "requests", "open_clip_torch")

# faiss: try GPU build first, fall back to CPU
try:
    _pip("faiss-gpu")
except Exception:
    try:
        _pip("faiss-cpu")
    except Exception:
        pass  # may already be installed under a different mechanism

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 2: Imports & Configuration

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:04:57.489282Z","iopub.execute_input":"2026-02-28T10:04:57.489508Z","iopub.status.idle":"2026-02-28T10:04:57.499500Z","shell.execute_reply.started":"2026-02-28T10:04:57.489489Z","shell.execute_reply":"2026-02-28T10:04:57.498841Z"}}
import os
import gc
import json
import math
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import torch
try:
    import faiss
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "faiss not found. Re-run Cell 1 to install it, or manually run:\n"
        "  pip install faiss-gpu   # if CUDA is available\n"
        "  pip install faiss-cpu   # CPU-only fallback"
    )
from PIL import Image
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/kaggle/input/datasets/miguelplanasdaz/hackudc2026")
WORK_DIR  = Path("/kaggle/working")
IMG_DIR   = WORK_DIR / "images"
PROD_DIR  = IMG_DIR / "products"
BUND_DIR  = IMG_DIR / "bundles"
SUBM_FILE = WORK_DIR / "submission.csv"

for d in [PROD_DIR, BUND_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Model selection ───────────────────────────────────────────────────────────
USE_MARQO      = True
MARQO_MODEL_ID = "marqo/marqo-fashionSigLIP"
MODEL_TAG      = "marqo" if USE_MARQO else "fclip"
EMB_FILE       = WORK_DIR / f"product_embeddings_{MODEL_TAG}.npy"
IDS_FILE       = WORK_DIR / f"product_ids_{MODEL_TAG}.json"

# ── Hyper-params ─────────────────────────────────────────────────────────────
TOP_K            = 15
EMBED_BATCH      = 64
DOWNLOAD_WORKERS = 32   # parallel download threads
IMG_SIZE         = 224  # FashionCLIP canonical input size
DOWNLOAD_TIMEOUT = 20   # seconds per request (increased)
DOWNLOAD_RETRIES = 3    # per-image retry attempts with backoff
K_PER_SEGMENT    = 150  # candidates per segment before category filter (increased)
AGGREGATION      = "sum" # score aggregation: "sum" or "max"
ENABLE_TEXT_RERANK  = True
TEXT_RERANK_ALPHA   = 0.20  # slightly less text weight (image more reliable)
TEXT_RERANK_TOP_N   = 50   # more candidates to re-rank

# HTTP headers to avoid CDN bot-blocks
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

print("\nConfiguration loaded.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 3: Data Loading

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:04:57.500508Z","iopub.execute_input":"2026-02-28T10:04:57.500785Z","iopub.status.idle":"2026-02-28T10:04:57.640323Z","shell.execute_reply.started":"2026-02-28T10:04:57.500755Z","shell.execute_reply":"2026-02-28T10:04:57.639597Z"}}
bundles_df = pd.read_csv(DATA_DIR / "bundles_dataset.csv")
products_df = pd.read_csv(DATA_DIR / "product_dataset.csv")
train_df = pd.read_csv(DATA_DIR / "bundles_product_match_train.csv")
test_df = pd.read_csv(DATA_DIR / "bundles_product_match_test.csv")

print("=== Dataset Statistics ===")
print(f"Bundles total  : {len(bundles_df):,}")
print(f"Products total : {len(products_df):,}")
print(f"Train pairs    : {len(train_df):,}")
print(f"Test bundles   : {len(test_df['bundle_asset_id'].unique()):,}")

# Ground-truth lookup: bundle_id → set of product_ids
train_gt: dict[str, set] = (
    train_df.dropna(subset=["product_asset_id"])
    .groupby("bundle_asset_id")["product_asset_id"]
    .apply(set)
    .to_dict()
)

# Per-bundle product counts
counts = [len(v) for v in train_gt.values()]
print(f"\nTrain bundles with GT  : {len(train_gt):,}")
print(f"Avg products/bundle    : {np.mean(counts):.2f}")
print(f"Max products/bundle    : {max(counts)}")
print(f"Min products/bundle    : {min(counts)}")

# URL lookup dicts
bundle_url:   dict[str, str] = dict(zip(bundles_df["bundle_asset_id"], bundles_df["bundle_image_url"]))
product_url:  dict[str, str] = dict(zip(products_df["product_asset_id"], products_df["product_image_url"]))
product_desc: dict[str, str] = dict(zip(products_df["product_asset_id"], products_df["product_description"].fillna("")))

# Category distribution
if "product_description" in products_df.columns:
    cat_counts = products_df["product_description"].value_counts().head(10)
    print("\nTop-10 product categories:")
    print(cat_counts.to_string())

# Test bundle IDs
test_bundle_ids = test_df["bundle_asset_id"].dropna().unique().tolist()
print(f"\nTest bundle IDs: {len(test_bundle_ids)}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 4: Image Download + Preview

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:04:57.641150Z","iopub.execute_input":"2026-02-28T10:04:57.641352Z","iopub.status.idle":"2026-02-28T10:06:10.901561Z","shell.execute_reply.started":"2026-02-28T10:04:57.641333Z","shell.execute_reply":"2026-02-28T10:06:10.900857Z"}}
def download_image(
    asset_id: str,
    url: str,
    out_dir: Path,
    timeout: int = DOWNLOAD_TIMEOUT,
    retries: int = DOWNLOAD_RETRIES,
) -> bool:
    """Download image with retries + exponential backoff. Returns True on success."""
    out_path = out_dir / f"{asset_id}.jpg"
    if out_path.exists():
        return True
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=DOWNLOAD_HEADERS)
            r.raise_for_status()
            # Validate that response is a non-empty image
            if len(r.content) < 500:
                raise ValueError(f"Response too small ({len(r.content)} bytes), likely an error page")
            out_path.write_bytes(r.content)
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
    return False


def batch_download(
    id_url_pairs: list[tuple[str, str]],
    out_dir: Path,
    desc: str = "Downloading",
    timeout: int = DOWNLOAD_TIMEOUT,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Parallel download with ThreadPoolExecutor.
    Returns (ok_ids, failed_pairs) where failed_pairs can be retried.
    """
    ok_ids: list[str] = []
    failed_pairs: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(download_image, aid, url, out_dir, timeout): (aid, url)
            for aid, url in id_url_pairs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            aid, url = futures[fut]
            if fut.result():
                ok_ids.append(aid)
            else:
                failed_pairs.append((aid, url))
    return ok_ids, failed_pairs


def load_image(asset_id: str, img_dir: Path) -> Image.Image | None:
    """Load a cached image as RGB PIL image."""
    p = img_dir / f"{asset_id}.jpg"
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


# ── Download products (~27K) ─────────────────────────────────────────────────
product_pairs = [(pid, url) for pid, url in product_url.items()]
_prod_ok, _prod_failed = batch_download(product_pairs, PROD_DIR, desc="Products pass-1")
print(f"\nProducts pass-1: {len(_prod_ok):,} ok, {len(_prod_failed):,} failed")

# Retry pass with longer timeout
if _prod_failed:
    print(f"Retrying {len(_prod_failed):,} failed products (60s timeout) …")
    _prod_ok2, _prod_failed2 = batch_download(
        _prod_failed, PROD_DIR, desc="Products pass-2", timeout=60
    )
    _prod_ok += _prod_ok2
    print(f"  Retry recovered: {len(_prod_ok2):,}  still failed: {len(_prod_failed2):,}")

valid_product_ids = _prod_ok
print(f"Products downloaded: {len(valid_product_ids):,} / {len(product_pairs):,} "
      f"({100*len(valid_product_ids)/len(product_pairs):.1f}%)")

# ── Download bundles (~2.3K) ─────────────────────────────────────────────────
bundle_pairs = [(bid, url) for bid, url in bundle_url.items()]
_bund_ok, _bund_failed = batch_download(bundle_pairs, BUND_DIR, desc="Bundles  pass-1")
print(f"\nBundles  pass-1: {len(_bund_ok):,} ok, {len(_bund_failed):,} failed")

if _bund_failed:
    print(f"Retrying {len(_bund_failed):,} failed bundles (60s timeout) …")
    _bund_ok2, _bund_failed2 = batch_download(
        _bund_failed, BUND_DIR, desc="Bundles  pass-2", timeout=60
    )
    _bund_ok += _bund_ok2
    print(f"  Retry recovered: {len(_bund_ok2):,}  still failed: {len(_bund_failed2):,}")

valid_bundle_ids = _bund_ok
print(f"Bundles downloaded : {len(valid_bundle_ids):,} / {len(bundle_pairs):,} "
      f"({100*len(valid_bundle_ids)/len(bundle_pairs):.1f}%)")

valid_product_set = set(valid_product_ids)
valid_bundle_set  = set(valid_bundle_ids)


def show_images(asset_ids: list[str], img_dir: Path, title: str = "", n_cols: int = 5):
    """Matplotlib grid preview of a sample of images."""
    imgs = [(aid, load_image(aid, img_dir)) for aid in asset_ids[:n_cols * 3]]
    imgs = [(aid, img) for aid, img in imgs if img is not None]
    if not imgs:
        print("No images to show.")
        return
    n_cols = min(n_cols, len(imgs))
    n_rows = math.ceil(len(imgs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()
    for ax, (aid, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_title(aid[:12], fontsize=7)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(WORK_DIR / f"preview_{title.replace(' ','_')}.png", bbox_inches="tight", dpi=80)
    plt.show()
    print(f"Saved preview to {WORK_DIR}/preview_{title.replace(' ','_')}.png")


show_images(valid_product_ids[:15], PROD_DIR, title="Product Sample")
show_images(valid_bundle_ids[:5],   BUND_DIR, title="Bundle Sample")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 5: FashionCLIP Loading

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:08:35.322487Z","iopub.execute_input":"2026-02-28T10:08:35.323027Z","iopub.status.idle":"2026-02-28T10:08:42.897941Z","shell.execute_reply.started":"2026-02-28T10:08:35.322997Z","shell.execute_reply":"2026-02-28T10:08:42.897223Z"}}
from fashion_clip.fashion_clip import FashionCLIP  # noqa: E402
import fashion_clip.fashion_clip as _fc_module      # noqa: E402

# ── Compatibility patch ───────────────────────────────────────────────────────
# fashion_clip passes `use_auth_token` to CLIPModel.from_pretrained, which was
# removed in transformers>=4.40. Patch _load_model to use `token` instead.
def _patched_load_model(self, name, device=None, auth_token=None):
    from transformers import CLIPModel, CLIPProcessor
    name = _fc_module._MODELS.get(name, name)
    token_kwargs = {"token": auth_token} if auth_token else {}
    model = CLIPModel.from_pretrained(name, **token_kwargs)
    processor = CLIPProcessor.from_pretrained(name, **token_kwargs)
    hash_val = _fc_module._model_processor_hash(name, model, processor)
    return model, processor, hash_val

FashionCLIP._load_model = _patched_load_model
# ─────────────────────────────────────────────────────────────────────────────

print("Loading FashionCLIP …")
fclip = FashionCLIP("fashion-clip")
# Move underlying model to GPU if available
if DEVICE == "cuda":
    fclip.model = fclip.model.to(DEVICE)
fclip.model.eval()
print("FashionCLIP loaded.")


def embed_images_fashionclip(image_paths: list[Path], batch_size: int = EMBED_BATCH) -> np.ndarray:
    """
    Embed a list of image paths with FashionCLIP.
    Returns float32 (N, 512) L2-normalised embeddings.
    """
    all_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding", leave=False):
        batch_paths = image_paths[i : i + batch_size]
        imgs: list[Image.Image] = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (IMG_SIZE, IMG_SIZE)))

        embs = _l2_norm(_clip_encode_images_raw(imgs))
        all_embs.append(embs)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_embs, axis=0)


print("embed_images_fashionclip() ready.")


# ── Low-level helpers that bypass fashion_clip's broken encode_images/encode_text ──
# fashion_clip's encode_images calls model.get_image_features() and tries to
# call .detach() on the result, but newer transformers returns a model-output
# object there. We call the processor + model directly instead.

def _clip_encode_images_raw(imgs: list[Image.Image]) -> np.ndarray:
    """
    Encode a list of PIL images with FashionCLIP's underlying CLIP model.
    Returns float32 (N, 512) embeddings (NOT normalised).
    """
    inputs = fclip.preprocess(images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    # Keep only pixel_values — get_image_features does not accept other keys
    pixel_inputs = {"pixel_values": inputs["pixel_values"]}
    with torch.no_grad():
        out = fclip.model.get_image_features(**pixel_inputs)
    # out may be a tensor or a ModelOutput; extract tensor either way
    if not isinstance(out, torch.Tensor):
        out = out.image_embeds if hasattr(out, "image_embeds") else out.last_hidden_state[:, 0]
    return out.cpu().float().numpy()


def _clip_encode_texts_raw(texts: list[str]) -> np.ndarray:
    """
    Encode a list of text strings with FashionCLIP's underlying CLIP model.
    Returns float32 (N, 512) embeddings (NOT normalised).
    """
    inputs = fclip.preprocess(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    text_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        out = fclip.model.get_text_features(**text_inputs)
    if not isinstance(out, torch.Tensor):
        out = out.text_embeds if hasattr(out, "text_embeds") else out.last_hidden_state[:, 0]
    return out.cpu().float().numpy()


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
    return arr / norms


# ── Marqo-FashionSigLIP via open_clip (recommended loader) ───────────────────
import open_clip  # noqa: E402

if USE_MARQO:
    print(f"Loading Marqo-FashionSigLIP via open_clip (hf-hub:{MARQO_MODEL_ID}) …")
    marqo_model, _, marqo_preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{MARQO_MODEL_ID}"
    )
    marqo_tokenizer = open_clip.get_tokenizer(f"hf-hub:{MARQO_MODEL_ID}")
    marqo_model.to(DEVICE).eval()
    print("Marqo-FashionSigLIP loaded.")

    # Free FashionCLIP VRAM
    del fclip
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("FashionCLIP unloaded to free VRAM.")


def _marqo_encode_images_raw(imgs: list) -> np.ndarray:
    """Encode PIL images with Marqo-FashionSigLIP. Returns (N, 512) float32."""
    tensors = torch.stack([marqo_preprocess(img) for img in imgs]).to(DEVICE)
    with torch.no_grad():
        out = marqo_model.encode_image(tensors)
    return out.cpu().float().numpy()


def _marqo_encode_texts_raw(texts: list[str]) -> np.ndarray:
    """Encode text strings with Marqo-FashionSigLIP. Returns (N, 512) float32."""
    tokens = marqo_tokenizer(texts).to(DEVICE)
    with torch.no_grad():
        out = marqo_model.encode_text(tokens)
    return out.cpu().float().numpy()


def _encode_images_raw(imgs: list) -> np.ndarray:
    """Unified image encoder: Marqo or FashionCLIP depending on USE_MARQO."""
    return _marqo_encode_images_raw(imgs) if USE_MARQO else _clip_encode_images_raw(imgs)


def _encode_texts_raw(texts: list[str]) -> np.ndarray:
    """Unified text encoder: Marqo or FashionCLIP depending on USE_MARQO."""
    return _marqo_encode_texts_raw(texts) if USE_MARQO else _clip_encode_texts_raw(texts)


def _embed_images(image_paths: list, batch_size: int = EMBED_BATCH) -> np.ndarray:
    """
    Embed a list of image paths with the active model (Marqo or FashionCLIP).
    Returns float32 (N, 512) L2-normalised embeddings.
    """
    all_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding", leave=False):
        batch_paths = image_paths[i : i + batch_size]
        imgs: list = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (IMG_SIZE, IMG_SIZE)))
        embs = _l2_norm(_encode_images_raw(imgs))
        all_embs.append(embs)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


print("Unified encode helpers ready.")


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 6: Product Embeddings + FAISS Index

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:08:46.502118Z","iopub.execute_input":"2026-02-28T10:08:46.502417Z","iopub.status.idle":"2026-02-28T10:34:30.837583Z","shell.execute_reply.started":"2026-02-28T10:08:46.502393Z","shell.execute_reply":"2026-02-28T10:34:30.836876Z"}}
def _ensure_valid_sets():
    """Rebuild valid_product/bundle sets from disk if empty (survives kernel restarts)."""
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
        print(f"Rebuilt valid_bundle_set from disk: {len(valid_bundle_set):,}")
    print(f"valid_product_set: {len(valid_product_set):,}  valid_bundle_set: {len(valid_bundle_set):,}")

_ensure_valid_sets()

if EMB_FILE.exists() and IDS_FILE.exists():
    print("Loading cached product embeddings …")
    product_embeddings = np.load(EMB_FILE)
    with open(IDS_FILE) as f:
        indexed_product_ids: list[str] = json.load(f)
    print(f"  Loaded {product_embeddings.shape} from cache.")
else:
    print("Computing product embeddings (first run, ~15 min on T4) …")
    # Only embed products we actually downloaded
    ordered_ids  = [pid for pid in valid_product_ids if pid in valid_product_set]
    ordered_paths = [PROD_DIR / f"{pid}.jpg" for pid in ordered_ids]
    product_embeddings = _embed_images(ordered_paths, batch_size=EMBED_BATCH)
    indexed_product_ids = ordered_ids
    np.save(EMB_FILE, product_embeddings)
    with open(IDS_FILE, "w") as f:
        json.dump(indexed_product_ids, f)
    print(f"  Saved embeddings: {product_embeddings.shape}")

# Reverse lookup
idx_to_pid: dict[int, str] = {i: pid for i, pid in enumerate(indexed_product_ids)}

# Build FAISS index
DIM = product_embeddings.shape[1]  # 512
print(f"\nBuilding FAISS IndexFlatIP (dim={DIM}, n={len(indexed_product_ids):,}) …")
index_cpu = faiss.IndexFlatIP(DIM)
index_cpu.add(product_embeddings)

if DEVICE == "cuda":
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        print("  Transferred index to GPU.")
    except Exception as e:
        print(f"  [WARN] GPU index failed ({e}), falling back to CPU.")
        index = index_cpu
else:
    index = index_cpu
    print("  Using CPU index.")

print(f"  FAISS ntotal = {index.ntotal:,}")
print(f"  indexed_product_ids count = {len(indexed_product_ids):,}")
print(f"  valid_product_ids count   = {len(valid_product_ids):,}")
print(f"  valid_product_set count   = {len(valid_product_set):,}")
if index.ntotal < 1_000:
    raise RuntimeError(
        f"Only {index.ntotal} products indexed — something went wrong with download or embedding. "
        "Delete the cache files and re-run:\n"
        f"  {EMB_FILE}\n  {IDS_FILE}"
    )
elif index.ntotal < 20_000:
    print(f"  [WARN] Only {index.ntotal:,} products indexed (expected ~27K). "
          "Some downloads may have failed — results may be degraded.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 7: Baseline Pipeline (whole-image retrieval)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:34:33.401442Z","iopub.execute_input":"2026-02-28T10:34:33.401758Z","iopub.status.idle":"2026-02-28T10:34:33.408230Z","shell.execute_reply.started":"2026-02-28T10:34:33.401702Z","shell.execute_reply":"2026-02-28T10:34:33.407425Z"}}
def predict_baseline(bundle_ids: list[str], k: int = TOP_K) -> dict[str, list[str]]:
    """Whole-image baseline: no segmentation, embed full bundle → top-k."""
    predictions: dict[str, list[str]] = {}
    for bid in tqdm(bundle_ids, desc="Baseline predict"):
        if bid not in valid_bundle_set:
            continue
        img = load_image(bid, BUND_DIR)
        if img is None:
            continue
        emb = _l2_norm(_encode_images_raw([img]))  # (1, 512)
        scores, indices = index.search(emb, k)
        top_pids = [idx_to_pid[int(i)] for i in indices[0] if int(i) in idx_to_pid]
        predictions[bid] = top_pids[:k]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return predictions

print("predict_baseline() defined.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 8: SegFormer Segmentation

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:34:35.765351Z","iopub.execute_input":"2026-02-28T10:34:35.765858Z","iopub.status.idle":"2026-02-28T10:34:38.940633Z","shell.execute_reply.started":"2026-02-28T10:34:35.765829Z","shell.execute_reply":"2026-02-28T10:34:38.939905Z"}}
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation  # noqa: E402
import torch.nn.functional as F  # noqa: E402

print("Loading SegFormer B2 clothes segmentation model …")
SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
seg_processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
seg_model.to(DEVICE).eval()
print("SegFormer loaded.")

# ATR dataset label map (17 classes, 0 = background)
ATR_LABELS = {
    0:  "Background",
    1:  "Hat",
    2:  "Hair",
    3:  "Sunglasses",
    4:  "Upper-clothes",
    5:  "Skirt",
    6:  "Pants",
    7:  "Dress",
    8:  "Belt",
    9:  "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

# Group labels into retrieval segments
SEGMENT_GROUPS: dict[str, list[int]] = {
    "upper_body": [4],
    "lower_body":  [5, 6],
    "dress":       [7],
    "shoes":       [9, 10],
    "bag":         [16],
    "hat":         [1],
    "scarf_belt":  [8, 17],
}

SEGMENT_TO_CATEGORIES: dict[str, set[str]] = {
    "upper_body": {"T-SHIRT", "SHIRT", "SWEATER", "WIND-JACKET", "TOPS AND OTHERS", "BLAZER", "SWEATSHIRT", "BABY T-SHIRT", "POLO SHIRT"},
    "lower_body": {"TROUSERS", "SKIRT", "BERMUDA", "BABY TROUSERS", "SHORTS", "LEGGINGS", "JEANS"},
    "dress":      {"DRESS", "OVERALL", "JUMPSUIT"},
    "shoes":      {"SHOE", "SHOES", "SNEAKER", "SNEAKERS", "BOOT", "BOOTS", "SANDAL", "SANDALS", "LOAFER"},
    "bag":        {"HAND BAG-RUCKSACK", "HANDBAG", "BAG", "BACKPACK", "PURSE", "CLUTCH"},
    "hat":        {"HAT", "CAP", "BEANIE"},
    "scarf_belt": {"SCARF", "BELT", "TIE"},
}

ATR_COLORS = plt.cm.get_cmap("tab20", 18)


@torch.no_grad()
def segment_image(pil_img: Image.Image) -> np.ndarray:
    """
    Run SegFormer on pil_img.
    Returns (H, W) int numpy array with ATR label indices,
    upsampled to original image resolution via bilinear interpolation.
    """
    inputs = seg_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    logits = seg_model(**inputs).logits  # (1, num_labels, H/4, W/4)

    # Upsample to original size
    orig_h, orig_w = pil_img.height, pil_img.width
    upsampled = F.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )  # (1, num_labels, H, W)

    seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    return seg_map


def extract_segment_crops(
    pil_img: Image.Image,
    seg_map: np.ndarray,
    min_frac: float = 0.05,
    padding: float = 0.05,
) -> dict[str, Image.Image]:
    """
    For each SEGMENT_GROUP, compute bounding box over union of constituent labels,
    apply fractional padding, and crop from the original image.
    Skips segments covering less than min_frac of the image area.

    Returns dict: {segment_name: cropped PIL.Image}
    Note: We use bounding-box crops (NOT masked) because FashionCLIP was trained
    on clean product images — zeroing out pixels creates OOD input.
    """
    W, H = pil_img.size
    total_pixels = H * W
    crops: dict[str, Image.Image] = {}

    for seg_name, label_ids in SEGMENT_GROUPS.items():
        mask = np.isin(seg_map, label_ids)
        if mask.sum() < min_frac * total_pixels:
            continue  # segment too small

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        y_min, y_max = int(rows.min()), int(rows.max())
        x_min, x_max = int(cols.min()), int(cols.max())

        # Add fractional padding
        pad_y = int((y_max - y_min + 1) * padding)
        pad_x = int((x_max - x_min + 1) * padding)
        y_min = max(0, y_min - pad_y)
        y_max = min(H - 1, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(W - 1, x_max + pad_x)

        crop = pil_img.crop((x_min, y_min, x_max + 1, y_max + 1))
        if crop.width > 5 and crop.height > 5:
            crops[seg_name] = crop

    return crops


def visualise_segmentation(pil_img: Image.Image, seg_map: np.ndarray, bundle_id: str = ""):
    """Save a colourmap visualisation of the segmentation alongside the original."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(pil_img)
    ax1.set_title("Original", fontsize=11)
    ax1.axis("off")

    cmap = ListedColormap([ATR_COLORS(i)[:3] for i in range(18)])
    im = ax2.imshow(seg_map, cmap=cmap, vmin=0, vmax=17)
    ax2.set_title("Segmentation", fontsize=11)
    ax2.axis("off")

    patches = [mpatches.Patch(color=ATR_COLORS(i)[:3], label=ATR_LABELS[i]) for i in range(18)]
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(WORK_DIR / f"seg_{bundle_id[:10]}.png", bbox_inches="tight", dpi=80)
    plt.show()


# Demo segmentation on first test bundle
demo_bid = next((bid for bid in test_bundle_ids if bid in valid_bundle_set), None)
if demo_bid:
    demo_img = load_image(demo_bid, BUND_DIR)
    if demo_img:
        demo_seg = segment_image(demo_img)
        visualise_segmentation(demo_img, demo_seg, bundle_id=demo_bid)
        demo_crops = extract_segment_crops(demo_img, demo_seg)
        print(f"Segments found in demo bundle: {list(demo_crops.keys())}")

        # Show crops
        if demo_crops:
            n = len(demo_crops)
            fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
            if n == 1:
                axes = [axes]
            for ax, (name, crop) in zip(axes, demo_crops.items()):
                ax.imshow(crop)
                ax.set_title(name, fontsize=9)
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(WORK_DIR / "demo_crops.png", bbox_inches="tight", dpi=80)
            plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 9: Text Re-ranking + Improved Pipeline (per-segment retrieval)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:34:42.259966Z","iopub.execute_input":"2026-02-28T10:34:42.260441Z","iopub.status.idle":"2026-02-28T10:34:42.268100Z","shell.execute_reply.started":"2026-02-28T10:34:42.260414Z","shell.execute_reply":"2026-02-28T10:34:42.267230Z"}}
def text_rerank(
    bundle_id: str,
    candidate_pids: list[str],
    crop_embs: np.ndarray,
    alpha: float = TEXT_RERANK_ALPHA,
) -> list[str]:
    """
    Linearly combine image similarity scores with text–image cross-similarity.
    alpha: weight given to text score (0 = pure image, 1 = pure text).

    crop_embs: (n_crops, 512) L2-normalised embeddings for the bundle's crops.
    """
    if not candidate_pids:
        return candidate_pids

    descriptions = [product_desc.get(pid, "") for pid in candidate_pids]
    non_empty    = [(i, d) for i, d in enumerate(descriptions) if d.strip()]

    if not non_empty:
        return candidate_pids

    indices_ne, descs_ne = zip(*non_empty)

    text_embs = _l2_norm(_encode_texts_raw(list(descs_ne)))  # (n_text, 512)

    # Cross-similarity: max over crops for each text embedding
    sim = crop_embs @ text_embs.T  # (n_crops, n_text)
    text_scores_ne = sim.max(axis=0)  # (n_text,)

    # Combine with image rank (normalise rank to [0,1])
    img_rank_score = np.array([1.0 - i / len(candidate_pids) for i in range(len(candidate_pids))])

    text_score_full = np.zeros(len(candidate_pids))
    for list_idx, orig_idx in enumerate(indices_ne):
        text_score_full[orig_idx] = float(text_scores_ne[list_idx])

    combined = (1 - alpha) * img_rank_score + alpha * text_score_full
    order    = np.argsort(combined)[::-1]
    return [candidate_pids[j] for j in order]


print("text_rerank() defined.")


def _category_match(desc: str, allowed: set) -> bool:
    """
    Returns True if `desc` (product description / category) matches any string in `allowed`.
    Uses substring matching (both directions) to handle plurals and compound names:
      - "SHOES" matches allowed={"SHOE"}
      - "SNEAKERS" matches allowed={"SNEAKER"}
      - "HAND BAG-RUCKSACK" matches allowed={"HANDBAG", "BAG"}
    Empty descriptions pass through (don't exclude unknown categories).
    """
    if not desc.strip():
        return True  # unknown category: do not exclude
    d = desc.strip().upper()
    for cat in allowed:
        c = cat.upper()
        if d == c or d.startswith(c) or c.startswith(d) or c in d or d in c:
            return True
    return False

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:34:45.771846Z","iopub.execute_input":"2026-02-28T10:34:45.772358Z","iopub.status.idle":"2026-02-28T10:36:38.760959Z","shell.execute_reply.started":"2026-02-28T10:34:45.772333Z","shell.execute_reply":"2026-02-28T10:36:38.760176Z"}}
def predict_segmented(
    bundle_ids: list[str],
    k: int = TOP_K,
    fallback_to_whole: bool = True,
) -> dict[str, list[str]]:
    """
    Per-segment retrieval pipeline:
    1. SegFormer → garment crops
    2. Active model encode each crop
    3. FAISS query per segment (top K_PER_SEGMENT)
    4. Category-aware filtering per segment
    5. Sum-score aggregation across segments
    6. Optional text re-ranking on top-TEXT_RERANK_TOP_N
    7. Sort descending → top-k
    8. Fallback to whole-image embedding if no segments detected
    """
    predictions: dict[str, list[str]] = {}

    for bid in tqdm(bundle_ids, desc="Segmented predict"):
        if bid not in valid_bundle_set:
            continue

        img = load_image(bid, BUND_DIR)
        if img is None:
            continue

        # ── Segmentation ────────────────────────────────────────────────────
        try:
            seg_map = segment_image(img)
            crops = extract_segment_crops(img, seg_map)
        except Exception as e:
            print(f"  [WARN] Segmentation failed for {bid}: {e}")
            crops = {}

        # ── Embedding ───────────────────────────────────────────────────────
        if crops:
            crop_list = list(crops.values())
            seg_names = list(crops.keys())
        else:
            crop_list = [img]
            seg_names = ["whole"]

        embs_norm = _l2_norm(_encode_images_raw(crop_list))  # (n_crops, 512)

        # ── FAISS query per segment ──────────────────────────────────────────
        scores_map: dict[str, float] = {}  # product_id → accumulated score
        n_filtered_fallback = 0  # diagnostic counter

        scores_arr, indices_arr = index.search(embs_norm, K_PER_SEGMENT)
        # (n_crops, K_PER_SEGMENT)

        for seg_name, seg_scores, seg_indices in zip(seg_names, scores_arr, indices_arr):
            allowed = SEGMENT_TO_CATEGORIES.get(seg_name, set())

            filtered: list[tuple[str, float]] = []
            unfiltered: list[tuple[str, float]] = []

            for sc, idx_val in zip(seg_scores, seg_indices):
                pid = idx_to_pid.get(int(idx_val))
                if pid is None:
                    continue
                unfiltered.append((pid, float(sc)))
                desc = product_desc.get(pid, "")
                if not allowed or _category_match(desc, allowed):
                    filtered.append((pid, float(sc)))

            # Fallback: if category filter emptied the results, use unfiltered with a
            # 50% score penalty to signal lower confidence.
            if filtered:
                to_add = filtered
            else:
                to_add = [(p, s * 0.50) for p, s in unfiltered[: K_PER_SEGMENT // 4]]
                n_filtered_fallback += 1

            # Sum aggregation: reward products matching multiple segments
            for pid, sc in to_add:
                scores_map[pid] = scores_map.get(pid, 0.0) + sc

        # ── Text re-ranking on top candidates ───────────────────────────────
        ranked_ext = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
        top_pids = [pid for pid, _ in ranked_ext[:TEXT_RERANK_TOP_N]]
        if ENABLE_TEXT_RERANK and top_pids:
            top_pids = text_rerank(bid, top_pids, embs_norm, alpha=TEXT_RERANK_ALPHA)

        predictions[bid] = top_pids[:k]

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return predictions


_ensure_valid_sets()  # rebuild from disk if kernel was restarted
print("Running segmented pipeline on test bundles …")
segmented_preds = predict_segmented(test_bundle_ids, k=TOP_K)
print(f"Segmented predictions for {len(segmented_preds)} bundles.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 10: Validation (Recall@15)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:37:32.207417Z","iopub.execute_input":"2026-02-28T10:37:32.207747Z","iopub.status.idle":"2026-02-28T10:51:08.849442Z","shell.execute_reply.started":"2026-02-28T10:37:32.207719Z","shell.execute_reply":"2026-02-28T10:51:08.848894Z"}}
def recall_at_k(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set],
    k: int = TOP_K,
) -> float:
    """
    Macro-averaged Recall@k over all bundles that have ground-truth labels.
    recall_i = |pred_i[:k] ∩ gt_i| / |gt_i|
    """
    total_recall = 0.0
    n = 0
    for bid, gt_set in ground_truth.items():
        if not gt_set:
            continue
        preds = predictions.get(bid, [])[:k]
        hit = len(set(preds) & gt_set)
        total_recall += hit / len(gt_set)
        n += 1
    return total_recall / n if n > 0 else 0.0


# ── Evaluate on training set (has ground truth) ──────────────────────────────
train_bundle_ids_with_gt = list(train_gt.keys())
print("Running baseline on train bundles for validation …")
baseline_train_preds = predict_baseline(train_bundle_ids_with_gt)
print("Running segmented on train bundles for validation …")
seg_train_preds = predict_segmented(train_bundle_ids_with_gt)

recall_base = recall_at_k(baseline_train_preds, train_gt)
recall_seg  = recall_at_k(seg_train_preds, train_gt)

print(f"\n{'='*40}")
print(f"Recall@{TOP_K}  Baseline  : {recall_base:.4f}")
print(f"Recall@{TOP_K}  Segmented : {recall_seg:.4f}")
print(f"Delta                  : {recall_seg - recall_base:+.4f}")
print(f"{'='*40}")

# ── Per-complexity breakdown ──────────────────────────────────────────────────
complexity_bins = {
    "easy (1-2)":   [bid for bid, gt in train_gt.items() if 1 <= len(gt) <= 2],
    "medium (3-5)": [bid for bid, gt in train_gt.items() if 3 <= len(gt) <= 5],
    "hard (6+)":    [bid for bid, gt in train_gt.items() if len(gt) >= 6],
}

print("\nPer-complexity Recall@15:")
base_vals, seg_vals, labels = [], [], []
for label, bids in complexity_bins.items():
    sub_gt = {b: train_gt[b] for b in bids if b in train_gt}
    r_base = recall_at_k(baseline_train_preds, sub_gt)
    r_seg  = recall_at_k(seg_train_preds, sub_gt)
    print(f"  {label:16s}  Baseline={r_base:.4f}  Segmented={r_seg:.4f}  n={len(bids)}")
    base_vals.append(r_base)
    seg_vals.append(r_seg)
    labels.append(label)

# Bar chart
x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w/2, base_vals, w, label="Baseline",  color="steelblue")
ax.bar(x + w/2, seg_vals,  w, label="Segmented", color="coral")
ax.set_ylabel("Recall@15")
ax.set_title("Recall@15 by Bundle Complexity")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig(WORK_DIR / "recall_comparison.png", bbox_inches="tight", dpi=100)
plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 11: Generate Submission

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:51:44.559041Z","iopub.execute_input":"2026-02-28T10:51:44.559358Z","iopub.status.idle":"2026-02-28T10:52:15.514118Z","shell.execute_reply.started":"2026-02-28T10:51:44.559332Z","shell.execute_reply":"2026-02-28T10:52:15.513478Z"}}
# Generate test predictions for both methods
baseline_preds = predict_baseline(test_bundle_ids, k=TOP_K)

# Auto-select best method
use_segmented = recall_seg >= recall_base
method_name   = "segmented" if use_segmented else "baseline"
final_preds   = segmented_preds if use_segmented else baseline_preds
print(f"Selected method: {method_name} (Recall@{TOP_K} = {recall_seg if use_segmented else recall_base:.4f})")

# Global fallback: top-15 most frequent products in training set
from collections import Counter  # noqa: E402

product_freq = Counter(
    pid
    for gt_set in train_gt.values()
    for pid in gt_set
    if pid in valid_product_set
)
fallback_pids = [pid for pid, _ in product_freq.most_common(TOP_K)]
print(f"Fallback products (most frequent in train): {fallback_pids[:5]} …")

# Build submission rows
rows: list[dict] = []
missing_count = 0
for bid in test_bundle_ids:
    preds = final_preds.get(bid, [])
    if not preds:
        preds = fallback_pids
        missing_count += 1
    # Pad to exactly TOP_K if needed
    if len(preds) < TOP_K:
        for fp in fallback_pids:
            if fp not in preds:
                preds.append(fp)
            if len(preds) >= TOP_K:
                break
    preds = preds[:TOP_K]
    for pid in preds:
        rows.append({"bundle_asset_id": bid, "product_asset_id": pid})

submission_df = pd.DataFrame(rows)
submission_df.to_csv(SUBM_FILE, index=False)
print(f"\nSubmission saved to {SUBM_FILE}")
print(f"  Rows: {len(submission_df):,}")
print(f"  Fallback bundles: {missing_count}")

# ── Sanity checks ────────────────────────────────────────────────────────────
print("\nRunning sanity checks …")

# 1. All 455 test bundles covered
covered = set(submission_df["bundle_asset_id"].unique())
missing = set(test_bundle_ids) - covered
assert len(missing) == 0, f"Missing bundles in submission: {missing}"
print(f"  [OK] All {len(test_bundle_ids)} test bundles present.")

# 2. Max 15 predictions per bundle
max_preds = submission_df.groupby("bundle_asset_id")["product_asset_id"].count().max()
assert max_preds <= TOP_K, f"Bundle has {max_preds} predictions (max={TOP_K})"
print(f"  [OK] Max predictions per bundle: {max_preds}")

# 3. All product IDs are valid (in the full catalog)
predicted_pids = set(submission_df["product_asset_id"])
# Validate against the full catalog (not just downloaded) — downloaded set may be partial
all_catalog_pids = set(products_df["product_asset_id"])
invalid_pids = predicted_pids - all_catalog_pids
if invalid_pids:
    print(f"  [WARN] {len(invalid_pids)} product IDs not in catalog — removing.")
    submission_df = submission_df[~submission_df["product_asset_id"].isin(invalid_pids)]

    # Refill bundles that now have < TOP_K predictions
    existing_by_bundle: dict[str, list[str]] = (
        submission_df.groupby("bundle_asset_id")["product_asset_id"]
        .apply(list).to_dict()
    )
    refill_rows: list[dict] = []
    for bid in test_bundle_ids:
        curr = existing_by_bundle.get(bid, [])
        for fp in fallback_pids:
            if len(curr) >= TOP_K:
                break
            if fp not in curr:
                curr.append(fp)
                refill_rows.append({"bundle_asset_id": bid, "product_asset_id": fp})
    if refill_rows:
        submission_df = pd.concat(
            [submission_df, pd.DataFrame(refill_rows)], ignore_index=True
        )
        print(f"  Refilled {len(refill_rows)} predictions from fallback.")
    submission_df.to_csv(SUBM_FILE, index=False)
else:
    print(f"  [OK] All {len(predicted_pids):,} product IDs are valid catalog entries.")

print("\nSubmission preview:")
print(submission_df.head(20).to_string(index=False))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 12: Text Re-ranking Demo

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:52:16.606498Z","iopub.execute_input":"2026-02-28T10:52:16.606955Z","iopub.status.idle":"2026-02-28T10:52:17.608671Z","shell.execute_reply.started":"2026-02-28T10:52:16.606921Z","shell.execute_reply":"2026-02-28T10:52:17.607946Z"}}
# text_rerank() is defined in Cell 9 — demo only here.
print("Demonstrating text re-ranking …")

demo_rerank_bids = [bid for bid in train_bundle_ids_with_gt[:3] if bid in seg_train_preds]

for bid in demo_rerank_bids:
    img = load_image(bid, BUND_DIR)
    if img is None:
        continue

    try:
        seg_map = segment_image(img)
        crops   = extract_segment_crops(img, seg_map)
    except Exception:
        crops = {}

    crop_imgs = list(crops.values()) if crops else [img]

    ce = _l2_norm(_encode_images_raw(crop_imgs))

    before = seg_train_preds.get(bid, [])[:TOP_K]
    after  = text_rerank(bid, before[:], ce)

    gt = train_gt.get(bid, set())
    hit_before = len(set(before) & gt)
    hit_after  = len(set(after) & gt)
    print(f"  {bid}: hits before={hit_before}, after text-rerank={hit_after} (gt={len(gt)})")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Summary

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-28T10:52:26.539479Z","iopub.execute_input":"2026-02-28T10:52:26.540177Z","iopub.status.idle":"2026-02-28T10:52:26.546297Z","shell.execute_reply.started":"2026-02-28T10:52:26.540150Z","shell.execute_reply":"2026-02-28T10:52:26.545604Z"}}
print("\n" + "=" * 60)
print("HACKUDC 2026 — SOLUTION SUMMARY")
print("=" * 60)
print(f"Products indexed         : {index.ntotal:,}")
print(f"Embedding dimension      : {DIM}")
print(f"FAISS backend            : {'GPU' if DEVICE == 'cuda' else 'CPU'}")
print(f"Segmentation model       : SegFormer B2 (ATR 17 classes)")
print(f"Retrieval model          : {'Marqo-FashionSigLIP' if USE_MARQO else 'FashionCLIP'} (512-dim)")
print(f"Baseline Recall@{TOP_K}    : {recall_base:.4f}")
print(f"Segmented Recall@{TOP_K}   : {recall_seg:.4f}")
print(f"Method selected          : {method_name}")
print(f"Submission file          : {SUBM_FILE}")
print(f"Submission rows          : {len(submission_df):,}")
print("=" * 60)