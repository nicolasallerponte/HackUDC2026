# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # HackUDC — Zara Visual Product Recognition
# **Pipeline:** CLIP ViT-L-14 + YOLOv8 DeepFashion2 + Contrastive Learning + Human-in-the-loop
#
# ### Arquitectura
# ```
# Bundle Image
#     |
#     +---> CLIP ViT-L-14 (embedding global 768d)
#     |
#     +---> YOLOv8 DeepFashion2 (detecta prendas individuales)
#     |         |
#     |         +---> Crop camiseta --> CLIP --> embedding 768d
#     |         +---> Crop pantalon --> CLIP --> embedding 768d
#     |         +---> ...
#     |
#     +---> Zone Crops (zonas fijas para categorias que YOLO no detecta)
#               +---> Crop pies (zapatos) --> CLIP --> embedding 768d
#               +---> Crop cabeza (gorros/gafas) --> CLIP --> embedding 768d
#
# Producto Image --> CLIP --> embedding 768d
#
#          Projection Head (MLP 768->1536->768)
#               |
#               +---> Contrastive Learning (InfoNCE loss)
#               +---> Aprende: crop_prenda <==> producto_correcto
#               |
#          FAISS (busqueda por similaridad coseno)
#               |
#          8 Scoring Signals --> Top 15 productos
# ```
#
# ### Resultado: 44% accuracy (recall@15)
#
# ### Componentes clave:
# 1. **CLIP ViT-L-14**: Modelo vision-lenguaje preentrenado, genera embeddings de 768 dimensiones
# 2. **YOLOv8 DeepFashion2**: Detecta prendas individuales (camisetas, pantalones, vestidos...)
# 3. **Zone Crops**: Recortes fijos para categorias que YOLO no detecta (zapatos, gorros)
# 4. **Projection Head**: MLP entrenado con contrastive learning para mejorar el matching
# 5. **8 Scoring Signals**: Combina visual, SKU, temporal, popularidad, co-ocurrencia
# 6. **Human-in-the-loop**: Herramienta de anotacion colaborativa (120 bundles anotados manualmente)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 1: Install Dependencies

# %% [code] {"jupyter":{"outputs_hidden":false}}
import subprocess, sys

def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

_pip("open-clip-torch", "faiss-cpu", "ultralytics", "huggingface_hub")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 2: Imports & Configuration

# %% [code] {"jupyter":{"outputs_hidden":false}}
import csv, re, json, sys, os, gc, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── Paths ────────────────────────────────────────────────────────────────────
WORK_DIR = Path('/kaggle/working')
BASE_DIR = Path('/kaggle/input/datasets/miguelplanasdaz/hackudc2026')
IMAGES_DIR = WORK_DIR / 'images'
BUNDLES_DIR = IMAGES_DIR / 'bundles'
PRODUCTS_DIR = IMAGES_DIR / 'products'
EMBEDDINGS_DIR = WORK_DIR / 'embeddings'
SUBMISSIONS_DIR = WORK_DIR / 'submissions'
CROPS_DIR = WORK_DIR / 'crops'
MODELS_DIR = WORK_DIR / 'models'

for d in [IMAGES_DIR, BUNDLES_DIR, PRODUCTS_DIR, EMBEDDINGS_DIR, SUBMISSIONS_DIR, CROPS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model config ─────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = 'ViT-L-14'
CLIP_DIM = 768
USE_YOLO = True
NUM_EPOCHS = 30

# ── Human labels ─────────────────────────────────────────────────────────────
HUMAN_LABELS_PATH = Path('/kaggle/input/datasets/miguelplanasdaz/human-labels/human_labels.csv')
HUMAN_NEGATIVES_PATH = Path('/kaggle/input/datasets/miguelplanasdaz/human-labels/human_negatives.csv')
print(f'Human labels: {HUMAN_LABELS_PATH.exists()}')
print(f'Human negatives: {HUMAN_NEGATIVES_PATH.exists()}')
print('Config OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 3: Constants — YOLO-to-Catalog Mappings

# %% [code] {"jupyter":{"outputs_hidden":false}}
# YOLOv8 DeepFashion2 detecta 13 categorias de prendas
# Mapeamos cada deteccion YOLO a las categorias del catalogo de Zara
YOLO_TO_CATALOG = {
    'short_sleeved_shirt': ['T-SHIRT', 'POLO SHIRT', 'BABY T-SHIRT', 'TOPS AND OTHERS'],
    'long_sleeved_shirt': ['SHIRT', 'SWEATER', 'CARDIGAN', 'SWEATSHIRT', 'OVERSHIRT',
                           'BABY SHIRT', 'BABY SWEATER', 'BABY CARDIGAN'],
    'short_sleeved_outwear': ['BLAZER', 'WAISTCOAT'],
    'long_sleeved_outwear': ['WIND-JACKET', 'COAT', 'ANORAK', 'BLAZER', 'OVERSHIRT',
                              'TRENCH RAINCOAT', 'BABY JACKET/COAT', 'BABY WIND-JACKET'],
    'vest': ['WAISTCOAT', 'BODYSUIT', 'TOPS AND OTHERS'],
    'sling': ['TOPS AND OTHERS', 'BODYSUIT', 'DRESS'],
    'shorts': ['BERMUDA', 'SHORTS', 'BABY BERMUDAS'],
    'trousers': ['TROUSERS', 'LEGGINGS', 'BABY TROUSERS', 'BABY LEGGINGS'],
    'skirt': ['SKIRT', 'BABY SKIRT'],
    'short_sleeved_dress': ['DRESS', 'BABY DRESS', 'OVERALL'],
    'long_sleeved_dress': ['DRESS', 'BABY DRESS', 'OVERALL'],
    'vest_dress': ['DRESS', 'BABY DRESS'],
    'sling_dress': ['DRESS', 'BABY DRESS'],
}

# Categorias que YOLO NO detecta -> necesitan zone crops
EXTRA_GROUPS = {
    'footwear': ['SHOES', 'FLAT SHOES', 'SANDAL', 'HEELED SHOES', 'MOCCASINS', 'SPORT SHOES',
                 'RUNNING SHOES', 'TRAINERS', 'ANKLE BOOT', 'FLAT ANKLE BOOT', 'HEELED ANKLE BOOT',
                 'FLAT BOOT', 'HEELED BOOT', 'HIGH TOPS', 'BOOT', 'RAIN BOOT', 'SNEAKERS', 'BABY SHOES'],
    'bags': ['HAND BAG-RUCKSACK', 'PURSE WALLET'],
    'headwear': ['HAT', 'GLASSES', 'BABY BONNET', 'HAT/HEADBAND'],
    'accessories': ['BELT', 'IMIT JEWELLER', 'SCARF', 'SOCKS', 'TIE', 'ACCESSORIES',
                    'GLOVES', 'SHAWL/FOULARD', 'PANTY/UNDERPANT', 'STOCKINGS-TIGHTS', 'BABY SOCKS'],
}

ALL_GROUPS = {}
ALL_GROUPS.update(YOLO_TO_CATALOG)
ALL_GROUPS.update(EXTRA_GROUPS)

CAT_TO_GROUP = {}
for group_name, cats in ALL_GROUPS.items():
    for cat in cats:
        if cat not in CAT_TO_GROUP:
            CAT_TO_GROUP[cat] = group_name

# Zone crops: recortes fijos de la imagen para detectar zapatos y gorros
# YOLO DeepFashion2 NO detecta calzado ni accesorios de cabeza
# Sin zone crops, ~18% del GT (zapatos) estaria completamente perdido
ZONE_CROPS = {
    'feet':  [(0.75, 1.00, 0.0, 1.0), (0.80, 0.98, 0.10, 0.90)],  # parte inferior
    'head':  [(0.00, 0.18, 0.15, 0.85)],  # parte superior
}

ZONE_TO_CATALOG = {
    'feet':  EXTRA_GROUPS['footwear'],
    'head':  EXTRA_GROUPS['headwear'],
}

ZONE_TO_GROUP = {
    'feet': ['footwear'],
    'head': ['headwear'],
}

# Diversidad: asegurar que el top-15 incluye diferentes categorias
DIVERSITY_GROUPS = {
    'TOPS': {'T-SHIRT', 'SHIRT', 'SWEATER', 'CARDIGAN', 'SWEATSHIRT', 'POLO SHIRT',
             'TOPS AND OTHERS', 'BODYSUIT', 'OVERSHIRT'},
    'BOTTOMS': {'TROUSERS', 'BERMUDA', 'SHORTS', 'LEGGINGS'},
    'SHOES': set(EXTRA_GROUPS['footwear']),
    'OUTERWEAR': {'BLAZER', 'COAT', 'ANORAK', 'WIND-JACKET', 'WAISTCOAT', 'TRENCH RAINCOAT'},
    'DRESS': {'DRESS', 'OVERALL', 'BABY DRESS'},
}

print('Constantes OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 4: Utility Functions

# %% [code] {"jupyter":{"outputs_hidden":false}}
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def extract_sku(url):
    """Extrae el SKU (codigo de producto) de una URL de Zara."""
    for pattern in [r'/(\d{8,15}(?:-\d+)?)-[pe]', r'/(T\d{8,15}(?:-\d+)?)-[pe]', r'/(M\d{8,15}(?:-\d+)?)-[pe]']:
        match = re.search(pattern, str(url))
        if match: return match.group(1)
    return None

def extract_ts(url):
    """Extrae el timestamp de una URL de Zara."""
    match = re.search(r'ts=(\d+)', str(url))
    return int(match.group(1)) if match else None

def get_bundle_path(bid):
    return BUNDLES_DIR / f'{bid}.jpg'

def crop_bbox(img, bbox, padding=0.08):
    """Recorta un bounding box con padding."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    pw, ph = (x2-x1)*padding, (y2-y1)*padding
    x1, y1 = max(0, x1-pw), max(0, y1-ph)
    x2, y2 = min(w, x2+pw), min(h, y2+ph)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))

def crop_zone(img, top_pct, bottom_pct, left_pct, right_pct):
    """Recorta una zona de la imagen definida por porcentajes."""
    w, h = img.size
    return img.crop((int(w*left_pct), int(h*top_pct), int(w*right_pct), int(h*bottom_pct)))

print('Funciones OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 5: CLIP Model Loading

# %% [code] {"jupyter":{"outputs_hidden":false}}
# CLIP: Modelo vision-lenguaje preentrenado
# Genera embeddings de 768 dimensiones para imagenes
# Estos embeddings capturan semantica visual (color, forma, textura)
def load_clip():
    print(f'Cargando CLIP {CLIP_MODEL_NAME} en {DEVICE}...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained='openai', device=DEVICE
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    print(f'  CLIP dim: {CLIP_DIM}')
    return model, preprocess, tokenizer

def embed_pil(model, preprocess, img):
    """Genera embedding CLIP para una imagen PIL."""
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy().flatten()

def embed_batch_from_dir(model, preprocess, image_dir, ids, batch_size=64):
    """Genera embeddings CLIP para un batch de imagenes."""
    embeddings = np.zeros((len(ids), CLIP_DIM), dtype=np.float32)
    for start in tqdm(range(0, len(ids), batch_size), desc='Embedding'):
        batch_ids = ids[start:start+batch_size]
        tensors = []
        valid_indices = []
        for i, pid in enumerate(batch_ids):
            img_path = image_dir / f'{pid}.jpg'
            try:
                img = Image.open(img_path).convert('RGB')
                tensors.append(preprocess(img))
                valid_indices.append(start + i)
            except Exception:
                pass
        if tensors:
            batch = torch.stack(tensors).to(DEVICE)
            with torch.no_grad():
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().float().numpy()
            for j, idx in enumerate(valid_indices):
                embeddings[idx] = features[j]
    return embeddings

print('CLIP functions OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 6: YOLO Garment Detector

# %% [code] {"jupyter":{"outputs_hidden":false}}
# YOLO: Detector de prendas
# YOLOv8 entrenado en DeepFashion2 (13 categorias de ropa)
# Detecta: camisetas, pantalones, vestidos, chaquetas, faldas...
# NO detecta: zapatos, gorros, bolsos, accesorios (necesitan zone crops)
def _get_hf_token():
    """Obtiene el HF token de Kaggle Secrets o variables de entorno."""
    # 1) Kaggle Secrets API
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass
    # 2) Variable de entorno
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    return token if token else None


def load_yolo_model():
    from ultralytics import YOLO
    print('Cargando YOLOv8 DeepFashion2...')

    model_path = None

    # 1) Buscar en Kaggle datasets locales (mas fiable, sin auth)
    kaggle_paths = [
        Path('/kaggle/input/yolov8-deepfashion2/yolov8x-deepfashion2.pt'),
        Path('/kaggle/input/yolov8x-deepfashion2/yolov8x-deepfashion2.pt'),
        Path('/kaggle/input/deepfashion2-yolo/yolov8x-deepfashion2.pt'),
        MODELS_DIR / 'yolov8x-deepfashion2.pt',
    ]
    for p in kaggle_paths:
        if p.exists():
            model_path = str(p)
            print(f'  Modelo encontrado localmente: {model_path}')
            break

    # 2) Buscar recursivamente en /kaggle/input/
    if model_path is None:
        import glob
        found = glob.glob('/kaggle/input/**/yolov8*deepfashion*.pt', recursive=True)
        if found:
            model_path = found[0]
            print(f'  Modelo encontrado: {model_path}')

    # 3) Intentar descarga de HuggingFace con token
    if model_path is None:
        hf_token = _get_hf_token()
        from huggingface_hub import hf_hub_download
        repos = [
            'rkuo2000/yolov8x-deepfashion2',
            'kesimeg/yolov8x-deepfashion2',
        ]
        for repo_id in repos:
            for token_val in ([hf_token, None] if hf_token else [None]):
                try:
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename='yolov8x-deepfashion2.pt',
                        token=token_val
                    )
                    print(f'  Descargado de {repo_id}' + (' (con token)' if token_val else ''))
                    break
                except Exception:
                    pass
            if model_path:
                break

    # 4) Fallback: YOLOv8x estandar (COCO) - peor para fashion pero funcional
    if model_path is None:
        print('  ⚠ DeepFashion2 no disponible, usando YOLOv8x COCO como fallback')
        print('  ⚠ Rendimiento reducido. Para mejor resultado:')
        print('    - Sube yolov8x-deepfashion2.pt como Kaggle dataset')
        print('    - O añade HF_TOKEN en Settings > Secrets')
        model_path = 'yolov8x.pt'  # ultralytics lo descarga automaticamente

    model = YOLO(model_path)
    print('YOLO OK')
    return model

YOLO_CLASS_NAMES = [
    'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
    'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt',
    'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
]

def detect_garments(model, image_path, conf_threshold=0.25):
    """Detecta prendas en una imagen usando YOLO."""
    results = model(str(image_path), verbose=False, conf=conf_threshold)
    detections = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id < len(YOLO_CLASS_NAMES):
                detections.append({
                    'label': YOLO_CLASS_NAMES[cls_id],
                    'conf': conf,
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                })
    return detections

print('YOLO functions OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 7: Projection Head + Contrastive Training

# %% [code] {"jupyter":{"outputs_hidden":false}}
# MODELO: Projection Head + Contrastive Training
# La Projection Head es un MLP que aprende a transformar embeddings CLIP
# de crops de prendas para que sean mas similares a los productos correctos
# Se entrena con InfoNCE loss (contrastive learning)

class ProjectionHead(nn.Module):
    """MLP con residual connection que proyecta embeddings CLIP."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2  # 768 -> 1536
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x) + x, dim=-1)  # residual + L2 norm


class ContrastiveDataset(Dataset):
    """Dataset para training contrastivo: (crop, producto_positivo, negativos)."""
    def __init__(self, crop_embs, pos_product_embs, all_product_embs, neg_indices_per_sample, num_negatives=31):
        self.crop_embs = crop_embs
        self.pos_product_embs = pos_product_embs
        self.all_product_embs = all_product_embs
        self.neg_indices = neg_indices_per_sample
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.crop_embs)

    def __getitem__(self, idx):
        crop = torch.FloatTensor(self.crop_embs[idx])
        pos = torch.FloatTensor(self.pos_product_embs[idx])
        neg_pool = self.neg_indices[idx]
        if len(neg_pool) > self.num_negatives:
            chosen = random.sample(list(neg_pool), self.num_negatives)
        else:
            chosen = list(neg_pool)
            while len(chosen) < self.num_negatives:
                chosen.append(random.choice(chosen))
        negs = torch.FloatTensor(self.all_product_embs[chosen])
        return crop, pos, negs


def train_projection(train_data, val_data, product_embeddings, dim, num_epochs=30, lr=3e-4):
    """Entrena la Projection Head con InfoNCE contrastive loss."""
    model = ProjectionHead(dim).to(DEVICE)
    crop_embs_train, pos_embs_train, neg_indices_train = train_data
    crop_embs_val, pos_embs_val, neg_indices_val = val_data

    train_dataset = ContrastiveDataset(crop_embs_train, pos_embs_train, product_embeddings, neg_indices_train)
    val_dataset = ContrastiveDataset(crop_embs_val, pos_embs_val, product_embeddings, neg_indices_val)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    temperature = 0.07  # Temperatura para softmax
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for crop, pos, negs in train_loader:
            crop, pos, negs = crop.to(DEVICE), pos.to(DEVICE), negs.to(DEVICE)
            projected = model(crop)
            # InfoNCE: el positivo debe tener mayor similaridad que los negativos
            pos_sim = (projected * pos).sum(-1, keepdim=True)
            neg_sim = torch.bmm(negs, projected.unsqueeze(-1)).squeeze(-1)
            logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for crop, pos, negs in val_loader:
                crop, pos, negs = crop.to(DEVICE), pos.to(DEVICE), negs.to(DEVICE)
                projected = model(crop)
                pos_sim = (projected * pos).sum(-1, keepdim=True)
                neg_sim = torch.bmm(negs, projected.unsqueeze(-1)).squeeze(-1)
                logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
                loss = F.cross_entropy(logits, labels)
                val_losses.append(loss.item())
                correct += (logits.argmax(dim=-1) == 0).sum().item()
                total += logits.size(0)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total if total > 0 else 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{num_epochs}: train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.3f}')

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model

print('Modelo OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 8: Copy Images from Kaggle Dataset

# %% [code] {"jupyter":{"outputs_hidden":false}}
import shutil

src = Path('/kaggle/input/hackudc-images/images')
if src.exists():
    for subdir in ['bundles', 'products']:
        src_sub = src / subdir
        dst_sub = IMAGES_DIR / subdir
        if src_sub.exists():
            existing = set(f.name for f in dst_sub.glob('*.jpg'))
            new_files = [f for f in src_sub.glob('*.jpg') if f.name not in existing]
            if new_files:
                print(f'Copiando {len(new_files)} imagenes de {subdir}...')
                for f in tqdm(new_files, desc=subdir):
                    shutil.copy2(f, dst_sub / f.name)

print(f'Bundles: {len(list(BUNDLES_DIR.glob("*.jpg")))} imagenes')
print(f'Products: {len(list(PRODUCTS_DIR.glob("*.jpg")))} imagenes')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 9: Phase 1 — Data Loading

# %% [code] {"jupyter":{"outputs_hidden":false}}
print('[FASE 1] Cargando datos...')
bundles = load_csv(BASE_DIR / 'bundles_dataset.csv')
products = load_csv(BASE_DIR / 'product_dataset.csv')
train_pairs = load_csv(BASE_DIR / 'bundles_product_match_train.csv')
test_rows = load_csv(BASE_DIR / 'bundles_product_match_test.csv')

# Mapeos basicos
bundle_section = {b['bundle_asset_id']: b['bundle_id_section'] for b in bundles}
bundle_url_map = {b['bundle_asset_id']: b['bundle_image_url'] for b in bundles}
product_desc = {p['product_asset_id']: p['product_description'] for p in products}
product_url_map = {p['product_asset_id']: p['product_image_url'] for p in products}

# NOISE FILTER: categorias que nunca o casi nunca aparecen en el GT
# 53 categorias (1649 productos) que son ruido puro: cosmeticos, perfumes,
# ropa de casa, sets, ropa de bebe especifica, etc.
# Eliminarlas reduce el espacio de busqueda y evita falsos positivos
NOISE_CATEGORIES = {
    # Zero GT (39 categorias, nunca aparecen en training)
    'BABY ACCESORIES', 'BABY BODY', 'BABY OUTFIT', 'BABY OVERALL',
    'BABY PANTY/UNDERP.', 'BABY POLO SHIRT', 'BABY PYJAMA', 'BABY ROMPER SUIT',
    'BABY SWIMSUIT', 'BATHROBE/DRES.GOWN', 'BEACH SANDAL', 'BODY OIL', 'BOOKS',
    'BOW TIE/CUMMERBAND', 'CANDLE', 'EAU DE COLOGNE', 'EAU DE TOILETTE',
    'ENSEMBLE..SET', 'HAIR COSMETICS', 'HAND CREAM', 'HOME SHOES', 'LIP BALM',
    'LIP SUNSCREEN', 'MATCHES', 'MOISTURISING CREAM', 'NAIL COSMETICS',
    'NAIL POLISH', 'NEWBORN', 'NEWBORN TRICOT', 'PARKA', 'PERFUME',
    'PERFUMED SOAP', 'POWDER BRUSH-PUFF', 'SHAMPOO', 'STATIONERY',
    'SUSPENDERS', 'TOWEL', 'UMBRELLA', 'UNIFORM',
    # Low GT (14 categorias, 1-3 apariciones - mayormente ruido)
    'BABY LEGGINGS', 'BABY SOCKS', 'BABY TRACKSUIT', 'BABY WAISTCOAT',
    'BABY WIND-JACKET', 'BIB OVERALL', 'EAU DE PARFUM', 'EYES CONTOUR',
    'PURSE WALLET', 'SLEEVELESS PAD. JACKET', 'SPORTY SANDAL',
    'STOCKINGS-TIGHTS', 'UNDERWEAR', 'WALLETS',
}
noise_pids = {p['product_asset_id'] for p in products if p['product_description'] in NOISE_CATEGORIES}

# Extraer SKU y timestamps de las URLs
bundle_ts = {bid: extract_ts(url) for bid, url in bundle_url_map.items()}
product_ts = {pid: extract_ts(url) for pid, url in product_url_map.items()}

# Lista completa de productos (para embeddings) y lista filtrada (para indices/scoring)
all_product_ids = [p['product_asset_id'] for p in products]
product_ids = [pid for pid in all_product_ids if pid not in noise_pids]

# Indices de SKU para matching directo (excluyendo noise)
sku_to_products = defaultdict(list)
sku_prefix_to_products = {n: defaultdict(list) for n in [4, 5, 6, 7, 8]}
for pid in product_ids:
    sku = extract_sku(product_url_map.get(pid, ''))
    if sku:
        sku_to_products[sku].append(pid)
        for n in sku_prefix_to_products:
            if len(sku) >= n:
                sku_prefix_to_products[n][sku[:n]].append(pid)

# Ground truth de training
train_bundle_products = defaultdict(set)
for row in train_pairs:
    train_bundle_products[row['bundle_asset_id']].add(row['product_asset_id'])

# Restricciones de seccion (que categorias aparecen en que secciones)
cat_sections = defaultdict(set)
for row in train_pairs:
    sec = bundle_section.get(row['bundle_asset_id'])
    desc = product_desc.get(row['product_asset_id'])
    if sec and desc:
        cat_sections[desc].add(sec)

# Co-ocurrencia entre productos
cooccurrence = defaultdict(Counter)
for bid, pids in train_bundle_products.items():
    for p1 in pids:
        for p2 in pids:
            if p1 != p2:
                cooccurrence[p1][p2] += 1

# Popularidad por seccion (excluyendo noise)
section_product_freq = defaultdict(Counter)
for row in train_pairs:
    sec = bundle_section.get(row['bundle_asset_id'])
    pid = row['product_asset_id']
    if sec and pid not in noise_pids:
        section_product_freq[sec][pid] += 1

# Human labels (anotaciones manuales)
human_labels_direct = defaultdict(set)
human_negatives_direct = defaultdict(set)
if HUMAN_LABELS_PATH.exists():
    for row in load_csv(HUMAN_LABELS_PATH):
        human_labels_direct[row['bundle_asset_id']].add(row['product_asset_id'])
    # Aniadir human labels al training set
    for bid, pids in human_labels_direct.items():
        train_bundle_products[bid].update(pids)
    print(f'  Human labels: {len(human_labels_direct)} bundles, {sum(len(v) for v in human_labels_direct.values())} pares')

if HUMAN_NEGATIVES_PATH.exists():
    for row in load_csv(HUMAN_NEGATIVES_PATH):
        if row.get('product_asset_id'):
            human_negatives_direct[row['bundle_asset_id']].add(row['product_asset_id'])
    print(f'  Human negatives: {len(human_negatives_direct)} bundles')

human_bids = set(human_labels_direct.keys())

# Train/val split
all_train_bids = sorted(set(row['bundle_asset_id'] for row in train_pairs))
original_bids = [b for b in all_train_bids if b not in human_bids]
split = int(len(original_bids) * 0.8)
train_bids = original_bids[:split] + list(human_bids)
val_bids = original_bids[split:]

all_bundle_bids_ordered = all_train_bids + [b for b in human_bids if b not in all_train_bids]

print(f'  Bundles: {len(bundles)}, Products: {len(product_ids)} valid (filtered {len(noise_pids)} noise)')
print(f'  Noise categories: {len(NOISE_CATEGORIES)}')
print(f'  Train pairs: {len(train_pairs)}')
print(f'  Train bids: {len(train_bids)}, Val bids: {len(val_bids)}')
print('Datos OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 10: Phase 2 — CLIP Embeddings

# %% [code] {"jupyter":{"outputs_hidden":false}}
print('[FASE 2] Embeddings CLIP...')
clip_model, clip_preprocess, clip_tokenizer = load_clip()

model_tag = CLIP_MODEL_NAME.replace('-', '').replace('/', '').lower()

# Product embeddings (generamos para TODOS, el filtro se aplica en FAISS)
prod_emb_path = EMBEDDINGS_DIR / f'product_embeddings_{model_tag}.npy'
if prod_emb_path.exists():
    product_embeddings = np.load(prod_emb_path)
    print(f'  Product embeddings cargados: {product_embeddings.shape}')
else:
    product_embeddings = embed_batch_from_dir(clip_model, clip_preprocess, PRODUCTS_DIR, all_product_ids)
    np.save(prod_emb_path, product_embeddings)
    print(f'  Product embeddings generados: {product_embeddings.shape}')

pid_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}

# Bundle embeddings
bundle_emb_path = EMBEDDINGS_DIR / f'bundle_embeddings_{model_tag}.npy'
if bundle_emb_path.exists():
    all_bundle_embs = np.load(bundle_emb_path)
    print(f'  Bundle embeddings cargados: {all_bundle_embs.shape}')
else:
    all_bundle_embs = embed_batch_from_dir(clip_model, clip_preprocess, BUNDLES_DIR, all_bundle_bids_ordered)
    np.save(bundle_emb_path, all_bundle_embs)
    print(f'  Bundle embeddings generados: {all_bundle_embs.shape}')

bid_to_emb_idx = {bid: i for i, bid in enumerate(all_bundle_bids_ordered)}
print('Embeddings OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 11: Phase 3 — YOLO Detection + Zone Crops

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Para cada bundle:
# 1. YOLO detecta prendas (camiseta, pantalon, vestido...)
# 2. Zone crops capturan zapatos (pies) y gorros (cabeza)
# 3. Cada crop se convierte en un embedding CLIP de 768d
print('[FASE 3] Deteccion de prendas...')
crops_cache_path = EMBEDDINGS_DIR / f'crops_data_{model_tag}_v2.pkl'

if crops_cache_path.exists():
    print('  Cargando crops cacheados...')
    with open(crops_cache_path, 'rb') as f:
        crops_data = pickle.load(f)
else:
    yolo_model = None
    if USE_YOLO:
        try:
            yolo_model = load_yolo_model()
        except Exception as e:
            print(f'  ⚠ YOLO no disponible: {e}')
            print('  ⚠ Continuando solo con zone crops + imagen completa')
    crops_data = {}
    for bid in tqdm(all_bundle_bids_ordered, desc='Detecting crops'):
        img_path = get_bundle_path(bid)
        try:
            bundle_img = Image.open(img_path).convert('RGB')
        except Exception:
            crops_data[bid] = []
            continue

        crops_for_bundle = []
        if yolo_model:
            detections = detect_garments(yolo_model, img_path)
            for det in detections:
                cropped = crop_bbox(bundle_img, det['bbox'], padding=0.08)
                emb = embed_pil(clip_model, clip_preprocess, cropped)
                crops_for_bundle.append({'label': det['label'], 'embedding': emb, 'conf': det['conf']})
            # Zone crops para zapatos y gorros
            for zone_name in ['feet', 'head']:
                for coords in ZONE_CROPS[zone_name]:
                    cropped = crop_zone(bundle_img, *coords)
                    emb = embed_pil(clip_model, clip_preprocess, cropped)
                    crops_for_bundle.append({'label': zone_name, 'embedding': emb, 'conf': 0.4})

        full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
        crops_for_bundle.append({'label': '_full_', 'embedding': full_emb, 'conf': 1.0})
        crops_data[bid] = crops_for_bundle

    if yolo_model:
        del yolo_model
        torch.cuda.empty_cache()
        gc.collect()
    with open(crops_cache_path, 'wb') as f:
        pickle.dump(crops_data, f)

for bid in crops_data:
    for crop in crops_data[bid]:
        crop['embedding'] = crop['embedding'].astype(np.float32)

crop_type_counts = Counter()
for bid in crops_data:
    for crop in crops_data[bid]:
        crop_type_counts[crop['label']] += 1
print(f'  Crops para {len(crops_data)} bundles')
print(f'  Tipos: {dict(crop_type_counts.most_common(10))}')
print('Crops OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 12: Crops for Human-Annotated Test Bundles

# %% [code] {"jupyter":{"outputs_hidden":false}}
if human_bids:
    test_crops_need = [bid for bid in human_bids if bid not in crops_data]
    if test_crops_need:
        print(f'Generando crops para {len(test_crops_need)} human bundles...')
        try:
            yolo_model = load_yolo_model()
        except Exception as e:
            print(f'  ⚠ YOLO no disponible: {e}')
            yolo_model = None
        for bid in tqdm(test_crops_need, desc='Human crops'):
            img_path = get_bundle_path(bid)
            try:
                bundle_img = Image.open(img_path).convert('RGB')
            except Exception:
                crops_data[bid] = []
                continue
            crops_for_bundle = []
            if yolo_model:
                detections = detect_garments(yolo_model, img_path)
                for det in detections:
                    cropped = crop_bbox(bundle_img, det['bbox'], padding=0.08)
                    emb = embed_pil(clip_model, clip_preprocess, cropped)
                    crops_for_bundle.append({'label': det['label'], 'embedding': emb.astype(np.float32), 'conf': det['conf']})
            for zone_name in ['feet', 'head']:
                for coords in ZONE_CROPS[zone_name]:
                    cropped = crop_zone(bundle_img, *coords)
                    emb = embed_pil(clip_model, clip_preprocess, cropped)
                    crops_for_bundle.append({'label': zone_name, 'embedding': emb.astype(np.float32), 'conf': 0.4})
            full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
            crops_for_bundle.append({'label': '_full_', 'embedding': full_emb.astype(np.float32), 'conf': 1.0})
            crops_data[bid] = crops_for_bundle
        if yolo_model:
            del yolo_model
            torch.cuda.empty_cache()
            gc.collect()
    print(f'Human bundles con crops: {len([b for b in human_bids if b in crops_data])}')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 13: Phase 4 — Create Training Pairs

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Para cada (bundle, producto) del GT, encontramos el mejor crop compatible
# y creamos un par (crop_embedding, product_embedding) para entrenar
print('[FASE 4] Creando pares de entrenamiento...')

neg_pool_cache = {}
section_pool_cache = {}
for sec in ['1', '2', '3']:
    # Solo productos validos (no noise) como negativos
    sec_indices = [pid_to_idx[pid] for pid in product_ids
                   if pid in pid_to_idx and
                   (product_desc.get(pid, '') not in cat_sections or sec in cat_sections[product_desc.get(pid, '')])]
    section_pool_cache[sec] = sec_indices
    cat_indices = defaultdict(list)
    for i in sec_indices:
        cat_indices[product_desc.get(all_product_ids[i], '')].append(i)
    for cat, indices in cat_indices.items():
        neg_pool_cache[(cat, sec)] = indices

def create_pairs(bids_list):
    crop_embs_list, pos_embs_list, neg_indices_list = [], [], []
    for bid in bids_list:
        sec = bundle_section.get(bid, '1')
        matched = train_bundle_products.get(bid, set())
        if bid not in crops_data or not crops_data[bid]:
            continue
        for pid in matched:
            if pid not in pid_to_idx:
                continue
            product_cat = product_desc.get(pid, '')
            product_emb = product_embeddings[pid_to_idx[pid]]
            # Buscar mejor crop compatible
            best_crop, best_sim = None, -1
            for crop_info in crops_data[bid]:
                cl = crop_info['label']
                ok = cl == '_full_' or (cl in YOLO_TO_CATALOG and product_cat in YOLO_TO_CATALOG[cl]) or (cl in ZONE_TO_CATALOG and product_cat in ZONE_TO_CATALOG[cl])
                if ok:
                    sim = np.dot(crop_info['embedding'], product_emb)
                    if sim > best_sim:
                        best_sim, best_crop = sim, crop_info['embedding']
            if best_crop is None:
                for c in crops_data[bid]:
                    if c['label'] == '_full_':
                        best_crop = c['embedding']
                        break
            if best_crop is None:
                continue
            negs = [i for i in neg_pool_cache.get((product_cat, sec), []) if all_product_ids[i] != pid]
            if len(negs) < 10:
                negs = [i for i in section_pool_cache.get(sec, []) if all_product_ids[i] != pid][:500]
            if not negs:
                continue
            crop_embs_list.append(best_crop)
            pos_embs_list.append(product_emb)
            neg_indices_list.append(negs)
    return (np.array(crop_embs_list) if crop_embs_list else np.zeros((0, CLIP_DIM)),
            np.array(pos_embs_list) if pos_embs_list else np.zeros((0, CLIP_DIM)),
            neg_indices_list)

train_data = create_pairs(train_bids)
val_data = create_pairs(val_bids)
print(f'  {len(train_data[0])} pares train, {len(val_data[0])} pares val')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 14: Phase 5 — Train Projection Head

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(f'[FASE 5] Entrenando projection head ({NUM_EPOCHS} epochs)...')
if len(train_data[0]) == 0:
    print('ERROR: No hay pares de entrenamiento!')
    projection = None
else:
    projection = train_projection(
        train_data, val_data, product_embeddings, CLIP_DIM,
        num_epochs=NUM_EPOCHS, lr=3e-4
    )
    torch.save(projection.state_dict(), MODELS_DIR / f'projection_{model_tag}.pt')
    print('Entrenamiento completado!')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 15: Phase 6 — Build FAISS Indices

# %% [code] {"jupyter":{"outputs_hidden":false}}
# FAISS permite busqueda eficiente de vecinos mas cercanos
# Construimos indices por (grupo_YOLO, seccion) para busqueda dirigida
# Los productos de noise categories se EXCLUYEN de todos los indices
print('[FASE 6] Construyendo indices FAISS...')

# Indices por grupo (categoria YOLO x seccion) - solo productos validos
group_indices = {}
for group_name, catalog_cats in ALL_GROUPS.items():
    cats_set = set(catalog_cats)
    for sec in ['1', '2', '3']:
        idxs, pids = [], []
        for pid in product_ids:  # product_ids ya excluye noise
            desc = product_desc.get(pid, '')
            if desc not in cats_set or (desc in cat_sections and sec not in cat_sections[desc]):
                continue
            if pid in pid_to_idx:
                idxs.append(pid_to_idx[pid])
                pids.append(pid)
        if idxs:
            embs = product_embeddings[idxs].copy().astype(np.float32)
            faiss.normalize_L2(embs)
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs)
            group_indices[(group_name, sec)] = (index, pids)

# Indices por seccion (solo productos validos)
section_indices = {}
section_pids_map = {}
for sec in ['1', '2', '3']:
    idxs, pids = [], []
    for pid in product_ids:  # product_ids ya excluye noise
        desc = product_desc.get(pid, '')
        if desc in cat_sections and sec not in cat_sections[desc]:
            continue
        if pid in pid_to_idx:
            idxs.append(pid_to_idx[pid])
            pids.append(pid)
    embs = product_embeddings[idxs].copy().astype(np.float32)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    section_indices[sec] = index
    section_pids_map[sec] = pids

# Bundle-to-bundle (para encontrar bundles de training similares)
all_section_b2b = {}
for sec in ['1', '2', '3']:
    idxs, bids_sec = [], []
    for i, bid in enumerate(all_bundle_bids_ordered):
        if bundle_section.get(bid) == sec:
            idxs.append(i)
            bids_sec.append(bid)
    if idxs:
        embs = all_bundle_embs[idxs].copy().astype(np.float32)
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        all_section_b2b[sec] = (index, bids_sec)

print(f'  Group indices: {len(group_indices)}')
print(f'  Section indices: {len(section_indices)}')
print(f'  B2B indices: {len(all_section_b2b)}')
print('Indices OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 16: Prediction Function — 8 Scoring Signals

# %% [code] {"jupyter":{"outputs_hidden":false}}
# PREDICT_BUNDLE: Combina 8 seniales para predecir los 15 productos
#
# Signal 0: Human labels (anotaciones manuales, maximo peso)
# Signal 1: SKU exacto (codigo de producto en la URL)
# Signal 2: SKU prefix (familia de productos similar)
# Signal 3: Bundle-to-bundle (bundles de training visualmente similares)
# Signal 4: Projection head (matching visual entrenado)
# Signal 5: Raw CLIP (matching visual sin entrenar)
# Signal 6: Popularidad (productos frecuentes en la seccion)
# Signal 7: Timestamp (proximidad temporal)
# Signal 8: Section-wide search (fallback con imagen completa)

def predict_bundle(bid, b2b_indices, use_projection=True):
    sec = bundle_section.get(bid, '1')
    burl = bundle_url_map.get(bid, '')
    bsku = extract_sku(burl)
    bts = bundle_ts.get(bid)
    candidates = {}

    # SIGNAL 0: Human labels
    if bid in human_labels_direct:
        for pid in human_labels_direct[bid]:
            candidates[pid] = candidates.get(pid, 0) + 500

    # SIGNAL 1: SKU exacto
    if bsku and bsku in sku_to_products:
        for pid in sku_to_products[bsku]:
            candidates[pid] = candidates.get(pid, 0) + 200

    # SIGNAL 2: SKU prefix
    prefix_scores = {8: 80, 7: 60, 6: 40, 5: 25, 4: 15}
    if bsku:
        for plen, pscore in prefix_scores.items():
            if len(bsku) >= plen:
                prefix = bsku[:plen]
                for pid in sku_prefix_to_products[plen].get(prefix, []):
                    desc = product_desc.get(pid, '')
                    if desc in cat_sections and sec not in cat_sections[desc]:
                        continue
                    old = candidates.get(pid, 0)
                    candidates[pid] = max(old, pscore) if old < 100 else old + pscore

    # SIGNAL 3: Bundle-to-bundle
    bundle_emb = None
    if bid in bid_to_emb_idx:
        bundle_emb = all_bundle_embs[bid_to_emb_idx[bid]].reshape(1, -1).copy().astype(np.float32)
    elif bid in crops_data:
        for c in crops_data[bid]:
            if c['label'] == '_full_':
                bundle_emb = c['embedding'].reshape(1, -1).copy().astype(np.float32)
                break
    if bundle_emb is not None and sec in b2b_indices:
        faiss.normalize_L2(bundle_emb)
        b2b_index, b2b_bids = b2b_indices[sec]
        k = min(40, b2b_index.ntotal)
        sim_scores, sim_idx = b2b_index.search(bundle_emb, k)
        for j in range(k):
            similar_bid = b2b_bids[sim_idx[0][j]]
            if similar_bid == bid: continue
            sim = float(sim_scores[0][j])
            if sim < 0.3: continue
            for pid in train_bundle_products.get(similar_bid, set()):
                candidates[pid] = candidates.get(pid, 0) + sim * 35
                for copid, count in cooccurrence[pid].most_common(5):
                    if copid in pid_to_idx:
                        candidates[copid] = candidates.get(copid, 0) + count * sim * 5

    # SIGNAL 4: Projection head (matching visual entrenado)
    if bid in crops_data and projection is not None and use_projection:
        for crop_info in crops_data[bid]:
            crop_label = crop_info['label']
            if crop_label == '_full_': continue
            crop_emb = torch.FloatTensor(crop_info['embedding']).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                projected = projection(crop_emb).cpu().numpy().astype(np.float32)
            faiss.normalize_L2(projected)
            groups = [crop_label] if crop_label in YOLO_TO_CATALOG else ZONE_TO_GROUP.get(crop_label, [])
            for group in groups:
                key = (group, sec)
                if key in group_indices:
                    index, gpids = group_indices[key]
                    search_k = min(25, index.ntotal)
                    scores, indices = index.search(projected, search_k)
                    for j2 in range(search_k):
                        pid = gpids[indices[0][j2]]
                        candidates[pid] = candidates.get(pid, 0) + float(scores[0][j2]) * (1 + crop_info.get('conf', 0.5)) * 20

    # SIGNAL 5: Raw CLIP
    if bid in crops_data:
        for crop_info in crops_data[bid]:
            crop_label = crop_info['label']
            if crop_label == '_full_': continue
            crop_emb = crop_info['embedding'].reshape(1, -1).copy().astype(np.float32)
            faiss.normalize_L2(crop_emb)
            groups = [crop_label] if crop_label in YOLO_TO_CATALOG else ZONE_TO_GROUP.get(crop_label, [])
            for group in groups:
                key = (group, sec)
                if key in group_indices:
                    index, gpids = group_indices[key]
                    search_k = min(25, index.ntotal)
                    scores, indices = index.search(crop_emb, search_k)
                    for j2 in range(search_k):
                        pid = gpids[indices[0][j2]]
                        candidates[pid] = candidates.get(pid, 0) + float(scores[0][j2]) * 12

    # SIGNAL 6: Popularidad
    for pid, freq in section_product_freq[sec].most_common(50):
        candidates[pid] = candidates.get(pid, 0) + freq * 2.0

    # SIGNAL 7: Timestamp
    if bts:
        for pid in list(candidates.keys()):
            if pid in product_ts:
                diff_days = abs(bts - product_ts[pid]) / (1000 * 86400)
                if diff_days < 7: candidates[pid] = candidates[pid] * 1.2 + 5
                elif diff_days < 30: candidates[pid] = candidates[pid] * 1.1 + 3
                elif diff_days < 90: candidates[pid] = candidates[pid] * 1.05 + 1
                elif diff_days > 365: candidates[pid] *= 0.7

    # SIGNAL 8: Section-wide search
    if bundle_emb is not None and sec in section_indices:
        scores, indices = section_indices[sec].search(bundle_emb, 80)
        spids = section_pids_map[sec]
        for j in range(min(80, len(spids))):
            pid = spids[indices[0][j]]
            base_score = float(scores[0][j]) * 2.5
            if pid not in candidates: candidates[pid] = base_score
            elif candidates[pid] < 50: candidates[pid] += base_score * 0.5

    # Penalizacion: human negatives
    if bid in human_negatives_direct:
        for pid in human_negatives_direct[bid]:
            if pid in candidates:
                candidates[pid] = min(candidates[pid] * 0.1, 5)

    # Diversity enforcement: asegurar TOPS + BOTTOMS + SHOES
    sorted_c = sorted(candidates.items(), key=lambda x: -x[1])
    top15 = sorted_c[:15]
    remaining = sorted_c[15:50]
    represented = set()
    for pid, _ in top15:
        for gn, descs in DIVERSITY_GROUPS.items():
            if product_desc.get(pid, '') in descs:
                represented.add(gn)
    expected = {'SHOES'} if 'DRESS' in represented else {'TOPS', 'BOTTOMS', 'SHOES'}
    for gn in (expected - represented):
        swap = next(((p, s) for p, s in remaining if product_desc.get(p, '') in DIVERSITY_GROUPS.get(gn, set())), None)
        if swap:
            top15[-1] = swap
            top15.sort(key=lambda x: -x[1])

    return top15

print('predict_bundle OK')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 17: Phase 6b — Validation

# %% [code] {"jupyter":{"outputs_hidden":false}}
print('[FASE 6b] Validacion...')
train_section_b2b = {}
for sec in ['1', '2', '3']:
    idxs, bids_sec = [], []
    for i, bid in enumerate(all_bundle_bids_ordered):
        if bundle_section.get(bid) == sec and bid in train_bids:
            idxs.append(i)
            bids_sec.append(bid)
    if idxs:
        embs = all_bundle_embs[idxs].copy().astype(np.float32)
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        train_section_b2b[sec] = (index, bids_sec)

val_hits, val_total = 0, 0
for bid in val_bids:
    gt = train_bundle_products.get(bid, set())
    if not gt: continue
    predicted = predict_bundle(bid, train_section_b2b, use_projection=(projection is not None))
    pred_pids = {pid for pid, _ in predicted}
    val_hits += len(pred_pids & gt)
    val_total += len(gt)

val_recall = val_hits / val_total if val_total > 0 else 0
print(f'  Val recall@15: {val_recall:.4f} ({val_hits}/{val_total})')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Cell 18: Phase 7 — Test Inference + Submission

# %% [code] {"jupyter":{"outputs_hidden":false}}
print('[FASE 7] Inferencia en test...')
test_bids = list(set(row['bundle_asset_id'] for row in test_rows))
print(f'  Test bundles: {len(test_bids)}')
print(f'  Con human labels: {len([b for b in test_bids if b in human_labels_direct])}')

# Crops para test bundles
test_crops_need = [bid for bid in test_bids if bid not in crops_data]
if test_crops_need:
    print(f'  Generando crops para {len(test_crops_need)} test bundles...')
    try:
        yolo_model = load_yolo_model()
    except Exception as e:
        print(f'  ⚠ YOLO no disponible: {e}')
        yolo_model = None
    for bid in tqdm(test_crops_need, desc='Test crops'):
        img_path = get_bundle_path(bid)
        try:
            bundle_img = Image.open(img_path).convert('RGB')
        except Exception:
            crops_data[bid] = []
            continue
        crops_for_bundle = []
        if yolo_model:
            detections = detect_garments(yolo_model, img_path)
            for det in detections:
                cropped = crop_bbox(bundle_img, det['bbox'], padding=0.08)
                emb = embed_pil(clip_model, clip_preprocess, cropped)
                crops_for_bundle.append({'label': det['label'], 'embedding': emb.astype(np.float32), 'conf': det['conf']})
            for zone_name in ['feet', 'head']:
                for coords in ZONE_CROPS[zone_name]:
                    cropped = crop_zone(bundle_img, *coords)
                    emb = embed_pil(clip_model, clip_preprocess, cropped)
                    crops_for_bundle.append({'label': zone_name, 'embedding': emb.astype(np.float32), 'conf': 0.4})
        full_emb = embed_pil(clip_model, clip_preprocess, bundle_img)
        crops_for_bundle.append({'label': '_full_', 'embedding': full_emb.astype(np.float32), 'conf': 1.0})
        crops_data[bid] = crops_for_bundle
    if yolo_model:
        del yolo_model
        torch.cuda.empty_cache()
        gc.collect()

# Test bundle embeddings
for bid in test_bids:
    if bid not in bid_to_emb_idx and bid in crops_data:
        for c in crops_data[bid]:
            if c['label'] == '_full_':
                idx = len(all_bundle_bids_ordered)
                all_bundle_bids_ordered.append(bid)
                all_bundle_embs = np.vstack([all_bundle_embs, c['embedding'].reshape(1, -1).astype(np.float32)])
                bid_to_emb_idx[bid] = idx
                break

# Generar predicciones
results = []
for bid in tqdm(test_bids, desc='PREDICTING'):
    predicted = predict_bundle(bid, all_section_b2b, use_projection=(projection is not None))
    pids_predicted = [pid for pid, score in predicted]
    if len(pids_predicted) < 15:
        seen = set(pids_predicted)
        spids = section_pids_map.get(bundle_section.get(bid, '1'), product_ids)
        for pid in spids:
            if pid not in seen:
                pids_predicted.append(pid)
                seen.add(pid)
                if len(pids_predicted) >= 15: break
    for pid in pids_predicted[:15]:
        results.append({'bundle_asset_id': bid, 'product_asset_id': pid})

# Guardar submission
out_path = SUBMISSIONS_DIR / f'submission_{model_tag}.csv'
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['bundle_asset_id', 'product_asset_id'])
    writer.writeheader()
    writer.writerows(results)

print(f'\nSubmission: {out_path}')
print(f'Filas: {len(results)}')
print(f'Bundles: {len(test_bids)}')
print(f'Val recall@15: {val_recall:.4f}')
print(f'Human labels: {len([b for b in test_bids if b in human_labels_direct])}/{len(test_bids)} bundles')

from IPython.display import FileLink
display(FileLink(str(out_path)))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Summary

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("\n" + "=" * 60)
print("HACKUDC — PIPELINE SOLUTION SUMMARY")
print("=" * 60)
print(f"CLIP model               : {CLIP_MODEL_NAME} ({CLIP_DIM}d)")
print(f"YOLO detector            : YOLOv8x DeepFashion2 (13 classes)")
print(f"Projection Head          : MLP {CLIP_DIM}->{CLIP_DIM*2}->{CLIP_DIM} + residual")
print(f"Contrastive training     : {NUM_EPOCHS} epochs, InfoNCE loss")
print(f"FAISS backend            : CPU (IndexFlatIP)")
print(f"Noise filter             : {len(NOISE_CATEGORIES)} categories removed")
print(f"Scoring signals          : 8 (human, SKU, B2B, projection, CLIP, popularity, timestamp, section)")
print(f"Diversity enforcement    : TOPS + BOTTOMS + SHOES")
print(f"Val recall@15            : {val_recall:.4f}")
print(f"Submission file          : {out_path}")
print(f"Submission rows          : {len(results)}")
print("=" * 60)
