from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from pathlib import Path
from collections import defaultdict

app = FastAPI(title="MXNJ — Outfit Intelligence")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── Load data ──────────────────────────────────────────────────────────────────
submission = pd.read_csv(BASE_DIR / "data/submission.csv")
products   = pd.read_csv(BASE_DIR / "data/products.csv")
bundles    = pd.read_csv(BASE_DIR / "data/bundles.csv")

# Fast lookup dicts
product_map = products.set_index("product_asset_id")[["product_image_url","product_description"]].to_dict("index")
bundle_map  = bundles.set_index("bundle_asset_id")[["bundle_image_url","bundle_id_section"]].to_dict("index")

# Bundle → predicted products
bundle_predictions: dict[str, list[str]] = defaultdict(list)
for _, row in submission.iterrows():
    bundle_predictions[row["bundle_asset_id"]].append(row["product_asset_id"])

all_bundle_ids = list(bundle_predictions.keys())

# Section number → human label
SECTION_LABELS = {1: "Woman", 2: "Man", 3: "Kid"}

def section_label(sec) -> str:
    try:
        return SECTION_LABELS.get(int(sec), f"Section {sec}")
    except (TypeError, ValueError):
        return str(sec)

# Category display order — head to toe, then bags and accessories
CATEGORY_ORDER = [
    "HAT", "HAT/HEADBAND", "GLASSES", "EARRINGS", "NECKLACE", "SCARF",
    "COAT", "PUFFER", "WIND-JACKET", "JACKET", "BLAZER", "WAISTCOAT",
    "SHIRT", "T-SHIRT", "BLOUSE", "TOP", "TOPS AND OTHERS", "BODY",
    "SWEATER", "SWEATSHIRT", "KNIT SWEATER",
    "DRESS", "JUMPSUIT", "OVERALL",
    "TROUSERS", "JEANS", "CARGO PANTS", "SHORTS", "BERMUDA", "BERMUDAS",
    "SKIRT", "LEGGINGS",
    "SHOES", "FLAT SHOES", "HEELED SHOES", "SNEAKERS", "TRAINERS",
    "ANKLE BOOT", "HEELED BOOT", "RAIN BOOT", "SANDAL", "SPORTY SANDAL",
    "MOCCASINS", "SOCKS",
    "HAND BAG-RUCKSACK", "BELT",
]

def category_sort_key(desc: str) -> int:
    desc_upper = (desc or "").upper()
    for i, cat in enumerate(CATEGORY_ORDER):
        if cat in desc_upper:
            return i
    return len(CATEGORY_ORDER)

# Fixed female bundle for hero top-right slot
HERO_FEMALE_BUNDLE = "B_4b729b800473"

# ── API ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/hero")
async def get_hero():
    """Returns exactly 4 bundles for the hero grid — separated from the main grid."""
    hero_ids = all_bundle_ids[:4]
    result   = []
    for bid in hero_ids:
        bdata = bundle_map.get(bid, {})
        result.append({
            "id":         bid,
            "image":      bdata.get("bundle_image_url", ""),
            "section":    section_label(bdata.get("bundle_id_section", "")),
            "n_products": len(bundle_predictions[bid]),
        })

    # Force female bundle into index 1 (top-right slot)
    if HERO_FEMALE_BUNDLE in bundle_map:
        bdata  = bundle_map[HERO_FEMALE_BUNDLE]
        female = {
            "id":         HERO_FEMALE_BUNDLE,
            "image":      bdata.get("bundle_image_url", ""),
            "section":    section_label(bdata.get("bundle_id_section", "")),
            "n_products": len(bundle_predictions.get(HERO_FEMALE_BUNDLE, [])),
        }
        result = [r for r in result if r["id"] != HERO_FEMALE_BUNDLE]
        result.insert(1, female)
        result = result[:4]

    return {"bundles": result}

@app.get("/api/bundles")
async def get_bundles(page: int = Query(1, ge=1), size: int = Query(12, ge=1, le=48)):
    start    = (page - 1) * size
    end      = start + size
    page_ids = all_bundle_ids[start:end]
    result   = []
    for bid in page_ids:
        bdata = bundle_map.get(bid, {})
        result.append({
            "id":         bid,
            "image":      bdata.get("bundle_image_url", ""),
            "section":    section_label(bdata.get("bundle_id_section", "")),
            "n_products": len(bundle_predictions[bid]),
        })
    return {
        "bundles": result,
        "total":   len(all_bundle_ids),
        "page":    page,
        "pages":   (len(all_bundle_ids) + size - 1) // size,
    }

@app.get("/api/bundle/{bundle_id}")
async def get_bundle(bundle_id: str):
    if bundle_id not in bundle_predictions:
        return JSONResponse({"error": "Bundle not found"}, status_code=404)

    bdata    = bundle_map.get(bundle_id, {})
    pred_ids = bundle_predictions[bundle_id]

    predicted = []
    for pid in pred_ids:
        pdata = product_map.get(pid, {})
        predicted.append({
            "id":          pid,
            "image":       pdata.get("product_image_url", ""),
            "description": pdata.get("product_description", "ITEM"),
        })

    predicted.sort(key=lambda p: category_sort_key(p["description"]))

    return {
        "id":        bundle_id,
        "image":     bdata.get("bundle_image_url", ""),
        "section":   section_label(bdata.get("bundle_id_section", "")),
        "predicted": predicted,
    }

@app.get("/api/stats")
async def get_stats():
    desc_counts = defaultdict(int)
    for pid in submission["product_asset_id"]:
        desc = product_map.get(pid, {}).get("product_description", "OTHER")
        desc_counts[desc] += 1
    top_categories = sorted(desc_counts.items(), key=lambda x: -x[1])[:8]
    return {
        "total_bundles":  len(all_bundle_ids),
        "total_products": submission["product_asset_id"].nunique(),
        "avg_per_bundle": round(len(submission) / len(all_bundle_ids), 1),
        "top_categories": [{"name": k, "count": v} for k, v in top_categories],
    }
