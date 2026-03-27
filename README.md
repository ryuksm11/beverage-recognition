# Beverage Recognition System

An end-to-end image classification pipeline that identifies packaged beverages
from photos using a fine-tuned CNN, then retrieves product specifications and
packaging details from a structured knowledge base.

---

## What it does

1. **Classifies** the beverage using EfficientNet-B0 (transfer learning, 9 classes)
2. **Reads the label** with OCR (Tesseract) to detect bottle size and flavor variant
3. **Returns** the product name, confidence score, detected size/flavor, and top-3 alternatives
4. **Looks up** brand details, ingredients, packaging, and manufacturer from a curated product database
5. **Displays** everything in a Streamlit web app

---

## Architecture

```
User Image (upload)
        │
        ▼
  ┌─────────────┐     ┌────────────────────────────────────┐
  │  Streamlit  │────▶│  inference/predict.py              │
  │  app/       │     │  EfficientNet-B0 → softmax top-k   │
  └─────────────┘     │  + OCR (size, flavor detection)    │
                      └──────────────┬─────────────────────┘
                                     │ predicted class + volume_ml + flavor
                                     ▼
                       ┌─────────────────────────────────────┐
                       │  inference/retriever.py             │
                       │  exact / fuzzy name lookup          │
                       │  packaging variant match by volume  │
                       └──────────────┬──────────────────────┘
                                      │ ProductInfo JSON
                                      ▼
                            ┌──────────────────────┐
                            │  UI renders:         │
                            │  • class + confidence│
                            │  • detected size     │
                            │  • product card      │
                            │  • top-k bar chart   │
                            └──────────────────────┘

Training pipeline (offline):
  Bing images → data_cleaner → augmentation → EfficientNet-B0
  → models/best_checkpoint.pth  (not tracked — see below)

Product DB (offline):
  OpenFoodFacts API + manual overrides → data/product_db/{class}.json
```

---

## Supported beverage classes

9 classes — index order matches ImageFolder alphabetical sort (ground truth for label encoding):

| Index | Class     | Brand origin  |
|-------|-----------|---------------|
| 0     | 7UP       | United States |
| 1     | Coca-Cola | United States |
| 2     | Fanta     | United States |
| 3     | Maaza     | India         |
| 4     | Mirinda   | United States |
| 5     | Pepsi     | United States |
| 6     | Red Bull  | Austria       |
| 7     | Sprite    | United States |
| 8     | Tropicana | United States |

---

## Model performance

Trained on EfficientNet-B0 with two-phase fine-tuning + Albumentations augmentation + background randomization (v2):

| Metric | v1 (baseline) | v2 (current) |
|--------|--------------|--------------|
| Test accuracy | 82.98% | **87.23%** |
| Per-class F1 (min) | 0.762 (Tropicana) | 0.788 (Pepsi) |
| Per-class F1 (max) | 0.889 (Red Bull) | 0.966 (Red Bull) |

All 9 classes met the ≥ 0.65 F1 gate.

| Class | F1 (v2) |
|-------|---------|
| 7UP | 0.867 |
| Coca-Cola | 0.857 |
| Fanta | 0.875 |
| Maaza | 0.897 |
| Mirinda | 0.848 |
| Pepsi | 0.788 |
| Red Bull | 0.966 |
| Sprite | 0.919 |
| Tropicana | 0.833 |

> **Note:** v2 was retrained with rembg background segmentation + random indoor backgrounds
> to reduce domain shift between studio training images and real-world phone photos.
> The app shows a warning and hides product details when confidence < 60%.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Via conda — do NOT use 3.13, PyTorch wheels are incomplete |
| conda | any | Anaconda or Miniconda |
| Tesseract | 5.x | Required for OCR — `brew install tesseract` (macOS) |
| Git | any | |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd "CNN Project end2end"
```

### 2. Create and activate the conda environment

```bash
conda create -n beverage-cnn python=3.11 -y
conda activate beverage-cnn
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract (macOS)

```bash
brew install tesseract
```

### 5. Create required directories

```bash
python setup_project.py
```

Creates `data/`, `models/`, `logs/`, and `__init__.py` package files.

---

## Training from scratch

Follow these steps in order to reproduce the trained model.

### Step 1 — Download training images

```bash
python training/downloader.py
```

Downloads ~200 images per class into `data/raw/{class_name}/`.
After download, manually inspect and remove obviously wrong images (blurry, multi-brand, non-packaged).

### Step 2 — Clean the dataset

```bash
python utils/data_cleaner.py
```

Removes corrupt files and exact duplicates (by MD5 hash).

### Step 3 — Split into train / val / test

```bash
python training/dataset.py
```

Stratified 70/15/15 split → `data/processed/`.

### Step 4 — Segment training images (optional — improves real-world accuracy)

Requires a separate conda env to avoid numpy conflicts:

```bash
conda create -n rembg-env python=3.11 -y
conda activate rembg-env
pip install "rembg[cpu]" pillow tqdm pyyaml
python scripts/segment_training_images.py   # saves RGBA PNGs to data/segmented/
conda activate beverage-cnn
```

### Step 5 — Download background images (optional — pairs with step 4)

```bash
python scripts/download_backgrounds.py      # ~240 indoor backgrounds → data/backgrounds/
```

### Step 6 — Smoke-test the training pipeline

```bash
python scripts/smoke_test_training.py
```

Runs 2 epochs on 10 images/class. Verifies the loop works before committing to a full run.

### Step 7 — Train the model

```bash
python training/train.py
```

Two-phase training:
- Phase 1 (10 epochs): classifier head only, EfficientNet base frozen
- Phase 2 (10 epochs): fine-tune last 3 EfficientNet blocks

If `data/segmented/` and `data/backgrounds/` exist, training automatically applies background-paste augmentation (70% probability per image).

Saves `models/best_checkpoint_v2.pth`, `models/confusion_matrix.png`, `models/training_curves.png`.

### Step 6 — Scrape the product knowledge base

```bash
python scraper/scrape_products.py

# Scrape a single class only
python scraper/scrape_products.py --class "Coca-Cola"
```

Queries OpenFoodFacts and merges with curated manual overrides.
Saves `data/product_db/{class_name}.json`.

---

## Running the app

### Verify the pipeline end-to-end

```bash
python scripts/verify_ui.py
```

Headless check: loads Predictor + ProductRetriever, validates all 9 product records and schema keys.

### Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 — upload a photo of a packaged beverage.

---

## Tests

```bash
pytest tests/ -v
```

| File | Tests | What it covers |
|---|---|---|
| `tests/test_dataset.py` | 10 | DataLoader shapes, split ratios, class balance |
| `tests/test_model.py` | 5 | Model output shape, frozen layers, checkpoint save/load |
| `tests/test_predict.py` | 13 | Prediction schema, confidence range, OCR volume detection |
| `tests/test_retriever.py` | 14 | Exact lookup, fuzzy lookup (typo/case), missing-key fallback, container type |

---

## Project structure

```
.
├── config/
│   └── config.yaml              # All hyperparams, paths, class names
├── data/
│   ├── raw/                     # Downloaded images (gitignored — large)
│   ├── processed/               # Train/val/test splits (gitignored — large)
│   └── product_db/              # 9 × curated JSON files (tracked)
├── models/
│   ├── best_checkpoint.pth      # Trained weights (gitignored — 16 MB binary)
│   ├── confusion_matrix.png     # Tracked
│   └── training_curves.png      # Tracked
├── scraper/
│   ├── product_schema.py        # ProductInfo + PackagingVariant TypedDicts
│   └── scrape_products.py       # OpenFoodFacts + manual overrides scraper
├── training/
│   ├── downloader.py            # Bing image crawler
│   ├── augmentation.py          # Train / eval transforms
│   ├── dataset.py               # BeverageDataset + DataLoaders
│   ├── model.py                 # EfficientNet-B0 builder
│   ├── train.py                 # Two-phase training loop
│   └── evaluate.py              # Metrics, confusion matrix, training curves
├── inference/
│   ├── predict.py               # Predictor class — top-k + OCR
│   └── retriever.py             # ProductRetriever — exact + fuzzy lookup
├── app/
│   └── streamlit_app.py         # Streamlit web UI
├── utils/
│   ├── config_loader.py         # load_config() — used everywhere
│   ├── logger.py                # get_logger() — used everywhere
│   ├── seed.py                  # set_global_seed() + get_device()
│   ├── data_cleaner.py          # Corrupt-file removal + deduplication
│   └── ocr_helper.py            # Tesseract OCR — volume + flavor extraction
├── tests/                       # pytest test suite (42 tests total)
├── scripts/
│   ├── smoke_test_training.py   # 2-epoch sanity check
│   ├── verify_ui.py             # Headless end-to-end pipeline check
│   ├── segment_training_images.py  # rembg background removal (run in rembg-env)
│   └── download_backgrounds.py     # Bing crawler — indoor background images
├── setup_project.py             # One-time directory scaffolding
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## Output schemas

### Prediction (`inference/predict.py`)

```json
{
  "class": "Tropicana",
  "confidence": 0.97,
  "flavor": "Orange",
  "volume_ml": 500,
  "top_k": [
    {"class": "Tropicana", "confidence": 0.97},
    {"class": "Maaza",     "confidence": 0.02},
    {"class": "Fanta",     "confidence": 0.01}
  ]
}
```

`flavor` and `volume_ml` are `null` if OCR cannot read them from the label.

### Product info (`data/product_db/{class_name}.json`)

```json
{
  "class_name": "Coca-Cola",
  "brand": "Coca-Cola",
  "brand_origin": "United States",
  "manufacturer": "The Coca-Cola Company",
  "local_manufacturer": "Hindustan Coca-Cola Beverages Pvt Ltd",
  "product_type": "Carbonated Soft Drink",
  "flavors": ["Original", "Zero Sugar", "Diet"],
  "ingredients": ["Carbonated Water", "Sugar", "Caramel Color", "Phosphoric Acid", "Caffeine"],
  "packaging": [
    {"type": "can",        "volume_ml": 330},
    {"type": "PET bottle", "volume_ml": 500},
    {"type": "PET bottle", "volume_ml": 1500}
  ],
  "website": "https://www.coca-cola.com",
  "image_url": null,
  "last_scraped": "2026-03-24T00:00:00Z"
}
```

`local_manufacturer` is `null` for Red Bull (imported, not bottled locally).

---

## What is and isn't tracked in git

| Path | Tracked | Reason |
|---|---|---|
| `data/product_db/*.json` | Yes | Small, curated, needed to run the app |
| `models/*.png` | Yes | Eval output images — lightweight |
| `data/raw/` | No | ~200 imgs × 9 classes — large, reproducible via downloader |
| `data/processed/` | No | Derived from raw — regenerate with `dataset.py` |
| `data/segmented/` | No | Generated by `segment_training_images.py` — large |
| `data/backgrounds/` | No | Downloaded by `download_backgrounds.py` — large |
| `data/test/` | No | Personal device photos — never commit |
| `models/*.pth` | No | 16 MB binary — regenerate with `train.py` |
| `logs/` | No | Runtime logs only |
| `.claude/` | No | Local AI assistant state — not project code |

To reproduce the full project from a fresh clone, run Steps 1–6 under "Training from scratch".

---

## Evaluation gates

| Metric | Threshold |
|---|---|
| Val accuracy after Phase 1 | ≥ 45% |
| Val accuracy after Phase 2 | ≥ 80% |
| Per-class F1 (all classes) | ≥ 0.65 |

---

## License

MIT
