# cxr-lung-segmentation

Reproducible experiments in chest X-ray lung segmentation (PyTorch + MONAI, trained on Kaggle GPUs).

## Learning roadmap

| Stage | Goal | Notebook |
|-------|------|----------|
| 0 | Understand segmentation, Dice/IoU metrics | — |
| 1 | Dataset EDA: sizes, mask stats, overlay masks | `notebooks/01_eda.ipynb` |
| 2 | Classical baseline (Otsu + morphology) | `notebooks/02_classical_baseline.ipynb` |
| 3 | U-Net from scratch in pure PyTorch | `notebooks/03_unet_pytorch.ipynb` |
| 4 | MONAI (transforms, pretrained backbones) comparison | `notebooks/04_monai_unet.ipynb` |
| 5 | Capstone: SIIM-ACR pneumothorax segmentation | `notebooks/05_siim_pneumothorax.ipynb` |

## What Stage 1 established (verified on the full 800-image dataset — Kaggle run `cxr-01-dataset-eda-montgomery-shenzhen` v3)

- **Splits**: Montgomery ships the only ground-truth masks — **138 image·mask·mask pairs** (leftMask/rightMask folder counts differ by ±1 because of junk files; the stem-based pairing + image-suffix filter handle it). Shenzhen is images only (662) — useful for unlabeled/classical experiments, not supervised training.
- **Geometry**: sizes are mixed and one image is portrait, and **no dimension is divisible by 16** (`4020`, `4892`) ⇒ the U-Net in Stage 3 will resize to a fixed 512×512 (image and mask with the *same* transform).
- **Masks**: all 276 mask files are 0/255 PNGs that load clean as {0, 1} — `load_grayscale`'s `/255` normalization does it; later notebooks should reuse that loader rather than re-derive it.
- **Appearance**: Montgomery pixel means ≈ 0.39–0.49 (stored "inverted" vs classic radiography conventions); Shenzhen images look visibly brighter.
- **Lung area**: the lung occupies ≈ **24–30%** of a CXR (median ~27%) — sanity anchor for Q3 and for what Dice predicts (an "all empty" mask scores Dice 0).

Datasets stay out of git; Kaggle notebooks attach the full data natively (Kaggle
mounts API-attached datasets nested at `/kaggle/input/datasets/<owner>/<slug>`,
so every notebook finds the dataset root by name in its first cell).

**Full data** (local copy or own Kaggle upload):
`uv run kaggle datasets download -d kmader/pulmonary-chest-xray-abnormalities -p data/ --unzip`

**Committed sample** — `data/samples/` (~44 MB) holds 4 Montgomery pairs (2 TB-neg, 2 TB-pos, left/right masks + clinical notes) and 4 Shenzhen images, mirroring the real folder layout, so notebooks can be smoke-tested locally on CPU without the 4 GB download.

## Structure

```
notebooks/       Jupyter notebooks (edited in VSCode, run on Kaggle)
src/             Reusable code: data utilities, losses, metrics
data/            Datasets via kaggle CLI (gitignored; data/samples/ is a committed mini-excerpt)
experiments.md   Log of results (Dice/IoU) per experiment
```

## Running on Kaggle

Notebooks are pushed with the Kaggle CLI (no browser). Each notebook has a
descriptor template in `kernels/*.kernel-metadata.json` (the file the CLI looks
up next to the notebook):

```bash
# one-time: put your API credentials in ~/.kaggle/  (access_token or kaggle.json,
# from kaggle.com → Settings → API; never committed)
cp kernels/01_eda.kernel-metadata.json notebooks/kernel-metadata.json
uv run kaggle kernels push -p notebooks                        # upload + start a run
uv run kaggle kernels status oxoooo/cxr-01-dataset-eda-montgomery-shenzhen # poll until "complete"
uv run kaggle kernels output oxoooo/cxr-01-dataset-eda-montgomery-shenzhen -p /tmp/kaggle-out/  # pull logs + outputs
```

(The kernel `id` must match the slug Kaggle derives from `title` — the CLI
warns and later 409s otherwise. Use the slug from the pushed code URL.)

The descriptor attaches the full dataset (auto-detected by each notebook's
first cell under the nested Kaggle mount) and toggles `enable_gpu` only for training stages.
Local edits stay source of truth — re-push to re-run. The co-located
`notebooks/kernel-metadata.json` holds your personal slug and is gitignored.

## Setup

```bash
uv sync                 # install dependencies from uv.lock
uv run kaggle --version # verify the Kaggle CLI
```

Kaggle API credentials go to `~/.kaggle/` (`access_token` or `kaggle.json`, per
the CLI's auth convention — never committed).
