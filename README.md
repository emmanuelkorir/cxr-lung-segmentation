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

Datasets stay out of git; Kaggle notebooks attach the full data natively (`/kaggle/input/pulmonary-chest-xray-abnormalities`).

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

## Setup

```bash
uv sync                 # install dependencies from uv.lock
uv run kaggle --version # verify the Kaggle CLI
```

Kaggle API token goes to `~/.kaggle/kaggle.json` (never committed).
