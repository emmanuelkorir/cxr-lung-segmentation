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

Datasets come from Kaggle via the CLI and are stored in `data/` (gitignored, never committed):
`uv run kaggle datasets download -d <dataset-slug> -p data/`

## Structure

```
notebooks/       Jupyter notebooks (edited in VSCode, run on Kaggle)
src/             Reusable code: data utilities, losses, metrics
data/            Datasets via kaggle CLI (gitignored)
experiments.md   Log of results (Dice/IoU) per experiment
```

## Setup

```bash
uv sync                 # install dependencies from uv.lock
uv run kaggle --version # verify the Kaggle CLI
```

Kaggle API token goes to `~/.kaggle/kaggle.json` (never committed).
