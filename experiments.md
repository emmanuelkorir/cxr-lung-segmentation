# Experiment log

| Date | Stage | Model / method | Dataset | Dice | IoU | Notes |
|------|-------|----------------|---------|------|-----|-------|

**Stage 1 (EDA) — findings, no model** (full dataset, Kaggle kernel `cxr-01-dataset-eda-montgomery-shenzhen` v3):
138 Montgomery pairs (the only supervised signal) + 662 Shenzhen images; every mask loads as {0, 1} after /255 normalization; lung fraction median ≈ 27% (range 24–30%); sizes mixed, no dimension divisible by 16 → downstream resize to 512×512.

**Bar to beat** (decided in `notebooks/01_eda.ipynb` Q5): Stage 2's classical Otsu + morphology baseline — Dice ≈ 0.75 visually means ~3/4 of the lung overlapped; Stage 3's U-Net must beat the classical number to justify GPU training.
