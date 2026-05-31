# Benchmarking
This folder contains code specific for probing/downstream evaluation. The normal nanopath loop is to train within the 1,000,000-sample and 1e18-FLOP caps, freeze the model backbone, run the fixed downstream probe suite, and use that result to decide whether a training idea is worth scaling. The benchmark definition stays fixed for fair model comparisons. For the most part we borrow the same approach used by THUNDER / PathoBench for downstream evaluations. Most probes hold out official test splits; `cptac_pda_os` instead uses PathoBench's official five-fold survival evaluation because that task is defined as cross-validation.

## Metric

`mean_probe_score` is the unweighted mean of one scalar per dataset and is the single score we use to assess relative performance of trained nanopath models.

```text
mean_probe_score = mean(
  break_his, bracs, mhist, pcam,
  monusac, consep, pannuke,
  ucla_lung,
  surgen,
  cptac_pda_os,
  pathorob
)
```

## Probe Families

| family | datasets | dataset scalar |
|---|---|---|
| Tile classification | `break_his`, `bracs`, `mhist`, `pcam` | mean of linear, KNN, and 16-shot SimpleShot macro F1|
| Segmentation | `monusac`, `consep`, `pannuke` | macro Jaccard from a small MaskTransformer head on frozen patch tokens |
| Progression | `ucla_lung` | AUROC from a balanced logistic linear probe on mean-pooled slide embeddings |
| Mutation | `surgen` | AUROC for PathoBench SR386 RAS mutation status |
| Survival | `cptac_pda_os` | Harrell's c-index from official PathoBench folds with fixed PCA(2) L2 CoxPH |
| Robustness | `pathorob` | mean PathoROB-style robustness index across camelyon and tolkach_esca |

All probes keep the backbone frozen. Probe heads are intentionally small: they measure representation quality, not downstream fine-tuning capacity.

Probe heads consume each model's native frozen feature dimension rather than projecting every backbone to a common width. This intentionally evaluates each checkpoint as a deployed feature extractor, but it also means dimensionality is part of the baseline comparison: DINOv2-small emits 384-d features, DINOv2-G/OpenMidnight/H-optimus-0/UNI-2-h/GigaPath emit 1536-d features, Virchow emits 2560-d CLS plus mean-patch features, Midnight-12K emits 3072-d CLS plus mean-patch features, and GenBio-PathFM emits 4608-d features.

Linear, KNN, segmentation-head, and logistic hyperparameters are selected on the same internal validation splits that define `mean_probe_score`. Thunder-derived tile classifiers keep Thunder-style linear/KNN/16-shot SimpleShot heads; SimpleShot precomputes 1000 deterministic support sets and majority-votes query predictions. PathoBench-derived slide classifiers use balanced logistic linear probing; SurGen uses sklearn's `liblinear` solver. CPTAC-PDA OS uses fixed `z-score -> PCA(2) -> CoxPH(alpha=100)` on each official PathoBench fold after case-level slide pooling; the head is fixed because Coxnet sweeps inflated random-init survival scores during validation.

## Runtime Strategy

The full suite is designed to stay lightweight for the standard small Nanopath model by keeping the benchmark small where it can be small, and precomputing expensive slide tiling.

- Whole-slide tasks use cached tile grids, so the final probe embeds JPEG/parquet tiles rather than opening full WSIs. PathoBench-derived slide tasks use a 20x, 512 px, 0-overlap tissue grid following the Trident/PathoBench tutorial contract. UCLA Lung and CPTAC-PDA OS embed the full cached grid; SurGen streams deterministic up-to-768-tile raster-spaced sub-bags per slide so the larger mutation task fits the final-probe window. UCLA Lung and SurGen use pre-extracted parquet caches by default; CPTAC-PDA OS downloads TCIA PathDB SVS files and builds its cache locally. The remaining preprocessing simplification is a deterministic thumbnail tissue mask instead of invoking Trident's HEST segmentation model during `prepare.py`.
- PCam is a fixed subset of the official train/valid H5 files, mirrored as small H5s with the same filenames/schema.
- Tile classifiers use `Resize((224, 224))` from `model.py::probe_transforms` for trained Nanopath checkpoints. Frozen baseline scripts set their own `probe.transform_policy`; Virchow and GigaPath use their official timm bicubic 224 center-crop. Patch-cache probes keep square resize because their inputs are already extracted tissue tiles.
- Segmentation runs in a background thread while classification, slide, survival, and robustness probes run in the main worker for DINOv2-style backbones. CUDA kernels still serialize, but CPU-heavy decode/head work overlaps with segmentation head training.
- The same loaded frozen backbone serves every probe in one subprocess, avoiding repeated model load overhead.
- Official held-out test splits are not read by most probes. `cptac_pda_os` is the exception: PathoBench defines the survival task by five official train/test folds, and Nanopath reports their mean c-index.

Recent H100 timings from the latest untouched baseline reruns:

| dataset | DINOv2-random | DINOv2-small | DINOv2-G | EXAONE-Path | Virchow | GigaPath | UNI-2-h | Midnight-12K | OpenMidnight | H-optimus-0 | GenBio-PathFM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bracs` | 157.4s | 158.8s | 168.4s | 172.1s | 183.5s | 182.0s | 166.8s | 167.8s | 169.0s | 166.5s | 151.6s |
| `break_his` | 18.5s | 19.1s | 19.5s | 14.2s | 18.7s | 18.8s | 18.0s | 18.4s | 18.1s | 19.1s | 14.2s |
| `mhist` | 13.1s | 12.8s | 24.4s | 16.2s | 23.8s | 19.6s | 22.4s | 24.6s | 23.4s | 25.4s | 20.8s |
| `pcam` | 24.6s | 23.7s | 44.0s | 30.5s | 34.9s | 38.3s | 43.0s | 44.9s | 43.9s | 45.5s | 42.4s |
| `pannuke` | 154.8s | 154.7s | 281.1s | 192.3s | 249.9s | 235.9s | 274.7s | 288.2s | 291.9s | 292.5s | 268.3s |
| `monusac` | 24.2s | 24.0s | 120.9s | 29.1s | 57.5s | 43.4s | 118.0s | 129.0s | 131.8s | 135.2s | 32.6s |
| `consep` | 4.2s | 4.8s | 36.9s | 4.4s | 26.1s | 13.0s | 34.3s | 32.6s | 36.0s | 35.6s | 5.4s |
| `ucla_lung` | 26.9s | 26.7s | 63.2s | 26.1s | 46.4s | 48.4s | 44.3s | 62.8s | 64.8s | 63.3s | 139.3s |
| `surgen` | 194.2s | 198.6s | 413.2s | 193.9s | 255.0s | 287.6s | 251.1s | 398.0s | 425.5s | 400.1s | 1131.9s |
| `cptac_pda_os` | 169.7s | 163.9s | 276.0s | 169.5s | 181.4s | 209.4s | 174.5s | 270.5s | 271.6s | 274.1s | 769.6s |
| `pathorob` | 20.0s | 19.6s | 67.5s | 20.9s | 43.2s | 50.7s | 43.1s | 67.0s | 67.4s | 67.5s | 191.0s |

GenBio-PathFM is a slow outlier because each RGB tile is encoded as three single-channel ViT-G passes and the heads consume native 4608-d features. GenBio-PathFM baseline also runs segmentation sequentially because its three channel-wise ViT-G passes are already GPU-bound, and background PanNuke contention made the baseline much slower without changing the metric.

## Dataset Summary

| dataset | task | tissue / organ | train units | val units | train tiles/images | val tiles/images | reference H100 time | source | Nanopath adaptation |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `break_his` | tile classification (~700×460 microscope captures) | breast | 936 images | 196 images | 936 | 196 | 20.4s | BreaKHis 40X / EVA-Thunder | Patient-disjoint 4-subtype 40X split; linear/KNN/16-shot SimpleShot; no test scoring |
| `bracs` | ROI classification (variable-size WSI crops) | breast | 3657 ROIs | 312 ROIs | 3657 | 312 | 179.9s | BRACS ROI FTP | Frozen-embedding linear/KNN/16-shot SimpleShot; no official test scoring |
| `mhist` | tile classification (224 px patches) | colorectal polyps | 1743 images | 432 images | 1743 | 432 | 26.3s | MHIST | Official train partition split internally; linear/KNN/16-shot SimpleShot; official test not read |
| `pcam` | tile classification (96 px patches) | lymph node metastasis | 3072 images | 768 images | 3072 | 768 | 50.0s | PCam Zenodo | Fixed train/valid subset; linear/KNN/16-shot SimpleShot; official test not read |
| `monusac` | segmentation | multi-organ nuclei | ~31 slides/fold | ~15 slides/fold | 209 total images | 3 folds | 91.5s | MoNuSAC train set | Deterministic 3-fold slide split of train package; no test data |
| `consep` | segmentation | colorectal nuclei | 18 ROIs/fold | 9 ROIs/fold | 27 total images | 3 folds | 18.5s | CoNSeP | Deterministic 3-fold split of official Train folder; Test folder not read |
| `pannuke` | segmentation | multi-organ nuclei | Fold1 | Fold2 | 2656 images | 2523 images | 309.5s | PanNuke folds | Fixed Fold1/Fold2 protocol; Fold3 not scored |
| `ucla_lung` | slide progression classification | lung | 60 slides/fold | 30 slides/fold | full 20x/512 grid | 3 folds | 63.6s | PathoBench `ucla_lung/progression_regression` / IDR idr0082 | 3-fold balanced logistic AUROC over fold-0 train using the full tissue grid; 22-slide test fold held out |
| `surgen` | mutation classification | colorectal | ~207 slides/fold | ~104 slides/fold | 1,167,089 cached tiles; up to 768 embedded/slide | 3 folds | 386.8s | PathoBench SR386 / SurGen, mirrored as pre-extracted HF parquet | 3-fold validation over PathoBench fold-0 train; fold-0 test sealed |
| `cptac_pda_os` | survival | pancreatic ductal adenocarcinoma | 77-78 cases/fold | 19-20 cases/fold | 146,896 cached tiles; full grid embedded | 5 official folds | 163.9s | PathoBench CPTAC-PDA OS / TCIA PathDB | official PathoBench folds; case-level pooling; train-fold z-score + PCA(2) + fixed L2 CoxPH |
| `pathorob` | robustness | breast lymph node + esophagus | n/a | n/a | 22,402 + 13,800 used patches | n/a | 74.5s | PathoROB HF datasets | Robustness index over camelyon/tolkach_esca; TCGA subset excluded |

## Files

Split metadata used directly by `probe.py`:

- `bracs.json`
- `break_his.json`
- `mhist.json`
- `ucla_lung.json`
- `surgen.json`
- `cptac_pda_os.json`

Datasets without JSON here are split directly by code: PCam uses fixed subset H5s, PanNuke uses Fold1/Fold2, MoNuSAC and CoNSeP use deterministic train-folder splits, and PathoROB reads its public parquet subsets directly.

Dataset-specific notes:

- [break_his.md](break_his.md)
- [bracs.md](bracs.md)
- [mhist.md](mhist.md)
- [pcam.md](pcam.md)
- [monusac.md](monusac.md)
- [consep.md](consep.md)
- [pannuke.md](pannuke.md)
- [ucla_lung.md](ucla_lung.md)
- [surgen.md](surgen.md)
- [cptac_pda_os.md](cptac_pda_os.md)
- [pathorob.md](pathorob.md)
