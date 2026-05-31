# CPTAC-PDA OS

`cptac_pda_os` is a pancreatic ductal adenocarcinoma case-level overall-survival probe from PathoBench. It contributes one scalar to `mean_probe_score`: Harrell's c-index averaged over the official five PathoBench folds.

## Source

- Dataset: [PathoBench](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) `cptac_pda/OS`
- Labels: `MahmoodLab/Patho-Bench`, file `cptac_pda/OS/k=all.tsv`
- Upstream metadata: `task_type: survival`, `metrics: cindex`, with `OS_event` and `OS_days`
- Slide source: [TCIA CPTAC-PDA](https://www.cancerimagingarchive.net/collection/cptac-pda/) pathology images exposed by [PathDB](https://pathdb.cancerimagingarchive.net/imagesearch?f%5B0%5D=collection%3Acptac-pda)
- Local cluster cache: `/data/CPTAC-PDA`

## Split

Nanopath vendors `cptac_pda_os.json`, derived directly from PathoBench `k=all.tsv`. The task is case-level, not slide-level: the JSON maps 97 CPTAC-PDA case ids to 227 PathDB SVS slide ids, with 71 observed events and 26 censored cases. PathoBench defines this survival task by five official folds, each with 77-78 train cases and 19-20 held-out cases, so Nanopath reports the mean c-index across those five folds instead of carving a separate validation split.

## Probe Implementation

`prepare.py download=True` can now build this cache from public official sources on a fresh clone. It downloads each needed `{slide_id}.svs` from TCIA PathDB, extracts a deterministic 20x, 512 px, 0-overlap tissue grid with the same lightweight thumbnail tissue mask used by the other PathoBench-derived slide probes, writes one resumable full-grid parquet per slide, then concatenates them into:

- `patches.parquet`: `case_id`, `slide_id`, `tile_idx`, `image`
- `labels.tsv`: `case_id`, `slide_id`, `OS_event`, `OS_days`
- `tiling_version.txt`: `pathobench_20x_512_v1_full`

The local cache currently contains 146,896 JPEG tiles. The PathDB server does not reliably expose `Content-Length`, so the CPTAC path intentionally streams the SVS files directly instead of using the size-checked downloader used for SurGen CZI files.

`probe.py` streams cached tiles with a no-crop square resize, mean-pools tile embeddings by slide, mean-pools slides by case, then fits a fixed low-dimensional survival head on each official fold:

```text
train-fold z-score -> train-fold PCA(2) -> CoxPHSurvivalAnalysis(alpha=100)
```

This intentionally differs from PathoBench's default Coxnet head. PathoBench evaluates fixed Trident embeddings with Coxnet, but Nanopath compares many custom frozen backbones and random-init controls. Validation on this exact CPTAC-PDA OS cache showed raw or merely z-scored Coxnet without PCA gave randomized DINOv2-small backbones inflated c-index around 0.56-0.58. The fixed PCA(2) CoxPH head kept 20 randomized-weight DINOv2-small reruns centered at 0.496, while preserving usable pretrained separation: EXAONE-Path-2.5 0.580, H-optimus-0 0.565, GenBio-PathFM 0.547, Virchow 0.544, GigaPath 0.544, UNI-2-h 0.540, OpenMidnight 0.530, DINOv2-small 0.522, Midnight-12K 0.515, DINOv2-small random 0.489, and DINOv2-giant 0.467. Tree and rank-SVM survival heads were less stable under the same random-init checks.

## Null Distribution Audit

![CPTAC-PDA OS null distributions](cptac_pda_os_null_distributions.png)

`plot_null_checks.py` generates the figure above. The orange null is a fresh current-code rerun that constructs a new DINOv2-small with randomized weights for each seed before calling `probe.py`: mean 0.496, std 0.013, max 0.538. Fixed checkpoints are shown as vertical references: DINOv2-small 0.522, DINOv2-giant 0.467, GigaPath 0.544, GenBio-PathFM 0.547, and H-optimus-0 0.565.

This is the core reliability check for the survival probe. Randomized DINOv2-small weights remain centered near chance, while the strongest pathology-pretrained references clear the randomized-weight null tail. DINOv2-giant landing below the null is useful context: this survival probe is not merely rewarding generic ViT scale, and should not be interpreted as a universal ordering of foundation-model quality by itself.

## Difference From Original Usage

This is PathoBench-derived but still adapted for Nanopath's fast single-GPU loop. Nanopath uses the official PathoBench case folds and c-index metric, but evaluates custom-backbone mean-pooled tile features instead of Trident embeddings and uses a fixed CoxPH/PCA2 head to avoid Coxnet's high-dimensional random-feature instability. The tissue mask is a lightweight deterministic thumbnail mask rather than Trident HEST segmentation, so this should be interpreted as a compact representation probe, not as a claim about the best possible CPTAC-PDA survival model.
