# TCGA tile input pipeline backed by Parquet shards. Each shard is a parquet
# file of `{path: string, jpeg: binary}` rows. We open the shards via pyarrow
# directly (NOT `datasets.load_dataset`, which copies into ~/.cache) so the
# ~120 GB of shards are mmap'd in place with zero duplication. Random access
# is resolved by per-shard ParquetFile.read_row_group; prepare.py packs each
# shard with PARQUET_ROW_GROUP_SIZE=64 rows/group. Smaller row groups reduce
# unused JPEG reads but increase Parquet metadata and index startup cost.
#
# Patients (not tiles) are hashed by TCGA barcode and the bottom `val_fraction`
# of the hash space is held out from training; train.py instantiates the dataset
# twice (`is_train=True` for the training loop, `is_train=False` for the
# lightweight DINO/iBOT/KDE validation pass), so the held-out patient slice
# stays cleanly out-of-distribution from optimization.
#
# Each view uses PIL crop/resize/flips (+ optional transpose for 90-degree rotations), then optional HED jitter, color jitter,
# grayscale/blur, and normalization. Optional student-only GPU noise follows: StudentAugment on globals, LocalAugment (site scrubbing) on locals.
#
# This file is the *pretraining* input pipeline only. The downstream probes
# (probe.py) do not import anything from here.

import hashlib
import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


HED_FROM_RGB = torch.tensor(
    [
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.1355218],
        [-0.60190736, -0.48041419, 1.57358807],
    ],
    dtype=torch.float32,
)
RGB_FROM_HED = torch.tensor(
    [
        [0.65, 0.7, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=torch.float32,
)
LOG_1E6 = float(np.log(1e-6))
TILE_SIZE = 224


# Patients (not tiles) are the split unit so train/val never share a case.
def patient_in_val(patient_id, seed, val_fraction):
    key = f"{seed}:{patient_id}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    return value < float(val_fraction)


# Path entries start with the SVS stem (TCGA-XX-XXXX-...); the first three dash parts are the patient barcode.
def patient_id_from_relpath(rel):
    return "-".join(rel.split("/", 1)[0].split("-")[:3])


# Lightweight stain-space jitter; this is the stain augmentation hook for pretraining tiles.
class HEDJitter(nn.Module):
    # Store conversion matrices as buffers so transforms move with the module dtype/device if needed.
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.register_buffer("hed_from_rgb", HED_FROM_RGB)
        self.register_buffer("rgb_from_hed", RGB_FROM_HED)

    # Perturb HED channels, then convert back to RGB while the crop is still in [0, 1].
    def forward(self, x):
        rgb = x.movedim(-3, -1).clamp_min(1e-6)
        hed = (torch.log(rgb) / LOG_1E6) @ self.hed_from_rgb.to(dtype=x.dtype)
        hed = hed.clamp_min(0.0)
        shape = (*x.shape[:-3], 1, 1, 3)
        shift = torch.randn(shape, dtype=x.dtype, device=x.device) * self.sigma
        scale = 1.0 + torch.randn(shape, dtype=x.dtype, device=x.device) * self.sigma
        hed = hed * scale + shift
        log_rgb = -(hed * (-LOG_1E6)) @ self.rgb_from_hed.to(dtype=x.dtype)
        return torch.exp(log_rgb).clamp_(0.0, 1.0).movedim(-1, -3)


class GPUAugment(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.hed = HEDJitter(data["hed_jitter"]) if data["hed_jitter"] > 0 else nn.Identity()
        for name, value in [("mean", data["mean"]), ("std", data["std"]),
                            ("jitter", [data["color_jitter"], data["color_jitter"], data["color_jitter_saturation"]])]:
            self.register_buffer(name, torch.tensor(value).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, views):
        import kornia as K
        shape = views.shape
        x = self.hed(views.flatten(0, 1).float() / 255)
        n = x.shape[0]
        factors = 1 + (torch.rand(n, 3, 1, 1, device=x.device) * 2 - 1) * self.jitter
        order = torch.rand(n, 3, device=x.device).argsort(1)
        # Kornia's high-level jitter shares order; functions preserve per-view orders.
        for position in range(3):
            brightness = K.enhance.adjust_brightness_accumulative(x, factors[:, 0, 0, 0])
            contrast = K.enhance.adjust_contrast_with_mean_subtraction(x, factors[:, 1, 0, 0])
            saturation = K.enhance.adjust_saturation_with_gray_subtraction(x, factors[:, 2, 0, 0])
            op = order[:, position, None, None, None]
            x = torch.where(op == 0, brightness, torch.where(op == 1, contrast, saturation))
        x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.1, K.color.rgb_to_grayscale(x), x)
        sigma = (0.1 + torch.rand(n, 1, device=x.device) * 1.7).expand(-1, 2)
        blurred = K.filters.gaussian_blur2d(x, (9, 9), sigma, border_type="reflect", separable=True)
        x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.35, blurred, x)
        return ((x - self.mean) / self.std).reshape(shape)


# Extra stain/focus noise on student global inputs only, so the student must match teacher targets from a perturbed copy.
# Operates on normalised crops: denormalise, HED jitter, random Gaussian blur, renormalise, in float32 outside autocast.
class StudentAugment(nn.Module):
    def __init__(self, data, hed_jitter, blur_prob, blur_sigma):
        super().__init__()
        self.hed = HEDJitter(float(hed_jitter)) if hed_jitter > 0 else nn.Identity()
        self.blur_prob, (self.blur_lo, self.blur_hi) = float(blur_prob), map(float, blur_sigma)
        for name in ("mean", "std"):
            self.register_buffer(name, torch.tensor(data[name]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, views):
        import kornia as K
        with torch.autocast(device_type=views.device.type, enabled=False):
            x = self.hed((views.float() * self.std + self.mean).clamp(0.0, 1.0))
            if self.blur_prob > 0:
                n = x.shape[0]
                sigma = (self.blur_lo + torch.rand(n, 1, device=x.device) * (self.blur_hi - self.blur_lo)).expand(-1, 2)
                blurred = K.filters.gaussian_blur2d(x, (9, 9), sigma, border_type="reflect", separable=True)
                x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < self.blur_prob, blurred, x)
            return ((x - self.mean) / self.std).to(views.dtype)


# Site-scrubbing noise on student local crops (the teacher never sees locals), so the student must reach teacher targets without the
# stain/scanner signature of the tile's site. Each op hits each crop with its own probability, on normalised (L*b, 3, H, W) crop-major views:
#   stain_mix: Reinhard transfer of LAB mean/std from another image's crop in the same local slot (batches are unique-slide, so real site stains);
#   rescale: down- then up-sample by one random factor in [0.5, 1) per call (20x vs 40x scans); jpeg: recompress at quality 30-90;
#   noise: Gaussian sensor noise with per-crop sigma up to 0.03.
class LocalAugment(nn.Module):
    def __init__(self, data, stain_mix, rescale, jpeg, noise):
        super().__init__()
        self.probs = (float(stain_mix), float(rescale), float(jpeg), float(noise))
        for name in ("mean", "std"):
            self.register_buffer(name, torch.tensor(data[name]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, views, b):
        import kornia as K
        with torch.autocast(device_type=views.device.type, enabled=False):
            x = (views.float() * self.std + self.mean).clamp(0.0, 1.0)
            n, size, ar = x.shape[0], x.shape[-1], torch.arange(views.shape[0], device=views.device)
            # Donor = a different image (so a different slide) in the same local slot; stats come from every crop before any op runs.
            lab = K.color.rgb_to_lab(x)
            mu, sd = lab.mean((-2, -1), keepdim=True), lab.std((-2, -1), keepdim=True) + 1e-4
            donor = ar // b * b + (ar % b + torch.randint(1, b, (n,), device=x.device)) % b
            scale = 0.5 + 0.5 * random.random()
            ops = (lambda c, i: K.color.lab_to_rgb((lab[i] - mu[i]) / sd[i] * sd[donor[i]] + mu[donor[i]]),
                   lambda c, i: F.interpolate(F.interpolate(c, scale_factor=scale, mode="bilinear", antialias=True), size=size, mode="bilinear"),
                   lambda c, i: K.enhance.jpeg_codec_differentiable(c, 30 + 60 * torch.rand(len(i), device=c.device)),
                   lambda c, i: (c + 0.03 * torch.rand(len(i), 1, 1, 1, device=c.device) * torch.randn_like(c)).clamp(0.0, 1.0))
            # Run each op only on its selected crops (JPEG on every local would be wasted compute).
            for p, op in zip(self.probs, ops):
                idx = (torch.rand(n, device=x.device) < p).nonzero().flatten()
                if len(idx):
                    x[idx] = op(x[idx], idx).clamp(0.0, 1.0)
            return ((x - self.mean) / self.std).to(views.dtype)


# Batch sampler with no repeated slide per batch: each batch draws distinct slides weighted by remaining tiles (so tiles
# are still visited ~uniformly), then pops each slide's next shuffled tile; the epoch ends when fewer than batch_size slides remain.
class UniqueSlideBatchSampler(torch.utils.data.Sampler):
    def __init__(self, slide_of, batch_size, seed):
        self.slide_of, self.batch_size, self.seed, self.epoch = torch.as_tensor(slide_of, dtype=torch.long), batch_size, seed, 0

    def __len__(self):
        return len(self.slide_of) // self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        perm = torch.randperm(len(self.slide_of), generator=g)
        order = perm[self.slide_of[perm].argsort(stable=True)]  # tiles grouped by slide, shuffled within each slide
        counts = torch.bincount(self.slide_of)
        starts, remaining = counts.cumsum(0) - counts, counts.clone()
        while int((remaining > 0).sum()) >= self.batch_size:
            slides = torch.multinomial(remaining.double(), self.batch_size, replacement=False, generator=g)
            remaining[slides] -= 1
            yield order[starts[slides] + remaining[slides]].tolist()


# Map-style TCGA tile dataset that emits global/local multi-view stacks for train.py.
class TCGATileDataset(Dataset):
    # Glob shards, build a (shard_idx, row_in_shard) index over the requested patient
    # split, and configure augmentations. `is_train=True` keeps the (1 - val_fraction)
    # majority of patient ids; `is_train=False` keeps the held-out `val_fraction` slice.
    def __init__(self, cfg, is_train=True):
        data = cfg["data"]
        train = cfg["train"]
        self.tissue_thresh = float(data["tissue_thresh"]) if is_train else 0.0
        dataset_dir = Path(data["dataset_dir"])
        self.shards = sorted(dataset_dir.glob("shard-*.parquet"))
        if not self.shards:
            raise FileNotFoundError(
                f"No parquet shards (shard-*.parquet) under {dataset_dir}. Run "
                f"`python prepare.py {cfg['config_path']} download=True` to fetch them from "
                f"the medarc/nanopath HF dataset before training."
            )
        # Lazy ParquetFile handles, opened on first __getitem__ in each worker
        # so fork-children own their own file positions.
        self._readers = [None] * len(self.shards)
        # Pull just the path column from each shard once to build the train index;
        # the JPEG bytes column stays on disk until __getitem__.
        in_split_shard = []
        in_split_row = []
        in_split_slide = []
        slide_codes = {}
        for shard_idx, shard_path in enumerate(self.shards):
            paths = pq.read_table(str(shard_path), columns=["path"], memory_map=True)["path"].to_pylist()
            for row_idx, p in enumerate(paths):
                # XOR with is_train: training keeps tiles where patient_in_val is False,
                # validation keeps the complement.
                if patient_in_val(patient_id_from_relpath(p), data["split_seed"], data["val_fraction"]) != is_train:
                    in_split_shard.append(shard_idx)
                    in_split_row.append(row_idx)
                    in_split_slide.append(slide_codes.setdefault(p.split("/", 1)[0], len(slide_codes)))
        if not in_split_shard:
            raise ValueError(f"no {'train' if is_train else 'val'} tiles found in {dataset_dir}; check val_fraction={data['val_fraction']}")
        # Two parallel int32 arrays (~32 MB total for 4M tiles) shared COW across DataLoader fork-workers.
        self.shard_of = np.asarray(in_split_shard, dtype=np.int32)
        self.row_of = np.asarray(in_split_row, dtype=np.int32)
        # Dense 0..S-1 slide code per tile; unique-slide batching reads it, and tissue rejections then resample within the slide.
        self.slide_of = np.asarray(in_split_slide, dtype=np.int32)
        self.unique_slide_batches = is_train and bool(train["unique_slide_batches"])
        mean, std = data["mean"], data["std"]
        self.global_views = int(train["global_views"])
        self.local_views = int(train["local_views"])
        # Global and local views differ only in crop scale/size; the stochastic tail is shared.
        # data.rot90 adds a random transpose to the H/V flips, so views cover all 8 square symmetries (90/270 rotations included).
        augment = [
            v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
            *([v2.RandomApply([v2.Lambda(lambda im: im.transpose(Image.Transpose.TRANSPOSE), Image.Image)])] if data["rot90"] else []),
            v2.ToImage(),
        ]
        if not train["gpu_augment"]:
            augment += [
                v2.ToDtype(torch.float32, scale=True),
                *([HEDJitter(data["hed_jitter"])] if data["hed_jitter"] > 0 else []),
                v2.ColorJitter(data["color_jitter"], data["color_jitter"], data["color_jitter_saturation"], 0.0),
                v2.RandomGrayscale(p=0.1),
                v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 1.8))], p=0.35),
                v2.Normalize(mean=mean, std=std),
            ]
        self.global_aug = v2.Compose([v2.RandomResizedCrop(train["global_size"], scale=tuple(data["global_crop_scale"]), antialias=True), *augment])
        self.local_aug = v2.Compose([v2.RandomResizedCrop(train["local_size"], scale=tuple(data["local_crop_scale"]), antialias=True), *augment])

    # Dataset length is the number of tiles in this train/val split.
    def __len__(self):
        return int(self.shard_of.shape[0])

    # Read one JPEG row, resampling until training tiles clear the tissue threshold.
    def __getitem__(self, idx):
        idx = int(idx)
        for _ in range(1000):
            shard_idx = int(self.shard_of[idx])
            row_idx = int(self.row_of[idx])
            reader = self._readers[shard_idx]
            if reader is None:
                reader = pq.ParquetFile(str(self.shards[shard_idx]), memory_map=True)
                self._readers[shard_idx] = reader
            # Read the actual group size so repacked shards keep the same sample indices.
            rg_size = reader.metadata.row_group(0).num_rows
            rg_idx = row_idx // rg_size
            row_in_rg = row_idx % rg_size
            table = reader.read_row_group(rg_idx, columns=["path", "jpeg"])
            rel = table["path"][row_in_rg].as_py()
            jpeg_bytes = table["jpeg"][row_in_rg].as_py()
            with Image.open(io.BytesIO(jpeg_bytes)) as img:
                tile = img.convert("RGB")
            if self.tissue_thresh <= 0:
                break
            # temporary until we precalculate the tile tissue percentage
            rgb = v2.functional.to_image(tile).float() / 255
            sat = (rgb.amax(0) - rgb.amin(0)) / (rgb.amax(0) + 1e-6)
            if float((sat > 0.07).float().mean()) >= self.tissue_thresh:
                break
            # Same-slide replacement keeps the sampler's no-repeated-slide guarantee intact.
            idx = random.choice(np.flatnonzero(self.slide_of == self.slide_of[idx])) if self.unique_slide_batches else random.randint(0, self.shard_of.shape[0] - 1)
        else:
            raise RuntimeError(f"no tile met tissue_thresh={self.tissue_thresh} after 1000 samples")
        slide_stem = rel.split("/", 1)[0]
        patient_id = patient_id_from_relpath(rel)
        slide_key = int.from_bytes(hashlib.blake2b(slide_stem.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        patient_key = int.from_bytes(hashlib.blake2b(patient_id.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF
        # Augmentations are stochastic per view; reproducibility comes from worker seeds.
        views = {"global_views": torch.stack([self.global_aug(tile) for _ in range(self.global_views)])}
        # local_views: 0 omits the key entirely, so collate never stacks an empty list.
        if self.local_views:
            views["local_views"] = torch.stack([self.local_aug(tile) for _ in range(self.local_views)])
        return {
            **views,
            "sample_idx": torch.tensor(int(idx), dtype=torch.int64),
            "slide_id": torch.tensor(slide_key, dtype=torch.int64),
            "patient_id": torch.tensor(patient_key, dtype=torch.int64),
        }
