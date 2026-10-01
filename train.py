# DINO/JEPA pretraining on TCGA tiles (single-GPU), initialized from DINOv2. Loss terms:
# DINO CLS self-distillation (teacher targets balanced by Sinkhorn-Knopp, centering, ME-MAX, or SimDINO's coding rate),
# I-JEPA masked-patch prediction, a KDE uniformity term on the
# L2-normalised CLS tokens, and an optional CLS isotropy term. YAML drives the tunable knobs (backbone variant,
# optimizer (AdamW or Muon), LR + LR scheduler, drop path, layerwise decay, KDE weight + concentration, isotropy weight,
# FLOP/sample budgets, batch size, CLS specialization, block expansion); other objective hyperparameters are hardcoded
# inline at their use sites.

import atexit
import contextlib
import fnmatch
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.flop_counter import FlopCounterMode

from dataloader import GPUAugment, StudentAugment, TCGATileDataset, TILE_SIZE, UniqueSlideBatchSampler
from model import CrossJEPAPredictor, DINOHead, FactoredDINOHead, JEPAPredictor, ViT, expand_blocks, load_pretrained, specialize_cls_weights
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Read the YAML recipe and fail before any GPU work if the parquet tile dataset is absent.
# expandvars is necessary to resolve `$USER` for checked-in configs.
def load_config():
    if len(sys.argv) < 2:
        raise ValueError("usage: python train.py <config.yaml> [output_dir=<path>] [seed=<int>]")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    # Run identity and confirmation seed are the only CLI overrides; recipes stay in YAML.
    for arg in sys.argv[2:]:
        key, _, value = arg.partition("=")
        if key == "output_dir":
            cfg["project"]["output_dir"] = os.path.expandvars(value)
        elif key == "seed":
            cfg["train"]["seed"] = int(value)
        else:
            raise ValueError(f"unsupported override {arg!r}; use output_dir=<path> or seed=<int>")
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not any(dataset_dir.glob("shard-*.parquet")):
        raise FileNotFoundError(
            f"No parquet shards (shard-*.parquet) under {dataset_dir}. Pull the 4M-tile "
            f"parquet dataset from medarc/nanopath on HF by running "
            f"`python prepare.py {cfg['config_path']} download=True`. Follow the data setup in "
            f"README.md before launching train.py."
        )
    return cfg


# Arm Labless before any GPU work so direct `python train.py ...` gets the same
# no-scope GitHub device login path as the SLURM launcher. Noninteractive runs
# train locally unless the launcher passed a preauthorized token file.
def maybe_arm_labless_autosubmit(cfg, repo_dir):
    token_path = os.environ.get("LABLESS_AUTOSUBMIT_FILE", "")
    eligible = (
        bool(cfg["probe"]["enabled"])
        and int(cfg["probe"]["count"]) > 0
        and int(cfg["train"]["max_train_samples"]) == 1_000_000
        and int(cfg["train"]["max_train_flops"]) == 1_000_000_000_000_000_000
    )
    if token_path:
        atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
        return token_path
    if not eligible:
        return ""
    if not sys.stdin.isatty():
        if not os.environ.get("SLURM_JOB_ID"):
            print(f"{console_prefix()} Labless  no interactive stdin; training will run without auto-submit.", flush=True)
        return ""
    print("This looks like a full Labless-eligible run. Leave the run name blank to train without auto-submit.", flush=True)
    run_name = input("Labless run name (<=20 chars): ").strip()
    notes = input("Labless experiment note (unique change + why): ").strip()
    if not run_name or len(run_name) > 20:
        print("Labless auto-submit skipped; run name is required and must be <=20 chars.", flush=True)
        return ""
    token_path = str(Path(str(Path(cfg["project"]["output_dir"]).expanduser().resolve()) + ".labless_autosubmit.json"))
    status = subprocess.run(
        [sys.executable, str(repo_dir / "labless" / "submit_to_labless.py"), "login_only=true", f"token_output={token_path}", f"run_name={run_name}", f"notes={notes}"],
        cwd=repo_dir,
        check=False,
    ).returncode
    if status != 0:
        print("Labless login did not complete; training will run without auto-submit.", flush=True)
        Path(token_path).unlink(missing_ok=True)
        return ""
    os.environ["LABLESS_AUTOSUBMIT_FILE"] = token_path
    atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
    return token_path


def finish_labless_autosubmit(token_path, output_dir, repo_dir):
    token_file = Path(token_path) if token_path else None
    if token_file is None or not token_file.exists():
        return
    token = json.loads(token_file.read_text())
    status = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "labless" / "submit_to_labless.py"),
            f"output_dir={output_dir.resolve()}",
            f"run_name={token['run_name']}",
            f"notes={token['notes']}",
            f"github_token_file={token_file}",
        ],
        cwd=repo_dir,
        check=False,
    ).returncode
    token_file.unlink(missing_ok=True)
    if status == 2:
        print(f"{console_prefix()} Labless  auto-submit skipped because the completed run did not satisfy submission restrictions.", flush=True)
    elif status != 0:
        raise SystemExit(status)


# Cosine schedule from `start` to `end` over fractional progress in [0, 1].
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(1.0, max(0.0, frac))))


# Sinkhorn-Knopp centring across this batch, used as DINO teacher targets; (N, F, K) input balances each factor's codebook independently.
def sinkhorn(x, temp):
    q = torch.exp(x.float() / temp).permute(1, 2, 0)  # (F, K, N)
    k, b = q.shape[1], q.shape[2]
    q /= q.sum((1, 2), keepdim=True)
    for _ in range(3):
        q /= q.sum(2, keepdim=True) * k
        q /= q.sum(1, keepdim=True) * b
    return (q * b).permute(2, 0, 1)


# Cross-entropy between teacher distribution and softmax(student / temp) over (..., F, K), summed over factors (= the joint-code CE).
# temp None is SimDINO: negative cosine between normalized student and teacher chunks, which is also linear in the teacher targets.
def dino_ce(student, teacher, temp):
    return -(teacher * (student if temp is None else F.log_softmax(student / temp, dim=-1))).sum((-2, -1)).mean()


# SimDINO coding rate of L2-normalized (N, F, d) chunks, per dimension and summed over factors: 0.5 logdet(I + d / (N eps^2) Z^T Z) / d grows
# as the batch spans more directions (~0.8 when isotropic at eps 0.5), so its per-sample gradient is on the cosine alignment's scale.
def coding_rate(z, eps=0.5):
    with torch.autocast("cuda", enabled=False):
        n, _, d = z.shape
        gram = torch.einsum("nfd,nfe->fde", z.float(), z.float()) * d / (n * eps ** 2)
        return 0.5 * torch.logdet(torch.eye(d, device=z.device) + gram).sum() / d


# Weight masked-patch regression by each image's inverse mask count.
def jepa_regression(student, teacher, weights):
    return F.smooth_l1_loss(student, teacher, reduction="none").mean(-1).mul(weights).sum()


# KDE uniformity loss on L2-normalised CLS tokens; the tent kernel peaks at chord distance `radius`, so pairs inside attract and pairs outside repel (radius 0 = plain vMF KDE).
def kde_loss(x, concentration, radius):
    x = F.normalize(x, p=2, dim=-1)
    cos_r = 1 - radius ** 2 / 2
    sim = concentration * (cos_r - (x @ x.T - cos_r).abs())
    sim.fill_diagonal_(-float("inf"))
    return torch.logsumexp(sim, dim=1).mean() - math.log(max(1, sim.shape[1] - 1))


# Anisotropy of L2-normalized CLS: D * Σ(p_i - 1/D)² over covariance eigen-fractions p, i.e. D / participation_ratio - 1 (0 = isotropic).
def isotropy_loss(x):
    with torch.autocast("cuda", enabled=False):
        z = F.normalize(x.float(), dim=-1)
        z = z - z.mean(0)
        c = z.T @ z
        return x.shape[1] * c.square().sum() / c.diagonal().sum().square() - 1


# Cross-factor redundancy of (N, F, d) head chunks: squared cross-correlation between each factor pair over the batch, per dimension
# (~d/N when factors are independent, ~canonical-corr mass when one chunk is a linear copy of another); 0 for a single factor.
def factor_decorrelation(z):
    with torch.autocast("cuda", enabled=False):
        z = z.float()
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        pairs = [(i, j) for i in range(z.shape[1]) for j in range(i + 1, z.shape[1])]
        return sum(((z[:, i].T @ z[:, j] / len(z)).square().sum() / z.shape[-1] for i, j in pairs), z.new_zeros(())) / max(1, len(pairs))


# I-JEPA masks contiguous square blocks to infer missing tissue context.
def make_block_mask(batch, grid, device, n_blocks, block_scale):
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    side = max(1, round(grid * block_scale ** 0.5))
    for i in range(batch):
        for _ in range(n_blocks):
            top, left = random.randint(0, grid - side), random.randint(0, grid - side)
            masks[i, top : top + side, left : left + side] = True
    masks = masks.flatten(1)
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# Toroidal distance from every grid index to the window [t, t+side): 0 inside, else the shorter way round.
def _axis_dist(pos, t, grid, side):
    delta = (pos - t) % grid
    return torch.where(delta < side, torch.zeros_like(delta), torch.minimum(delta - side + 1, grid - delta))


# Cross JEPA context is n toroidal side x side blocks per crop, concatenated with overlaps kept so every crop has
# n * side**2 tokens, plus k unseen targets outside their union. Target weights exp(-8 * decay * d) use Chebyshev
# distance to the nearest block normalised per row: decay>0 hugs the context, decay<0 pushes deep.
# Weighted sampling without replacement is one topk over Exp(1) / w keys.
def make_region_idx(batch, grid, device, n_blocks, side, k, decay):
    pos = torch.arange(grid, device=device)
    ty, tx = torch.randint(grid, (2, batch, n_blocks, 1), device=device)
    rows, cols = (ty + pos[:side]) % grid, (tx + pos[:side]) % grid
    keep = (rows[..., None] * grid + cols[..., None, :]).flatten(1)
    d = torch.maximum(_axis_dist(pos, ty, grid, side)[..., :, None], _axis_dist(pos, tx, grid, side)[..., None, :]).amin(1).flatten(1).float()
    d = d / d.amax(-1, keepdim=True).clamp(min=1.0)
    keys = torch.empty_like(d).exponential_() * torch.exp(8.0 * decay * d)
    keys.scatter_(1, keep, float("inf"))
    return keep, keys.topk(k, dim=-1, largest=False).indices


# AdamW parameter groups with layer-wise LR decay on the backbone:
# block i gets lr * layerwise_decay^(depth - 1 - i); patch_embed gets the deepest decay
# multiplied by patch_embed_lr_mult; biases and norms get no weight decay; the head's
# final weight-norm last_layer parameters get an LR-freeze for the first dino.freeze_last_layer_fraction.
# With `muon`, hidden matrices (transformer blocks + DINO head MLP) get group["muon"] = number of stacked sub-matrices
# (fused qkv=3, kv=2, else 1); 0 means AdamW. Embeddings, tokens, norms, biases, proj_in/proj, last_layer, and CLS-specialized
# (`.cls.`) matrices stay AdamW, the latter because their one-token-per-image gradients are too low-rank to orthogonalize.
# `cls_lr_mult` (None = follow layer-wise decay) sets the LR of `.cls.` backbone copies.
def build_param_groups(student_backbone, student_dino_head, student_predictor, layerwise_decay, patch_embed_lr_mult, muon, cls_lr_mult=None):
    depth = len(student_backbone.blocks)
    # Coalesce params that share (lr_mult, wd_mult, last_layer) into a single group each (~30 groups
    # instead of one-per-param), so AdamW's foreach path fuses the step across many tensors rather than
    # launching per-parameter kernels. Per-param lr/wd are unchanged, so the optimization is numerically identical.
    coalesced = {}
    modules = ((student_backbone, "backbone"), (student_dino_head, "dino_head"), (student_predictor, "jepa_predictor"))
    for module, kind in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if kind == "backbone" and ".cls." in name and cls_lr_mult is not None:
                lr_mult = cls_lr_mult
            elif kind == "backbone" and name.startswith("blocks."):
                lr_mult = layerwise_decay ** (depth - 1 - int(name.split(".")[1]))
            elif kind == "backbone" and name.startswith("patch_embed."):
                lr_mult = (layerwise_decay ** depth) * patch_embed_lr_mult
            wd_mult = 0.0 if name.endswith("bias") or "norm" in name or p.ndim < 2 else 1.0
            split = 3 if "qkv" in name else 2 if name.endswith(".kv.weight") else 1
            split = split if muon and p.ndim == 2 and ".cls." not in name and (name.startswith("blocks.") or kind == "dino_head" and name.startswith("mlp.")) else 0
            key = (lr_mult, wd_mult, "last_layer" in name or kind == "dino_head" and name == "prototypes", split)
            coalesced.setdefault(key, {"params": [], "lr_mult": lr_mult, "wd_mult": wd_mult, "last_layer": key[2], "muon": split})["params"].append(p)
    return list(coalesced.values())


# Per-iteration quintic coefficients (a, b, c) for a*x + b*x^3 + c*x^5; varying them per step tightens the composite sign approximation.
NS_COEFFS = ((3.6, -6.77, 3.69), (3.6, -7.43, 4.42), (3.6, -8.14, 5.33), (3.6, -8.2, 5.71), (3.6 / 1.4, -4.27 / 1.4, 2.1 / 1.4))


# Quintic Newton-Schulz: pushes every singular value of a (S, m, n) stack toward ~1, i.e. approximately U V^T.
def newton_schulz(G):
    X = G.mT if G.shape[-2] > G.shape[-1] else G
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in NS_COEFFS:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if G.shape[-2] > G.shape[-1] else X


# Muon: Nesterov momentum (0.95) orthogonalized per sub-matrix, so every direction of a weight moves equally.
# The 0.2*sqrt(max(m, n)) scale matches AdamW's update RMS (Moonlight), so dino.lr, layerwise decay, and the WD schedule carry over.
class Muon(torch.optim.Optimizer):
    def __init__(self, params, compile_ns): super().__init__(params, {"lr": 1.0, "weight_decay": 0.0}); self.ns = torch.compile(newton_schulz, disable=not compile_ns)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                buf.lerp_(p.grad, 0.05)
                G = p.grad.lerp(buf, 0.95).view(group["muon"], -1, p.shape[1])
                update = self.ns(G.bfloat16()).reshape_as(p).to(p.dtype)
                p.mul_(1 - group["lr"] * group["weight_decay"]).add_(update, alpha=-group["lr"] * 0.2 * max(G.shape[-2:]) ** 0.5)


# EMA-update teacher modules from student modules with a single multiplicative decay.
# Params are fused into two _foreach kernels (mul then add) instead of a Python per-tensor loop;
# numerically identical (pt = pt*m + ps*(1-m) per tensor). Called under torch.no_grad() by the caller.
def update_ema(student_module, teacher_module, momentum):
    teacher_params, student_params = list(teacher_module.parameters()), list(student_module.parameters())
    torch._foreach_mul_(teacher_params, momentum)
    torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)
    for bs, bt in zip(student_module.buffers(), teacher_module.buffers()):
        bt.copy_(bs)


# Orchestrates one pretraining run: setup, train+probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    repo_dir = Path(__file__).resolve().parent
    labless_autosubmit_file = maybe_arm_labless_autosubmit(cfg, repo_dir)
    train_cfg = cfg["train"]
    dino_cfg = cfg["dino"]
    save_every = train_cfg["save_every"]
    save_checkpoints = save_every is not None
    device = torch.device("cuda")
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    variant = cfg["model"]["type"]
    # model.cls_specialization: null keeps the plain ViT; {qkv_blocks, lr_mult} gives CLS its own norms/LayerScales (all blocks) and qkv (first qkv_blocks).
    # lr_mult: null keeps the CLS copies on layer-wise decay like their patch twins; a number bypasses it with lr * lr_mult.
    cls_spec = cfg["model"]["cls_specialization"]
    if cls_spec is not None and set(cls_spec) != {"qkv_blocks", "lr_mult"}:
        raise ValueError(f"model.cls_specialization keys {sorted(cls_spec)} invalid; expected null or exactly qkv_blocks, lr_mult")
    # model.block_expansion: null keeps DINOv2's depth; {chunks, freeze_base, freeze_embeddings} inserts `chunks` identity-initialized block
    # copies (one per depth/chunks blocks); freeze_base: true trains only those new blocks among the transformer blocks, and
    # freeze_embeddings: true also freezes patch_embed, pos_embed, and the cls/register/mask tokens, so backward stops at the first new block.
    expansion = cfg["model"]["block_expansion"]
    if expansion is not None and set(expansion) != {"chunks", "freeze_base", "freeze_embeddings"}:
        raise ValueError(f"model.block_expansion keys {sorted(expansion)} invalid; expected null or exactly chunks, freeze_base, freeze_embeddings")
    student_backbone = ViT(variant=variant, drop_path_rate=dino_cfg["drop_path_rate"])
    if cls_spec is not None:
        specialize_cls_weights(student_backbone, int(cls_spec["qkv_blocks"]))
    student_backbone = load_pretrained(student_backbone)
    if expansion is not None:
        new_blocks = expand_blocks(student_backbone, int(expansion["chunks"]))
        for blk in student_backbone.blocks:
            blk.requires_grad_(blk in new_blocks or not expansion["freeze_base"])
        # Everything outside blocks.* and the final norm is the token-embedding path.
        for name, p in student_backbone.named_parameters():
            p.requires_grad_(p.requires_grad and (name.startswith(("blocks.", "norm.")) or not expansion["freeze_embeddings"]))
    student_backbone = student_backbone.to(device)
    teacher_backbone = deepcopy(student_backbone)
    teacher_backbone.train(False)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    # dino.head.type picks the prototype layer; its sibling keys are exactly that type's arguments.
    #   weight_norm: {}. DINOv2's single 131072-way weight-normed last_layer.
    #   factored: {prototypes, factors, decorr_weight, separate_mlps}. `prototypes` Parameter vectors split evenly into `factors` codebooks over equal
    #             bottleneck chunks; decorr_weight penalizes cross-factor chunk correlation (0 = off), ramped in with LR warmup; separate_mlps gives
    #             each factor its own head MLP instead of slicing one shared output.
    head_cfg = dict(dino_cfg["head"])
    head_type = head_cfg.pop("type")
    head_args = {"weight_norm": set(), "factored": {"prototypes", "factors", "decorr_weight", "separate_mlps"}}
    if head_type not in head_args or set(head_cfg) != head_args[head_type]:
        raise ValueError(f"dino.head type={head_type!r} with keys {sorted(head_cfg)} invalid; expected one of {head_args}")
    if head_type == "weight_norm":
        student_dino_head, codebook_size = DINOHead(student_backbone.embed_dim, 131072, dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3).to(device), 131072
    else:
        student_dino_head = FactoredDINOHead(student_backbone.embed_dim, int(head_cfg["prototypes"]), dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3, int(head_cfg["factors"]), bool(head_cfg["separate_mlps"])).to(device)
        codebook_size = int(head_cfg["prototypes"]) // int(head_cfg["factors"])
    decorr_weight = 0.0 if head_type == "weight_norm" else float(head_cfg["decorr_weight"])
    # dino.temp_scale multiplies both DINO temperatures: `auto` = sqrt(ln 131072 / ln K), since softmax sharpness is relative to ~1/sqrt(2 ln K)
    # and this keeps smaller codebooks as far from collapse (1.0 for weight_norm); a number fixes it (1.0 = unscaled).
    temp_scale = math.sqrt(math.log(131072) / math.log(codebook_size)) if dino_cfg["temp_scale"] == "auto" else float(dino_cfg["temp_scale"])
    # dino.balance.type picks what keeps the DINO head from collapsing onto a few prototypes; its sibling keys are exactly that type's arguments.
    #   sinkhorn: {}. Sinkhorn-Knopp equipartitions each batch's teacher targets over every codebook.
    #   center: {}. DINO v1: teacher logits minus an EMA (momentum 0.9) of their batch mean, then the sharpened softmax.
    #   memax: {weight}. MSN: teacher softmax at a fixed 0.025 temperature, no balancing; weight * -entropy of the student's batch-mean
    #          prediction (all global + local views) spreads prototype use. The `balance` log is this negative entropy for every non-SimDINO type.
    #   simdino: {weight}. Prototypes unused: student head chunks align (cosine) to the teacher's, and weight * -coding_rate prevents collapse.
    balance_cfg = dict(dino_cfg["balance"])
    balance_type = balance_cfg.pop("type")
    balance_args = {"sinkhorn": set(), "center": set(), "memax": {"weight"}, "simdino": {"weight"}}
    if balance_type not in balance_args or set(balance_cfg) != balance_args[balance_type]:
        raise ValueError(f"dino.balance type={balance_type!r} with keys {sorted(balance_cfg)} invalid; expected one of {balance_args}")
    balance_weight = float(balance_cfg.get("weight", 0.0))
    student_temp = None if balance_type == "simdino" else 0.1 * temp_scale
    head_out = int(balance_type == "simdino")  # DINO heads return (logits, normalized chunks); SimDINO trains on the chunks
    center = torch.zeros(1 if head_type == "weight_norm" else int(head_cfg["factors"]), codebook_size, device=device)
    global_grid = train_cfg["global_size"] // student_backbone.patch_size
    global_patches = global_grid ** 2
    # dino.jepa.type picks the masking protocol + predictor; its sibling keys are exactly that type's arguments.
    #   block: {depth, width, blocks, block_scale}. Student sees the full grid with mask_token blocks; self-attention predictor.
    #   cross: {depth, width, context_regions: [{blocks, side, count}, ...], targets, target_decay, context_cls}. Student sees only
    #          `count` contexts of `blocks` side x side squares per global view; cross-attention predictor regresses `targets` unseen patches.
    #          context_cls picks the CLS the predictor reads: student (JEPA trains the student CLS), teacher (frozen EMA CLS of the full clean crop), or none.
    jepa_cfg = dict(dino_cfg["jepa"])
    jepa_type = jepa_cfg.pop("type")
    jepa_args = {"block": {"depth", "width", "blocks", "block_scale"}, "cross": {"depth", "width", "context_regions", "targets", "target_decay", "context_cls"}}
    if jepa_type not in jepa_args or set(jepa_cfg) != jepa_args[jepa_type]:
        raise ValueError(f"dino.jepa type={jepa_type!r} with keys {sorted(jepa_cfg)} invalid; expected one of {jepa_args}")
    if jepa_type == "block":
        context_regions = [(1, global_grid, 1)]
        student_predictor = JEPAPredictor(student_backbone.embed_dim, int(jepa_cfg["depth"]), int(jepa_cfg["width"])).to(device)
    else:
        if any(set(region) != {"blocks", "side", "count"} for region in jepa_cfg["context_regions"]):
            raise ValueError("dino.jepa.context_regions entries must have exactly the keys {blocks, side, count}")
        context_regions = [(int(r["blocks"]), int(r["side"]), int(r["count"])) for r in jepa_cfg["context_regions"]]
        # Targets come from outside the block union, so the no-overlap context size bounds k.
        max_targets = global_patches - max((n * side ** 2 for n, side, _ in context_regions), default=global_patches)
        if not context_regions or any(not 0 < side <= global_grid or n < 1 or count < 1 for n, side, count in context_regions) or not 0 < int(jepa_cfg["targets"]) <= max_targets:
            raise ValueError(f"dino.jepa cross needs sides in 1..{global_grid}, blocks and counts >= 1, and targets in 1..{max_targets}")
        if jepa_cfg["context_cls"] not in {"student", "teacher", "none"}:
            raise ValueError(f"dino.jepa.context_cls={jepa_cfg['context_cls']!r} invalid; expected student, teacher, or none")
        student_predictor = CrossJEPAPredictor(student_backbone.embed_dim, int(jepa_cfg["depth"]), int(jepa_cfg["width"])).to(device)
    total_regions = sum(count for _, _, count in context_regions)
    visible_global_patches = global_patches if jepa_type == "block" else sum(count * n * side ** 2 for n, side, count in context_regions)
    teacher_dino_head = deepcopy(student_dino_head)
    for p in teacher_dino_head.parameters():
        p.requires_grad = False
    backbone_activated_params = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    # Param groups carry per-parameter LR/WD multipliers (LWD + patch_embed + biases-no-WD); dino.optimizer is adamw or muon,
    # and muon keeps AdamW for everything that isn't a hidden matrix, so opts = [AdamW] or [AdamW, Muon].
    muon = {"adamw": False, "muon": True}[dino_cfg["optimizer"]]
    param_groups = build_param_groups(student_backbone, student_dino_head, student_predictor, dino_cfg["layerwise_decay"], dino_cfg["patch_embed_lr_mult"], muon, None if cls_spec is None else cls_spec["lr_mult"])
    opt = torch.optim.AdamW([g for g in param_groups if not g["muon"]], lr=1.0, betas=(0.9, dino_cfg["adam_beta2"]), fused=train_cfg["fused_adamw"])
    opts = [opt, Muon([g for g in param_groups if g["muon"]], train_cfg["compile"])] if muon else [opt]
    step = 0
    batch_size = int(train_cfg["batch_size"])
    max_train_samples = int(train_cfg["max_train_samples"])
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
    for key, name in [("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")]:
        os.environ.setdefault(key, str(wandb_dir.parent / name))
    # Compiled backward keeps at most this fraction of the default-saved activations, recomputing the cheapest ops (1.0 = off).
    # Read at each graph's first compile, so it must be set before any compiled module runs; it overlaps with block checkpointing.
    if train_cfg["activation_checkpointing"] and train_cfg["activation_memory_budget"] < 1.0:
        raise ValueError("use either train.activation_checkpointing or train.activation_memory_budget < 1.0, not both")
    torch._functorch.config.activation_memory_budget = float(train_cfg["activation_memory_budget"])
    # Compile calls in place so checkpoint keys and parameter ownership stay unchanged.
    for module in (student_backbone, teacher_backbone, student_dino_head, teacher_dino_head, student_predictor):
        module.compile(dynamic=isinstance(module, (DINOHead, FactoredDINOHead)), disable=not train_cfg["compile"])
    sinkhorn_fn = torch.compile(sinkhorn, dynamic=True, disable=not train_cfg["compile"])
    dino_ce_fn = torch.compile(dino_ce, dynamic=True, disable=not train_cfg["compile"])
    jepa_loss_fn = torch.compile(jepa_regression, dynamic=True, disable=not train_cfg["compile"])
    wandb_name = cfg["project"]["name"]
    if labless_autosubmit_file:
        wandb_name = json.loads(Path(labless_autosubmit_file).read_text()).get("run_name") or wandb_name
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    latest_checkpoint_path = output_dir / "latest.pt"
    # Fresh launches always start from scratch and wipe output_dir.
    resume_path = Path(train_cfg["resume"]) if train_cfg["resume"] else None
    if resume_path is None and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    # Validate cached probes before resume can replace the saved source snapshot.
    probe_state = prepare_probe_state(cfg, output_dir) if probe_enabled(cfg) else None
    wandb_meta = None
    if resume_path is not None:
        print(f"{console_prefix()} Resume  loading checkpoint: {resume_path}", flush=True)
        # Resume restores training progress, optimizer state, and wandb identity.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        student_backbone.load_state_dict(checkpoint["model"])
        teacher_backbone.load_state_dict(checkpoint["model_ema"])
        student_dino_head.load_state_dict(checkpoint["dino_head"])
        teacher_dino_head.load_state_dict(checkpoint["dino_head_ema"])
        student_predictor.load_state_dict(checkpoint["predictor"])
        center.copy_(checkpoint["center"])
        # The requested backend owns step placement, even when the saved backend differs.
        for group in checkpoint["opt"]["param_groups"]:
            group.update(fused=train_cfg["fused_adamw"], foreach=None)
        for state in checkpoint["opt"]["state"].values():
            state["step"] = state["step"].to(device if train_cfg["fused_adamw"] else "cpu")
        opt.load_state_dict(checkpoint["opt"])
        for o, state in zip(opts[1:], checkpoint["muon"]):
            o.load_state_dict(state)
        step = int(checkpoint["step"])
        examples_seen = int(checkpoint["examples_seen"])
        visible_patch_presentations = int(checkpoint["visible_patch_presentations"])
        train_flops = int(checkpoint["train_flops"])
        wandb_meta = dict(checkpoint["wandb"])
    wandb_init = {
        "project": "nanopath",
        "name": wandb_name,
        "dir": str(wandb_dir),
        "config": cfg,
        "settings": wandb.Settings(
            console="wrap",
            x_file_stream_transmit_interval=5,
        ),
    }
    if wandb_meta is not None:
        wandb_init["id"] = wandb_meta["id"]
        wandb_init["resume"] = "must"
    wandb_run = wandb.init(**wandb_init)
    wandb_run.config.update({"pillow": {"version": PIL.__version__, "path": PIL.__file__}}, allow_val_change=True)
    for key in ("probe/target_flops", "probe/wall_seconds"):
        wandb_run.define_metric(key, hidden=True, overwrite=True)
    print(
        f"{console_prefix()} Run  start: {wandb_name}  "
        f"config: {cfg['config_path']}  batch_size: {batch_size}  max_train_samples: {max_train_samples}  "
        f"seed: {train_cfg['seed']}  "
        f"max_train_flops: {train_cfg['max_train_flops']}  "
        f"probe_count: {cfg['probe']['count']}  warmup_fraction: {dino_cfg['warmup_fraction']}  "
        f"optimizer: {dino_cfg['optimizer']}  lr: {dino_cfg['lr']}  adam_beta2: {dino_cfg['adam_beta2']}  kde_loss_weight: {dino_cfg['kde_loss_weight']}  "
        f"kde_concentration: {dino_cfg['kde_concentration']}  kde_radius: {dino_cfg['kde_radius']}  iso_loss_weight: {dino_cfg['iso_loss_weight']}  drop_path: {dino_cfg['drop_path_rate']}  "
        f"layerwise_decay: {dino_cfg['layerwise_decay']}  pillow: {PIL.__version__} ({PIL.__file__})",
        flush=True,
    )
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    git_remote = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=repo_dir, text=True, capture_output=True, check=False).stdout.strip()
    source_id = f"nanopath-source-{wandb_run.id}"
    artifact_ignore = [
        line.strip() for line in (repo_dir / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ] + [".git/", "baselines/", "slurm/", "AGENTS.md", "CLAUDE.md"]
    ignored_roots = [output_dir.resolve(), wandb_dir.resolve()]

    def artifact_ignored(path):
        if any(path.resolve().is_relative_to(root) for root in ignored_roots):
            return True
        rel_path = path.relative_to(repo_dir)
        if any(part.startswith(".") for part in rel_path.parts):
            return True
        rel, name = rel_path.as_posix(), path.name
        for pat in artifact_ignore:
            pat = pat.rstrip("/") if pat.endswith("/") else pat
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat) or rel == pat or rel.startswith(pat + "/"):
                return True
        return False

    source_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if not artifact_ignored(Path(root) / d))
        for name in sorted(files):
            path = Path(root) / name
            if artifact_ignored(path):
                continue
            rel = path.relative_to(repo_dir)
            source_files.append((path, rel))
    source_snapshot_dir = output_dir / "labless_source"
    if source_snapshot_dir.exists():
        shutil.rmtree(source_snapshot_dir)
    for path, rel in source_files:
        target = source_snapshot_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    wandb_meta = {"entity": wandb_run.entity, "project": "nanopath", "id": wandb_run.id, "name": wandb_name, "url": wandb_run.url,
                  "mode": getattr(wandb_run.settings, "mode", ""), "source_artifact": source_id,
                  "source_dir": str(source_snapshot_dir), "git": {"commit": git_commit, "remote": git_remote}}
    augment = torch.nn.Identity()
    if train_cfg["gpu_augment"]:
        augment = GPUAugment(cfg["data"]).to(device)
        if train_cfg["compile"]:
            augment.compile()
            # Compile before timing; preserve the training RNG state through warmup.
            rng = torch.cuda.get_rng_state(device)
            for views, size in [(train_cfg["global_views"], train_cfg["global_size"]), (train_cfg["local_views"], train_cfg["local_size"])]:
                if views:
                    augment(torch.zeros(batch_size, views, 3, size, size, dtype=torch.uint8, device=device))
            torch.cuda.synchronize(device)
            torch.cuda.set_rng_state(rng, device)
    # Optional data.student_augment: {hed_jitter, blur_prob, blur_sigma: [lo, hi]}; absent means the student sees the teacher's pixels.
    student_augment = torch.nn.Identity()
    if "student_augment" in cfg["data"]:
        student_augment_cfg = cfg["data"]["student_augment"]
        if set(student_augment_cfg) != {"hed_jitter", "blur_prob", "blur_sigma"}:
            raise ValueError(f"data.student_augment keys {sorted(student_augment_cfg)} invalid; expected exactly hed_jitter, blur_prob, blur_sigma")
        student_augment = StudentAugment(cfg["data"], **student_augment_cfg).to(device)
    train_ds = TCGATileDataset(cfg, is_train=True)
    val_ds = TCGATileDataset(cfg, is_train=False)

    # Train shuffles + drops partials; the loop never starts a batch that would exceed
    # max_train_samples, so every optimizer step keeps the configured batch size.
    loader_kwargs = {
        "batch_size": batch_size,
        "drop_last": True,
        "num_workers": train_cfg["num_workers"],
        "pin_memory": True,
        "prefetch_factor": train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
        "persistent_workers": train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0,
    }
    # train.unique_slide_batches swaps plain shuffling for a batch sampler that never puts two tiles from one slide in a batch.
    train_loader = (DataLoader(train_ds, batch_sampler=UniqueSlideBatchSampler(train_ds.slide_of, batch_size, train_cfg["seed"]), **{k: v for k, v in loader_kwargs.items() if k not in ("batch_size", "drop_last")})
                    if train_cfg["unique_slide_batches"] else DataLoader(train_ds, shuffle=True, **loader_kwargs))
    # The val split is stored slide-contiguous, so read a fixed random subset (seeded by split_seed, identical across runs and evals)
    # to give each val batch many slides; unshuffled batches held 1-2 slides and skewed every batch-level val metric.
    val_subset = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(cfg["data"]["split_seed"]))[:int(train_cfg["val_batches"]) * batch_size]
    val_loader = DataLoader(Subset(val_ds, val_subset.tolist()), shuffle=False, **loader_kwargs)

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    local_patches = (train_cfg["local_size"] // student_backbone.patch_size) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (TILE_SIZE // student_backbone.patch_size) ** 2
    seen_ids = {"sample": set(), "slide": set(), "patient": set()}
    pending_ids = {key: set() for key in seen_ids}

    # cpu_state(m) materializes an on-CPU copy of a module's state_dict for torch.save.
    def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # Full checkpoint (latest.pt) covers explicit train.resume whereas probe checkpoint is a slim
    # weights-only ckpt, given probe.py does not need optimizer or projection heads.
    def checkpoint_payload(next_step, full):
        payload = {"model": cpu_state(student_backbone), "model_ema": cpu_state(teacher_backbone), "step": next_step, "config": cfg}
        if not full:
            return payload
        return {**payload, "dino_head": cpu_state(student_dino_head), "dino_head_ema": cpu_state(teacher_dino_head), "predictor": cpu_state(student_predictor), "center": center.cpu(),
                "opt": opt.state_dict(), "muon": [o.state_dict() for o in opts[1:]], "examples_seen": examples_seen,
                "visible_patch_presentations": visible_patch_presentations, "train_flops": train_flops, "wandb": wandb_meta}

    def save_latest_checkpoint(checkpoint_step):
        nonlocal last_saved_step
        print(f"{console_prefix()} Checkpoint  [{checkpoint_step}]  save: latest.pt", flush=True)
        tmp_path = latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload(checkpoint_step, full=True), tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
        for stale_checkpoint_path in output_dir.glob("step_*.pt"):
            stale_checkpoint_path.unlink()
        last_saved_step = checkpoint_step

    # Count unique tiles/slides/patients for data-coverage diagnostics.
    def flush_unique_counts():
        for key, seen in seen_ids.items():
            seen.update(pending_ids[key])
            pending_ids[key].clear()
        unique_tiles_seen = len(seen_ids["sample"])
        return {
            "unique_slides_seen": len(seen_ids["slide"]),
            "unique_patients_seen": len(seen_ids["patient"]),
            "unique_tiles_seen": unique_tiles_seen,
            "unique_patches_seen": unique_tiles_seen * unique_tile_patch_count,
        }

    # One JEPA draw per student pass over the global views, as (count, keep_idx, mask_idx, masks, mask_w) groups.
    # block is a single full-grid group; cross has one group per context scale since sequence lengths differ.
    def draw_masks(n_crops):
        if jepa_type == "block":
            masks, mask_idx, mask_w = make_block_mask(n_crops, global_grid, device, int(jepa_cfg["blocks"]), float(jepa_cfg["block_scale"]))
            return [(1, None, mask_idx, masks, mask_w)]
        return [(count, *make_region_idx(n_crops * count, global_grid, device, n, side, int(jepa_cfg["targets"]), float(jepa_cfg["target_decay"])), None, None)
                for n, side, count in context_regions]

    # Augment a collated (B, V, 3, H, W) view stack and flatten crop-major, so [crop0_img0, crop0_img1, ..., crop1_img0, ...]
    # chunks cleanly per crop for teacher/student alignment. None when the dataloader omitted the key (local_views: 0).
    def load_views(batch, key):
        return augment(batch[key].to(device, non_blocking=True)).transpose(0, 1).flatten(0, 1) if key in batch else None

    # Compute (dino_loss, jepa_loss, kde, iso, decorr, bal) for one batch of (gf, lf) crops with the given mask groups + schedule
    # values; iso, decorr, and bal are unweighted so they double as diagnostics. Used by both the train step and evaluate() (no_grad).
    def compute_losses(gf, lf, b, mask_groups, t_temp, k_scale, ckpt=False):
        # Tensor schedule values do not specialize Sinkhorn to each temperature.
        t_temp = torch.tensor(t_temp, device=gf.device)
        with torch.no_grad():
            t = teacher_backbone(gf)
            t_cls = teacher_dino_head(t["cls"])[head_out].chunk(train_cfg["global_views"])
            t_swap = torch.cat((t_cls[1], t_cls[0]))
            if balance_type == "sinkhorn":
                t_prob = sinkhorn_fn(t_swap, t_temp)
            elif balance_type == "simdino":
                t_prob = t_swap
            else:  # memax never moves the center off 0, leaving the plain sharpened softmax
                t_prob = F.softmax((t_swap.float() - center) / t_temp, dim=-1)
            # Update after use, as in DINO; evaluate() puts the student in eval mode, so val batches never move the center.
            if balance_type == "center" and student_backbone.training:
                center.lerp_(t_swap.float().mean(0), 0.1)
            t_prob = t_prob.view(2, b, *t_cls[0].shape[1:])
        L, R, D = train_cfg["local_views"], total_regions, student_backbone.embed_dim
        # Each context region is one more student-teacher pair, so the multi-crop mean is over 2L + 2R pairs.
        global_loss, jepa_loss, kde, iso, decorr, bal, p_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Summed student softmax over the batch (F, K); detached unless memax trains on it, so a 131k-prototype head keeps no softmax graph.
        anchor_probs = lambda x: F.softmax((x if balance_type == "memax" else x.detach()).float() / student_temp, dim=-1).sum(0)
        for count, keep_idx, mask_idx, masks, mask_w in mask_groups:
            # Each region copy draws its own student noise; the teacher and JEPA targets stay on the clean crop.
            sv = student_augment(gf if count == 1 else gf.repeat(count, 1, 1, 1))
            sg = student_backbone(sv, masks=masks, checkpoint=ckpt, keep_idx=keep_idx)
            t_flat = t_prob.flatten(0, 1)
            s_logits, s_chunks = student_dino_head(sg["cls"])
            global_loss = global_loss + dino_ce_fn((s_logits, s_chunks)[head_out], t_flat if count == 1 else t_flat.repeat(count, 1, 1), student_temp) * 2 * count / (2 * L + 2 * R)
            if keep_idx is None:
                patch_target = F.layer_norm(t["patches"].flatten(0, 1), (D,))[mask_idx]
                patch_prediction = student_predictor(sg["patches"]).flatten(0, 1)[mask_idx]
                jepa = jepa_loss_fn(patch_prediction, patch_target, mask_w) / max(1, b * 2)
            else:
                # Gather targets per region chunk so the teacher grid is never duplicated count times.
                tp = F.layer_norm(t["patches"], (D,))
                patch_target = torch.cat([tp.gather(1, m[..., None].expand(-1, -1, D)) for m in mask_idx.chunk(count)])
                # Queries are the backbone's own mask_token (trained by JEPA) + patch pos (detached, so only the encoder path trains it).
                query_table = student_backbone.mask_token + student_backbone.patch_pos_embed(global_grid, global_grid).detach()
                queries = query_table.expand(mask_idx.shape[0], -1, -1).gather(1, mask_idx[..., None].expand(-1, -1, D))
                context_cls = {"student": sg["cls"], "teacher": t["cls"].repeat(count, 1), "none": None}[jepa_cfg["context_cls"]]
                context = torch.cat([*([] if context_cls is None else [context_cls[:, None]]), sg["registers"], sg["patches"]], dim=1)
                jepa = F.smooth_l1_loss(student_predictor(context, queries), patch_target)
            jepa_loss = jepa_loss + jepa * count / R
            kde = kde + dino_cfg["kde_loss_weight"] * k_scale * sum(kde_loss(x, dino_cfg["kde_concentration"], dino_cfg["kde_radius"]) for x in sg["cls"].chunk(train_cfg["global_views"] * count)) / R
            # Isotropy acts on backbone CLS, in front of the DINO head, so the head can still collapse its bottleneck for prototype matching.
            iso = iso + sum(isotropy_loss(x) for x in sg["cls"].chunk(train_cfg["global_views"] * count)) / (train_cfg["global_views"] * R)
            # Factor decorrelation acts on the head's per-factor bottleneck chunks, per batch of distinct images, so factored codebooks stop copying each other.
            decorr = decorr + sum(factor_decorrelation(x) for x in s_chunks.chunk(train_cfg["global_views"] * count)) / (train_cfg["global_views"] * R)
            # Balance term: SimDINO's negative coding rate per batch of distinct images, else accumulate the student's predictions for ME-MAX.
            if head_out:
                bal = bal + sum(-coding_rate(z) for z in s_chunks.chunk(train_cfg["global_views"] * count)) / (train_cfg["global_views"] * R)
            else:
                p_sum = p_sum + anchor_probs(s_logits)
        local_loss = 0.0
        if lf is not None:
            sl_cls = student_dino_head(student_backbone(lf, checkpoint=ckpt)["cls"])[head_out]
            # CE is linear in targets; keep the original reduction order for eager recipes.
            local_loss = (dino_ce_fn(sl_cls.view(L, b, *sl_cls.shape[1:]), t_prob.sum(0), student_temp) * L if train_cfg["compile"]
                          else sum(dino_ce_fn(x, y, student_temp) for x in sl_cls.chunk(L) for y in t_prob)) / (2 * L + 2 * R)
            p_sum = p_sum if head_out else p_sum + anchor_probs(sl_cls)
        if not head_out:
            # ME-MAX (MSN): negative entropy of the student's mean prediction over every anchor view (globals + locals) in the batch, summed over factors.
            p_bar = p_sum / (b * (2 * R + (0 if lf is None else L)))
            bal = (p_bar * p_bar.clamp_min(1e-12).log()).sum()
        return local_loss + global_loss, jepa_loss, kde, iso, decorr, bal

    # Held-out validation pass: same DINO + JEPA + KDE + isotropy + decorrelation losses on `val_batches` of the val split.
    # Schedule terms (teacher_temp, kde_scale) drift over training, so read val curves as same-step
    # diagnostics. RNG is snapshotted/restored so val masks don't perturb the next training step.
    def evaluate(eval_step, eval_teacher_temp, eval_kde_scale):
        for m in (student_backbone, student_dino_head, student_predictor):
            m.eval()
        py_rng, cpu_rng, cuda_rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        random.seed(train_cfg["seed"] + eval_step)
        torch.manual_seed(train_cfg["seed"] + eval_step)
        sums = torch.zeros(7, device=device)
        n_batches = 0
        for vb_idx, vbatch in enumerate(val_loader):
            if vb_idx >= int(train_cfg["val_batches"]):
                break
            gf, lf = load_views(vbatch, "global_views"), load_views(vbatch, "local_views")
            b = vbatch["global_views"].shape[0]
            with torch.no_grad(), autocast:
                dino_l, jepa_l, kde_v, iso_v, decorr_v, bal_v = compute_losses(gf, lf, b, draw_masks(b * train_cfg["global_views"]), eval_teacher_temp, eval_kde_scale)
            total_v = dino_l + jepa_l + kde_v + balance_weight * bal_v + dino_cfg["iso_loss_weight"] * iso_v + decorr_weight * decorr_v
            sums += torch.tensor([float(dino_l), float(jepa_l), float(kde_v), float(iso_v), float(decorr_v), float(bal_v), float(total_v)], device=device)
            n_batches += 1
        random.setstate(py_rng)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        return dict(zip(("dino", "jepa", "kde", "iso", "decorr", "balance", "total"), (sums / max(1, n_batches)).tolist()))

    # Ingest completed probe result JSONs into metrics.jsonl and wandb.
    def log_probe_results():
        if probe_state is not None:
            collect_probe_results(probe_state, wandb_run, metrics_path)

    # Queue a probe at `checkpoint_step` for the given sample target; no-op if already done.
    def run_probe_at(checkpoint_step, target_samples):
        if probe_state is None or (probe_state["paths"]["results_dir"] / f"step_{checkpoint_step:07d}.json").exists():
            log_probe_results()
            return
        queue_probe_job(probe_state, checkpoint_payload(checkpoint_step, full=False), checkpoint_step, train_flops, min(1.0, target_samples / max_train_samples))
        log_probe_results()

    # Queue the furthest crossed sample milestone so delayed probes do not run on stale checkpoints.
    def maybe_run_probe(checkpoint_step):
        nonlocal next_probe_idx
        if probe_state is None or next_probe_idx >= len(probe_targets) or examples_seen < probe_targets[next_probe_idx]:
            return
        while next_probe_idx + 1 < len(probe_targets) and examples_seen >= probe_targets[next_probe_idx + 1]:
            next_probe_idx += 1
        run_probe_at(checkpoint_step, probe_targets[next_probe_idx])
        next_probe_idx += 1

    log_probe_results()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_train_samples = math.ceil(max_train_samples * dino_cfg["warmup_fraction"])
    # Probe targets are sample milestones: one tile counts once even with many global/local crops.
    probe_count = int(cfg["probe"]["count"]) if probe_enabled(cfg) else 0
    probe_targets = [math.ceil(max_train_samples * (i + 1) / probe_count) for i in range(probe_count)]
    if len(set(probe_targets)) != len(probe_targets):
        raise ValueError(f"probe.count={probe_count} is too large for max_train_samples={max_train_samples}")
    next_probe_idx = 0
    if probe_state is not None:
        completed = [round(float(json.loads(p.read_text()).get("target_fraction", -1)) * max_train_samples) for p in probe_state["paths"]["results_dir"].glob("step_*.json")]
        if completed:
            next_probe_idx = sum(target <= max(completed) for target in probe_targets)
    train_loop_started_at = time.monotonic()
    last_saved_step = step
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if train_cfg["bf16"] else contextlib.nullcontext()
    # Per-step FLOPs are measured once on the first step (a scaled eager forward + backward probe plus the eager
    # opt.step) and reused for every subsequent step since the shapes don't change.
    # Counts the EMA teacher forward + all objective heads, not just the backbone, so the
    # 1e18 leaderboard cap reflects real GPU work.
    measured_flops_per_step = None

    while examples_seen + batch_size <= max_train_samples and train_flops < max_train_flops:
        for batch in train_loader:
            if examples_seen + batch_size > max_train_samples or train_flops >= max_train_flops:
                break
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            student_backbone.train()
            student_dino_head.train()
            student_predictor.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            gf, lf = load_views(batch, "global_views"), load_views(batch, "local_views")
            visible_now = batch_size * (train_cfg["global_views"] * visible_global_patches + train_cfg["local_views"] * local_patches)
            # dino.schedule_progress drives every schedule below: `samples` runs each over the full sample budget, while `pace` keeps the
            # legacy sample progress at the block baseline's FLOP pace (0.1862 of the budget at the 1M cap), so FLOP-cheaper architectures
            # share one trajectory and the arcs deliberately stop early. Real FLOPs still enforce the cap either way.
            sample_frac = min(1.0, examples_seen / max_train_samples)
            prog = {"pace": min(1.0, 0.1862 * sample_frac), "samples": sample_frac}[dino_cfg["schedule_progress"]]
            warmup = min(1.0, examples_seen / max(1, warmup_train_samples))
            if warmup < 1.0:
                lr = dino_cfg["lr"] * warmup
            else:
                lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], (prog - dino_cfg["warmup_fraction"]) / max(1e-9, 1 - dino_cfg["warmup_fraction"]))
            wd = cosine_schedule(*dino_cfg["weight_decay"], prog)  # [start, end]; the applied shrink per step is lr * wd
            # memax follows MSN: targets sharpened by T = 0.25 against the 0.1 student temperature, i.e. a fixed 0.025 teacher temperature.
            teacher_temp = temp_scale * (0.025 if balance_type == "memax" else 0.04 + min(1.0, prog / 0.2727) * (0.07 - 0.04))
            last_layer_lr = 0.0 if prog < dino_cfg["freeze_last_layer_fraction"] else lr
            for group in (g for o in opts for g in o.param_groups):
                base_lr = last_layer_lr if group["last_layer"] else lr
                group["lr"] = base_lr * group["lr_mult"]
                group["weight_decay"] = wd * group["wd_mult"]
            mask_groups = draw_masks(batch_size * train_cfg["global_views"])
            kde_scale = min(1.0, max(0.0, (prog - 0.1) / 0.4))
            if measured_flops_per_step is None:
                # Compiled kernels are opaque to FlopCounterMode, so forward + backward FLOPs are counted once on an eager,
                # no-update pass over a 32-image slice and scaled to the batch (linear, bar negligible b^2 KDE terms). The real
                # steps all run compiled, so eager memory never caps the batch size. RNG is restored so the probe's masks,
                # drop path, and student noise leave the training stream untouched.
                probe_b = min(batch_size, 32)
                rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
                probe_views = [None if v is None else v.view(-1, batch_size, *v.shape[1:])[:, :probe_b].flatten(0, 1) for v in (gf, lf)]
                with FlopCounterMode(display=False) as probe_ctx, torch.compiler.set_stance("force_eager"):
                    with autocast:
                        p_dino, p_jepa, p_kde, p_iso, p_decorr, p_bal = compute_losses(*probe_views, probe_b, draw_masks(probe_b * train_cfg["global_views"]), teacher_temp, kde_scale, ckpt=activation_checkpointing)
                        probe_loss = p_dino + p_jepa + p_kde + balance_weight * p_bal + warmup * (dino_cfg["iso_loss_weight"] * p_iso + decorr_weight * p_decorr)
                    probe_loss.backward()
                for o in opts:
                    o.zero_grad(set_to_none=True)
                random.setstate(rng[0])
                torch.random.set_rng_state(rng[1])
                torch.cuda.set_rng_state(rng[2], device)
            with autocast:
                dino_loss_value, jepa_loss, kde, iso, decorr, bal = compute_losses(
                    gf, lf, batch_size, mask_groups, teacher_temp, kde_scale,
                    ckpt=activation_checkpointing,
                )
                # Isotropy and factor decorrelation ramp in with LR warmup so they don't reshape the pretrained space before the LR is live;
                # the balance term is the DINO head's anti-collapse mechanism, so it is on from step 0.
                total_loss = dino_loss_value + jepa_loss + kde + balance_weight * bal + warmup * (dino_cfg["iso_loss_weight"] * iso + decorr_weight * decorr)
            for o in opts:
                o.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                [*student_backbone.parameters(), *student_dino_head.parameters(), *student_predictor.parameters()],
                dino_cfg["clip_grad"],
            )
            # The optimizer step is batch-independent (only Muon's Newton-Schulz matmuls count), so the first one is counted eagerly as-is.
            opt_ctx = FlopCounterMode(display=False) if measured_flops_per_step is None else contextlib.nullcontext()
            with opt_ctx, torch.compiler.set_stance("force_eager" if measured_flops_per_step is None else "default"):
                for o in opts:
                    o.step()
            if measured_flops_per_step is None:
                measured_flops_per_step = int(probe_ctx.get_total_flops()) * batch_size // probe_b + int(opt_ctx.get_total_flops())
                print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}  (fwd+bwd counted at batch {probe_b}, scaled to {batch_size})", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                # Per-step momentum is rescaled to a 128-image reference batch, so the teacher lags the student by a fixed number of samples at any batch size.
                m = cosine_schedule(0.994, 1.0, prog) ** (batch_size / 128)
                update_ema(student_backbone, teacher_backbone, m)
                update_ema(student_dino_head, teacher_dino_head, m)
            examples_seen += batch_size
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            if should_log:
                reduced = {
                    "dino": float(dino_loss_value.detach()),
                    "jepa": float(jepa_loss.detach()),
                    "kde": float(kde.detach()),
                    "iso": float(iso.detach()),
                    "decorr": float(decorr.detach()),
                    "balance": float(bal.detach()),
                    "total": float(total_loss.detach()),
                }
                step_seconds = time.monotonic() - batch_started_at  # Loss transfers wait for GPU completion.
                unique_counts = flush_unique_counts()
                now = time.time()
                elapsed = max(1e-6, now - last_time)
                items_per_sec = (examples_seen - last_examples) / elapsed
                visible_patches_per_sec = (visible_patch_presentations - last_visible_patch_presentations) / elapsed
                flops_per_sec = (train_flops - last_train_flops) / elapsed
                train_loop_wall_seconds = time.monotonic() - train_loop_started_at
                last_time = now
                last_examples = examples_seen
                last_visible_patch_presentations = visible_patch_presentations
                last_train_flops = train_flops
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                flop_steps_remaining = math.ceil(max(0, max_train_flops - train_flops) / max(1, step_train_flops))
                sample_steps_remaining = max(0, max_train_samples - examples_seen) // batch_size
                steps_remaining = min(flop_steps_remaining, sample_steps_remaining)
                total_steps_estimate = completed_step + steps_remaining
                eta_seconds = int(max(0.0, steps_remaining * console_gap_ms / 1000.0 / steps_since_console))
                eta_string = f"{eta_seconds // 3600}:{(eta_seconds % 3600) // 60:02d}:{eta_seconds % 60:02d}"
                current_lr = opt.param_groups[0]["lr"]
                train_log = {
                    "step": completed_step,
                    **reduced,
                    "items_per_sec": items_per_sec,
                    "visible_patches_per_sec": visible_patches_per_sec,
                    "flops_per_sec": flops_per_sec,
                    "wall_seconds": train_loop_wall_seconds,
                    "step_seconds": step_seconds,
                    "data_seconds": data_seconds,
                    "console_gap_ms": console_gap_ms,
                    "eta_seconds": eta_seconds,
                    "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
                    "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
                    "lr": current_lr,
                    "wd": wd,
                    "teacher_temp": teacher_temp,
                    "teacher_momentum": m,
                    "kde_scale": kde_scale,
                    "cls_participation_ratio": student_backbone.embed_dim / (reduced["iso"] + 1),  # batch-capped at 127 for 128 images
                    "batch_size": batch_size,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                    "grad_norm": float(grad_norm.detach()),
                }
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f}  total: {reduced['total']:.4f}  "
                    f"dino: {reduced['dino']:.4f}  jepa: {reduced['jepa']:.4f}  kde: {reduced['kde']:.4f}  iso: {reduced['iso']:.4f}  decorr: {reduced['decorr']:.4f}  balance: {reduced['balance']:.4f}  "
                    f"grad_norm: {train_log['grad_norm']:.4f}  flops/s: {flops_per_sec:.3e}  "
                    f"time: {step_seconds:.6f}  data: {data_seconds:.6f}  "
                    f"max mem: {int(gpu_peak_mem_gb * 1024)}",
                    flush=True,
                )
                last_console_step = completed_step
                last_console_monotonic = console_now
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(train_log) + "\n")
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_log.items() if key != "step"},
                    step=completed_step,
                )
                log_probe_results()
                torch.cuda.reset_peak_memory_stats(device)
            if save_checkpoints and completed_step % save_every == 0:
                # Atomic rename keeps the previous good latest.pt intact if a
                # kill lands mid-save.
                save_latest_checkpoint(completed_step)
            # Probe at intermediate sample milestones (probe.count > 1); the final probe
            # always runs after the loop exits, regardless of milestones.
            maybe_run_probe(completed_step)
            if completed_step % int(train_cfg["eval_every"]) == 0 or train_flops >= max_train_flops or examples_seen + batch_size > max_train_samples:
                val = evaluate(completed_step, teacher_temp, kde_scale)
                val_log = {"step": completed_step, **{f"val_{k}": v for k, v in val.items()}}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(val_log) + "\n")
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=completed_step)
                print(f"{console_prefix()} Validation  [{completed_step}]  total: {val['total']:.4f}  dino: {val['dino']:.4f}  jepa: {val['jepa']:.4f}  kde: {val['kde']:.4f}  iso: {val['iso']:.4f}  decorr: {val['decorr']:.4f}  balance: {val['balance']:.4f}", flush=True)
                # Reset rate clocks after validation so the next train log is train-rate only.
                last_console_step, last_console_monotonic = completed_step, time.monotonic()
                last_time, last_examples, last_visible_patch_presentations, last_train_flops = time.time(), examples_seen, visible_patch_presentations, train_flops
            step = completed_step
            data_wait_started_at = time.monotonic()
            if train_flops >= max_train_flops or examples_seen + batch_size > max_train_samples:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else "max_train_samples"
    final_unique_counts = flush_unique_counts()
    if step > 0:
        # Final probes have their own readers; close pretraining workers before they compete for CPU/IO.
        if train_cfg["num_workers"] > 0 and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()
            train_loader._iterator = None
        # Probes get their own short-lived checkpoint via run_probe_at; only persist latest.pt
        # at end-of-run when periodic saving is on (save_every set) so smoke runs leave nothing.
        if save_checkpoints and step != last_saved_step:
            save_latest_checkpoint(step)
        run_probe_at(step, examples_seen)
    log_probe_results()
    # Summary is the small, stable artifact downstream scripts and humans compare across runs.
    summary = {
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "train_seed": int(train_cfg["seed"]),
        "data_split_seed": int(cfg["data"]["split_seed"]),
        "wandb": wandb_meta,
        "slurm_job_id": slurm_job_id,
        "backbone_activated_params": backbone_activated_params,
        "batch_size": batch_size,
        "unique_slide_batches": train_cfg["unique_slide_batches"],
        "activation_memory_budget": train_cfg["activation_memory_budget"],
        "max_train_samples": max_train_samples,
        "max_train_flops": max_train_flops,
        "train_loop_wall_seconds": train_loop_wall_seconds,
        "stop_reason": stop_reason,
        "steps_completed": step,
        "tile_presentations": examples_seen,
        "visible_patch_presentations": visible_patch_presentations,
        **final_unique_counts,
        "train_flops": train_flops,
        "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
        "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
        # Average throughput over the train loop; wall time is diagnostic, not an eligibility cap.
        "flops_per_sec": train_flops / max(1.0, train_loop_wall_seconds),
        "visible_patches_per_sec": visible_patch_presentations / max(1.0, train_loop_wall_seconds),
        "warmup_fraction": dino_cfg["warmup_fraction"],
        "warmup_train_samples": warmup_train_samples,
        "optimizer": dino_cfg["optimizer"],
        "lr": dino_cfg["lr"],
        "schedule_progress": dino_cfg["schedule_progress"],
        "adam_beta2": dino_cfg["adam_beta2"],
        "kde_loss_weight": dino_cfg["kde_loss_weight"],
        "kde_concentration": dino_cfg["kde_concentration"],
        "kde_radius": dino_cfg["kde_radius"],
        "dino_head": dino_cfg["head"],
        "dino_balance": dino_cfg["balance"],
        "temp_scale": temp_scale,
        "iso_loss_weight": dino_cfg["iso_loss_weight"],
        "drop_path_rate": dino_cfg["drop_path_rate"],
        "layerwise_decay": dino_cfg["layerwise_decay"],
        "weight_decay": dino_cfg["weight_decay"],
        "cls_specialization": cls_spec,
        "block_expansion": expansion,
        "probe_target_samples": probe_targets,
        "probe_target_fractions": [None if max_train_samples == 0 else target / max_train_samples for target in probe_targets],
        **({} if probe_state is None else completed_probe_summary(output_dir)),
    }
    if probe_state is not None and "final_score" not in summary:
        raise ValueError("probe.enabled is true but final_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{console_prefix()} Summary  "
        f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
        f"final_score: {summary.get('final_score')}",
        flush=True,
    )
    for key, value in summary.items():
        wandb_run.summary[key] = value
    wandb_run.finish()
    finish_labless_autosubmit(labless_autosubmit_file, output_dir, repo_dir)


if __name__ == "__main__":
    main()
