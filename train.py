# Continual DINOv2 pretraining on unlabelled images (single-GPU). Three loss terms:
# DINO CLS self-distillation (Sinkhorn-Knopp centred teacher targets),
# I-JEPA patch-feature regression, and a KDE uniformity term on the
# L2-normalised CLS tokens. YAML drives the tunable knobs (backbone variant,
# LR + LR scheduler, drop path, layerwise decay, KDE weight + concentration,
# FLOP/sample budgets, batch size); other DINOv2 hyperparameters are hardcoded
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from dataloader import IMAGE_SIZE, ParquetImageDataset, hed_jitter_batch
from model import FactoredDINOHead, GradScale, JEPAPredictor, SpecializedDinoV2ViT, load_dinov2_pretrained, make_prototype_regularizer
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Read the YAML recipe and fail before any GPU work if the parquet dataset is absent.
# expandvars is necessary to resolve `$USER` for checked-in configs.
def load_config():
    if len(sys.argv) < 2:
        raise ValueError("usage: python train.py <config.yaml> [output_dir=<path>]")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    # Optional `key=value` overrides after the config; only output_dir is supported,
    # since it's the run identifier and routinely set per-submission from the CLI.
    for arg in sys.argv[2:]:
        key, _, value = arg.partition("=")
        if key != "output_dir":
            raise ValueError(f"unsupported override {arg!r}; only output_dir=<path> is supported")
        cfg["project"]["output_dir"] = os.path.expandvars(value)
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not any(dataset_dir.glob("shard-*.parquet")):
        raise FileNotFoundError(
            f"No parquet shards (shard-*.parquet) under {dataset_dir}. Fetch the configured "
            f"dataset by running `python prepare.py {cfg['config_path']} download=True`. "
            f"Follow the data setup in README.md before launching train.py."
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
    print("This looks like a full Labless-eligible run. Leave either prompt blank to train without auto-submit.", flush=True)
    run_name = input("Labless run name (<=20 chars): ").strip()
    notes = input("Labless notes: ").strip()
    if not run_name or not notes or len(run_name) > 20:
        print("Labless auto-submit skipped; run name and notes are required, and run name must be <=20 chars.", flush=True)
        return ""
    token_path = str(Path(str(Path(cfg["project"]["output_dir"]).expanduser().resolve()) + ".labless_autosubmit.json"))
    status = subprocess.run(
        [sys.executable, str(repo_dir / "labless" / "submit_to_labless.py"), "login_only=true", f"token_output={token_path}", f"run_name={run_name}", f"notes={notes}"],
        cwd=repo_dir,
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
    ).returncode
    token_file.unlink(missing_ok=True)
    if status == 2:
        print(f"{console_prefix()} Labless  auto-submit skipped because the completed run did not satisfy submission restrictions.", flush=True)
    elif status != 0:
        raise SystemExit(status)


# Cosine schedule from `start` to `end` over fractional progress in [0, 1].
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(1.0, max(0.0, frac))))


# Sinkhorn-Knopp centring across this batch, used for DINO teacher targets. Input is (N, F, K): each of
# the F codebooks is an independent balancing problem over its own K prototypes, solved in parallel here.
# At F=1 this is numerically identical to the unfactorised version it replaces.
def sinkhorn(x, temp):
    q = torch.exp(x.float() / temp).permute(1, 2, 0)  # (F, K, N)
    k, b = q.shape[1], q.shape[2]
    q /= q.sum((1, 2), keepdim=True)
    for _ in range(3):
        q /= q.sum(2, keepdim=True) * k  # prototype marginals, per factor
        q /= q.sum(1, keepdim=True) * b  # sample marginals, per factor
    return (q * b).permute(2, 0, 1)  # (N, F, K)


# KDE uniformity loss on L2-normalised CLS tokens.
def kde_loss(x, concentration):
    x = F.normalize(x, p=2, dim=-1)
    sim = concentration * (x @ x.T)
    sim.fill_diagonal_(-float("inf"))
    return torch.logsumexp(sim, dim=1).mean() - math.log(max(1, sim.shape[1] - 1))


# Toroidal distance from every grid index to the window [t, t+side): 0 inside, else the shorter way round.
def _axis_dist(pos, t, grid, side):
    delta = (pos - t) % grid  # (B, grid), offset from the window start
    return torch.where(delta < side, torch.zeros_like(delta), torch.minimum(delta - side + 1, grid - delta))


# I-JEPA context/target geometry, drawn together because the targets are the context's complement.
# Everything is toroidal, so a region wraps across the image border rather than being clipped and every
# position is equally likely to be covered (no centre bias, no edge-pinned artifacts).
#
# Context: one contiguous side x side window at a random per-sample offset. The student encoder sees
# exactly these V = side**2 patches, so V is constant and the gather stays rectangular. Ascending order
# preserves the grid-order invariant on x_norm_patchtokens.
#
# Targets: k patches drawn without replacement from outside the window, so every target is unseen and
# none is solvable by copying a visible token. Weights are exp(-decay * d) with d the Chebyshev distance
# to the window normalised by its own per-row maximum, which puts `decay` in e-folds across the full
# available depth and makes it mean the same thing at every side. decay=0 is uniform, decay>0 pulls
# targets against the context boundary, decay<0 pushes them deep into the masked region.
#
# Weighted sampling without replacement is the exponential race (Efraimidis-Spirakis): with keys e_i / w_i
# for e_i ~ Exp(1), the k smallest are exactly the draw, so it stays one topk with no loop. Context keys
# are +inf, which is the w_i = 0 case and is what enforces the exclusion.
def make_region_idx(batch, grid, device, side, k, decay):
    pos = torch.arange(grid, device=device)
    ty, tx = torch.randint(grid, (2, batch, 1), device=device)
    rows, cols = (ty + pos[:side]) % grid, (tx + pos[:side]) % grid  # (B, side)
    keep = (rows[..., None] * grid + cols[:, None, :]).flatten(1).sort(dim=-1).values  # (B, side**2)
    d = torch.maximum(_axis_dist(pos, ty, grid, side)[:, :, None], _axis_dist(pos, tx, grid, side)[:, None, :])
    d = d.flatten(1).float()  # (B, grid**2)
    d = d / d.amax(-1, keepdim=True).clamp(min=1.0)
    keys = torch.empty_like(d).exponential_() * torch.exp(8.0 * decay * d)
    keys.scatter_(1, keep, float("inf"))
    return keep, keys.topk(k, dim=-1, largest=False).indices  # (B, side**2), (B, k)


# AdamW parameter groups with layer-wise LR decay on the backbone:
# block i gets lr * layerwise_decay^(depth - 1 - i); patch_embed gets the deepest decay
# multiplied by patch_embed_lr_mult; biases, norms, and token/positional embeddings (backbone
# cls/register/pos/mask tokens, predictor query/pos/cond_emb) get no weight decay; the head's
# DINO final weight-norm last_layer parameters get an LR-freeze for the first dino.freeze_last_layer_fraction.
def build_param_groups(student_backbone, student_dino_head, student_predictor, layerwise_decay, patch_embed_lr_mult, cls_lr_mult=1.0):
    depth = len(student_backbone.blocks)
    # Coalesce params that share (lr_mult, wd_mult, last_layer) into a single group each (~30 groups
    # instead of one-per-param), so AdamW's foreach path fuses the step across many tensors rather than
    # launching per-parameter kernels. Per-param lr/wd are unchanged, so the optimization is numerically identical.
    coalesced = {}
    # ndim >= 2 embedding-like params the bias/norm/ndim rule misses; the predictor's pos table is a
    # query's only positional identity, so decaying it directly erodes the JEPA queries.
    no_wd_names = {"pos_embed", "cls_token", "register_tokens", "mask_token", "query", "pos", "cond_emb.weight"}
    modules = ((student_backbone, "backbone"), (student_dino_head, "dino_head"), (student_predictor, "jepa_predictor"))
    for module, kind in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if kind == "backbone" and name.startswith("blocks."):
                # Layer-wise decay exists to protect *pretrained* low-level features. The CLS-specialized
                # copies (".cls." branch of every Specialized wrapper) are new capacity wearing a
                # pretrained initialization: they start as exact duplicates of their patch twin and are
                # worth nothing until they diverge, which is purely LR-driven. Under decay they are also
                # penalised hardest exactly where they matter -- qkv specialization lives in the first
                # `qkv_blocks` blocks, which at decay 0.7 train at 2-6% of base LR -- so they could never
                # diverge within the sample budget. Exempt them; the ".patch." branch is the untouched
                # pretrained tensor and keeps its decay.
                lr_mult = cls_lr_mult if ".cls." in name else layerwise_decay ** (depth - 1 - int(name.split(".")[1]))
            elif kind == "backbone" and name.startswith("patch_embed."):
                lr_mult = (layerwise_decay ** depth) * patch_embed_lr_mult
            wd_mult = 0.0 if name.endswith("bias") or "norm" in name or p.ndim < 2 or name in no_wd_names else 1.0
            # FactoredDINOHead holds its whole pool in one `prototypes` Parameter, so the freeze that used
            # to key on the weight-norm `last_layer` name has to look for that instead.
            key = (lr_mult, wd_mult, kind == "dino_head" and "prototypes" in name)
            coalesced.setdefault(key, {"params": [], "lr_mult": lr_mult, "wd_mult": wd_mult, "last_layer": key[2]})["params"].append(p)
    return list(coalesced.values())


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
    # FINO metadata-guidance: select factors + signs (float; + encourage M+ / - suppress M-). fino_meta (built or
    # copied beside the dataset by prepare.py) holds per-factor barcode maps + cardinalities (n) / vector dims.
    fino_cfg = cfg["fino"] if (cfg.get("fino") or {}).get("enabled") else None
    fino_disc = [(f, float(s)) for f, s in fino_cfg.get("discrete", [])] if fino_cfg else []
    fino_cont = [(f, float(s)) for f, s in fino_cfg.get("continuous", [])] if fino_cfg else []
    fino_meta = json.loads((Path(cfg["data"]["dataset_dir"]) / "fino_meta.json").read_text()) if fino_cfg else {"n": {}, "cont_dim": {}}
    # FINO two-phase: freeze the backbone (except patch_embed) for the first this-fraction of the run so the DINO/JEPA
    # heads + metadata prototypes/predictors converge against a fixed target before they steer the encoder. 0 = off.
    freeze_backbone_frac = float(dino_cfg.get("freeze_backbone_fraction", 0.0))
    # JEPA-T: optionally condition the JEPA predictor on a discrete factor (must be in fino.discrete so its per-sample
    # label rides in the batch). cond_col indexes that factor's column in batch["meta_disc"].
    jepa_cond = fino_cfg.get("jepa_cond") if fino_cfg else None
    cond_col = [f for f, _ in fino_disc].index(jepa_cond) if jepa_cond else None
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
    student_backbone = load_dinov2_pretrained(SpecializedDinoV2ViT(variant=variant, drop_path_rate=dino_cfg["drop_path_rate"], qkv_blocks=cfg["model"]["qkv_blocks"])).to(device)
    teacher_backbone = deepcopy(student_backbone)
    teacher_backbone.train(False)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    # Product-factorised prototypes: the bottleneck splits into head_factors chunks, each with its own
    # head_prototypes codebook, giving head_prototypes**head_factors joint codes while Sinkhorn only ever
    # balances head_prototypes bins at a time. Defaults reproduce the single 131072-way head exactly.
    n_factors, n_prototypes = int(dino_cfg.get("head_factors", 1)), int(dino_cfg.get("head_prototypes", 131072))
    # Prototype-bank penalty, weighted inside the regularizer so compute_losses just adds the scalar.
    prototype_reg = str(dino_cfg.get("prototype_reg", "none"))
    regularizer = make_prototype_regularizer(prototype_reg, float(dino_cfg.get("prototype_reg_weight", 0.0)), float(dino_cfg.get("prototype_reg_eps", 0.5)))
    student_dino_head = FactoredDINOHead(student_backbone.embed_dim, n_prototypes, dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3, n_factors, regularizer).to(device)
    teacher_dino_head = deepcopy(student_dino_head)
    global_grid = train_cfg["global_size"] // student_backbone.patch_size
    global_patches = global_grid ** 2
    # Context-region scales: [[side, count], ...], `count` independent regions of that side sampled per
    # global view. Multi-crop emulated by index gathering; heterogeneous sides give the student both
    # near-complete and small views, so DINO keeps a global-to-global term alongside local-to-global.
    context_regions = [(int(side), int(count)) for side, count in train_cfg["context_regions"]]
    for side, count in context_regions:
        if not 0 < side <= global_grid or count < 1:
            raise ValueError(f"train.context_regions entry [{side}, {count}] invalid: side must be in 1..{global_grid} (the {global_grid}x{global_grid} patch grid) and count >= 1")
    total_regions = sum(count for _, count in context_regions)
    # Targets come from the complement, so the largest context sets the ceiling on k.
    jepa_targets = int(dino_cfg["jepa_targets"])
    max_complement = global_patches - max(side ** 2 for side, _ in context_regions)
    if not 0 < jepa_targets <= max_complement:
        raise ValueError(f"dino.jepa_targets={jepa_targets} invalid: targets are drawn without replacement from outside the context, so it must be in 1..{max_complement} (the {global_patches}-patch grid minus the largest context region)")
    # Target-depth curriculum: cosine from _decay to _decay_end over reg_frac, so it inherits reg_key.
    # Omitting the end value holds the decay constant, which keeps a fixed-lambda run config-identical.
    jepa_decay_start = float(dino_cfg["jepa_target_decay"])
    jepa_decay_end = float(dino_cfg.get("jepa_target_decay_end", jepa_decay_start))

    view_jitter = float(dino_cfg.get("student_view_jitter", 0.0))
    norm_mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)

    def jitter_view(x):
        with torch.autocast(device_type="cuda", enabled=False):
            rgb = hed_jitter_batch((x.float() * norm_std + norm_mean).clamp_(0.0, 1.0), view_jitter)
            return ((rgb - norm_mean) / norm_std).to(x.dtype)

    # One (keep_idx, mask_idx) pair per scale; each is (b * global_views * count, ...). Separate forwards,
    # since scales have different sequence lengths.
    def make_region_indices(n_crops, decay):
        pairs = [make_region_idx(n_crops * count, global_grid, device, side, jepa_targets, decay) for side, count in context_regions]
        keep, mask = zip(*pairs)
        return keep, mask
    # n_pos sizes the predictor's query position table: one row per global-view patch position.
    student_predictor = JEPAPredictor(student_backbone.embed_dim, global_patches, depth=int(dino_cfg["jepa_pred_depth"]), width=int(dino_cfg["jepa_pred_width"]), n_cond=(fino_meta["n"][jepa_cond] if jepa_cond else 0)).to(device)
    for p in teacher_dino_head.parameters():
        p.requires_grad = False
    backbone_activated_params = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    # FINO continuous-factor predictors (phi -> vector regressors); their params join the optimizer.
    predictors = {f: nn.Sequential(nn.Linear(student_backbone.embed_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, fino_meta.get("cont_dim", {}).get(f, 1))).to(device) for f, _ in fino_cont}
    # AdamW param groups carry per-parameter LR/WD multipliers (LWD + patch_embed + biases-no-WD).
    param_groups = build_param_groups(student_backbone, student_dino_head, student_predictor, dino_cfg["layerwise_decay"], dino_cfg["patch_embed_lr_mult"], dino_cfg.get("cls_lr_mult", 1.0))
    if predictors:
        param_groups.append({"params": [p for m in predictors.values() for p in m.parameters()], "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    opt = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, dino_cfg["adam_beta2"]))
    # FINO prototype banks: one unit vector per discrete-factor value, EMA-updated from teacher CLS in compute_losses.
    protos = {f: F.normalize(torch.randn(fino_meta["n"][f], student_backbone.embed_dim, device=device), dim=-1) for f, _ in fino_disc} if fino_cfg else {}
    # FINO grad-equalisation EMA bank (one running grad-norm per factor); init 1.0 -> s_t~1 early. Not checkpointed
    # (mu=0.99 -> ~100-step memory, re-warms quickly on resume). Used only when fino.grad_equalize is set.
    grad_eq_ema = {f: torch.ones((), device=device) for f, _ in (fino_disc + fino_cont)} if fino_cfg else {}
    step = 0
    batch_size = int(train_cfg["batch_size"])
    max_train_samples = int(train_cfg["max_train_samples"])
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
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
        opt.load_state_dict(checkpoint["opt"])
        if fino_cfg:
            protos = {k: v.to(device) for k, v in checkpoint["protos"].items()}
            for f, mdl in predictors.items():
                mdl.load_state_dict(checkpoint["predictors"][f])
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
    for key in ("probe/target_flops", "probe/wall_seconds"):
        wandb_run.define_metric(key, hidden=True, overwrite=True)
    print(
        f"{console_prefix()} Run  start: {wandb_name}  "
        f"config: {cfg['config_path']}  batch_size: {batch_size}  max_train_samples: {max_train_samples}  "
        f"max_train_flops: {train_cfg['max_train_flops']}  "
        f"probe_count: {cfg['probe']['count']}  warmup_fraction: {dino_cfg['warmup_fraction']}  "
        f"lr: {dino_cfg['lr']}  adam_beta2: {dino_cfg['adam_beta2']}  kde_loss_weight: {dino_cfg['kde_loss_weight']}  "
        f"kde_concentration: {dino_cfg['kde_concentration']}  drop_path: {dino_cfg['drop_path_rate']}  "
        f"layerwise_decay: {dino_cfg['layerwise_decay']}",
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
    train_ds = ParquetImageDataset(cfg, is_train=True)
    val_ds = ParquetImageDataset(cfg, is_train=False)
    probe_state = prepare_probe_state(cfg, output_dir) if probe_enabled(cfg) else None

    # Train shuffles + drops partials; the loop never starts a batch that would exceed
    # max_train_samples, so every optimizer step keeps the configured batch size.
    loader_kwargs = dict(batch_size=batch_size, drop_last=True, num_workers=train_cfg["num_workers"], pin_memory=True,
                         prefetch_factor=train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
                         persistent_workers=train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    local_patches = (train_cfg["local_size"] // student_backbone.patch_size) ** 2
    # Patches the student actually encodes per global view: only the contiguous context region.
    visible_global_patches = sum(count * side ** 2 for side, count in context_regions)
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    sample_patch_count = (IMAGE_SIZE // student_backbone.patch_size) ** 2
    # Whatever grouping levels the dataset adapter declares; the first is the image itself.
    coverage_keys = train_ds.coverage_keys
    primary_coverage = next(iter(coverage_keys))
    seen_ids = {name: set() for name in coverage_keys}
    pending_ids = {name: set() for name in coverage_keys}

    # cpu_state(m) materializes an on-CPU copy of a module's state_dict for torch.save.
    def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # Full checkpoint (latest.pt) covers explicit train.resume whereas probe checkpoint is a slim
    # weights-only ckpt, given probe.py does not need optimizer or projection heads.
    def checkpoint_payload(next_step, full):
        payload = {"model": cpu_state(student_backbone), "model_ema": cpu_state(teacher_backbone), "step": next_step, "config": cfg}
        if not full:
            return payload
        return {**payload, "dino_head": cpu_state(student_dino_head), "dino_head_ema": cpu_state(teacher_dino_head),
                "predictor": cpu_state(student_predictor), "opt": opt.state_dict(),
                "examples_seen": examples_seen, "visible_patch_presentations": visible_patch_presentations,
                "train_flops": train_flops, "wandb": wandb_meta,
                **({"protos": {k: v.cpu() for k, v in protos.items()}, "predictors": {f: cpu_state(m) for f, m in predictors.items()}} if fino_cfg else {})}

    def save_latest_checkpoint(checkpoint_step):
        nonlocal last_saved_step
        print(f"{console_prefix()} Checkpoint  [{checkpoint_step}]  save: latest.pt", flush=True)
        tmp_path = latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload(checkpoint_step, full=True), tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
        for stale_checkpoint_path in output_dir.glob("step_*.pt"):
            stale_checkpoint_path.unlink()
        last_saved_step = checkpoint_step

    # Data-coverage diagnostics: unique count per grouping level the adapter declares, plus the
    # patch count implied by the number of distinct images.
    def flush_unique_counts():
        for name in seen_ids:
            seen_ids[name].update(pending_ids[name])
            pending_ids[name].clear()
        counts = {name: len(seen) for name, seen in seen_ids.items()}
        return {**counts, "unique_patches_seen": counts[primary_coverage] * sample_patch_count}

    # Compute (dino_loss, jepa_loss, kde) for one batch of global crops with the given per-scale
    # context/target indices + schedule values. Used by both the train step and evaluate() (no_grad).
    #
    # Teacher: the full global view, every patch -- 2b rows, one forward, shared by every scale below.
    # Student: context regions at several SCALES, `count` of each per global view, plus `lf` local crops.
    def compute_losses(gf, lf, b, keep_idx, mask_idx, t_temp, k_scale, ckpt=False, meta=None, cond=None):
        gv, total_r, n_local = train_cfg["global_views"], total_regions, train_cfg["local_views"]
        with torch.no_grad():
            t = teacher_backbone(gf)
            # Sinkhorn centres over all gv*b teacher rows at once, per factor. Prototype count per factor
            # sets how thin that statistic is: gv*b rows spread over n_prototypes bins, not over the joint
            # n_prototypes**n_factors code space, which is the point of factorising.
            t_prob = sinkhorn(teacher_dino_head(t["x_norm_clstoken"]), t_temp).view(gv, b, n_factors, -1)
        tp = F.layer_norm(t["x_norm_patchtokens"], (student_backbone.embed_dim,))
        cls_tokens, jepa_terms = [], []
        for (_, r), ki, mi in zip(context_regions, keep_idx, mask_idx):
            sv = gf.repeat(r, 1, 1, 1)
            sg = student_backbone(jitter_view(sv) if view_jitter else sv, keep_idx=ki, checkpoint=ckpt)
            cls_tokens.append(sg["x_norm_clstoken"])
            # K is fixed and every slot is a distinct unseen patch, so this is a plain mean.
            # Gathering per region chunk keeps the teacher features unduplicated: only the (2b*r, K, D)
            # result is materialised, not r copies of the (2b, 256, D) source.
            target = torch.cat([tp.gather(1, m[..., None].expand(-1, -1, tp.shape[-1])) for m in mi.chunk(r)])
            pred = student_predictor(sg["x_norm_patchtokens"], mi, None if cond is None else cond.repeat(gv * r))
            jepa_terms.append(r * F.smooth_l1_loss(pred, target))
        cls_all = torch.cat(cls_tokens)  # (gv*b * total_r, D), scale-major
        sg_cls = student_dino_head(cls_all)

        # Every ordered cross-view pair: student view v scored against teacher view w != v, averaged over
        # the gv*(gv-1) pairs and then over regions. At gv=2 this is exactly the old swap pair.
        # log_softmax is taken once per student view and reused across the gv-1 teachers it is scored
        # against; a per-pair cross-entropy would instead save gv*(gv-1) copies of a (b, F, K) tensor
        # for backward, which at gv=4 is ~800 MB per region unfactorised, for no numerical difference.
        # log_softmax is over K alone, so each codebook is its own distribution; summing the per-factor
        # cross-entropies over F is exactly the joint CE, since the code factorises across chunks.
        def dino_cross(x):
            ls = F.log_softmax(x.view(gv, b, n_factors, -1) / 0.1, dim=-1)
            pairs = [(v, w) for v in range(gv) for w in range(gv) if v != w]
            return sum(-(t_prob[w] * ls[v]).sum((-2, -1)).mean() for v, w in pairs) / len(pairs)

        dino_loss = sum(dino_cross(x) for x in sg_cls.chunk(total_r)) / total_r
        # Local crops: every (local, teacher view) pair, no exclusion since a local never IS a teacher view.
        # The two branches are recombined by pair count, so the result stays a flat mean over all student-
        # teacher pairs -- at total_r=1 that is exactly the 1/(2L+2) multi-crop normalisation.
        # Locals feed DINO only: KDE and FINO stay on the region CLS rows they are calibrated for.
        if n_local:
            sl_cls = student_dino_head(student_backbone(lf, checkpoint=ckpt)["x_norm_clstoken"])
            ls = F.log_softmax(sl_cls.view(n_local, b, n_factors, -1) / 0.1, dim=-1)
            local_loss = sum(-(t_prob[w] * ls[v]).sum((-2, -1)).mean() for v in range(n_local) for w in range(gv)) / (n_local * gv)
            n_gpairs, n_lpairs = total_r * gv * (gv - 1), n_local * gv
            dino_loss = (n_gpairs * dino_loss + n_lpairs * local_loss) / (n_gpairs + n_lpairs)
        # Region-count weighted so the result is a plain mean over all regions regardless of how the
        # scales are split; identical to a flat mean when every scale has the same count.
        jepa_loss = sum(jepa_terms) / total_r
        # One KDE term per (region, view) group of b distinct images -- uniformity is only meaningful
        # across different images, never across regions of the same one. Averaged over regions so
        # kde_loss_weight keeps its calibrated meaning (a sum over global views) as the region count varies.
        kde = dino_cfg["kde_loss_weight"] * k_scale * sum(kde_loss(x, dino_cfg["kde_concentration"]) for x in cls_all.chunk(gv * total_r)) / total_r
        # FINO metadata guidance on the CLS token (train-only; meta=None in eval), orthogonal to the JEPA patch
        # objective. lambda_meta=0.03/branch; GradScale gates the encoder gradient by the DANN ramp gamma with the
        # per-factor sign (+ M+ encourage / - M- suppress). fp32 island (1/tau=0.023 too sharp for bf16); missing
        # factors masked. Discrete: L2-normed student CLS vs EMA prototype bank (clone-rebind keeps the backward-saved
        # bank valid). Continuous: an MLP regresses the z-scored value.
        meta_loss = cls_all.new_zeros(())
        if meta is not None:
            gamma, md, mc = meta  # md (B,n_disc) int64 (-1 missing); mc {factor: (B,dim) float, nan missing}
            phi_s = F.normalize(cls_all.float(), dim=-1)
            phi_t = F.normalize(t["x_norm_clstoken"].float(), dim=-1)
            terms = []  # (factor, per-branch loss 0.03*L_t); combined below, optionally gradient-equalized
            with torch.autocast(device_type="cuda", enabled=False):
                for j, (f, sign) in enumerate(fino_disc):
                    # phi_s has 2b*total_r rows (one per region across all scales), phi_t only 2b -- the
                    # prototype bank is updated from the teacher, so its labels tile by global_views alone.
                    lab = md[:, j].repeat(gv * total_r); ok = lab >= 0  # repeat, NOT interleave
                    lab_t = md[:, j].repeat(gv); ok_t = lab_t >= 0
                    if ok.any():
                        logits = (GradScale.apply(phi_s[ok], sign * gamma) @ protos[f].t()) / 0.023
                        terms.append((f, 0.03 * F.cross_entropy(logits, lab[ok])))
                        with torch.no_grad():
                            pt, lt = phi_t[ok_t], lab_t[ok_t]
                            upd = torch.zeros_like(protos[f]).index_add_(0, lt, pt)
                            cnt = torch.zeros(protos[f].shape[0], 1, device=device).index_add_(0, lt, torch.ones_like(pt[:, :1]))
                            seen = cnt.squeeze(1) > 0; new = protos[f].clone()
                            new[seen] = F.normalize(0.99 * new[seen] + 0.01 * (upd[seen] / cnt[seen]), dim=-1); protos[f] = new
                # FINO Eq.3 regresses continuous factors from the RAW backbone CLS; phi_s is L2-normalized (needed only
                # for the cosine discrete branch and it strips the radial magnitude). raw_cls=True feeds the raw CLS.
                cls_cont = cls_all.float() if fino_cfg.get("raw_cls") else phi_s
                for f, sign in fino_cont:
                    val = mc[f].repeat(gv * total_r, 1); ok = ~torch.isnan(val).any(dim=1)
                    if ok.any():
                        cpred = predictors[f](GradScale.apply(cls_cont[ok], sign * gamma))
                        terms.append((f, 0.03 * F.mse_loss(cpred, val[ok])))
                # FINO Alg A.3 per-branch gradient equalisation: rescale each branch by n_bar/EMA(||dL_t/dCLS||) so the
                # discrete-CE and continuous-MSE gradients reach the encoder at matched magnitudes (detached -> reweight
                # only; geometric-mean target; no-op for <2 branches). grad_eq_ema = per-factor EMA bank (mu=0.99).
                if fino_cfg.get("grad_equalize") and len(terms) > 1:
                    g = {f: torch.autograd.grad(L, cls_all, retain_graph=True)[0].norm() for f, L in terms}
                    for f in g: grad_eq_ema[f] = 0.99 * grad_eq_ema[f] + 0.01 * g[f].detach().float()
                    nbar = torch.exp(torch.stack([grad_eq_ema[f].log() for f, _ in terms]).mean())
                    meta_loss = sum((nbar / grad_eq_ema[f]).detach() * L for f, L in terms)
                else:
                    for _, L in terms: meta_loss = meta_loss + L
        # Penalises the prototype parameter, not the data, so it is returned separately and evaluate()
        # leaves it out of the val totals.
        return dino_loss, jepa_loss, kde, meta_loss, student_dino_head.prototype_reg()

    # Held-out validation pass: same DINO + JEPA + KDE losses on `val_batches` of the val split.
    # Schedule terms (teacher_temp, kde_scale, jepa decay) drift over training, so read val curves as
    # same-step diagnostics. RNG is snapshotted/restored so val masks don't perturb the next training step.
    def evaluate(eval_step, eval_teacher_temp, eval_kde_scale, eval_decay):
        for m in (student_backbone, student_dino_head, student_predictor):
            m.eval()
        py_rng, cpu_rng, cuda_rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        random.seed(train_cfg["seed"] + eval_step)
        torch.manual_seed(train_cfg["seed"] + eval_step)
        sums = torch.zeros(4, device=device)
        n_batches = 0
        for vb_idx, vbatch in enumerate(val_loader):
            if vb_idx >= int(train_cfg["val_batches"]):
                break
            vg = vbatch["global_views"].to(device, non_blocking=True)
            vl = vbatch["local_views"].to(device, non_blocking=True)
            b = vg.shape[0]
            with torch.no_grad(), autocast:
                gf, lf = vg.transpose(0, 1).flatten(0, 1), vl.transpose(0, 1).flatten(0, 1)
                keep_idx, mask_idx = make_region_indices(b * train_cfg["global_views"], eval_decay)
                dino_l, jepa_l, kde_v, _, _ = compute_losses(gf, lf, b, keep_idx, mask_idx, eval_teacher_temp, eval_kde_scale)
            sums += torch.tensor([float(dino_l), float(jepa_l), float(kde_v), float(dino_l + jepa_l + kde_v)], device=device)
            n_batches += 1
        random.setstate(py_rng)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        return dict(zip(("dino", "jepa", "kde", "total"), (sums / max(1, n_batches)).tolist()))

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
    # Probe targets are sample milestones: one image counts once even with many global/local crops.
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
    # Per-step FLOPs are measured once via FlopCounterMode on the first wrapped step (forward +
    # backward + opt.step) and reused for every subsequent step since the shapes don't change.
    # Counts the EMA teacher forward + DINO/JEPA heads, not just the backbone, so the
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
            for name, batch_key in coverage_keys.items():
                pending_ids[name].update(int(x) for x in batch[batch_key].tolist())
            global_views = batch["global_views"].to(device, non_blocking=True)
            local_views = batch["local_views"].to(device, non_blocking=True)
            visible_now = batch_size * (train_cfg["global_views"] * visible_global_patches + train_cfg["local_views"] * local_patches)
            # LR warmup uses the 1M-sample cap; decay/WD/teacher/freeze/KDE default to the public FLOP budget.
            # But this run hits the sample cap at ~19% of the FLOP budget, so a FLOP-keyed cosine only traverses ~0.11
            # of its arc (LR never anneals, KDE peaks at 0.22, WD ~0.05). lr_key/reg_key="sample" re-key the decay/reg
            # schedules to SAMPLE progress so they complete over the actual 1M-sample run (same fix as the FINO gamma ramp).
            frac = min(1.0, train_flops / max_train_flops)
            sfrac = min(1.0, examples_seen / max_train_samples)
            lr_frac = sfrac if dino_cfg.get("lr_key") == "sample" else frac
            reg_frac = sfrac if dino_cfg.get("reg_key") == "sample" else frac
            warmup = min(1.0, examples_seen / max(1, warmup_train_samples))
            if warmup < 1.0:
                lr = dino_cfg["lr"] * warmup
            else:
                lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], (lr_frac - dino_cfg["warmup_fraction"]) / max(1e-9, 1 - dino_cfg["warmup_fraction"]))
            wd = cosine_schedule(0.04, 0.2, reg_frac)
            teacher_temp = 0.04 + min(1.0, reg_frac / 0.2727) * (0.07 - 0.04)
            last_layer_lr = 0.0 if frac < dino_cfg["freeze_last_layer_fraction"] else lr
            for group in opt.param_groups:
                base_lr = last_layer_lr if group["last_layer"] else lr
                group["lr"] = base_lr * group["lr_mult"]
                group["weight_decay"] = wd * group["wd_mult"]
            jepa_decay = cosine_schedule(jepa_decay_start, jepa_decay_end, reg_frac)
            keep_idx, mask_idx = make_region_indices(batch_size * train_cfg["global_views"], jepa_decay)
            kde_scale = min(1.0, max(0.0, (reg_frac - 0.1) / 0.4))
            # Wrap forward + backward + opt.step in FlopCounterMode on the first step only;
            # subsequent steps reuse measured_flops_per_step (fixed shapes => fixed cost).
            flop_ctx = FlopCounterMode(display=False) if measured_flops_per_step is None else contextlib.nullcontext()
            with flop_ctx:
                with autocast:
                    # Crop-major flatten: collate shape is (B, V, 3, H, W) but DINO wants per-crop chunks
                    # so [crop0_img0, crop0_img1, ..., crop1_img0, ...] for clean teacher/student alignment.
                    gf, lf = global_views.transpose(0, 1).flatten(0, 1), local_views.transpose(0, 1).flatten(0, 1)
                    # FINO DANN ramp keyed to nanopath's SAMPLE budget (NOT FLOPs — sample-capped at ~19% of the FLOP
                    # cap, so a flop-keyed ramp stalls gamma at ~0.75*gamma_max). Counted from the backbone-unfreeze
                    # point: gamma=0 through the frozen Phase 1 (banks warm), then ramps to full gamma_max by the cap.
                    ramp = max(0.0, (examples_seen / max_train_samples - freeze_backbone_frac) / max(1e-6, 1.0 - freeze_backbone_frac))
                    meta = ((fino_cfg["gamma_max"] * (2.0 / (1.0 + math.exp(-10.0 * ramp)) - 1.0),
                             batch["meta_disc"].to(device, non_blocking=True),
                             {f: batch["mc_" + f].to(device, non_blocking=True) for f, _ in fino_cont}) if fino_cfg else None)
                    cond = batch["meta_disc"][:, cond_col].to(device, non_blocking=True) if jepa_cond else None
                    dino_loss_value, jepa_loss, kde, meta_loss, proto_reg = compute_losses(
                        gf, lf, batch_size, keep_idx, mask_idx, teacher_temp, kde_scale,
                        ckpt=activation_checkpointing, meta=meta, cond=cond,
                    )
                    total_loss = dino_loss_value + jepa_loss + kde + meta_loss + proto_reg
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if examples_seen / max_train_samples < freeze_backbone_frac:  # Phase 1: backbone frozen (patch_embed + heads + metadata still train)
                    for n, p in student_backbone.named_parameters():
                        if not n.startswith("patch_embed"): p.grad = None
                grad_norm = nn.utils.clip_grad_norm_(
                    [*student_backbone.parameters(), *student_dino_head.parameters(), *student_predictor.parameters()],
                    dino_cfg["clip_grad"],
                )
                opt.step()
            if measured_flops_per_step is None:
                measured_flops_per_step = int(flop_ctx.get_total_flops())
                print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                m = cosine_schedule(0.994, 1.0, reg_frac)
                update_ema(student_backbone, teacher_backbone, m)
                update_ema(student_dino_head, teacher_dino_head, m)
            step_seconds = time.monotonic() - batch_started_at
            examples_seen += batch_size
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            if should_log:
                reduced = {
                    "dino": float(dino_loss_value.detach()),
                    "jepa": float(jepa_loss.detach()),
                    "kde": float(kde.detach()),
                    "total": float(total_loss.detach()),
                }
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
                    "prototype_reg_loss": float(proto_reg),
                    "jepa_target_decay": jepa_decay,
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
                    f"dino: {reduced['dino']:.4f}  jepa: {reduced['jepa']:.4f}  kde: {reduced['kde']:.4f}  "
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
                val = evaluate(completed_step, teacher_temp, kde_scale, jepa_decay)
                val_log = {"step": completed_step, **{f"val_{k}": v for k, v in val.items()}}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(val_log) + "\n")
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=completed_step)
                print(f"{console_prefix()} Validation  [{completed_step}]  total: {val['total']:.4f}  dino: {val['dino']:.4f}  jepa: {val['jepa']:.4f}  kde: {val['kde']:.4f}", flush=True)
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
        if train_cfg["num_workers"] > 0:
            if train_loader._iterator is not None:
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
        "wandb": wandb_meta,
        "slurm_job_id": slurm_job_id,
        "backbone_activated_params": backbone_activated_params,
        "batch_size": batch_size,
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
        "lr": dino_cfg["lr"],
        "adam_beta2": dino_cfg["adam_beta2"],
        "kde_loss_weight": dino_cfg["kde_loss_weight"],
        "kde_concentration": dino_cfg["kde_concentration"],
        "head_factors": n_factors,
        "head_prototypes": n_prototypes,
        "prototype_reg": prototype_reg,
        "prototype_reg_weight": float(dino_cfg.get("prototype_reg_weight", 0.0)),
        "jepa_targets": jepa_targets,
        "jepa_target_decay": jepa_decay_start,
        "jepa_target_decay_end": jepa_decay_end,
        "drop_path_rate": dino_cfg["drop_path_rate"],
        "layerwise_decay": dino_cfg["layerwise_decay"],
        "probe_target_samples": probe_targets,
        "probe_target_fractions": [None if max_train_samples == 0 else target / max_train_samples for target in probe_targets],
        **({} if probe_state is None else completed_probe_summary(output_dir)),
    }
    if probe_state is not None and "final_probe_score" not in summary:
        raise ValueError("probe.enabled is true but final_probe_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{console_prefix()} Summary  "
        f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
        f"final_probe_score: {summary.get('final_probe_score')}",
        flush=True,
    )
    for key in summary.keys():
        wandb_run.summary[key] = summary[key]
    wandb_run.finish()
    finish_labless_autosubmit(labless_autosubmit_file, output_dir, repo_dir)


if __name__ == "__main__":
    main()
