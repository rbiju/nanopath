# Run one randomized DINOv2-small null eval for a single benchmark probe.
# Usage: CUDA_VISIBLE_DEVICES=0 python benchmarking/eval_randomized_dinov2_small_null.py break_his 1000 out.json

from pathlib import Path
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe
from model import DinoV2ViT


dataset, seed, out_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
cfg = yaml.safe_load(os.path.expandvars((Path(__file__).resolve().parents[1] / "configs/main.yaml").read_text()))
cfg["project"]["output_dir"] = str(out_path.parent / f"work_seed_{seed}")
cfg["project"]["family"] = "null_check"
cfg["model"]["type"] = "dinov2_vits14_reg"
cfg["probe"]["model_weights"] = "ema"
cfg["probe"]["transform_policy"] = "resize_crop_224"
groups = {k: [] for k in probe.TASK_FIELDS}
for request_key, (_, supported) in probe.TASK_FIELDS.items():
    if dataset in supported:
        groups[request_key] = [dataset]
cfg["probe"].update({cfg_key: groups[request_key] for request_key, (cfg_key, _) in probe.TASK_FIELDS.items()})
out_path.parent.mkdir(parents=True, exist_ok=True)
work = Path(cfg["project"]["output_dir"])
work.mkdir(parents=True, exist_ok=True)

torch.manual_seed(seed)
model = DinoV2ViT(variant="dinov2_vits14_reg")
for p in (model.cls_token, model.register_tokens, model.pos_embed, model.mask_token):
    torch.nn.init.trunc_normal_(p, std=0.02)
weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
checkpoint_path = work / "checkpoint.pt"
request_path = work / "request.json"
result_path = work / "result.json"
torch.save({"model": weights, "model_ema": weights, "step": 0, "config": cfg}, checkpoint_path)
request_path.write_text(json.dumps({"checkpoint_step": 0, "train_step": seed, "target_flops": 0, "target_fraction": 1.0, "checkpoint_path": str(checkpoint_path), "request_path": str(request_path), "result_path": str(result_path), "job_id": f"random-dinov2-small-{dataset}-{seed}", **groups}, indent=2) + "\n")

probe.MEAN_PROBE_DATASETS[:] = [dataset]
started = time.monotonic()
probe.run_probe_job(request_path)
result = json.loads(result_path.read_text())
out_path.write_text(json.dumps({"dataset": dataset, "seed": seed, "score": result["metrics"][f"probe_{dataset}_score"], "wall_seconds": time.monotonic() - started, "metrics": result["metrics"]}, indent=2) + "\n")
checkpoint_path.unlink()
request_path.unlink()
result_path.unlink()
work.rmdir()
