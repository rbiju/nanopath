# Run the full frozen-probe suite on the untouched EXAONE-Path-2.5 patch encoder.
# Defaults to the MedARC cluster checkpoint path; pass checkpoint_path=/path off-cluster.
# The local repo at checkpoint_path is expected to ship LG AI Research's `exaonepath/`
# package so we can build the ViT-B/14 backbone without depending on transformers.

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch.nn as nn
import yaml
from safetensors.torch import load_file

from probe import TASK_FIELDS, completed_probe_summary, prepare_probe_state


# Probe contract: `forward` returns DINOv2-style x_norm_* dict, `encode_image` and
# `probe_features` mirror DinoV2ViT. EXAONE has no register tokens.
class ExaonePathPatchEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit, self.registers = vit, 0

    def forward(self, x):
        # get_intermediate_layers(n=1) returns the final-LayerNorm [B, 1+256, 768] tokens, which
        # is exactly what EXAONE's own model(x) emits as the cls feature — we feed them raw, like
        # every other baseline (the survival probe z-scores features itself, so no L2-norm needed).
        seq = self.vit.get_intermediate_layers(x, n=1)[0]
        return {"x_norm_clstoken": seq[:, 0], "x_norm_patchtokens": seq[:, 1:]}

    def encode_image(self, x):
        return self(x)["x_norm_patchtokens"]

    def probe_features(self, x):
        return self(x)["x_norm_clstoken"]


def load_probe_model(checkpoint_path, device):
    # Build vit_base via the upstream `exaonepath/` package shipped in the repo dir, then
    # strict-load patch-encoder/model.safetensors (keys are `backbone.*` from the PatchEncoder wrapper).
    repo = Path(checkpoint_path)
    sys.path.insert(0, str(repo))
    from exaonepath.models.patch_transformer import vit_base
    backbone = vit_base(patch_size=14, img_size=[224])
    state = load_file(str(repo / "patch-encoder" / "model.safetensors"))
    backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in state.items()}, strict=True)
    return ExaonePathPatchEncoder(backbone).to(device).eval()


def main():
    usage = "usage: python baselines/exaone_path_25_baseline.py [config.yaml] [checkpoint_path=/path] [output_dir=/path]"
    config_path = REPO_DIR / "configs" / "main.yaml"
    checkpoint_path = Path("/data/exaone_path_2.5")
    output_dir = Path(os.path.expandvars("/data/$USER/nanopath/baselines/exaone-path-2.5"))
    for arg in sys.argv[1:]:
        if arg.endswith((".yaml", ".yml")):
            config_path = Path(arg)
        else:
            key, _, value = arg.partition("=")
            if key == "checkpoint_path":
                checkpoint_path = Path(os.path.expandvars(value))
            elif key == "output_dir":
                output_dir = Path(os.path.expandvars(value))
            else:
                raise SystemExit(usage)
    print(f"checkpoint_path={checkpoint_path} (override with checkpoint_path=/path if not using MedARC defaults)", flush=True)

    cfg = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    cfg["config_path"] = str(config_path.resolve())
    cfg["project"]["name"] = "baseline-exaone-path-2.5"
    cfg["project"]["family"] = "baseline"
    cfg["project"]["recipe_id"] = "exaone-path-2.5-vitb14-untouched"
    cfg["project"]["output_dir"] = str(output_dir)
    # EXAONE-Path-2.5 trained with ImageNet normalization (see exaonepath/feature_extraction).
    cfg["data"]["mean"] = [0.485, 0.456, 0.406]
    cfg["data"]["std"] = [0.229, 0.224, 0.225]
    cfg["model"]["type"] = "exaone_path_2.5"
    cfg["probe"]["enabled"] = True
    cfg["probe"]["model_weights"] = "ema"
    cfg["probe"]["count"] = 1
    cfg["probe"]["model_loader"] = "baselines.exaone_path_25_baseline:load_probe_model"
    cfg["probe"]["transform_policy"] = "resize_crop_224"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started_at = time.monotonic()
    state = prepare_probe_state(cfg, output_dir)
    request = {
        "checkpoint_step": 0,
        "train_step": 0,
        "target_flops": 0,
        "target_fraction": 1.0,
        "checkpoint_path": str(checkpoint_path),
        "request_path": str(state["paths"]["probe_dir"] / "step_0000000.request.json"),
        "result_path": str(state["paths"]["results_dir"] / "step_0000000.json"),
        "job_id": f"{os.environ.get('SLURM_JOB_ID', 'local')}-exaone-path-2.5",
        "config": cfg,
        **{key: list(state["data"][key]) for key in TASK_FIELDS},
    }
    Path(request["request_path"]).write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    env.pop("WANDB_SERVICE", None)
    env["PYTHONPATH"] = str(REPO_DIR)
    subprocess.run([sys.executable, str(REPO_DIR / "probe.py"), request["request_path"]], env=env, check=True)

    result = json.loads(Path(request["result_path"]).read_text())
    event = {
        "event": "probe",
        "step": 0,
        "target_flops": 0,
        "target_fraction": 1.0,
        "probe_wall_seconds": float(result["wall_seconds"]),
        **{key: float(value) for key, value in result["metrics"].items()},
    }
    (output_dir / "metrics.jsonl").write_text(json.dumps(event) + "\n")
    summary = {
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "checkpoint_path": str(checkpoint_path),
        "backbone_activated_params": 85_706_496,  # ViT-B/14 (sum(p.numel() for p in vit_base().parameters()))
        "steps_completed": 0,
        "train_flops": 0,
        "total_wall_seconds": time.monotonic() - started_at,
        **completed_probe_summary(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"baseline metrics: {output_dir / 'metrics.jsonl'}")
    print(f"mean_probe_score: {event['mean_probe_score']:.6f}")


if __name__ == "__main__":
    main()
