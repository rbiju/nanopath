# labless integration

This folder contains the nanopath-to-labless bridge. The goal is simple: after
you train a model, one command publishes the run to the public nanopath tracker.

```bash
RUN_DIR=/data/$USER/nanopath/main/my-run
./labless/submit_to_labless.py output_dir=$RUN_DIR \
    run_name=kde-crops \
    notes="what changed and why"
```

## What the submit script does

`submit_to_labless.py` should be run from the nanopath repo root after
`train.py` finishes. It:

1. Reads `summary.json` and `metrics.jsonl` from `output_dir`.
2. Extracts the final `mean_probe_score` and probe submetrics.
3. Uses the local `output_dir/labless_source` snapshot written by `train.py` and
   diffs that source against the current main commit for `train.py`, `model.py`,
   `dataloader.py`, `prepare.py`, and the config YAML used by the run.
4. Records hardware, Python version, optional W&B run link, and the full changed
   path list from the saved source snapshot.
5. Writes the submission payload to `output_dir/labless_submission.json`.
6. Opens GitHub's device sign-in flow and posts it to
   `https://api.labless.dev/api/nano-projects/nanopath/submissions` with the
   resulting bearer token.

The labless backend stores the submission as a run with saved source context and
an optional W&B run link. It derives the public contributor from the verified
GitHub login, rejects scoped OAuth tokens, and accepts at most 10 submissions per
login per 24 hours. The website fetches the API data and the SVG plot from
`api.labless.dev`, so the run appears in the project log, run table, and plot
without opening a pull request.

## Submit a completed run

Run training first:

```bash
RUN_DIR=/data/$USER/nanopath/main/my-run
sbatch submit/train_1gpu.sbatch configs/main.yaml output_dir=$RUN_DIR
# or directly on a GPU machine:
python train.py configs/main.yaml output_dir=$RUN_DIR
```

Then point the submit script at the same run directory:

```bash
./labless/submit_to_labless.py \
    output_dir=$RUN_DIR \
    run_name=kde-crops \
    notes="changed the crop schedule and kept all probe paths untouched"
```

Completed submissions require both `summary.json` and `metrics.jsonl`. The run
is shown as `pending` until the organizer validates it. A copied config such as
`configs/new_config.yaml` is accepted if the completed `summary.json` reports
`max_train_samples: 1000000`, `tile_presentations <= 1000000`, and
`max_train_flops: 1e18`; short local configs are rejected even if they are not named smoke.
Use the same config you prepared and trained with; off the MedARC cluster, copy
the config and point its data paths at writable local storage before training.
Smoke runs are local setup checks only and are not accepted by labless.

## Submit a baseline/reference run

Tracked reference baseline scripts write the same `summary.json` and
`metrics.jsonl` files as `train.py`, so they can be submitted the same way:

```bash
python baselines/dinov2_small_baseline.py configs/main.yaml
./labless/submit_to_labless.py \
  output_dir=/data/$USER/nanopath/baselines/dinov2-small \
  notes="reran the frozen DINOv2-small reference"
```

The submit script detects `summary.family == "baseline"` and marks the run as
`tier=baseline`. Labless currently tracks GenBio-PathFM plus DINOv2 giant and
small references; other nanopath baselines, including the separate Virchow and
GigaPath scripts, can stay in the repo README without becoming Labless reference
rows. The nanopath leaderboard still ranks validated completed full runs by score.

## Useful options

Arguments are `key=value`; there is no `argparse`.

| key | use |
|---|---|
| `output_dir` | Required run directory. |
| `run_name` | Short plot label, 20 characters or fewer. |
| `notes` | Short explanation of what changed and why. |
| `wandb_url` | Optional W&B run URL for linking the external dashboard; private or unlisted W&B URLs are accepted because labless only validates URL shape. |
| `tier` | `full` or `baseline`; inferred when omitted. |
| `hardware` | Override detected hardware string. |
| `source_dir` / `source_commit` | Manual repair knobs for copied or older runs; normally inferred from `output_dir/labless_source` and `summary.json`. |
| `dry_run=true` | Write `labless_submission.json` without posting. |
| `api_url` | Use a local labless backend for testing. |
| `main_run_id` / `main_commit` | Local testing override for the live main lookup; both are required when either is set. |

For real submissions, the script prompts you to open
`https://github.com/login/device` and enter a short code. Dry runs write the
payload locally without signing in or posting.

## Validation rules

The benchmark score is only meaningful when evaluation stays fixed. The script
marks submissions invalid if the saved source snapshot changed:

- `probe.py`
- anything under `benchmarking/`

The config YAML may only change train/model/data tunables and local
`probe.dataset_roots`. Labless rejects changes to the locked probe suite keys
such as dataset lists, probe count, and model weights. Helper code changes must
stay inside the flat reviewed surface (`train.py`, `model.py`, `dataloader.py`,
`prepare.py`, or `configs/*.yaml`); hidden helper modules such as `losses.py`
are rejected.

The current checkout can change after training; labless uses the source snapshot
from the run, not the present working tree, when building the review diff. W&B
may be online or offline because source review never depends on the W&B API.

## What becomes public

The payload intentionally makes the run inspectable. It includes:

- verified GitHub login and notes
- final metric and probe submetrics
- run family, recipe id, and tier (`baseline` for frozen reference scripts)
- source snapshot id, optional git remote, commit, full changed source path list,
  changed review files, and a capped review-file snapshot for server-built diffs
- hardware, Python version, and optional GPU summary
- W&B run URL

The public API redacts local machine paths, hostnames, users, repo roots, and
local artifact paths from legacy and new rows.

The review snapshot is only collected for `train.py`, `model.py`,
`dataloader.py`, `prepare.py`, and the config YAML used by the run. Labless
builds capped patches server-side when it compares two logged snapshots. Binary
or large-file suffixes are omitted from patches and listed in the payload. Local
hostnames, users, working directories, repo roots, artifact paths, model weights,
and raw data are not posted.

## Maintainer validation

New completed full runs appear on the plot as `pending`. A maintainer can
replicate a promising run, then mark it `validated` in labless. The public
leader label is the highest scoring validated run. Maintainers mark a separate
`main` state with the full git commit pushed to the project repo, so the submit
script can diff the saved source snapshot directly against current main.
Failed runs are not accepted as public submissions.
