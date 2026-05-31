#!/usr/bin/env bash
#SBATCH --job-name=main_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --partition=n
#SBATCH --qos=normal # low | normal | top

set -euo pipefail

# Single-GPU SLURM launcher. Optional first arg overrides the YAML config
# (defaults to configs/main.yaml). Any extra args after that are forwarded
# to train.py as overrides, e.g.
#   sbatch submit/train_1gpu.sbatch configs/main.yaml output_dir=/data/$USER/nanopath/main/myrun
# Logs land in repo-local slurm/<jobid>.{out,err}.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/main.yaml}"
[[ $# -gt 0 ]] && shift

cd "${REPO_DIR}"
# Worktrees share the venv from the main checkout at /admin/home/$USER/nanopath.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"

python train.py "${CONFIG_FILE}" "$@"
