#!/usr/bin/env bash
#SBATCH --job-name=nb-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=normal

cd /admin/home/raymond.biju/repos/nanopath
source .venv/bin/activate
PORT=$((8800 + SLURM_JOB_ID % 200))
TOKEN=$(openssl rand -hex 16)
echo "JUPYTER_URL=http://$(hostname):${PORT}/?token=${TOKEN}"
jupyter server --no-browser --ip=0.0.0.0 --port="$PORT" --IdentityProvider.token="$TOKEN" --ServerApp.root_dir="$PWD"
