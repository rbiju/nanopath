#!/usr/bin/env bash
#SBATCH --job-name=nb-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --partition=c
#SBATCH --account=training
#SBATCH --qos=normal

cd /admin/home/raymond.biju/repos/nanopath
source .venv/bin/activate
PORT=$((8800 + SLURM_JOB_ID % 200))
TOKEN=$(openssl rand -hex 16)
echo "JUPYTER_URL=http://$(hostname):${PORT}/?token=${TOKEN}"
jupyter server --no-browser --ip=0.0.0.0 --port="$PORT" --IdentityProvider.token="$TOKEN" --ServerApp.root_dir="$PWD"
