#!/bin/bash
#SBATCH -J VectorDB-Retrieval-Guarantee_smoke
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output=slurm_jobs/slurm_logs/%x-%j-%N.log

set -xeuo pipefail

REPO_DIR="$HOME/PycharmProjects/vectordb-retrieval"
VENV_DIR="$HOME/scratch/vector-db-venv"
LOG_DIR="$REPO_DIR/slurm_jobs/slurm_logs"
mkdir -p "$LOG_DIR"

cd "$REPO_DIR"

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt

python scripts/run_full_benchmark.py --config configs/covertree_v2_2_smoke.yaml
