#!/bin/bash
#SBATCH -J Codex-All-MSM100K-ReuseCT
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH -t 18:00:00
#SBATCH --output=slurm_jobs/slurm_logs/%x-%j-%N.log
#SBATCH --mail-type=BEGIN,END,FAIL
set -xeu

REPO_DIR="$HOME/PycharmProjects/vectordb-retrieval"
VENV_DIR="$HOME/scratch/vector-db-venv"
LOG_DIR="$REPO_DIR/slurm_jobs/slurm_logs"
mkdir -p "$LOG_DIR"

if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory '$REPO_DIR' not found."
    exit 1
fi

SETUP_SCRIPT="$REPO_DIR/scripts/setup_uv_env.sh"
if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "Error: setup script '$SETUP_SCRIPT' not found."
    exit 1
fi

bash "$SETUP_SCRIPT" "$REPO_DIR" "$VENV_DIR" "requirements.txt"

cd "$REPO_DIR"
"$VENV_DIR/bin/python" -u scripts/run_full_benchmark.py --config configs/benchmark_all_datasets_msm100k_covertree_reuse.yaml
