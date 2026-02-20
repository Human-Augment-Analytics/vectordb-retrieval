#!/bin/bash
#SBATCH -J Codex-NomSMA-CTv2_2                 # Job name
#SBATCH -N 1 --ntasks-per-node=1               # Number of nodes
#SBATCH --cpus-per-task=24                     # Number of CPU
#SBATCH --mem-per-cpu=16G                      # Memory per CPU
#SBATCH -t 15:00:00                            # Duration
#SBATCH --output=slurm_jobs/slurm_logs/%x-%j-%N.log  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail preferences
set -xeu


# SLURM launcher that delegates uv environment setup to scripts/setup_uv_env.sh
# and then runs the benchmark with the venv Python interpreter.


###########################################################
#       USER-CONFIGURABLE SECTION                         #
###########################################################
REPO_DIR="$HOME/PycharmProjects/vectordb-retrieval"
VENV_DIR="$HOME/scratch/vector-db-venv"
LOG_DIR="$REPO_DIR/slurm_jobs/slurm_logs"
mkdir -p "$LOG_DIR"
###########################################################

# Check if the repository directory exists.
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

# Run benchmark using the requested non-MSMARCO covertree config.
# Use unbuffered output so SLURM logs stream progress promptly.
"$VENV_DIR/bin/python" -u scripts/run_full_benchmark.py --config configs/benchmark_nomsma_covertree_v2_2.yaml
