#!/bin/bash
#SBATCH -J Codex-Covertree-All              # Job name
#SBATCH -N 1 --ntasks-per-node=1            # Number of nodes
#SBATCH --cpus-per-task=16                  # CPU cores
#SBATCH --mem=64G                           # Total memory
#SBATCH -t 24:00:00                         # Max runtime (MSMARCO phase exceeded 12h)
#SBATCH --output=slurm_jobs/slurm_logs/%x-%j-%N.log  # Log file
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail preferences
set -xeu

###########################################################
#       USER-CONFIGURABLE SECTION                         #
###########################################################
# Repository directory.
REPO_DIR="$HOME/PycharmProjects/vectordb-retrieval"
# Virtual environment directory.
VENV_DIR="$HOME/scratch/vector-db-venv"
LOG_DIR="$REPO_DIR/slurm_jobs/slurm_logs"
mkdir -p "$LOG_DIR"
###########################################################

# Ensure uv is available.
if ! command -v uv &> /dev/null; then
    echo "uv command not found. Installing with curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Validate repository path.
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory '$REPO_DIR' not found."
    exit 1
fi

cd "$REPO_DIR"

if [ ! -f "requirements.txt" ]; then
    echo "Error: 'requirements.txt' not found in '$REPO_DIR'."
    exit 1
fi

echo "Creating uv environment in '$VENV_DIR'..."
uv venv "$VENV_DIR"

echo "Sourcing the new environment..."
source "$VENV_DIR/bin/activate"

echo "Installing requirements from requirements.txt..."
uv pip install -r requirements.txt

echo "Environment ready. Running benchmark..."
python scripts/run_full_benchmark.py --config configs/benchmark_all_covertree.yaml
