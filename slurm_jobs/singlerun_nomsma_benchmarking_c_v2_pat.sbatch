#!/bin/bash
#SBATCH -J VectorDB-Retrieval-Guarantee_FULL        # Job name
#SBATCH -N 1 --ntasks-per-node=1                # Number of nodes
#SBATCH --cpus-per-task=24                       # Number of CPU
#SBATCH --mem-per-cpu=8G                        # Memory per CPU
#SBATCH -t 15:00:00                              # Duration
#SBATCH --output=slurm_jobs/slurm_logs/%x-%j-%N.log  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
set -xeu


# A script to set up a uv virtual environment and install dependencies.
# This script is designed for environments where uv is installed via a curl command.


###########################################################
#       USER-CONFIGURABLE SECTION                         #
###########################################################
# Specify the path to your repository directory.
REPO_DIR="$HOME/PycharmProjects/vectordb-retrieval"
# -# Specify the name for the virtual environment directory.
VENV_DIR="$HOME/scratch/vector-db-venv"
LOG_DIR="$REPO_DIR/slurm_jobs/slurm_logs"
mkdir -p "$LOG_DIR"
###########################################################

# Check if uv is installed, and install it if not using the curl method.
if ! command -v uv &> /dev/null
then
    echo "uv command not found. Installing with curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add the installation directory to the PATH for the current session.
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check if the repository directory exists.
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory '$REPO_DIR' not found."
    exit 1
fi

# Navigate into the repository directory.
cd "$REPO_DIR"

# Check if requirements.txt exists.
if [ ! -f "requirements.txt" ]; then
    echo "Error: 'requirements.txt' not found in '$REPO_DIR'."
    exit 1
fi

# Create a uv virtual environment with the specified name.
echo "Creating uv environment in '$VENV_DIR'..."
uv venv "$VENV_DIR"

# Source the newly created virtual environment.
echo "Sourcing the new environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies from requirements.txt.
echo "Installing requirements from requirements.txt..."
uv pip install -r requirements.txt

echo "Setup complete. The virtual environment is now active."
echo "To activate this environment in the future, run: source $VENV_DIR/bin/activate"

cd $REPO_DIR

# Run benchmarking setup
python scripts/run_full_benchmark.py --config configs/benchmark_nomsma_c_v2.yaml
