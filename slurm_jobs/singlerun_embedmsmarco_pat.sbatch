#!/bin/bash
#SBATCH -J MSMARCO_embed                # Job name
#SBATCH -N1 --gres=gpu:V100:1
#SBATCH --cpus-per-task=4                   # Number of CPUs per task
#SBATCH -t 0:60:00                          # Job duration (hh:mm:ss) - Reduced for sampling, adjust if needed
#SBATCH --output=$HOME/PycharmProjects/vectordb-retrieval/slurm_jobs/slurm_logs/%x-%j.log   # Output and error messages file (includes job ID)
#SBATCH --mail-type=BEGIN,END,FAIL          # When to send email notifications
set -xeu

LOG_DIR="$HOME/PycharmProjects/vectordb-retrieval/slurm_jobs/slurm_logs"
mkdir -p "$LOG_DIR"


# A script to set up a uv virtual environment and install dependencies.
# This script is designed for environments where uv is installed via a curl command.


###########################################################
#       USER-CONFIGURABLE SECTION                         #
###########################################################
# Specify the path to your repository directory.
REPO_DIR="$HOME/PycharmProjects/vectordb-retrieval"
# -# Specify the name for the virtual environment directory.
VENV_DIR="$HOME/scratch/vector-db-venv"
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
python src/dataprep/embed_msmarco.py
