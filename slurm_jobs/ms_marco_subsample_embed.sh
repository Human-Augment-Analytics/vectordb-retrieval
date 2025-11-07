#!/bin/bash
#SBATCH -J MSMARCO_embed                # Job name  
#SBATCH -N1 --gres=gpu:V100:1   
#SBATCH --cpus-per-task=4                   # Number of CPUs per task
#SBATCH -t 0:60:00                          # Job duration (hh:mm:ss) - Reduced for sampling, adjust if needed
#SBATCH --output=slurm_jobs/slurm_logs/%x-%j.log   # Output and error messages file (includes job ID)
#SBATCH --mail-type=BEGIN,END,FAIL          # When to send email notifications
#SBATCH --mail-user=apramov3@gatech.edu     

set -xeu # Exit on error, print commands

# --- Configuration ---
SHARED_DRIVE_PATH="/storage/ice-shared/cs8903onl/vectordb-retrieval"
PROJECT_DIR="/home/hice1/apramov3/vectordb-retrieval/"
VENV_PATH=".venv/bin/activate"  
LOG_DIR="${PROJECT_DIR%/}/slurm_jobs/slurm_logs"
mkdir -p "$LOG_DIR"

# --- Environment Setup ---
echo "Setting up environment..."
echo "Changing to project directory: ${PROJECT_DIR}"
cd "$PROJECT_DIR" || { echo "Error: Could not change to project directory ${PROJECT_DIR}"; exit 1; }

# Activate your Python virtual environment
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Virtual environment activated."
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

echo "Running the embedding script..."
python src/dataprep/embed_msmarco.py

echo "Script finished."

 
