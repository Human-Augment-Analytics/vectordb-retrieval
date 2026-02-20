#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <repo_dir> <venv_dir> [requirements_file]"
    exit 1
fi

REPO_DIR="$1"
VENV_DIR="$2"
REQ_INPUT="${3:-requirements.txt}"

# Ensure common user bin locations are on PATH in non-interactive jobs.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "uv command not found. Installing with curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv is still not available on PATH after installation."
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory '$REPO_DIR' not found."
    exit 1
fi

if [[ "$REQ_INPUT" = /* ]]; then
    REQUIREMENTS_FILE="$REQ_INPUT"
else
    REQUIREMENTS_FILE="$REPO_DIR/$REQ_INPUT"
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: requirements file '$REQUIREMENTS_FILE' not found."
    exit 1
fi

if [ -d "$VENV_DIR" ]; then
    echo "Environment already exists at '$VENV_DIR'; skipping venv creation."
else
    echo "Creating uv environment in '$VENV_DIR'..."
    mkdir -p "$(dirname "$VENV_DIR")"
    uv venv --python 3.13.2 "$VENV_DIR"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Error: '$VENV_DIR' does not contain a valid virtual environment."
    exit 1
fi

echo "Using uv at: $(command -v uv)"
echo "Installing requirements from '$REQUIREMENTS_FILE'..."
uv pip install --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS_FILE"

echo "Setup complete."
echo "To activate this environment manually: source $VENV_DIR/bin/activate"
