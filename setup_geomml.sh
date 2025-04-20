#!/bin/bash

set -e  # Exit on any error

# --- Config ---
INSTALL_DIR="$HOME/micromamba-bin"
MAMBA_ENV_NAME="geomml"
ENV_FILE="$(dirname "$0")/env.yaml"  # Use env.yaml in same folder as script

# --- Step 1: Install micromamba ---
echo ">>> Downloading micromamba..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -f bin/micromamba ]; then
  wget -q https://micro.mamba.pm/api/micromamba/linux-64/latest -O micromamba.tar.bz2
  tar -xvjf micromamba.tar.bz2 bin/micromamba
  chmod +x bin/micromamba
fi

# Add micromamba to PATH in .bashrc if not already added
if ! grep -q "$INSTALL_DIR/bin" ~/.bashrc; then
  echo 'export PATH="$HOME/micromamba-bin/bin:$PATH"' >> ~/.bashrc
fi

export PATH="$INSTALL_DIR/bin:$PATH"

# --- Step 2: Shell integration ---
echo ">>> Initializing shell integration..."
micromamba shell init -s bash -y

source ~/.bashrc  # Load micromamba shell hook
export PATH="$INSTALL_DIR/bin:$PATH"  # Just in case

# --- Step 3: Create environment ---
echo ">>> Creating micromamba environment '$MAMBA_ENV_NAME'..."
micromamba create -n "$MAMBA_ENV_NAME" -f "$ENV_FILE" -y

# --- Step 4: Activate environment ---
echo ">>> Activating environment..."
micromamba activate "$MAMBA_ENV_NAME"

echo "✅ Environment '$MAMBA_ENV_NAME' created and activated successfully!"