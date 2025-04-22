#!/bin/bash
set -e  # Exit on any error

# --- Config ---
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ENV_FILE="$SCRIPT_DIR/env.yaml"
INSTALL_DIR="$HOME/micromamba-bin"
MAMBA_ENV_NAME="geomml"

# --- Step 1: Detect architecture ---
ARCH=$(uname -m)
echo ">>> Detected architecture: $ARCH"
if [[ "$ARCH" == "x86_64" ]]; then
  PLATFORM="linux-64"
elif [[ "$ARCH" == "aarch64" ]]; then
  PLATFORM="linux-aarch64"
else
  echo "❌ Unsupported architecture: $ARCH"
  exit 1
fi

# --- Step 2: Install micromamba ---
echo ">>> Downloading micromamba for $PLATFORM..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -f bin/micromamba ]; then
  wget -q "https://micro.mamba.pm/api/micromamba/$PLATFORM/latest" -O micromamba.tar.bz2
  tar -xvjf micromamba.tar.bz2 bin/micromamba
  chmod +x bin/micromamba
fi

# Add micromamba to PATH in .bashrc if not already added
if ! grep -q "$INSTALL_DIR/bin" ~/.bashrc; then
  echo 'export PATH="$HOME/micromamba-bin/bin:$PATH"' >> ~/.bashrc
fi

export PATH="$INSTALL_DIR/bin:$PATH"

# --- Step 3: Shell integration ---
echo ">>> Initializing shell integration..."
micromamba shell init -s bash -y || true

# Activate the shell hook in the current session (requires the script to be sourced)
eval "$(micromamba shell hook --shell bash)"

# --- Step 4: Create environment ---
echo ">>> Creating micromamba environment '$MAMBA_ENV_NAME' using $ENV_FILE ..."
if [ ! -f "$ENV_FILE" ]; then
  echo "❌ Error: Environment file not found at $ENV_FILE"
  exit 1
fi

micromamba create -n "$MAMBA_ENV_NAME" -f "$ENV_FILE" -y

# --- Step 5: Activation Instructions ---
echo "✅ Environment '$MAMBA_ENV_NAME' created successfully!"
echo "👉 To activate the environment in this session, run: micromamba activate $MAMBA_ENV_NAME"
echo "👉 Or simply run a command with it: micromamba run -n $MAMBA_ENV_NAME <command>"