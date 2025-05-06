#!/usr/bin/env bash
set -euo pipefail   # safer bash

# ─── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.yaml"
INSTALL_DIR="$HOME/micromamba-bin"
MAMBA_ENV_NAME="geomml"
ENV_PREFIX="$INSTALL_DIR/envs/$MAMBA_ENV_NAME"

# ─── Step 1: Detect architecture ─────────────────────────────────────
ARCH=$(uname -m)
echo ">>> Detected architecture: $ARCH"
case "$ARCH" in
  x86_64)  PLATFORM="linux-64"     ;;
  aarch64) PLATFORM="linux-aarch64";;
  *) echo "❌ Unsupported architecture: $ARCH"; exit 1 ;;
esac

# ─── Step 2: Install micromamba ──────────────────────────────────────
echo ">>> Installing micromamba for $PLATFORM..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [[ ! -x bin/micromamba ]]; then
  wget -q "https://micro.mamba.pm/api/micromamba/$PLATFORM/latest" -O micromamba.tar.bz2
  tar -xvjf micromamba.tar.bz2 bin/micromamba
  chmod +x bin/micromamba
fi

# Put micromamba on PATH for future shells (idempotent)
if ! grep -q 'micromamba-bin/bin' ~/.bashrc; then
  echo 'export PATH="$HOME/micromamba-bin/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$INSTALL_DIR/bin:$PATH"

# ─── Step 3: Shell integration ───────────────────────────────────────
echo ">>> Initializing shell integration..."
micromamba shell init -s bash -y >/dev/null || true

export MAMBA_ROOT_PREFIX="$INSTALL_DIR"

eval "$(micromamba shell hook --shell bash)"

# ─── Step 4: Create environment ──────────────────────────────────────
echo ">>> Creating environment '$MAMBA_ENV_NAME' from $ENV_FILE ..."
[[ -f "$ENV_FILE" ]] || { echo "❌ env.yaml not found at $ENV_FILE"; exit 1; }

micromamba create -p "$ENV_PREFIX" -f "$ENV_FILE" -y

# ─── Step 5: (Optional) Activate if script is sourced ────────────────
# Detect whether the script was *sourced* or *executed*
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo ">>> Activating $ENV_PREFIX in this shell..."
  micromamba activate -p "$ENV_PREFIX"
  echo "✅ Environment '$MAMBA_ENV_NAME' active."
else
  echo "✅ Environment created at $ENV_PREFIX"
  echo "👉 Open a new shell or run:"
  echo "   micromamba activate -p \"$ENV_PREFIX\""
fi