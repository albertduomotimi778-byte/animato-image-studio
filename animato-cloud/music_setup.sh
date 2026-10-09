#!/usr/bin/env bash
# Installs the music engine (ACE-Step 1.5 turbo, CPU) into ~/acestep-local — once; the
# workflows cache that folder, so later runs skip all of this.
# WITH_LM=true also fetches the small song-planning model (the video renderer uses it).
set -e
ROOT="$HOME/acestep-local"; REPO="$ROOT/repo"; VENV="$ROOT/venv"; PY="$VENV/bin/python"; CKPT="$REPO/checkpoints"
mkdir -p "$ROOT"
if [ ! -d "$REPO/acestep" ]; then git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "$REPO"; fi
if [ ! -x "$PY" ]; then python3 -m venv "$VENV"; fi
if ! "$PY" -c "import acestep, torch, diffusers, transformers" 2>/dev/null; then
  echo "[songgen] [installing] engine packages (first run only)"
  "$PY" -m pip install -q --disable-pip-version-check --upgrade pip
  "$PY" -m pip install -q --disable-pip-version-check torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip install -q --disable-pip-version-check "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "matplotlib>=3.7.5" "scipy>=1.10.1" "soundfile>=0.13.1" "loguru>=0.7.3" "einops>=0.8.1" "accelerate>=1.12.0" "fastapi>=0.110.0" diskcache "uvicorn[standard]>=0.27.0" "numba>=0.63.1" "vector-quantize-pytorch>=1.27.15" "torchao>=0.16.0,<0.17.0" toml "peft>=0.18.0" lycoris-lora modelscope "typer-slim>=0.21.1" "pytorch-wavelets>=1.3.0" "pywavelets>=1.9.0" "setuptools<72" huggingface_hub "gradio==6.2.0" "lightning>=2.0.0" "tensorboard>=2.20.0"
  "$PY" -m pip install -q --disable-pip-version-check --no-deps -e "$REPO"
fi
if [ ! -d "$CKPT/acestep-v15-turbo" ] || [ ! -d "$CKPT/vae" ] || [ ! -d "$CKPT/Qwen3-Embedding-0.6B" ]; then
  echo "[songgen] [downloading model] (first run only)"
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='$CKPT', allow_patterns=['acestep-v15-turbo/*','vae/*','Qwen3-Embedding-0.6B/*','*.json','*.txt'])"
fi
if [ "${WITH_LM:-}" = "true" ] && [ ! -d "$CKPT/acestep-5Hz-lm-0.6B" ]; then
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-5Hz-lm-0.6B', local_dir='$CKPT/acestep-5Hz-lm-0.6B')" || echo "planning model download failed (songs are made without it)"
fi
du -sh "$ROOT" || true
