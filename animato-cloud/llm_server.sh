#!/usr/bin/env bash
# Starts the app's own AI model on this GitHub runner — no API keys, no AI services.
#
#   llama.cpp (prebuilt CPU release, or built from source if the download fails)
#   + Qwen3-4B-Instruct-2507 (Q4_K_M GGUF, ~2.5 GB) as an OpenAI-compatible server
#   on http://127.0.0.1:8080/v1. With "vision" as the first argument it also starts
#   Qwen2.5-VL-3B-Instruct + its mmproj on :8081 (looking at pictures); "vision-only"
#   starts just that one (the brain does this on first use).
#
# Everything lives in ~/animato-llm (cache that folder with actions/cache).
# Exports LOCAL_LLM_URL (and LOCAL_VLM_URL) to $GITHUB_ENV for the next steps.
set -euo pipefail
DIR="$HOME/animato-llm"
LLAMA_TAG="${LLAMA_TAG:-b11435}"
TEXT_REPO="${LLM_TEXT_REPO:-unsloth/Qwen3-4B-Instruct-2507-GGUF}"
TEXT_FILE="${LLM_TEXT_FILE:-Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
VL_REPO="${LLM_VL_REPO:-ggml-org/Qwen2.5-VL-3B-Instruct-GGUF}"
VL_FILE="${LLM_VL_FILE:-Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf}"
VL_PROJ="${LLM_VL_PROJ:-mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf}"
CTX="${LLM_CTX:-12288}"
mkdir -p "$DIR/models" "$DIR/bin"

server_bin() { find "$DIR/bin" -type f -name llama-server -perm -u+x 2>/dev/null | head -n 1; }

if [ -z "$(server_bin)" ]; then
  echo "Installing llama.cpp ($LLAMA_TAG)…"
  ok=0
  for asset in "llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" "llama-${LLAMA_TAG}-bin-ubuntu-x64.zip"; do
    url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${asset}"
    if curl -fsSL --retry 3 -o "$DIR/$asset" "$url"; then
      case "$asset" in *.zip) unzip -qo "$DIR/$asset" -d "$DIR/bin" ;; *) tar -xzf "$DIR/$asset" -C "$DIR/bin" ;; esac
      rm -f "$DIR/$asset"; ok=1; break
    fi
  done
  if [ "$ok" != 1 ] || [ -z "$(server_bin)" ]; then
    # Newest release instead of the pinned one.
    api=$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest || true)
    url=$(printf '%s' "$api" | grep -o '"browser_download_url": *"[^"]*bin-ubuntu-x64\.\(tar\.gz\|zip\)"' | head -n 1 | sed 's/.*"\(https[^"]*\)"/\1/')
    if [ -n "$url" ] && curl -fsSL --retry 3 -o "$DIR/llama.pkg" "$url"; then
      case "$url" in *.zip) unzip -qo "$DIR/llama.pkg" -d "$DIR/bin" ;; *) tar -xzf "$DIR/llama.pkg" -C "$DIR/bin" ;; esac
      rm -f "$DIR/llama.pkg"
    fi
  fi
  if [ -z "$(server_bin)" ]; then
    echo "Prebuilt llama.cpp unavailable — building it (a few minutes, once)…"
    sudo apt-get install -y -qq build-essential cmake > /dev/null
    rm -rf "$DIR/src" && git clone --depth 1 https://github.com/ggml-org/llama.cpp "$DIR/src"
    cmake -S "$DIR/src" -B "$DIR/src/build" -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$DIR/src/build" --config Release -j "$(nproc)" --target llama-server > /dev/null
    mkdir -p "$DIR/bin/src" && cp -r "$DIR/src/build/bin/." "$DIR/bin/src/"
    rm -rf "$DIR/src"
  fi
fi
BIN="$(server_bin)"; BIN_DIR="$(dirname "$BIN")"
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

fetch() { # repo file
  local out="$DIR/models/$2"
  if [ ! -s "$out" ]; then
    echo "Downloading $2…"
    curl -fL --retry 4 --retry-delay 3 -o "$out.part" "https://huggingface.co/$1/resolve/main/$2"
    mv "$out.part" "$out"
  fi
}
THREADS="$(nproc)"
start() { # port args…
  local port="$1"; shift
  nohup "$BIN" --host 127.0.0.1 --port "$port" -t "$THREADS" -np 1 --cache-reuse 256 "$@" > "$DIR/server-$port.log" 2>&1 &
  for i in $(seq 1 180); do
    if curl -fs "http://127.0.0.1:$port/health" > /dev/null 2>&1; then echo "AI model ready on :$port ($i s)"; return 0; fi
    sleep 1
  done
  echo "::error::the AI model did not start"; tail -n 40 "$DIR/server-$port.log"; return 1
}
if [ "${1:-}" != "vision-only" ]; then
  fetch "$TEXT_REPO" "$TEXT_FILE"
  start 8080 -m "$DIR/models/$TEXT_FILE" -c "$CTX"
  [ -n "${GITHUB_ENV:-}" ] && echo "LOCAL_LLM_URL=http://127.0.0.1:8080/v1" >> "$GITHUB_ENV"
fi

if [ "${1:-}" = "vision" ] || [ "${1:-}" = "vision-only" ]; then
  fetch "$VL_REPO" "$VL_FILE"; fetch "$VL_REPO" "$VL_PROJ"
  start 8081 -m "$DIR/models/$VL_FILE" --mmproj "$DIR/models/$VL_PROJ" -c 4096
  [ -n "${GITHUB_ENV:-}" ] && echo "LOCAL_VLM_URL=http://127.0.0.1:8081/v1" >> "$GITHUB_ENV"
fi
echo "LLAMA_SERVER_BIN=$BIN" >> "${GITHUB_ENV:-/dev/null}"
