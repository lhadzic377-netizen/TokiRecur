#!/usr/bin/env bash
# Build llama.cpp with the Toki Recur KV cache patch applied.
# Targets ROCm on the same host (Strix Halo / gfx1151 in particular).
#
# Usage:
#   ./scripts/build_llama_cpp.sh
#
# Env vars:
#   LLAMA_CPP_SRC   path to a llama.cpp checkout (default: ~/llama.cpp)
#   LLAMA_CPP_BLD   build directory (default: $LLAMA_CPP_SRC/build-tr)
#   HIP_ARCH        gfx target (default: gfx1151)
#   UMA             1 to enable unified memory (default: 1 on APU)

set -euo pipefail

LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-$HOME/llama.cpp}"
LLAMA_CPP_BLD="${LLAMA_CPP_BLD:-$LLAMA_CPP_SRC/build-tr}"
HIP_ARCH="${HIP_ARCH:-gfx1151}"
UMA="${UMA:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/patches/llama-server-kv-endpoints.diff"

if [[ ! -d "$LLAMA_CPP_SRC" ]]; then
    echo "error: $LLAMA_CPP_SRC does not exist" >&2
    exit 1
fi

cd "$LLAMA_CPP_SRC"

# 1. Apply the patch if not already applied.
if grep -q "sv_kv_save" examples/server/server.cpp 2>/dev/null; then
    echo "patch already applied"
else
    echo "applying patch: $PATCH"
    git apply --check "$PATCH" || {
        echo "patch does not apply cleanly; check versions" >&2
        exit 1
    }
    git apply "$PATCH"
fi

# 2. Configure.
mkdir -p "$LLAMA_CPP_BLD"
cd "$LLAMA_CPP_BLD"

CMAKE_ARGS=(
    -DGGML_HIP=ON
    -DAMDGPU_TARGETS="$HIP_ARCH"
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_NATIVE=OFF
    -DBUILD_SHARED_LIBS=ON
)

if [[ "$UMA" == "1" ]]; then
    CMAKE_ARGS+=(-DGGML_HIP_UMA=ON)
    echo "UMA enabled ($HIP_ARCH)"
else
    echo "UMA disabled"
fi

cmake "${CMAKE_ARGS[@]}" ..

# 3. Build.
cmake --build . --config Release --target llama-server -j"$(nproc)"

echo "llama-server built: $LLAMA_CPP_BLD/bin/llama-server"
