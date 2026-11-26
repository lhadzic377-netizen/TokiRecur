#!/usr/bin/env bash
# Build llama.cpp for this box (ROCm / Strix Halo gfx1151 in particular).
#
# Usage:
#   ./scripts/build_llama_cpp.sh
#
# Env vars:
#   LLAMA_CPP_SRC     path to a llama.cpp checkout (default: ~/llama.cpp)
#   LLAMA_CPP_BLD     build directory (default: $LLAMA_CPP_SRC/build-tr)
#   HIP_ARCH          gfx target (default: gfx1151)
#   UMA               1 to enable unified memory (default: 1 on APU)
#   LLAMA_CPP_UPDATE  1 to fetch and reset the checkout to origin/master first
#   LLAMA_CPP_INSTALL 1 to install the result into the toki state bin dir
#   JOBS              build parallelism (default: nproc)

set -euo pipefail

LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-$HOME/llama.cpp}"
LLAMA_CPP_BLD="${LLAMA_CPP_BLD:-$LLAMA_CPP_SRC/build-tr}"
HIP_ARCH="${HIP_ARCH:-gfx1151}"
UMA="${UMA:-1}"
UPDATE="${LLAMA_CPP_UPDATE:-0}"
INSTALL="${LLAMA_CPP_INSTALL:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/patches/llama-server-kv-endpoints.diff"
STATE_BIN="${TOKI_STATE_BIN:-$HOME/.local/state/tokirecur/bin}"
# nproc is the LOGICAL CPU count (32 on this box). A HIP build with that
# many jobs just fights the GPU compiler and everyone else on the machine.
# Keep the quiet default low; JOBS=... when the box is yours alone.
JOBS="${JOBS:-8}"
if [[ "$JOBS" -gt "$(nproc)" ]]; then
    JOBS="$(nproc)"
fi

if [[ ! -d "$LLAMA_CPP_SRC" ]]; then
    echo "error: $LLAMA_CPP_SRC does not exist" >&2
    exit 1
fi

cd "$LLAMA_CPP_SRC"

# 0. Update the checkout to the latest upstream master (opt-in).
if [[ "$UPDATE" == "1" ]]; then
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        echo "error: $LLAMA_CPP_SRC is not a git checkout" >&2
        exit 1
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "error: $LLAMA_CPP_SRC has uncommitted changes:" >&2
        git status --porcelain >&2
        echo "commit or stash them first (or update a clean checkout)." >&2
        exit 1
    fi
    echo "updating: fetching origin/master"
    git fetch --depth=1 origin master
    git checkout --detach FETCH_HEAD
    echo "updating: now at $(git log --oneline -1)"
fi

# 1. Apply the KV endpoints patch if the checkout is old enough to need it.
#    Upstream shipped the `--slot-save-path` endpoints natively (the server
#    code now lives under tools/server/), so modern checkouts skip the patch.
if grep -q 'action == "save"' tools/server/server-context.cpp 2>/dev/null \
    || grep -q "sv_kv_save" examples/server/server.cpp 2>/dev/null; then
    echo "KV save/load endpoints already present (patch not needed)"
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
    # The prebuilt web UI is validated against the checkout it was built
    # from, and a source build has no matching assets: the build failed
    # on the asset check. The API server does not need the UI, so it is
    # off - and the checkout stays pristine (no local patch needed).
    -DLLAMA_USE_PREBUILT_UI=OFF
)

if [[ "$UMA" == "1" ]]; then
    CMAKE_ARGS+=(-DGGML_HIP_UMA=ON)
    echo "UMA enabled ($HIP_ARCH)"
else
    echo "UMA disabled"
fi

cmake "${CMAKE_ARGS[@]}" ..

# 3. Build.
cmake --build . --config Release --target llama-server -j"$JOBS"

echo "llama-server built: $LLAMA_CPP_BLD/bin/llama-server"

# 4. Install into the toki state bin dir (opt-in). The registry's
#    `binary:` entries point there, so this is what makes the new build
#    the one toki starts.
if [[ "$INSTALL" == "1" ]]; then
    mkdir -p "$STATE_BIN"
    cp -a "$LLAMA_CPP_BLD/bin/." "$STATE_BIN/"
    # The ABI version moves with every release (libllama.so.1.0.0 ->
    # 1.1.0), so the old ones are dead weight that only confuses a
    # later ldd probe. Keep exactly the libraries this build shipped.
    installed="$(cd "$LLAMA_CPP_BLD/bin" && ls -1)"
    while IFS= read -r stale; do
        if ! grep -qxF "$stale" <<<"$installed"; then
            echo "removing stale library: $stale"
            rm -f "$STATE_BIN/$stale"
        fi
    done < <(cd "$STATE_BIN" && ls -1 | grep -E '\.so\.[0-9]')
    echo "installed: $STATE_BIN/llama-server"
fi

"$LLAMA_CPP_BLD/bin/llama-server" --version 2>&1 | head -2
