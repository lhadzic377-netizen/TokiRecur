#!/usr/bin/env bash
#
# Build stable-diffusion.cpp (sd-cli + sd-server) from source for this OS,
# mirroring build_llama_cpp.sh. sd-server is the image-generation stack
# behind `toki model use` for image models (Qwen-Image family): the daemon
# is the text path and cannot generate images, and llama-server cannot
# load a diffusion GGUF, so image stacks run on their own GGML server.
#
#   ./scripts/build_sd_cpp.sh            build + install into ~/.local/state/tokirecur/bin
#   SD_SRC=~/sd.cpp ./scripts/build_sd_cpp.sh
#   HIP_ARCH=gfx1151 ./scripts/build_sd_cpp.sh
#
# Environment:
#   SD_SRC     path to a stable-diffusion.cpp checkout (default: ~/sd.cpp)
#   HIP_ARCH   gfx target (default: gfx1151 — this machine's Radeon 8060S)

set -euo pipefail

SD_SRC="${SD_SRC:-$HOME/sd.cpp}"
HIP_ARCH="${HIP_ARCH:-gfx1151}"
BUILD="$SD_SRC/build"

if [[ ! -d "$SD_SRC/.git" ]]; then
    echo "cloning stable-diffusion.cpp (with submodules) to $SD_SRC"
    git clone --depth 1 --recursive https://github.com/leejet/stable-diffusion.cpp.git "$SD_SRC"
else
    git -C "$SD_SRC" submodule update --init --recursive
fi

mkdir -p "$BUILD"
cd "$BUILD"
cmake .. \
    -DSD_HIPBLAS=ON \
    -DAMDGPU_TARGETS="$HIP_ARCH" \
    -DGGML_HIP_UMA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DSD_BUILD_EXAMPLES=ON
cmake --build . --config Release --target sd-cli sd-server -j"$(nproc)"

BIN_DIR="$HOME/.local/state/tokirecur/bin"
mkdir -p "$BIN_DIR"
install -m 0755 "$BUILD/bin/sd-cli" "$BIN_DIR/sd-cli"
install -m 0755 "$BUILD/bin/sd-server" "$BIN_DIR/sd-server"
echo "installed sd-cli + sd-server -> $BIN_DIR"
