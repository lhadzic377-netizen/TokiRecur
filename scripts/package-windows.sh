#!/usr/bin/env bash
# Build the Windows test package: a zip with the wheel, install.ps1 and
# the toki wrappers.
#
#   scripts/package-windows.sh                 # online install (deps from PyPI)
#   scripts/package-windows.sh --offline 3.12  # + wheelhouse for CPython 3.12
#   scripts/package-windows.sh --out DIR
#
# --offline needs network on THIS machine (it downloads win_amd64 wheels)
# but the resulting zip installs with no network on the Windows box.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-"$ROOT/.venv/bin/python"}
OUT="$ROOT/dist"
OFFLINE_PY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offline)
            OFFLINE_PY="${2:?--offline needs a Python version, e.g. 3.12}"
            shift 2
            ;;
        --out)
            OUT="${2:?--out needs a directory}"
            shift 2
            ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

VERSION=$("$PY" -c "import tokirecur; print(tokirecur.__version__)")
NAME="toki-windows-$VERSION"
if [[ -n "$OFFLINE_PY" ]]; then
    NAME="$NAME-py$OFFLINE_PY-offline"
fi
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/toki-win-XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

echo "building tokirecur $VERSION"
mkdir -p "$STAGE/$NAME/wheel"
"$PY" -m pip wheel --no-deps --no-build-isolation -w "$STAGE/$NAME/wheel" "$ROOT" >/dev/null

cp "$ROOT/scripts/windows/install.ps1" \
   "$ROOT/scripts/windows/toki.cmd" \
   "$ROOT/scripts/windows/toki.ps1" \
   "$ROOT/scripts/windows/README-WINDOWS.md" \
   "$STAGE/$NAME/"
cp "$ROOT/LICENSE" "$STAGE/$NAME/" 2>/dev/null || true

if [[ -n "$OFFLINE_PY" ]]; then
    echo "downloading win_amd64 dependency wheels for CPython $OFFLINE_PY"
    mkdir -p "$STAGE/$NAME/wheelhouse"
    # pip evaluates environment markers against the HOST, not --platform,
    # so a plain `pip download <wheel>` tries to fetch POSIX-only uvloop
    # and fails. Ask for the direct dependencies explicitly (markers
    # evaluated for the target) and spell out uvicorn's Windows extras.
    mapfile -t DEPS < <("$PY" - "$ROOT/pyproject.toml" "$OFFLINE_PY" <<'PY'
import sys
import tomllib
from packaging.requirements import Requirement

path, target = sys.argv[1], sys.argv[2]
data = tomllib.load(open(path, "rb"))
env = {
    "python_version": target,
    "python_full_version": target + ".0",
    "sys_platform": "win32",
    "platform_system": "Windows",
    "platform_python_implementation": "CPython",
    "os_name": "nt",
}
out: list[str] = []
for raw in data["project"]["dependencies"]:
    req = Requirement(raw)
    if req.marker is not None and not req.marker.evaluate(env):
        continue
    if req.name.lower() == "uvicorn" and "standard" in req.extras:
        out.append("uvicorn")
        out.extend(["httptools", "python-dotenv", "watchfiles", "websockets"])
        continue
    req.extras = set()
    req.marker = None
    out.append(str(req))
if tuple(int(x) for x in target.split(".")) < (3, 11):
    out.append("tomli")
# Markers are evaluated for the HOST when pip downloads transitive
# dependencies, so Windows-only deps of the direct set are missed:
# tqdm -> colorama. Add them explicitly.
out.append("colorama")
print("\n".join(out))
PY
)
    echo "  ${DEPS[*]}"
    "$PY" -m pip download \
        --only-binary=:all: \
        --platform win_amd64 \
        --python-version "$OFFLINE_PY" \
        --implementation cp \
        --abi "cp${OFFLINE_PY//./}" \
        -d "$STAGE/$NAME/wheelhouse" \
        "${DEPS[@]}" >/dev/null
fi

mkdir -p "$OUT"
ZIP="$OUT/$NAME.zip"
rm -f "$ZIP"
"$PY" - "$STAGE" "$NAME" "$ZIP" <<'PY'
import sys
import zipfile
from pathlib import Path

stage, name, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
root = stage / name
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(stage).as_posix())
print(f"{out}  ({out.stat().st_size / 1_048_576:.1f} MiB)")
with zipfile.ZipFile(out) as zf:
    for info in zf.infolist():
        print(f"  {info.filename}  ({info.file_size / 1024:.0f} KiB)")
PY
