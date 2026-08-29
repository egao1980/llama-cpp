#!/usr/bin/env bash
# Build ggml-org/llama.cpp shared libs + libllamastack into lib/<os>-<arch>/.
# Env: LLAMA_CPP_REF (default master), DEST_DIR, JOBS
#      LLAMA_CPP_SKIP_CMAKE=1 — restage + relink shim only (reuse $BUILD)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${LLAMA_CPP_REF:-master}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
SRC_URL="https://github.com/ggml-org/llama.cpp.git"

uname_s="$(uname -s)"
uname_m="$(uname -m)"
case "$uname_s" in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  *) echo "unsupported OS: $uname_s (Windows: follow vllm-cpp later)" >&2; exit 1 ;;
esac
case "$uname_m" in
  x86_64|amd64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "unsupported arch: $uname_m" >&2; exit 1 ;;
esac

OUT="${DEST_DIR:-$ROOT/lib/${os}-${arch}}"
SRC="$ROOT/build/llama.cpp"
BUILD="$ROOT/build/${os}-${arch}"

mkdir -p "$ROOT/build"
if [[ "${LLAMA_CPP_SKIP_CMAKE:-}" != "1" ]]; then
  if [[ -d "$SRC/.git" ]]; then
    git -C "$SRC" fetch --depth 1 origin "$REF"
    git -C "$SRC" checkout --force FETCH_HEAD
  else
    rm -rf "$SRC"
    git clone --depth 1 --branch "$REF" "$SRC_URL" "$SRC" \
      || { git clone --depth 1 "$SRC_URL" "$SRC"
           git -C "$SRC" fetch --depth 1 origin "$REF"
           git -C "$SRC" checkout --force FETCH_HEAD; }
  fi

  cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_TOOLS=OFF
  )
  if [[ "$os" == "darwin" ]]; then
    cmake_args+=(-DGGML_METAL=ON)
  fi

  cmake -S "$SRC" -B "$BUILD" "${cmake_args[@]}"
  cmake --build "$BUILD" --target llama -j"$JOBS"
fi

mkdir -p "$OUT"
# cmake writes libllama.dylib → libllama.0.dylib → libllama.X.Y.Z.dylib.
# find -type f skips the unversioned / SONAME links; -lllama needs them.
stage_lib() {
  local dest="$OUT/$(basename "$1")"
  if [[ ! -e "$dest" && ! -L "$dest" ]]; then
    cp -a "$1" "$OUT/"
  fi
}
if [[ -d "$BUILD/bin" ]]; then
  shopt -s nullglob
  for f in "$BUILD/bin/"libllama* "$BUILD/bin/"libggml* \
           "$BUILD/bin/"ggml-metal.metallib; do
    [[ -e "$f" || -L "$f" ]] && cp -a "$f" "$OUT/"
  done
  shopt -u nullglob
fi
while IFS= read -r -d '' f; do
  stage_lib "$f"
done < <(find "$BUILD" \( -type f -o -type l \) \( \
    -name 'libllama*.dylib' -o -name 'libllama*.so*' \
    -o -name 'libggml*.dylib' -o -name 'libggml*.so*' \
    -o -name 'ggml-metal.metallib' \
  \) -print0)

SHIM_OUT="$OUT/libllamastack"
if [[ "$os" == "darwin" ]]; then
  SHIM_OUT="${SHIM_OUT}.dylib"
  cc -dynamiclib -fPIC \
    -I"$ROOT/include" -I"$SRC/include" -I"$SRC/ggml/include" \
    "$ROOT/native/llama-stack.c" \
    -L"$OUT" -lllama -lggml \
    -install_name @rpath/libllamastack.dylib \
    -Wl,-rpath,@loader_path \
    -o "$SHIM_OUT"
else
  SHIM_OUT="${SHIM_OUT}.so"
  cc -shared -fPIC \
    -I"$ROOT/include" -I"$SRC/include" -I"$SRC/ggml/include" \
    "$ROOT/native/llama-stack.c" \
    -L"$OUT" -lllama -lggml -lm \
    -Wl,-rpath,'$ORIGIN' \
    -o "$SHIM_OUT"
fi

test -e "$SHIM_OUT"
ls -la "$OUT"
echo "staged $OUT"
