#!/usr/bin/env bash
# Build ggml-org/llama.cpp shared libs + libllamastack into lib/<os>-<arch>/.
# Published linux/amd64 overlay is CUDA+Vulkan (flavor=gpu). CPU/cuda/vulkan-only
# on that pair are local: lib/linux-amd64-<flavor>/. linux/arm64 = CPU.
# darwin = Metal. Windows → build-llama.ps1 (published = CUDA+Vulkan).
# Env: LLAMA_CPP_REF, LLAMA_CPP_FLAVOR=cpu|cuda|vulkan|gpu,
#      LLAMA_CPP_CUDA_ARCHITECTURES, DEST_DIR, JOBS, LLAMA_CPP_SKIP_CMAKE=1
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
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    if command -v pwsh >/dev/null 2>&1; then
      exec pwsh -File "$ROOT/scripts/build-llama.ps1"
    fi
    echo "Windows build needs pwsh (scripts/build-llama.ps1)" >&2
    exit 1
    ;;
  *) echo "unsupported OS: $uname_s" >&2; exit 1 ;;
esac
case "$uname_m" in
  x86_64|amd64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "unsupported arch: $uname_m" >&2; exit 1 ;;
esac

if [[ -z "${LLAMA_CPP_FLAVOR:-}" ]]; then
  if [[ "$os" == "linux" && "$arch" == "amd64" ]]; then
    FLAVOR=gpu
  else
    FLAVOR=cpu
  fi
else
  FLAVOR="$LLAMA_CPP_FLAVOR"
fi
case "$FLAVOR" in
  cpu|cuda|vulkan|gpu) ;;
  *) echo "LLAMA_CPP_FLAVOR must be cpu|cuda|vulkan|gpu (got: $FLAVOR)" >&2; exit 1 ;;
esac

want_cuda=0
want_vulkan=0
case "$FLAVOR" in
  cuda|gpu) want_cuda=1 ;;
esac
case "$FLAVOR" in
  vulkan|gpu) want_vulkan=1 ;;
esac

if ((want_cuda)) && [[ "$os" != "linux" ]]; then
  echo "CUDA in this script is linux-only (got $os/$arch; Windows → build-llama.ps1)" >&2
  exit 1
fi

# Published path is lib/<os>-<arch>/. Non-default flavors on linux/amd64 are local-only.
if [[ "$os" == "linux" && "$arch" == "amd64" && "$FLAVOR" != "gpu" ]]; then
  suffix="-$FLAVOR"
else
  suffix=""
fi
OUT="${DEST_DIR:-$ROOT/lib/${os}-${arch}${suffix}}"
SRC="$ROOT/build/llama.cpp"
BUILD="$ROOT/build/${os}-${arch}-${FLAVOR}"

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
    -DGGML_NATIVE=OFF
    -DGGML_HIP=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_TOOLS=OFF
  )
  if ((want_cuda)); then
    cmake_args+=(
      -DGGML_CUDA=ON
      -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CPP_CUDA_ARCHITECTURES:-80;86;89;90a}"
    )
  else
    cmake_args+=(-DGGML_CUDA=OFF)
  fi
  if ((want_vulkan)); then
    cmake_args+=(-DGGML_VULKAN=ON)
  else
    cmake_args+=(-DGGML_VULKAN=OFF)
  fi
  if [[ "$os" == "darwin" ]]; then
    cmake_args+=(-DGGML_METAL=ON -DGGML_BLAS=ON)
  else
    cmake_args+=(-DGGML_METAL=OFF -DGGML_BLAS=OFF)
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
  cc -dynamiclib -fPIC -DLLAMA_STACK_BUILD \
    -I"$ROOT/include" -I"$SRC/include" -I"$SRC/ggml/include" \
    "$ROOT/native/llama-stack.c" \
    -L"$OUT" -lllama -lggml \
    -install_name @rpath/libllamastack.dylib \
    -Wl,-rpath,@loader_path \
    -o "$SHIM_OUT"
else
  SHIM_OUT="${SHIM_OUT}.so"
  cc -shared -fPIC -DLLAMA_STACK_BUILD \
    -I"$ROOT/include" -I"$SRC/include" -I"$SRC/ggml/include" \
    "$ROOT/native/llama-stack.c" \
    -L"$OUT" -lllama -lggml -lm \
    -Wl,-soname,libllamastack.so.0 \
    -Wl,-rpath,'$ORIGIN' \
    -o "$SHIM_OUT"
  cp -f "$SHIM_OUT" "$OUT/libllamastack.so.0"
fi

# Materialize SONAME / unversioned names as regular files so the packager
# (read-file-octets) and .asd inventory do not depend on cmake's versioned names.
materialize_symlinks() {
  local f tmp
  shopt -s nullglob
  for f in "$OUT"/lib*.dylib "$OUT"/lib*.so "$OUT"/lib*.so.*; do
    [[ -L "$f" ]] || continue
    tmp="$(mktemp "$OUT/.mat.XXXXXX")"
    cp -L "$f" "$tmp"
    rm -f "$f"
    mv "$tmp" "$f"
  done
  shopt -u nullglob
}

drop_ggml_patch_versions() {
  # Drop libllama.0.3.0 / libggml*.0.22.0 only. Never libcudart.so.12.x.
  local f b
  for f in "$OUT"/libllama* "$OUT"/libggml*; do
    [[ -e "$f" || -L "$f" ]] || continue
    b="$(basename "$f")"
    if [[ "$b" =~ \.[0-9]+\.[0-9]+\.[0-9]+\.dylib$ ]] \
       || [[ "$b" =~ \.so\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      rm -f "$f"
    fi
  done
}

materialize_symlinks
drop_ggml_patch_versions
chmod a+rX "$OUT"/* 2>/dev/null || true

if [[ "$os" == "linux" ]] && command -v patchelf >/dev/null; then
  for f in "$OUT"/lib*.so "$OUT"/lib*.so.*; do
    [[ -f "$f" && ! -L "$f" ]] || continue
    patchelf --set-rpath '$ORIGIN' "$f"
  done
fi

if ((want_cuda)); then
  chmod +x "$ROOT/scripts/stage-cuda-runtime.sh"
  "$ROOT/scripts/stage-cuda-runtime.sh" "$OUT"
  if [[ ! -e "$OUT/libggml-cuda.so" && ! -e "$OUT/libggml-cuda.so.0" ]]; then
    echo "CUDA flavor missing libggml-cuda.so*" >&2
    ls -la "$OUT" >&2
    exit 1
  fi
fi

if ((want_vulkan)); then
  if [[ ! -e "$OUT/libggml-vulkan.so" && ! -e "$OUT/libggml-vulkan.so.0" \
        && ! -e "$OUT/libggml-vulkan.dylib" ]]; then
    echo "Vulkan flavor missing libggml-vulkan*" >&2
    ls -la "$OUT" >&2
    exit 1
  fi
fi

test -e "$SHIM_OUT"
ls -la "$OUT"
echo "staged $OUT flavor=$FLAVOR"
