#!/usr/bin/env bash
# Copy user-mode CUDA runtime (cudart + cublas) into the overlay and retarget
# DT_NEEDED to $ORIGIN. Never ship libcuda (the NVIDIA driver).
set -euo pipefail

OUT="${1:?usage: stage-cuda-runtime.sh <overlay-dir>}"
test -d "$OUT"

is_driver() {
  case "$1" in
    libcuda.so|libcuda.so.*) return 0 ;;
  esac
  return 1
}

copy_user_mode() {
  local src="$1"
  local name real rbase
  name="$(basename "$src")"
  # ggml-cuda DT_NEEDED libcuda.so.1 (cuda_driver / VMM). Host provides it.
  if is_driver "$name"; then
    return 0
  fi
  # ldconfig gives SONAME symlinks (libcudart.so.12 -> libcudart.so.12.8.90).
  # cp -a leaves a dangling link; packager and [[ -e ]] need a regular file.
  real="$(readlink -f "$src" 2>/dev/null || true)"
  if [[ -z "$real" || ! -f "$real" ]]; then
    echo "cannot resolve CUDA library $src" >&2
    exit 1
  fi
  rbase="$(basename "$real")"
  if [[ ! -f "$OUT/$rbase" || -L "$OUT/$rbase" ]]; then
    rm -f "$OUT/$rbase"
    cp -L "$real" "$OUT/$rbase"
  fi
  if [[ "$name" != "$rbase" ]]; then
    rm -f "$OUT/$name"
    cp -L "$real" "$OUT/$name"
  fi
}

resolve_soname() {
  local needed="$1"
  local found=""
  found="$(ldconfig -p 2>/dev/null | awk -v n="$needed" '$1 == n {print $NF; exit}')"
  if [[ -z "$found" ]]; then
    local d
    for d in /usr/local/cuda/lib64 /usr/local/cuda/lib /usr/lib/x86_64-linux-gnu; do
      if [[ -e "$d/$needed" ]]; then
        found="$d/$needed"
        break
      fi
    done
  fi
  printf '%s' "$found"
}

collect_needed() {
  local lib
  for lib in "$OUT"/libllamastack.so* "$OUT"/libllama.so* "$OUT"/libggml*.so*; do
    [[ -e "$lib" ]] || continue
    patchelf --print-needed "$lib" 2>/dev/null || true
  done | sort -u
}

while read -r needed; do
  [[ -n "$needed" ]] || continue
  is_driver "$needed" && continue
  case "$needed" in
    libcudart.so.*|libcublas.so.*|libcublasLt.so.*)
      found="$(resolve_soname "$needed")"
      if [[ -z "$found" ]]; then
        echo "CUDA runtime $needed not found via ldconfig / CUDA lib dirs" >&2
        exit 1
      fi
      copy_user_mode "$found"
      ;;
  esac
done < <(collect_needed)

for required in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
  if [[ ! -e "$OUT/$required" ]]; then
    found="$(resolve_soname "$required")"
    if [[ -n "$found" ]]; then
      copy_user_mode "$found"
    fi
  fi
done

# Resolve transitive NEEDED of the staged CUDA libs themselves (same names).
while read -r needed; do
  [[ -n "$needed" ]] || continue
  is_driver "$needed" && continue
  case "$needed" in
    libcudart.so.*|libcublas.so.*|libcublasLt.so.*)
      if [[ ! -e "$OUT/$needed" ]]; then
        found="$(resolve_soname "$needed")"
        [[ -n "$found" ]] || { echo "missing $needed" >&2; exit 1; }
        copy_user_mode "$found"
      fi
      ;;
  esac
done < <(
  for lib in "$OUT"/libcudart.so* "$OUT"/libcublas.so* "$OUT"/libcublasLt.so*; do
    [[ -e "$lib" ]] || continue
    patchelf --print-needed "$lib" 2>/dev/null || true
  done | sort -u
)

for required in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
  if [[ ! -f "$OUT/$required" || -L "$OUT/$required" ]]; then
    echo "overlay missing required CUDA user-mode library $required (need a regular file)" >&2
    ls -la "$OUT" >&2
    exit 1
  fi
done

if ls "$OUT"/libcuda.so* >/dev/null 2>&1; then
  echo "refusing to ship libcuda (driver)" >&2
  exit 1
fi

shopt -s nullglob
for so in "$OUT"/libllamastack.so* "$OUT"/libllama.so* "$OUT"/libggml*.so* \
          "$OUT"/libcudart.so* "$OUT"/libcublas.so* "$OUT"/libcublasLt.so*; do
  [[ -e "$so" ]] || continue
  patchelf --set-rpath '$ORIGIN' "$so"
done

echo "staged CUDA user-mode runtime into $OUT"
