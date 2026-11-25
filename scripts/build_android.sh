#!/usr/bin/env bash
# Cross-compile the runtime for Android / Wear OS (arm64-v8a).
#
#   ANDROID_NDK=/path/to/ndk scripts/build_android.sh [arm-arch]
#
# arm-arch defaults to armv8.2-a+dotprod+fp16. CPUs with int8 matrix multiply can use
# armv8.6-a+dotprod+fp16+i8mm. Binaries built for features the CPU lacks crash with SIGILL.
# Output: dist/android-arm64-*/ with littlebit, llama-server (static binaries) and liblittlebit.so.
set -euo pipefail
source "$(dirname "$0")/common.sh"
NDK="${ANDROID_NDK:-${ANDROID_NDK_HOME:-}}"
[ -f "$NDK/build/cmake/android.toolchain.cmake" ] || die "set ANDROID_NDK to an NDK directory (tested: r27c)"
ARCH="${1:-armv8.2-a+dotprod+fp16}"
TAG="android-arm64-${ARCH//[^a-z0-9]/_}"
BUILD="$ROOT/build-$TAG"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM="android-${ANDROID_API:-30}" -DANDROID_STL=c++_static \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="$ARCH" -DGGML_OPENMP=OFF \
    -DLITTLEBIT_TESTS="${LITTLEBIT_TESTS:-OFF}" "${@:2}"
cmake --build "$BUILD" -j "$JOBS"
OUT="$ROOT/dist/$TAG"
mkdir -p "$OUT"
cp "$BUILD/bin/littlebit" "$BUILD/bin/llama-server" "$BUILD"/bin/liblittlebit.so "$OUT/"
STRIP=("$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip)
"${STRIP[0]}" "$OUT"/*
log "built $OUT"
