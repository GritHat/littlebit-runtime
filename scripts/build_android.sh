#!/usr/bin/env bash
# Cross-compile the runtime for Android and Wear OS.
#
#   ANDROID_NDK=/path/to/ndk scripts/build_android.sh [arm-arch] [cmake options]
#   ANDROID_NDK=/path/to/ndk WATCH=1 scripts/build_android.sh       # statically linked command line tool
#
# arm-arch defaults to armv8.2-a+dotprod+fp16. CPUs with int8 matrix multiply can use armv8.6-a+dotprod+fp16+i8mm.
# Binaries built for features the CPU lacks crash with SIGILL.
#
# Wear OS watches often have a 64-bit kernel but only 32-bit system libraries. WATCH=1 builds a static arm64 binary that
# runs there from adb shell. ANDROID_ABI=armeabi-v7a builds for 32-bit processes, such as apps on those watches; it is
# several times slower.
#
# Output: dist/android-*/
set -euo pipefail
source "$(dirname "$0")/common.sh"
NDK="${ANDROID_NDK:-${ANDROID_NDK_HOME:-}}"
[ -f "$NDK/build/cmake/android.toolchain.cmake" ] || die "set ANDROID_NDK to an NDK directory (tested: r27c)"
ABI="${ANDROID_ABI:-arm64-v8a}"
OPTS=()
case $ABI in
    arm64-v8a)
        ARCH="${1:-armv8.2-a+dotprod+fp16}"
        OPTS+=("-DGGML_CPU_ARM_ARCH=$ARCH") ;;
    armeabi-v7a)
        ARCH=armv7
        OPTS+=(-DANDROID_ARM_NEON=ON -DGGML_LLAMAFILE=OFF) ;; # llamafile's kernels do not compile for 32-bit ARM
    *) die "unsupported ANDROID_ABI $ABI" ;;
esac
TAG="android-$ABI-${ARCH//[^a-z0-9]/_}"
[ "$ABI" = arm64-v8a ] && TAG="android-arm64-${ARCH//[^a-z0-9]/_}"
if [ "${WATCH:-0}" = 1 ]; then
    TAG+=-static
    OPTS+=(-DCMAKE_EXE_LINKER_FLAGS=-static -DLITTLEBIT_SERVER=OFF -DLITTLEBIT_SHARED_LIB=OFF)
fi
BUILD="$ROOT/build-$TAG"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-${ANDROID_API:-30}" -DANDROID_STL=c++_static \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLITTLEBIT_TESTS="${LITTLEBIT_TESTS:-OFF}" "${OPTS[@]}" "${@:2}"
cmake --build "$BUILD" -j "$JOBS"
OUT="$ROOT/dist/$TAG"
mkdir -p "$OUT"
for f in littlebit llama-server liblittlebit.so; do
    if [ -f "$BUILD/bin/$f" ]; then cp "$BUILD/bin/$f" "$OUT/"; fi
done
STRIP=("$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip)
"${STRIP[0]}" "$OUT"/*
log "built $OUT"
