#!/usr/bin/env bash
# Push a model and the Android binaries to a phone or Wear OS watch over adb, and run it.
#
#   scripts/deploy_android.sh model.gguf [littlebit arguments, default: bench]
#   scripts/deploy_android.sh model.gguf run -p "The capital of France is" -n 32
#
# Select a device with ANDROID_SERIAL and another build with BIN=dist/android-... A watch may suspend the program when
# its screen turns off; keep it awake and on the charger.
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || die "usage: $0 model.gguf [command and options]"
MODEL=$1; shift
BIN="${BIN:-$ROOT/dist/android-arm64-armv8_2_a_dotprod_fp16}"
[ -x "$BIN/littlebit" ] || die "$BIN/littlebit not found; run scripts/build_android.sh"
[ $# -gt 0 ] || set -- bench
DEV=/data/local/tmp/littlebit
NAME=$(basename "$MODEL")
adb shell mkdir -p "$DEV"
adb push "$BIN/littlebit" "$DEV/"
SIZE=$(wc -c < "$MODEL" | tr -d ' ')
if [ "$(adb shell "cat '$DEV/$NAME' 2>/dev/null | wc -c" | tr -d ' \r')" != "$SIZE" ]; then
    adb push "$MODEL" "$DEV/$NAME"
fi
adb shell "grep -E 'MemTotal|MemAvailable' /proc/meminfo"
CMD=$1; shift
ARGS=""
for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done
adb shell "cd $DEV && chmod 755 littlebit && ./littlebit $CMD $(printf '%q' "$NAME")$ARGS"
