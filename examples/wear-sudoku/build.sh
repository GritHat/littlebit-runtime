#!/usr/bin/env bash
# Build the Wear OS demo without Gradle:
#   BUILD_TOOLS=.../build-tools/34.0.0 ANDROID_JAR=.../platforms/android-34/android.jar examples/wear-sudoku/build.sh
# Needs the static arm64 command line tool (WATCH=1 scripts/build_android.sh), two Sudoku models and the test puzzles.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
: "${BUILD_TOOLS:?set BUILD_TOOLS}" "${ANDROID_JAR:?set ANDROID_JAR}"
BIN=${BIN:-$ROOT/dist/android-arm64-armv8_2_a_dotprod_fp16-static/littlebit}
LB=${LB_MODEL:-$ROOT/models/ncm/easy-lb0.55.gguf}; DENSE=${DENSE_MODEL:-$ROOT/models/ncm/easy-dense-q8.gguf}
PUZZLES=${PUZZLES:-$ROOT/models/ncm/data/sudoku/easy}
OUT=${OUT:-$ROOT/build/wear-sudoku}; KEY=${KEY:-$OUT/debug.keystore}
rm -rf "$OUT/stage" "$OUT/classes" && mkdir -p "$OUT/stage/assets" "$OUT/stage/lib/armeabi-v7a" "$OUT/classes"
# Wear OS apps on this watch are 32-bit, but its kernel runs the static 64-bit tool, which is several times faster.
cp "$BIN" "$OUT/stage/lib/armeabi-v7a/liblittlebit_cli.so"
cp "$LB" "$OUT/stage/assets/easy-lb0.55.gguf"; cp "$DENSE" "$OUT/stage/assets/easy-dense-q8.gguf"
python3 - "$PUZZLES" "$OUT/stage/assets/puzzles.txt" <<'PY'
import pickle, sys
class Safe(pickle.Unpickler):
    def find_class(self, m, n): raise pickle.UnpicklingError(f"blocked {m}.{n}")
d = sys.argv[1]
puz, sol = (Safe(open(f"{d}/{f}", "rb")).load()[:200] for f in ("puzzles.pickle", "solutions.pickle"))
g = lambda t: "".join(str(int(x)) for x in t if int(x) <= 9)
open(sys.argv[2], "w").write("".join(f"{g(p)} {g(s)}\n" for p, s in zip(puz, sol)))
PY
"$BUILD_TOOLS/aapt2" compile --dir "$HERE/res" -o "$OUT/res.zip"
"$BUILD_TOOLS/aapt2" link -o "$OUT/base.apk" -I "$ANDROID_JAR" --manifest "$HERE/AndroidManifest.xml" \
    --min-sdk-version 30 --target-sdk-version 34 -A "$OUT/stage/assets" -0 gguf "$OUT/res.zip"
javac --release 11 -encoding UTF-8 -nowarn -cp "$ANDROID_JAR" -d "$OUT/classes" $(find "$HERE/src" -name '*.java')
"$BUILD_TOOLS/d8" --min-api 30 --lib "$ANDROID_JAR" --output "$OUT" $(find "$OUT/classes" -name '*.class')
cp "$OUT/base.apk" "$OUT/unsigned.apk"
(cd "$OUT" && zip -q -j unsigned.apk classes.dex && cd stage && zip -q -0 -r ../unsigned.apk lib)
"$BUILD_TOOLS/zipalign" -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
[ -f "$KEY" ] || keytool -genkeypair -keystore "$KEY" -storepass android -keypass android -alias debug -keyalg RSA \
    -keysize 2048 -validity 10000 -dname "CN=Android Debug" > /dev/null 2>&1
"$BUILD_TOOLS/apksigner" sign --ks "$KEY" --ks-pass pass:android --out "$OUT/littlebit-sudoku.apk" "$OUT/aligned.apk"
echo "$OUT/littlebit-sudoku.apk"
