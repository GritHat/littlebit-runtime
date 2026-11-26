# Sudoku on a Wear OS watch

A minimal Wear OS app that solves Sudoku puzzles on the watch with a Next-Concept model and shows the result:
the given digits in white, the model's in blue (red where they are wrong). A tap solves the next of 200 easy test
puzzles; a long press switches between the LittleBit model (0.55 bits per weight, 6.8 MB) and the dense one (Q8_0, 31 MB).

Measured on a Galaxy Watch Ultra 2, one thread: 0.44 to 0.62 s per puzzle with LittleBit, 0.53 to 0.76 s with the dense
model, all puzzles of the recording solved.

## Build

No Gradle or Android Studio: the Android platform and build tools, the static command line tool and two models.

```
WATCH=1 ANDROID_NDK=/path/to/ndk scripts/build_android.sh
BUILD_TOOLS=/path/to/build-tools/34.0.0 ANDROID_JAR=/path/to/platforms/android-34/android.jar \
    examples/wear-sudoku/build.sh                                  # writes build/wear-sudoku/littlebit-sudoku.apk
adb install build/wear-sudoku/littlebit-sudoku.apk
```

The models default to `models/ncm/easy-lb0.55.gguf` and `models/ncm/easy-dense-q8.gguf` (see [NCM.md](../../docs/NCM.md);
`LB_MODEL` and `DENSE_MODEL` override them), the puzzles to the official test set in `models/ncm/data/sudoku/easy`. The
APK is signed with a debug key made on first use. The build-tools' `d8` fails on inner classes compiled by javac 24, so
the app has none.

## How it works

Apps on this watch are 32-bit, but its kernel runs 64-bit programs, and the 32-bit build of the runtime is several times
slower. So the APK carries the static 64-bit `littlebit` tool as `lib/armeabi-v7a/liblittlebit_cli.so`; Android
extracts it to the app's native library directory, from which apps may execute files, and the app runs
`littlebit ncm model.gguf -t 1` there, writing puzzles to its input and reading solutions from its output. Other apps
should link `liblittlebit` and use [littlebit_ncm.h](../../include/littlebit_ncm.h) instead.

The LittleBit model was trained with Samsung's CC BY-NC 4.0 code, so an APK containing it is for non-commercial use.
