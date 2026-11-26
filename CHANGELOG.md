# Changelog

## Unreleased

- Renamed from LittleBitOllama to littlebit-runtime; the Python package from littlebit-gguf to littlebit-tools.
- Engine, converter and C API for Next-Concept Modeling (NCM) models, dense or with LittleBit layers.
- The C API is split into `littlebit.h`, `littlebit_llm.h` and `littlebit_ncm.h`; either engine can be left out of a build.
- LittleBit Sudoku NCM results at 0.1 to 1 bit per weight, with three seeds and a rounding baseline (`tools/ncm/ptq_sudoku.py`);
  `train_sudoku.py` gains `--val`, `--resume` and `--state-every`.
- A Qwen3-0.6B trained at 0.55 bits per weight, measured on the watch.

## 0.1.0

First release: llama.cpp patch (format version 1), `littlebit` command line tool, C API, converter, size planner,
Ollama and Android build scripts. Tested on x86-64, an Android phone and a Wear OS watch.
