# Changelog

## Unreleased

- Renamed from LittleBitOllama to littlebit-runtime; the Python package from littlebit-gguf to littlebit-tools.
- Engine, converter and C API for Next-Concept Modeling (NCM) models, dense or with LittleBit layers.
- The C API is split into `littlebit.h`, `littlebit_llm.h` and `littlebit_ncm.h`; either engine can be left out of a build.

## 0.1.0

First release: llama.cpp patch (format version 1), `littlebit` command line tool, C API, converter, size planner,
Ollama and Android build scripts. Tested on x86-64, an Android phone and a Wear OS watch.
