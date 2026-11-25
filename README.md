# littlebit-runtime

A runtime for models compressed with [LittleBit](https://github.com/SamsungLabs/LittleBit), whose linear layers take
between 0.1 and 1 bit per weight. It has two engines: one for Llama, Qwen and Gemma language models, which adds LittleBit
layers to [llama.cpp](https://github.com/ggml-org/llama.cpp) and through it to [Ollama](https://github.com/ollama/ollama),
and one for [Next-Concept models](docs/NCM.md). An independent project, not affiliated with Samsung.

**Tested on a phone and a watch.** Qwen3-8B at 0.55 bits per weight runs at about 9 tokens/s on a Nothing Phone (4a) Pro,
from a 1.43 GB file. The same 8-billion-parameter model also runs on a Galaxy Watch Ultra 2, at 6 seconds per token
because it does not fit in memory; an untrained 0.5-billion-parameter model, as a speed test, reaches
42 tokens/s there. See [the measurements](docs/DEVICES.md).

- `littlebit`, a command line tool: run, chat, benchmark, and serve an OpenAI-compatible API
- `liblittlebit`, a shared library with a small [C API](include) for embedding
- `littlebit-convert`, which turns LittleBit checkpoints into GGUF
- `littlebit-size`, which estimates what fits on a device
- scripts to build it for Android and into Ollama
- `littlebit ncm` and `littlebit-ncm-convert`, for Next-Concept models (Sudoku and TinyStories), dense or with LittleBit layers

The 8B file decodes at 18 to 20 tokens/s on a 2018 desktop CPU.

> **Status: early.** Tested on x86-64 CPUs, an Android phone and a Wear OS watch. No GPU has been tried.
> A LittleBit model comes from training, not conversion: see [Models](#models).

## Install

Needs CMake 3.18+, a C++17 compiler, git, and Python 3.10+ for conversion.

```
cmake -B build                       # downloads llama.cpp b12257 and applies the patch
cmake --build build -j
ctest --test-dir build               # optional; needs numpy, pyyaml, tqdm, requests and pytest
scripts/setup_python.sh              # converter, in .venv
```

`cmake --install build --prefix ...` installs the tools, the library and its header.

## Use

```
littlebit-convert path/to/checkpoint --outtype bf16 --outfile model-bf16.gguf
scripts/quantize_dense.sh model-bf16.gguf model.gguf         # embedding Q4_K, lm_head Q6_K

build/bin/littlebit run   model.gguf -p "The capital of France is" --temp 0
build/bin/littlebit chat  model.gguf
build/bin/littlebit serve model.gguf --port 8080
```

Checkpoints without tokenizer files need `--tokenizer-dir path/to/base-model`.

**Ollama:** `scripts/setup_ollama.sh` (needs Go) builds an Ollama with the patch; `scripts/ollama_create.sh model.gguf
name` registers a model with it. A stock Ollama cannot load these files.

**Android and Wear OS:** see [docs/DEVICES.md](docs/DEVICES.md).

## Models

LittleBit models are produced by quantization-aware training with
[Samsung's code](https://github.com/SamsungLabs/LittleBit). The converter reads its checkpoints for Llama, Qwen2/2.5,
Qwen3, Gemma 2 and Gemma 3. The runtime refuses other architectures.

The trained model used for validation is a third-party upload,
[LittleBitLLM/littlebit-qwen3-8b-0.55bpw](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw)
(CC BY-NC 4.0). It continues plain text but does not answer chat prompts, here or in Samsung's implementation.

`tools/llm/make_littlebit_checkpoint.py` runs only the initialization step of LittleBit. Its output exercises the runtime
but is not a usable model.

## How it works

LittleBit replaces a weight matrix by two low-rank sign matrices and a few scale vectors:
`y = u1 ⊙ U((v1 ⊙ u2) ⊙ V(v2 ⊙ x))`, optionally plus a second such branch. The sign matrices are stored as ggml's existing
`Q1_0` type with `u1` and `v1 ⊙ u2` folded into its block scales, so a branch costs one elementwise product and its
matrix products run on ggml's optimized 1-bit kernels. The [patch](patches/llama.cpp) hooks the two functions through
which llama.cpp creates layer weights and multiplies by them. [docs/FORMAT.md](docs/FORMAT.md) specifies the files.

## Validation

**Trained model.** Qwen3-8B, 0.55 bpw, against Samsung's PyTorch code on the same 8 × 512 tokens of wikitext-2
(`tests/llm/test_trained_qwen3_8b.sh`):

| | PyTorch, fp32 | this runtime |
|---|---|---|
| perplexity, bf16 embedding and lm_head | 32.91 | 32.90 |
| perplexity, Q4_K embedding and Q6_K lm_head | | 33.13 |
| KL divergence per token | | 0.0015 mean, 0.013 p99 |
| same top token | | 97.8 % |

The 252 linear layers (6.95 billion weights) take 536 MiB: 0.65 bits per weight, above the 0.55 target because ranks are
padded to 128 and blocks carry a scale.

**Untrained initializations** of real models, made and evaluated with Samsung's code (`tests/llm/run_all.sh`). KL divergence
from Samsung's PyTorch output:

| model | bits per weight | dense F32 weights | F32 factors | Q1_0 factors |
|---|---|---|---|---|
| SmolLM2-135M (llama) | 1.0, residual | 0.00008 | 0.00006 | 0.12 |
| SmolLM2-135M (llama) | 0.1, residual | 0.0009 | 0.0002 | 0.10 |
| Qwen2.5-0.5B | 0.55, residual | 0.000002 | 0.000001 | 0.002 |
| Gemma-3-270M | 1.0 | 0.031 | 0.011 | 9.9 |

With F32 factors the runtime follows Samsung's code at least as closely as ordinary dense weights do. `Q1_0` adds 8-bit
activations, as every quantized llama.cpp format does on the CPU: about 1 % error per layer (at most 1.2 %). All 714 layers match
a numpy model of exactly that rounding to 1.3e-3 or better, most to 1e-7. How much it matters depends on the model:
little for the trained Qwen3-8B, a lot for the untrained Gemma. No trained LittleBit Gemma or Llama was available to test.

**Test suite.** `ctest` checks tiny random models of all five architectures against their dense equivalents, every
kernel, plain and repacked, against numpy, rejection of malformed files, the C API and the command line. It passes on
x86-64, on Linux and macOS ARM64 in CI, and natively on an Android phone and a Wear OS watch. An earlier version of the suite also passed
under address and undefined-behaviour sanitizers.

**Between machines.** Greedy output for a 48-token continuation was identical on the PC and the phone, but that is not
guaranteed: moving every input scale by one fp32 rounding step changes the logits by 1.2 % on Qwen2.5-0.5B, as much as
changing CPU or kernel set does. Each variant is equally far (1.1 %) from the result without 8-bit rounding.

## Layout

| | shared | language models (llama.cpp) | Next-Concept models (ggml) |
|---|---|---|---|
| engine | `src/core` | `src/llm` | `src/ncm` |
| C API | `littlebit.h` | `littlebit_llm.h` | `littlebit_ncm.h` |
| converter | `python/littlebit_gguf/{q1,ckpt}.py` | `convert.py` | `ncm.py` |
| tests | `tests/core` | `tests/llm` | `tests/ncm` |
| training and reference code | | `tools/llm` | `tools/ncm` |

The engines depend on the shared part and not on each other. `-DLITTLEBIT_LLM=OFF` or `-DLITTLEBIT_NCM=OFF` leaves one
out: a command line tool with only the NCM engine is 2.3 MB on x86-64. `tools/` may import Samsung's CC BY-NC code; nothing
else does, apart from comparison tests.

## Limitations

- GPU backends have `Q1_0` kernels upstream, but none has been tested with LittleBit layers.
- Phi, OPT and mixture-of-experts models are not supported, nor are LoRA adapters on LittleBit layers.
- `littlebit chat` uses llama.cpp's built-in chat templates; `littlebit serve` supports Jinja templates.
- The patch is tied to llama.cpp b12257, the version Ollama pinned when this was written.

## License

[Apache-2.0](LICENSE), see [NOTICE](NOTICE); the llama.cpp patch is [MIT](patches/llama.cpp/LICENSE). Samsung's LittleBit code and models trained with it are CC BY-NC 4.0, which forbids commercial
use; see [THIRD_PARTY.md](THIRD_PARTY.md). Not affiliated with Samsung, Ollama or ggml-org.
