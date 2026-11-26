# Phones and watches

Tested on one phone, a Nothing Phone (4a) Pro (SM7750, 8 GB, Android 17): the test suite passes natively with both
builds below. Also tested on a Galaxy Watch Ultra 2 (SM-L715F, 1.7 GB RAM, Android 17, about 0.4 GB free), where the
suite passes natively as a 64-bit and as a 32-bit build.

## Measured: Qwen3-8B at 0.55 bits per weight, 1.43 GB, on the phone

| threads | 1 | 2 | 3 | 4 | 5 (default) | 6 | 8 |
|---|---|---|---|---|---|---|---|
| prompt, tok/s | 13.7 | 22.6 | 32.9 | 43.1 | 50.6 | 37.5 | 32.2 |
| decode, tok/s | 4.0 | 5.7 | 7.5 | 8.9 | 8.9 | 7.4 | 8.2 |

Single runs with the i8mm build and 128 + 64 tokens; the dotprod build decodes as fast and reads prompts about 10 %
slower. The battery went from 25 to 35 degrees C over the few minutes of benchmarks, without thermal throttling being
reported. Longer sessions were not tested.

| | decode, tok/s | anonymous memory | peak RSS |
|---|---|---|---|
| default | 9 | 1.13 GB | 2.4 GB for a second while loading, then 2.0 GB |
| `--no-mmap` | 9 | 1.53 GB | 1.5 GB |
| `--no-repack` | 5 | 0.13 GB | 1.5 GB |

Anonymous memory is what the system cannot drop; the rest is file-backed and can be read again, which costs speed.
The default therefore leaves more memory available than `--no-mmap` despite its higher RSS. `--no-repack` is the option
for devices with very little memory.

## Measured on the watch

| model | threads | prompt, tok/s | decode, tok/s |
|---|---|---|---|
| Qwen3-0.6B, trained, 0.55 bpw, 127 MB, 64-bit | 1 (default) | 95 | 32 |
| same | 4 | 72 to 104 | 32 |
| same | 5 | 72 to 88 | 9 to 22 |
| same, `--no-repack` | 1 | 20 | 15 |
| Qwen2.5-0.5B shape, 110 MB, 64-bit | 1 (default) | 131 | 42 |
| same, `--no-repack` | 1 | 26 | 17 |
| same, 32-bit build | 1 | 8 | 3.7 |
| same, 32-bit build | 4 | 15 | 6.7 |
| Qwen3-8B, 1.27 GB, `--no-repack` | 5 | 0.35 | 0.16 |

The Qwen3-0.6B rows are three runs each of 128 + 64 tokens (one for `--no-repack`). A 400-token run sustained 29.8 tok/s.
It peaked at 531 MB resident, 405 MB of it anonymous, with 327 MB reported available before the run, and its greedy
output for a 40-token prompt matched the PC's. The model is described under [Training a small model](#training-a-small-model).

The Qwen2.5-0.5B-shaped model is an untrained initialization: its speed is real, its text is not. A 400-token run sustained 39.6 tok/s.
The 8B continued a prompt correctly (12 tokens) with less than 0.1 GB of anonymous memory, but each token reads
870 MB of weights and only about 0.6 GB could be cached, so it is limited by the storage (roughly 170 MB/s). For usable
speed the weights read per token must fit in free memory. No published trained LittleBit model of that size was found.

The battery stayed below 34 degrees C throughout, on the charger. It suspends running programs when the screen turns off, which stalls a
run started over adb.

## Training a small model

Qwen3-0.6B was trained at 0.55 bits per weight with `tools/llm/train.py` on one rented RTX 4090:

```
python tools/llm/train.py --littlebit LittleBit --out out --eff-bit 0.55 --steps 20000 --docs 400000
```

20,000 steps of 2 × 2048 tokens (164 million tokens of C4 and wikitext-2 train) took 8 hours; the rented machine ran for
8.6 hours at $0.89 per hour, setup and a pilot included.

| | wikitext-2 test perplexity |
|---|---|
| Qwen3-0.6B, bf16 | 20.97 |
| LittleBit 0.55 bpw, PyTorch | 54.79 |
| same, 8 × 512 tokens: PyTorch / this runtime / with Q4_K embedding and Q6_K lm_head | 70.08 / 70.11 / 72.55 |

The 196 LittleBit layers take 36.9 MiB (0.70 bpw as stored); the file is 127 MB, most of it the embedding and lm_head
of the 151,936-token vocabulary. The runtime matches PyTorch to a KL divergence of 0.0004 per token (98.4 % same top
token) with a bf16 embedding and lm_head, 0.021 (89.5 %) with the quantized ones.

The model writes English sentences but no correct facts: greedy decoding repeats itself ("The capital of France is a
country that is a country that is ..."), sampling does not, and neither answers questions. It is a measure of speed and
memory, not a usable model. [mgiachetti/little-bit](https://github.com/mgiachetti/little-bit) reports 36.4 for the same
model and rate after 107 million tokens, with a distillation loss adapted to Qwen3's attention sinks.

## What fits

`littlebit-size` adds up weights, KV cache and a rough compute buffer:

```
$ littlebit-size llama-3.1-8b --sweep --residual --ram 2G
llama-3.1-8b: Arch(hidden=4096, ffn=14336, layers=32, heads=32, kv_heads=8, head_dim=128, vocab=128256, tied=False)
dense f16 weights: 14.96 GiB; ctx=2048 kv=q8_0 emb=q4_k out=q4_k residual=True
eff_bit   bpw*   linear    embed   output  weights       kv    total  MiB   fits
   1.00  1.153    959.6    281.8    281.8   1524.2    136.0   1689.2   yes
   0.70  0.810    673.7    281.8    281.8   1238.3    136.0   1403.3   yes
   0.55  0.636    529.4    281.8    281.8   1094.0    136.0   1259.0   yes
   0.30  0.359    298.5    281.8    281.8    863.1    136.0   1028.1   yes
   0.10  0.152    126.4    281.8    281.8    691.0    136.0    856.0   yes
* bpw of the factorized linears as stored (includes fp16 block scales and 128-padding of the latent rank)
```

At these rates the dense token embedding and lm_head dominate, so their quantization and the vocabulary size of the base
model matter more than the bit rate of the linear layers. `scripts/quantize_dense.sh` re-quantizes them. For Qwen3-8B, whose
perplexity is 33.1 with a Q6_K lm_head, the lm_head types Q4_K, Q3_K and Q2_K (the last with a Q2_K embedding) give
33.7, 35.8 and 44.3 on the same 8 x 512 tokens. The estimate is
for the model process alone; check `MemAvailable` in `/proc/meminfo` on the device.

On x86-64, where nothing is repacked, the same file peaks at 1.53 GB resident with or without memory mapping.

## Build and run

A Wear OS app that runs a Sudoku model on the watch is in [examples/wear-sudoku](../examples/wear-sudoku).

```
ANDROID_NDK=/path/to/android-ndk scripts/build_android.sh
scripts/deploy_android.sh model.gguf                       # benchmark
scripts/deploy_android.sh model.gguf run -p "Hello" -n 32
```

The default target is `armv8.2-a+dotprod+fp16`. Pass `armv8.6-a+dotprod+fp16+i8mm` for CPUs with int8 matrix multiply.
A binary built for features the CPU lacks crashes with an illegal instruction.

Wear OS watches often have a 64-bit kernel but only 32-bit system libraries. `WATCH=1 scripts/build_android.sh` builds a
static 64-bit binary that runs there all the same; run it with
`BIN=dist/android-arm64-armv8_2_a_dotprod_fp16-static scripts/deploy_android.sh ...`. Apps on such watches are 32-bit and
would need `ANDROID_ABI=armeabi-v7a` and its much lower speed. To connect, enable *Developer options* and *Wireless debugging*, then `adb pair` and
`adb connect`, and select the watch with `ANDROID_SERIAL`. A phone app would bundle `liblittlebit.so` and call
[`littlebit_llm.h` or `littlebit_ncm.h`](../include) through JNI.

## Threads

ggml's workers spin while waiting, so one busy core slows every operation. Qwen3-8B on an 8-core i7-9700K with a
desktop session running:

| threads | 1 | 2 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| decode, tok/s | 5.4 | 9.8 | 16.6 | 18.4 | 6.5 | 2.5 |

By default the runtime leaves one or two cores free, and on CPUs with several core types uses those within 80 % of the
fastest one's capacity. Override with `-t`. Ollama chooses for itself: set `num_thread`.
