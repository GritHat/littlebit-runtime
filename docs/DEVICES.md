# Phones and watches

Tested on one phone, a Nothing Phone (4a) Pro (SM7750, 8 GB, Android 17): the test suite passes natively with both
builds below. Not tested on a watch.

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
model matter more than the bit rate of the linear layers. `scripts/quantize_dense.sh` re-quantizes them. The estimate is
for the model process alone; check `MemAvailable` in `/proc/meminfo` on the device.

On x86-64, where nothing is repacked, the same file peaks at 1.53 GB resident with or without memory mapping.

## Build and run

```
ANDROID_NDK=/path/to/android-ndk scripts/build_android.sh
scripts/deploy_android.sh model.gguf                       # benchmark
scripts/deploy_android.sh model.gguf run -p "Hello" -n 32
```

The default target is `armv8.2-a+dotprod+fp16`. Pass `armv8.6-a+dotprod+fp16+i8mm` for CPUs with int8 matrix multiply.
A binary built for features the CPU lacks crashes with an illegal instruction.

Wear OS watches take the same binaries: enable *Developer options* and *Wireless debugging*, then `adb pair` and
`adb connect`, and select the watch with `ANDROID_SERIAL`. An app would bundle `liblittlebit.so` and call
[`littlebit.h`](../include/littlebit.h) through JNI.

## Threads

ggml's workers spin while waiting, so one busy core slows every operation. Qwen3-8B on an 8-core i7-9700K with a
desktop session running:

| threads | 1 | 2 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| decode, tok/s | 5.4 | 9.8 | 16.6 | 18.4 | 6.5 | 2.5 |

By default the runtime leaves one or two cores free, and on CPUs with several core types uses those within 80 % of the
fastest clock. Override with `-t`. Ollama chooses for itself: set `num_thread`.
