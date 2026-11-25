# Phones and watches

Nothing here has run on a physical phone or watch yet. What has been done: the Android binaries build with NDK r27c,
and the test suite passes on ARM64 under QEMU (Cortex-A76 for the default build, `max` for the i8mm build). Emulation
proves correctness, not speed.

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

Measured on x86-64: the 1.43 GB Qwen3-8B file peaks at 1.53 GB resident with a 512-token context, with and without
memory mapping. Mapped pages are file-backed, so the kernel can drop them under memory pressure, at the price of reading
them again. On ARM the factors are copied into a repacked layout for speed; `--no-repack` keeps them mapped.

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

ggml's workers spin while waiting, so one busy core slows every operation. Qwen3-8B on an 8-core i7-9700K with a desktop
session running:

| threads | 1 | 2 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| decode, tok/s | 5.4 | 9.8 | 16.6 | 18.4 | 6.5 | 2.5 |

By default the runtime leaves one or two cores free, and on CPUs with several core types uses those within 80 % of the
fastest clock. Override with `-t`. Ollama chooses for itself: set `num_thread`.
