# LittleBit GGUF format (version 1)

A LittleBit GGUF is an ordinary llama.cpp GGUF of the base architecture (`llama`, `qwen2`, `qwen3`,
`gemma2` or `gemma3`; others are rejected). The only difference is that a dense 2D weight `<base>.weight` may be **absent**, with
binary latent factors stored next to it instead. Tensors that LittleBit leaves dense (token embedding,
`output`/lm_head, norms, biases) are unchanged and may use any GGML type.

## The layer

LittleBit (`quantization/modules/littlebit.py`) replaces `W [out x in]` by

```
y = u1 ⊙ ( U_b · ( (v1 ⊙ u2) ⊙ ( V_b · (v2 ⊙ x) ) ) )          U_b ∈ {±1}^{out×r},  V_b ∈ {±1}^{r×in}
  + u1_R ⊙ ( U_R,b · ( (v1_R ⊙ u2_R) ⊙ ( V_R,b · (v2_R ⊙ x) ) ) )   (only with --residual)
```

## Tensors

For a LittleBit layer whose dense weight would be `<base>.weight` with GGML shape `[in, out]`:

| tensor            | type | GGML shape        | content |
|-------------------|------|-------------------|---------|
| `<base>.lb_v2`    | F32  | `[in]`            | `v2 · β` |
| `<base>.lb_v`     | Q1_0 | `[in_pad, r]`     | row *k*: signs of `V_b[k,:]`, block scale `d = (v1·u2)[k] · α / β` |
| `<base>.lb_v2_r`  | F32  | `[in]`            | residual branch, same as above |
| `<base>.lb_v_r`   | Q1_0 | `[in_pad, r]`     | residual branch, same as above |
| `<base>.lb_u`     | Q1_0 | `[n_br·r_pad, out]` | row *j*: `[signs U_b[j,:] ‖ signs U_R,b[j,:]]`, block scales `u1[j]/α` and `u1_R[j]/α_R` |

* `Q1_0` is upstream GGML's 1-bit type: blocks of 128 values, `{fp16 d; uint8 qs[16]}`, LSB first,
  bit = 1 → `+d`, bit = 0 → `−d`. (LittleBit's own packer uses bit = 1 → −1; the converter inverts.)
* `in_pad = ceil(in/128)·128`, `r_pad = ceil(r/128)·128`, `n_br` = 1 or 2 (residual).
  Padded activations are exactly zero at run time (`ggml_pad`), so padded sign bits are irrelevant.
  Each branch occupies whole Q1_0 blocks of an `lb_u` row, which is why the two branches can have
  different per-row scales.
* `α` and `β` are the powers of two that bring `max|u1|` and `max|(v1·u2)·α|` into [1, 2) (per branch). The
  compensating factor lives in the fp32 `lb_v2`, so the layer function is unchanged.
* `lb_v`, `lb_v_r` and `lb_u` may also be `F16` or `F32` (`littlebit-convert --factor-type`), with padded entries zero.
  This is far larger and meant as a reference: `Q1_0` matmuls quantize activations to 8 bits on the CPU.
* Architecture-specific row permutations (e.g. llama's RoPE q/k permutation) are applied to the rows of
  `U` (and `u1`); the converter derives them by running the architecture's own converter on an index
  vector.

Run-time graph (`src/llama-littlebit.cpp`):

```
h   = mul_mat(lb_v,   pad(x ⊙ lb_v2,   in_pad))     → pad(h,   r_pad)
h_r = mul_mat(lb_v_r, pad(x ⊙ lb_v2_r, in_pad))     → pad(h_r, r_pad)   (residual)
y   = mul_mat(lb_u, concat(h, h_r))
```

Size: `n_br·(r·in_pad·1.125 + 32·in + out·r_pad·1.125)` bits.

## Metadata

| key | type | meaning |
|-----|------|---------|
| `littlebit.version`    | u32  | `1`, required; other values are rejected |
| `littlebit.block_size` | u32  | `128` |
| `littlebit.eff_bit`    | f32  | target bits/weight used for training (informational) |
| `littlebit.residual`   | bool | informational |
| `littlebit.itq`        | bool | LittleBit-2 Joint-ITQ init (informational) |
| `littlebit.quant_func` | str  | informational |

The runtime detects LittleBit layers by the presence of `<base>.lb_u`; all metadata except `littlebit.version` is
informational.

## Loader behaviour (patch)

`llama_model_base::create_tensor()` – the single choke point through which every architecture creates its
weights – checks, for each requested 2D `*.weight`, whether `<base>.lb_u` exists. If so, it creates the
factor tensors (normal buffer-type selection, mmap, offload) and returns a data-less *proxy* tensor of
the dense shape. `llm_graph_context::build_lora_mm()` – the single choke point for weight matmuls – maps
the proxy back to its factors. A file containing both a dense weight and factors for the same layer is
rejected.
