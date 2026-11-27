# Next-Concept Modeling

[Next-Concept Modeling](https://github.com/HazyResearch/ncm) (arXiv 2610.14162) generates a whole text at once: a transformer
predicts one level of discrete codes after another, from coarse to fine, all codes of a level in parallel, and a small
decoder turns the sum of the code embeddings into tokens. llama.cpp generates one token after another and cannot run such
models, so this runtime has an engine of its own for them (`src/ncm`), on ggml and with the same LittleBit layers.

Supported: the Sudoku and TinyStories models. The converter refuses the others (LM1B, OpenWebText, WMT14, XSum), which use
further variations of the architecture. CPU only.

## Use

```
littlebit-ncm-convert --checkpoint-dir artifacts/sudoku/easy --config ncm/config/eval/sudoku-easy.yaml --outfile sudoku.gguf
littlebit ncm sudoku.gguf -p 53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79

littlebit-ncm-convert --checkpoint-dir artifacts/tinystories --config ncm/config/eval/tinystories.yaml --outfile stories.gguf
littlebit ncm stories.gguf --temp 0.7 --seed 1
```

`--config` matters: the `config.yaml` files published with the Sudoku checkpoints do not enable the soft decisions that the
official evaluation uses, its `config/eval/*.yaml` do. `--student student.pt` takes the generator from a LittleBit state
dict (`tools/ncm/train_sudoku.py`). In C, see `littlebit_ncm_*` in [littlebit_ncm.h](../include/littlebit_ncm.h). A sudoku answer is
the model's output and is not checked.

Weight matrices may also be in ggml's other block formats (Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K). The converter
writes at most Q8_0; `tools/ncm/requant.py` re-quantizes its output. TinyStories in Q4_0 is 113 MB instead of 162 MB, with
a peak RSS of 220 MB instead of 268 MB on x86-64 (4 threads, about the same speed); its stories read like those of the
Q8_0 file (compared by eye on the same seeds, no metric).

## Validation

`tests/ncm/test_ncm.py` compares tiny random models, with dense and with LittleBit layers, to a numpy implementation.
`tools/ncm/compare.py` compares a Sudoku model with the official PyTorch code; TinyStories was compared in the same way.
Relative error of the logits of each level, on the CPU:

| model | weights | error | same codes |
|---|---|---|---|
| Sudoku easy, dense | F32 | at most 2e-6 | |
| Sudoku easy, dense | Q8_0 | at most 1.4 % | |
| Sudoku easy, LittleBit | F32 factors | at most 1.3e-6 | |
| Sudoku easy, LittleBit | Q1_0 factors, Q8_0 rest | at most 1.3 % | |
| TinyStories, dense | F32 | at most 8e-6 | 100 % |
| TinyStories, dense | Q8_0 | 0.3 to 3.3 % | 97 to 100 % per level |

The Sudoku and TinyStories rows are two to four puzzles and one greedy text. Sudoku models use soft decisions, so there are
no codes to compare. On the 2,000 test puzzles (`tools/ncm/score.py`):

| Sudoku | file | this runtime | PyTorch |
|---|---|---|---|
| easy, dense, Q8_0 (PyTorch fp32) | 31 MB | 99.60 % | 99.55 % |
| easy, LittleBit 0.975 bpw | 8.6 MB | 98.30 % | 98.30 % |
| easy, LittleBit 0.545 bpw | 6.8 MB | 96.25 % | 96.35 % |
| easy, LittleBit 0.277 bpw | 6.4 MB | 88.00 % | 88.00 % |
| easy, LittleBit 0.188 bpw | 6.2 MB | 67.80 % | 68.15 % |
| easy, LittleBit 0.125 bpw | 6.1 MB | 51.10 % | 51.05 % |
| easy, LittleBit 0.545 bpw after 200 of 2,000 training steps | 6.8 MB | 86.50 % | 86.60 % |
| medium, dense, Q8_0 | 31 MB | 97.85 % | 97.75 % |
| medium, LittleBit 0.545 bpw | 6.8 MB | 81.30 % | 81.00 % |
| hard, dense, Q8_0 | 31 MB | 90.15 % | 89.95 % |
| hard, LittleBit 0.545 bpw | 6.8 MB | 23.25 % | 23.35 % |

## LittleBit Sudoku models

`tools/ncm/train_sudoku.py` trains the 32 linear layers of the generator's transformer blocks (25.2 million of its
25.5 million weights) as LittleBit layers, distilling from the dense model on puzzles it generates; the published test
puzzles are only used for scoring. The input projection, embeddings, heads and the tokenizer stay dense, 4.5 MB in Q8_0,
which is why the files differ little in size. All runs on one RTX 2080 SUPER, batch 64, 2,000 steps (47 to 63 minutes).
Puzzles solved out of the 2,000 test puzzles, in PyTorch:

| bpw (target) | learning rate | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|---|
| 0.975 (1.0) | 2e-4 | 98.30 % | 98.00 % | 98.25 % | 98.18 % |
| 0.545 (0.55) | 2e-4 | 96.35 % | 95.95 % | 96.55 % | 96.28 % |
| 0.277 (0.3) | 1e-3 | 88.00 % | 87.80 % | 88.20 % | 88.00 % |
| 0.188 (0.2) | 1e-3 | 68.15 % | 64.90 % | 68.55 % | 67.20 % |

Easy puzzles. The seeds change the initialization and the order of training data, not the 20,000 generated grids.
At 0.1 the rank is clamped to 8 and the rate is 0.125. With 2,000 steps at 2e-4 it solved no puzzle (82 % of cells
right); 600-step probes, scored by the cells right in 500 held-out generated puzzles, chose 1e-3 over 2e-4 and 3e-3
and the hidden-state loss over none, and 10,000 steps then reached 51.05 %. That run stayed between 35 and 38 % from step 4,000 to 9,000 and
gained the rest in the last 1,000 steps, as the learning rate went to zero. Every run was still improving at its end.
Medium and hard 0.55 (one seed, 2e-4) reached 81.00 % and 23.35 %, against 97.75 % and 89.95 % for the dense models.

**Against rounding.** `tools/ncm/ptq_sudoku.py` rounds the same layers of the dense model without training: per block
of 32 (or 16) weights along a row, with an fp16 scale (s) or scale and offset (a), chosen to minimize the error of the
block. Bits per weight include the scales.

| | bpw | easy | medium | hard |
|---|---|---|---|---|
| dense fp32 | 32 | 99.55 % | 97.75 % | 89.95 % |
| 4 bits, s | 4.5 | 99.40 % | 97.45 % | 88.95 % |
| 3 bits, a | 4.0 | 99.20 % | 96.60 % | 84.60 % |
| 2 bits, a, blocks of 16 | 4.0 | 98.70 % | 88.05 % | 32.70 % |
| 2 bits, a | 3.0 | 97.25 % | 32.40 % | 0 % |
| 2 bits, s | 2.5 | 94.70 % | 1.35 % | 0 % |
| LittleBit, trained | 0.545 | 96.28 % (3 seeds) | 81.00 % | 23.35 % |
| LittleBit, trained 3x longer (6,000 steps) | 0.545 | | 84.85 % | 32.80 % |
| 2 bits, a, trained the same way (2,000 steps) | 3.0 | 99.25 % | 96.10 % | 79.70 % |

At four bits rounding costs at most one point. Below three bits rounding fails on the harder puzzles, where LittleBit at
0.55 bits still solves 81 % and 23 %. But training is what makes the difference, not the format: the same layers rounded
to 2 bits in blocks of 32 (3.0 bits per weight with the scales) and trained the same way (`tools/ncm/train_sudoku.py
--eff-bit 0 --wbits 2`, straight-through rounding) solve 99.25 %, 96.10 % and 79.70 %, far above LittleBit at 0.55 bits
on medium and hard. Training LittleBit three times longer helps (84.85 % and 32.80 %, still improving at the end) but
does not close the gap. LittleBit's advantage is size: 0.55 against 3.0 bits per weight. Calibrated methods such as
GPTQ were not tried. LittleBit is trained with Samsung's CC BY-NC 4.0 code, so
these models may not be used commercially.

## Speed

Nothing Phone (4a) Pro, 5 threads, dotprod build, mean of 20 puzzles or single texts:

| model | file | time | peak RSS |
|---|---|---|---|
| Sudoku, LittleBit 0.55 bpw | 6.8 MB | 91 ms per puzzle | 19 MB |
| Sudoku, dense Q8_0 | 31 MB | 102 ms per puzzle | 43 MB |
| TinyStories, dense Q8_0 | 162 MB | about 0.5 s per 128-token text | |

Galaxy Watch Ultra 2, awake on its charger, static 64-bit build, 10 puzzles, range of three runs:

| model | threads | time per puzzle |
|---|---|---|
| Sudoku, LittleBit 0.55 bpw | 1 (default) | 430 to 434 ms |
| same | 4 | 457 to 467 ms |
| same | 5 | 504 to 559 ms |
| Sudoku, dense Q8_0 | 1, pinned to the fast core | 521 to 526 ms |

The watch has one fast core and four slow ones, so more threads do not help. An earlier run, with the screen off
for part of it, took between 0.6 and 54 seconds per puzzle. With the 200-step checkpoint the answers to 10 puzzles were
identical to the PC's.

On the phone with one thread the LittleBit Sudoku model takes 290 ms. It is little faster than the dense one: in a
profile, the 1-bit matrix products are 43 % of the cycles. The gain is in size.

## File format

A GGUF with `general.architecture` = `ncm` and `ncm.version` = 1. LittleBit layers are stored as in
[FORMAT.md](FORMAT.md), with base names `blk.N.{qkv,proj,up,down}`. Other matrices are F32, F16 or Q8_0.

| tensors | content |
|---|---|
| `in`, `level_embd`, `l0_query` | input projection, level embeddings, learned level-0 input (if `ncm.query_level0`) |
| `level0_unigram` | distribution of the level-0 codes (otherwise) |
| `blk.N.*`, `ln_f.*` | generator |
| `head.K.*` | output heads; with `ncm.tied_heads`, a projection into code space and a bias, and `ncm.head_scale` |
| `codebook.K` | code embeddings |
| `phi.K.*` | level convolutions as a matrix over the concatenation of positions t-1, t, t+1, with the identity blend folded in |
| `l0_table` | finished contribution of every level-0 code, where that is smaller than the upsampling weights |
| `l0_up.J.*` | otherwise, the learned level-0 upsampling (`ncm.l0_strides`) |
| `token_embd`, `dec.*` | token embedding and decoder of the tokenizer |

The TinyStories level-0 codes are around 1e10 and scaled down by the upsampling, so that path is kept in F32.
