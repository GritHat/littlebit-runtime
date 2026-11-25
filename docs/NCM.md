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

| Sudoku easy | this runtime | PyTorch |
|---|---|---|
| dense, Q8_0, 31 MB | 99.60 % | 99.55 % |
| an early LittleBit checkpoint, 6.8 MB | 86.50 % | 86.60 % |

## Speed

Nothing Phone (4a) Pro, 5 threads, dotprod build, mean of 20 puzzles or single texts:

| model | file | time | peak RSS |
|---|---|---|---|
| Sudoku, LittleBit 0.55 bpw | 6.8 MB | 91 ms per puzzle | 19 MB |
| Sudoku, dense Q8_0 | 31 MB | 102 ms per puzzle | 43 MB |
| TinyStories, dense Q8_0 | 162 MB | about 0.5 s per 128-token text | |

With one thread the LittleBit Sudoku model takes 290 ms. It is little faster than the dense one: in a
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
