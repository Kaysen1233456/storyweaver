<div align="center">

# StoryWeaver · 50M

**A decoder-only language model implemented entirely from scratch in PyTorch.**

No `nn.Transformer`. No `AutoModel`. Every layer written by hand.

[中文说明](README_zh.md) · [Experiment Log](EXPERIMENTS.md)

</div>

---

## What is this?

StoryWeaver is a 50.23M-parameter GPT-style language model built from zero — byte-level BPE tokenizer, RMSNorm, RoPE, SwiGLU, multi-head self-attention, AdamW, cross-entropy and top-p sampling, all implemented without high-level abstractions.

The point is not scale. The point is that **every tensor is visible**: you can trace a token from the BPE merge table, through six attention blocks, to the logits and the loss, without a single framework black box.

It is trained on a 60/40 mix of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [Project Gutenberg](https://www.gutenberg.org/), then instruction-tuned on 20K synthetic instruction–story pairs.

## Capabilities

| Capability | Input | Output |
|---|---|---|
| **Story continuation** | `Once upon a time,` | A coherent 3-paragraph children's story |
| **Instruction following** | `Write a story about a brave little dog.` | A themed story with characters, conflict, resolution |

**Example** (`storyweaver-sft-50m`, temperature 0.8):

> **Instruction**: Write a story about a brave little dog.
>
> Once upon a time, there was a little dog named Spot. Spot loved to play in the park. One day, he found a big tree and decided to climb it. He was very careful not to fall. As Spot climbed the tree, he saw a little bird. The bird was sad because it could not fly. Spot wanted to help the bird... Spot and the bird became good friends. They played together all day long.

**Limitations** — stated honestly:

- English only (corpus is English children's stories + public-domain novels)
- 256-token context; generation beyond that degrades
- No question answering, reasoning, code, or math
- Writes like a children's story author, because that is what it read

## Architecture

```
token ids (B, T)
  → Embedding(16000, 640)
  → 6 × TransformerBlock          # Pre-RMSNorm
  → RMSNorm
  → Linear(640, 16000)            # lm_head, no weight tying
  → logits (B, T, 16000)
```

Each block:

```
x = x + MHA(RMSNorm(x))           # RoPE applied to Q/K, causal mask
x = x + SwiGLU(RMSNorm(x))        # SiLU gating, d_ff = 1728
```

### Hand-written components

| File | What it implements | Equivalent |
|---|---|---|
| `train_bpe.py` | Byte-level BPE: regex pre-tokenization, pair counting, deterministic merge ranking | tiktoken |
| `tokenizer.py` | Encode / decode with special tokens, streaming iterable | `AutoTokenizer` |
| `linear.py` / `embedding.py` | Linear & embedding with truncated-normal init | `nn.Linear` / `nn.Embedding` |
| `RMSNorm.py` | No mean subtraction, learnable gain, FP32 compute | Llama `RMSNorm` |
| `RoPE.py` | Interleaved rotary, precomputed sin/cos buffers | Llama rotary embedding |
| `SwiGLU.py` | `w2(silu(w1(x)) * w3(x))` | Llama MLP |
| `MultiHeadSelfAttention.py` | einsum attention + `torch.triu` causal mask | `F.scaled_dot_product_attention` |
| `AdamW.py` | Decoupled weight decay, bias correction | `torch.optim.AdamW` |
| `function.py` | Cross-entropy, gradient clipping, warmup+cosine, top-p sampling | — |

## Model Specs

| | |
|---|---|
| Parameters | **50.23 M** |
| Layers | 6 |
| d_model | 640 |
| Attention heads | 10 (d_k = 64) |
| FFN hidden | 1,728 |
| Vocabulary | 16,000 (byte-level BPE) |
| Context length | 256 |
| Positional encoding | RoPE (θ = 10,000) |
| Normalization | Pre-RMSNorm |

## Checkpoints

| Name | Steps | Best for | Validation |
|---|---|---|---|
| `storyweaver-base-50m` | 8,500 | Base model, story continuation | balanced **3.0161** |
| `storyweaver-sft-50m` | 800 (SFT) | Instruction following | SFT val **1.7130** |

Both are full checkpoints (`model_state_dict` + `optimizer_state_dict` + `iteration`).

## Evaluation

Evaluated on **808 fixed validation windows** (same windows every run, so numbers are comparable across experiments).

| Metric | Base | SFT |
|---|---|---|
| TinyStories loss | 1.9403 | 1.9525 |
| Gutenberg loss (11 books) | 4.6586 | **4.5622** |
| BPB (TinyStories) | 0.6744 | — |
| BPB (Gutenberg) | 1.5815 | — |
| Balanced `0.6·TS + 0.4·GB` | **3.0161** | — |

The 0.6/0.4 weighting mirrors the training corpus ratio, so the metric tracks the actual training distribution rather than an arbitrary average.

## Quick Start

```bash
pip install torch regex tqdm numpy
```

**Instruction mode** (SFT checkpoint):

```bash
python generate_run.py --preset v16k \
  --save_path weights/storyweaver-sft-50m.pth \
  --instruction "Write a story about a brave little dog." \
  --max_length 250 --temperature 0.8
```

**Continuation mode** (base checkpoint):

```bash
python generate_run.py --preset v16k \
  --save_path weights/storyweaver-base-50m.pth \
  --prompt "Once upon a time," \
  --max_length 250 --temperature 0.8
```

## Training Notes

| Stage | Setup |
|---|---|
| Corpus | 41.4M tokens — TinyStories 60% + Gutenberg 40% (11 books, √-weighted allocation, 8% held out per book) |
| Base training | AdamW, lr 1e-3 → 1e-4, warmup 404, dropout 0.1, weight decay 0.1, β₂ 0.95, 10,116 steps |
| SFT | 20K instruction–story pairs, 400 steps (0.32 epoch), lr 1e-4 → 1e-5, prompt tokens masked with `-100` |

SFT stops at 400 steps deliberately: beyond that point instruction quality improves marginally while base-model capability degrades (catastrophic forgetting). Full curves in [EXPERIMENTS.md](EXPERIMENTS.md).

## Project Structure

```
├─ train_bpe.py / tokenizer.py     # tokenizer pipeline
├─ TransformerLM.py                # model
├─ train_run.py                    # base training (balanced eval)
├─ sft_run.py                      # instruction tuning
├─ generate_run.py                 # sampling (continuation / instruction)
├─ compare_models.py               # side-by-side generation
├─ EXPERIMENTS.md                  # full experiment log
└─ weights/                        # checkpoints (downloaded separately)
```

## Acknowledgements

Architecture follows [Stanford CS336](https://stanford-cs336.github.io/spring2025/) Assignment 1 and the Llama family. Training corpora: [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023) and [Project Gutenberg](https://www.gutenberg.org/).

## License

MIT
