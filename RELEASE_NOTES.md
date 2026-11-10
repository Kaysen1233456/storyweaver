## StoryWeaver 50M v1.0

A 50.23M-parameter decoder-only language model implemented entirely from scratch in PyTorch — no `nn.Transformer`, no `AutoModel`.

- 6 layers, d_model 640, 10 heads, 16K byte-level BPE vocabulary
- Trained on 41.4M tokens (TinyStories 60% + Gutenberg 40%)
- Balanced evaluation on 808 fixed windows: **3.0161** (TS 1.9403 / GB 4.6586)

### Checkpoints

| File | Steps | Best for | Metric |
|---|---|---|---|
| `storyweaver-base-50m.pth` | 8,500 | Base model — story continuation, starting point for SFT | balanced **3.0161** (TS 1.9403 / GB 4.6586) |
| `storyweaver-sft-50m.pth` | 800 (SFT) | Instruction following | SFT val **1.7130**, base preserved (GB 4.5773 vs 4.5905 before) |

Both are full checkpoints: `model_state_dict` + `optimizer_state_dict` + `iteration`. Load with `--preset v16k`.

### Quick start

```
python generate_run.py --preset v16k \
  --save_path weights/storyweaver-sft-50m.pth \
  --instruction "Write a story about a brave little dog." \
  --max_length 250 --temperature 0.8
```

See the [README](https://github.com/Kaysen1233456/storyweaver#readme) for architecture details and full evaluation.
