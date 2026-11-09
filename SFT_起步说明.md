# SFT 起步说明（基于自己训出的基座）

> 这份文档是给你接手「微调训练 + 手撕」用的交接单。基座权重由 `train_run.py` 训出，原版 `8_LLM_From_Scratch/` 一个字没动。

## 一、你现在拿到的东西

| 资产 | 路径 | 说明 |
|---|---|---|
| 基座 checkpoint | `models/model.pth` | `model_state_dict` / `optimizer_state_dict` / `iteration` 三件套 |
| 词表 | `data/TinyStoriesV2-GPT4-train_small_vocab.json` + `_merges.txt` | 10000 词表、9731 条 merge，**必须与 checkpoint 配套** |
| 训练 token | `data/TinyStoriesV2-GPT4-train_small_tokens.bin`（int32，98MB） | 25,542,544 tokens |
| 验证 token | `data/TinyStoriesV2-GPT4-valid_tokens.bin`（int32，21MB） | 5,482,312 tokens |
| 训练脚本 | `8_LLM_From_Scratch_run/train_run.py` | 已接 warmup+cosine、梯度裁剪、早停、定期存档 |
| 生成脚本 | `8_LLM_From_Scratch_run/generate_run.py` | 词表/上下文/device 与训练对齐，top-p 采样 |
| 编码脚本 | `8_LLM_From_Scratch_run/encode_corpus_fast.py` | 流式 + rank 字典 + 词缓存，100MB 语料 47 秒 |

解释器固定用这个（有 torch 2.11+cu128、cuda=True）：

```
C:\Users\Raymond James\AppData\Local\Programs\Python\Python312\python.exe
```

## 二、三条硬约束（踩了会白训）

1. **词表绑定**：embedding 是 `10000 × 512`，微调和推理都必须加载同一套 `*_small_vocab.json` / `*_small_merges.txt`。换词表 = 换 token id 空间，权重直接失效。
2. **上下文长度 256**：训练用 `context_length=256`，RoPE 的 sin/cos 按 256 预计算且 `persistent=False` 不进 checkpoint。生成或微调超过 256，`token_positions` 会被 `clamp` 到 255，位置编码静默退化。
3. **它只会接龙**：预训练目标就是 next-token prediction，没有指令微调。要它能问答，必须走 SFT——这正是你接下来要手撕的部分。

## 三、SFT 必须补的三件事

1. **数据格式**：prompt / response 对，拼成 `prompt + response + <|endoftext|>`。TinyStories 里 `<|endoftext|>` 的 id 是 256，可以当结束符。
2. **loss mask**：`function.py` 的 `cross_entropy` 现在对全序列所有位置平均。SFT 必须只对 response 位置算 loss——把 prompt 位置的 target 置为 `-100`，并在交叉熵里加 `ignore_index=-100`。
3. **超参**：lr 从 `3e-4` 降到 `1e-4 ~ 5e-5`，几百到几千步即可，别把基座训崩。

## 四、改动落点（照着撕）

- `function.py::cross_entropy`：加 `ignore_index` 支持，`gather` 后把 `targets == -100` 的位置置 0 再除以有效 token 数。
- 新建 `sft_run.py`：`load_checkpoint` → 构造带 mask 的 batch → `model(inputs)` → masked loss → `clip_gradient` → `optimizer.step()`，主体结构和 `train_run.py` 一致，只是数据换成 prompt/response 对。
- `generate_run.py`：把 `--prompt` 换成你的指令模板，`--max_length` 别超过 256。

## 五、验证命令

```bash
cd "C:\Users\Raymond James\PyTorch-DeepLearning-Practice-Projects-main\8_LLM_From_Scratch_run"

# 续训（从 3000 步继续，先手动把 load_checkpoint 加进 train_run.py）
python train_run.py --max_iters 8000 --save_path ../models/model_long.pth

# 生成
python generate_run.py --prompt "Once upon a time," --max_length 100 --temperature 0.9
```

## 六、已知遗留问题（原版就有，副本里只修了必要项）

- `--d_ff` 仍是死参数：`SwiGLU` 内部固定算 `1408`，命令行传 1344 无效。
- RoPE 仍是逐 head 的 Python for 循环，是主要性能瓶颈，可向量化。
- `Tokenizer.str_encode` 仍是 O(n²) 合并；副本的编码脚本用 rank 字典 + 缓存绕开了它，但 Tokenizer 本体没改。
- 训练循环仍是随机采样窗口（`np.random.randint`），不是顺序遍历语料。
