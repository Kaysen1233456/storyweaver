# SFT 起步说明（23M 旧版归档）

> **历史交接单。** 当前仓库已经实现 `sft_run.py`、共享 prompt 模板和 masked loss，发布版是 50.23M / 16K 词表；当前使用方法以 `README_zh.md` 为准。下面保留的是早期 23.09M / 10K 基座阶段的设计背景。

## 一、你现在拿到的东西

| 资产 | 路径 | 说明 |
|---|---|---|
| 基座 checkpoint | `models/model.pth` | `model_state_dict` / `optimizer_state_dict` / `iteration` 三件套 |
| 词表 | `data/TinyStoriesV2-GPT4-train_small_vocab.json` + `_merges.txt` | 10000 词表、9731 条 merge，**必须与 checkpoint 配套** |
| 训练 token | `data/TinyStoriesV2-GPT4-train_small_tokens.bin`（int32，98MB） | 25,542,544 tokens |
| 验证 token | `data/TinyStoriesV2-GPT4-valid_tokens.bin`（int32，21MB） | 5,482,312 tokens |
| 训练脚本 | `8_LLM_From_Scratch_run/train_run.py` | warmup+cosine、梯度裁剪、周期评估、rolling/best 存档 |
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

# 续训（当前 train_run.py 用 --resume；max_iters 是目标总步数）
python train_run.py --max_iters 8000 --resume ../models/model.pth --run_name model_long

# 生成
python generate_run.py --prompt "Once upon a time," --max_length 100 --temperature 0.9
```

## 六、当时的遗留问题（当前状态）

- `--d_ff` 已真正接入 `SwiGLU`；23M 旧 checkpoint 的真实值是 1408。
- RoPE 已跨 head 向量化，因果 mask 已预计算复用。
- `Tokenizer.str_encode` 已改用 rank 字典，并保留旧编码结果兼容。
- 训练循环仍采用随机窗口采样；这是当前设计选择，不是顺序 epoch 遍历。
