# 在魔搭 A10 24G（Linux）上跑这套模型（23M 旧版归档）

> **历史文档，不能直接用于当前 StoryWeaver 50M 发布版。** 本文记录早期 23.09M / 10K 词表 bundle；当前发布模型是 50.23M / 16K 词表，推理请以 `README.md` 的 `--preset v16k` 命令为准。下面的 bundle 路径和硬件估算只在你仍持有那套旧资产时有效。

## 0. 包内结构（关键：三个目录必须同级）

```
llm_bundle/
├─ 8_LLM_From_Scratch_run/     # 全部代码
├─ data/                       # 词表 + tokens.bin
│   ├─ TinyStoriesV2-GPT4-train_small_vocab.json
│   ├─ TinyStoriesV2-GPT4-train_small_merges.txt
│   ├─ TinyStoriesV2-GPT4-train_small_tokens.bin   (25,542,544 tokens)
│   └─ TinyStoriesV2-GPT4-valid_tokens.bin          (5,482,312 tokens)
└─ models/
    ├─ model.pth               # 265MB，权重 + AdamW 状态 + iteration=3000（续训用）
    ├─ model_weights.pth       # 89MB，纯权重（推理/微调用，推荐）
    └─ train_log.txt           # 3000 步完整 loss 曲线
```

脚本里所有路径都是 `../data/`、`../models/`，**相对 `8_LLM_From_Scratch_run/`**，所以必须在 `llm_bundle/` 这一层运行，不要单独把代码目录挪走。

## 1. 上传到魔搭

三种方式，按方便程度排：

**A. 网页上传（最稳）**：魔搭 → 数据集 → 创建数据集 → 上传 `llm_from_scratch_a10.tar.gz`。

**B. modelscope CLI**（语法随版本变化，先看帮助再执行）：

```bash
pip install -U modelscope
modelscope upload --help
# 按 --help 给出的参数上传到你的数据集仓库
```

**C. Notebook 直传**：魔搭 Notebook 的 JupyterLab 左侧文件树可直接拖拽上传，500MB 以内可接受。

## 2. 云端解压

```bash
mkdir -p ~/work && cd ~/work
tar -xzf llm_from_scratch_a10.tar.gz
cd llm_bundle
ls            # 应看到 8_LLM_From_Scratch_run  data  models
```

## 3. 环境检查

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.x.x True NVIDIA A10

pip install regex tqdm psutil numpy      # torch 一般镜像已带，不要盲目升级
```

## 4. 先验证推理（1 分钟）

```bash
cd 8_LLM_From_Scratch_run

python generate_run.py \
  --save_path ../models/model_weights.pth \
  --device cuda:0 \
  --max_length 120 \
  --temperature 0.7
```

`generate_run.py` 同时兼容 `model.pth`（完整 checkpoint）和 `model_weights.pth`（纯权重），会自动识别。

## 5. 续训：A10 24G 的推荐配置

A10 24G 显存是 3060 的 4 倍，可以把原版默认的 batch 64 × ctx 512 真正跑起来：

```bash
cd 8_LLM_From_Scratch_run

python train_run.py \
  --device cuda:0 \
  --batch_size 64 \
  --context_length 512 \
  --max_iters 20000 \
  --warmup_ratio 0.025 \
  --save_every 2000 \
  --save_dir ../models \
  --run_name model_a10 \
  --resume ../models/model.pth
```

`train_run.py` 已支持 `--resume`，上面的命令会恢复模型和优化器状态。若你有意更换 batch/ctx 并重置优化器，应显式开始新运行并只导入模型权重，而不是假装无缝续训。

**显存参考**：batch 64 × ctx 512 × vocab 10000 的 logits 约 1.31 GB，加上激活和梯度预计 8~11 GB，24G 有余量。想再快可以试 batch 96，但先用 64 跑稳。

**速度参考**：3060 Laptop 是 0.30 s/step，A10 预计 0.12~0.18 s/step。全量语料 1 epoch = 16,562 步，约 35~50 分钟。

## 6. 换上下文长度是安全的

从 ctx 256 的 checkpoint 直接切到 512 训练没问题：RoPE 的 sin/cos 是 `persistent=False` 的 buffer，不进 checkpoint，构造模型时会按新的 `context_length` 重新预计算。参数形状与 ctx 无关。

## 7. 想跑全量 2.23 GB 语料

包内只带了 100MB 子集的 token，全量需要自己下载并编码：

```bash
cd ~/work/llm_bundle/data

# 下载（HF 慢的话换镜像：export HF_ENDPOINT=https://hf-mirror.com）
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

# 编码（用同一套词表，输出 train_tokens.bin/.txt）
python ../8_LLM_From_Scratch_run/encode_corpus_fast.py \
  --input ./TinyStoriesV2-GPT4-train.txt \
  --out_stem ./TinyStoriesV2-GPT4-train_tokens
```

然后在训练时把 `--train_data` 指向 `../data/TinyStoriesV2-GPT4-train_tokens.bin`。全量约 542.7 M tokens，1 epoch ≈ 16,562 步（batch 64 × ctx 512）。

## 8. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `FileNotFoundError: ../data/...` | 没在 `llm_bundle/` 这一层运行，或把代码目录单独挪走了 |
| `torch.load` 报 `weights_only` 相关错误 | 当前脚本使用 `weights_only=True`；只支持张量/字典形式的安全 checkpoint |
| 加载权重报设备不匹配 | `load_checkpoint` 已加 `map_location`；`generate_run.py` 用 `--device` 指定 |
| 生成超过 256 个 token 后质量骤降 | ctx 仍是 256；把 `--context_length` 与训练保持一致，或按第 6 节放大 |
| 显存 OOM | 先降 `--batch_size` 到 32，再降 `--context_length` 到 384 |
| 词表对不上、输出乱码 | 必须用包内 `*_small_vocab.json` / `*_small_merges.txt`，换词表等于换 id 空间 |

## 9. 关键参数速查

| 参数 | 值 |
|---|---|
| 模型 | 4 层 / d_model 512 / 16 头 / d_k 32 / d_ff 1408 / vocab 10000 / 23.09M 参数 |
| 已训步数 | 3000（loss 2.2627，val 2.1514，ppl ≈ 8.6） |
| 词表 | 10000 条，9731 merges，`<|endoftext|>` id = 256 |
| 优化器 | 手写 AdamW，betas (0.9,0.999)，eps 1e-8，wd 0.01 |
| 调度 | warmup → cosine 到 min_lr |
| 梯度裁剪 | max_norm = 1.0 |
