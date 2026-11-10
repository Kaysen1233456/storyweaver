<div align="center">

# StoryWeaver · 50M

**纯 PyTorch 从零实现的 decoder-only 语言模型**

没有 `nn.Transformer`，没有 `AutoModel`，每一层都是手写的。

[English](README.md) · [实验记录](EXPERIMENTS.md)

</div>

---

## 这是什么

StoryWeaver 是一个 50.23M 参数的 GPT 风格语言模型，从字节级 BPE 词表、RMSNorm、RoPE、SwiGLU、多头自注意力，到 AdamW、交叉熵和 top-p 采样，**全部手写，不调用任何高层封装**。

重点不是规模，而是**每个张量都看得见**：你可以追踪一个 token 从 BPE 合并表、穿过 6 个注意力块、到 logits 和 loss 的完整路径，中间没有任何框架黑盒。

训练语料是 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 与 [Project Gutenberg](https://www.gutenberg.org/) 按 60/40 混合，之后用 2 万条合成的"指令—故事"对做指令微调。

## 能力

| 能力 | 输入 | 输出 |
|---|---|---|
| **故事续写** | `Once upon a time,` | 一段结构完整的三段式儿童故事 |
| **指令遵循** | `Write a story about a brave little dog.` | 围绕该主题、有角色/冲突/结局的故事 |

**示例**（`storyweaver-sft-50m`，temperature 0.8）：

> **指令**：Write a story about a brave little dog.
>
> Once upon a time, there was a little dog named Spot. Spot loved to play in the park. One day, he found a big tree and decided to climb it. He was very careful not to fall. As Spot climbed the tree, he saw a little bird. The bird was sad because it could not fly. Spot wanted to help the bird... Spot and the bird became good friends. They played together all day long.

**局限（如实说明）**：

- 只支持英文（语料是英文儿童故事 + 公版小说）
- 上下文 256 token，超出后生成质量明显下降
- 不会问答、推理、写代码、做数学
- 文风像儿童故事作者——因为它读的就是这些

## 架构

```
token ids (B, T)
  → Embedding(16000, 640)
  → 6 × TransformerBlock          # Pre-RMSNorm
  → RMSNorm
  → Linear(640, 16000)            # lm_head，不做权重绑定
  → logits (B, T, 16000)
```

单个 Block：

```
x = x + MHA(RMSNorm(x))           # RoPE 作用于 Q/K，因果掩码
x = x + SwiGLU(RMSNorm(x))        # SiLU 门控，d_ff = 1728
```

### 手写组件一览

| 文件 | 实现内容 | 对应的框架组件 |
|---|---|---|
| `train_bpe.py` | 字节级 BPE：正则预分词、pair 计数、确定性合并排序 | tiktoken |
| `tokenizer.py` | 编解码、特殊 token、流式迭代 | `AutoTokenizer` |
| `linear.py` / `embedding.py` | 线性层与词嵌入，截断正态初始化 | `nn.Linear` / `nn.Embedding` |
| `RMSNorm.py` | 不减均值、可学习增益、FP32 计算 | Llama `RMSNorm` |
| `RoPE.py` | 交错式旋转位置编码，预计算 sin/cos buffer | Llama rotary embedding |
| `SwiGLU.py` | `w2(silu(w1(x)) * w3(x))` | Llama MLP |
| `MultiHeadSelfAttention.py` | einsum 注意力 + `torch.triu` 因果掩码 | `F.scaled_dot_product_attention` |
| `AdamW.py` | 解耦权重衰减、偏差修正 | `torch.optim.AdamW` |
| `function.py` | 交叉熵、梯度裁剪、warmup+cosine、top-p 采样 | — |

### 为什么这么设计

- **Pre-RMSNorm**：归一化放在子层之前，残差通路保持"干净"，深层训练更稳定（GPT-2 之后的主流做法）
- **RoPE**：把位置信息编码成旋转角度，天然表达相对位置，且不增加参数
- **SwiGLU**：SiLU 门控比 ReLU/GELU 在同等参数下效果更好，Llama 系标配
- **不做权重绑定**：`lm_head` 与 `embedding` 独立，参数量多 10M，但小模型上解耦更稳

## 模型规格

| | |
|---|---|
| 参数量 | **50.23 M** |
| 层数 | 6 |
| d_model | 640 |
| 注意力头 | 10（d_k = 64） |
| FFN 隐层 | 1,728 |
| 词表 | 16,000（字节级 BPE） |
| 上下文长度 | 256 |
| 位置编码 | RoPE（θ = 10,000） |
| 归一化 | Pre-RMSNorm |

## 权重

| 名称 | 步数 | 适用场景 | 验证指标 | 下载 |
|---|---|---|---|---|
| `storyweaver-sft-50m` | 800（SFT） | 指令遵循 | SFT val **1.7130** | [v1.0](https://github.com/Kaysen1233456/storyweaver/releases/download/v1.0/storyweaver-sft-50m.pth) |
| `storyweaver-base-50m` | 8,500 | 基座模型，故事续写 | balanced **3.0161** | 可用 `train_run.py` 复现 |

完整 checkpoint 格式：`model_state_dict` + `optimizer_state_dict` + `iteration`。

## 评估

在 **808 个固定验证窗口**上评估（每次跑同一批窗口，跨实验可比）。

| 指标 | 基座 | SFT |
|---|---|---|
| TinyStories loss | 1.9403 | 1.9525 |
| Gutenberg loss（11 本书平均） | 4.6586 | **4.5622** |
| BPB（TinyStories） | 0.6744 | — |
| BPB（Gutenberg） | 1.5815 | — |
| Balanced `0.6·TS + 0.4·GB` | **3.0161** | — |

0.6/0.4 的权重对应训练语料的配比，所以这个指标反映的是"模型在真实训练分布上的平均表现"，而不是拍脑袋的加权平均。

## 快速开始

```bash
pip install torch regex tqdm numpy
```

**指令模式**（SFT 权重）：

```bash
python generate_run.py --preset v16k \
  --save_path weights/storyweaver-sft-50m.pth \
  --instruction "Write a story about a brave little dog." \
  --max_length 250 --temperature 0.8
```

**续写模式**（基座权重）：

```bash
python generate_run.py --preset v16k \
  --save_path weights/storyweaver-base-50m.pth \
  --prompt "Once upon a time," \
  --max_length 250 --temperature 0.8
```

## 训练要点

| 阶段 | 配置 |
|---|---|
| 语料 | 41.4M tokens —— TinyStories 60% + Gutenberg 40%（11 本书，按 √字符数分配配额，每本留 8% 验证） |
| 基座训练 | AdamW，lr 1e-3 → 1e-4，warmup 404，dropout 0.1，weight decay 0.1，β₂ 0.95，10,116 步 |
| 指令微调 | 2 万条"指令—故事"对，400 步（0.32 epoch），lr 1e-4 → 1e-5，prompt 部分 target 置 `-100` |

SFT 只跑 400 步是刻意的：再往后指令质量提升有限，基座能力却开始退化（灾难性遗忘）。完整曲线见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 目录结构

```
├─ train_bpe.py / tokenizer.py     # 分词器链路
├─ TransformerLM.py                # 模型
├─ train_run.py                    # 基座训练（均衡评估）
├─ sft_run.py                      # 指令微调
├─ generate_run.py                 # 采样（续写 / 指令）
├─ compare_models.py               # 多模型对比生成
├─ EXPERIMENTS.md                  # 完整实验记录
└─ weights/                        # 权重（单独下载）
```

## 致谢

架构参考 [Stanford CS336](https://stanford-cs336.github.io/spring2025/) Assignment 1 与 Llama 系列。训练语料来自 [TinyStories](https://arxiv.org/abs/2305.07759)（Eldan & Li, 2023）与 [Project Gutenberg](https://www.gutenberg.org/)。

## 许可

MIT
