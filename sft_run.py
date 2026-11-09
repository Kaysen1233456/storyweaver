"""SFT 训练入口：加载预训练基座, 只对回答部分计分, 同时监控基座能力有没有被训坏。

数据(由 build_sft_ts.py 生成)
  *.bin: 每条 = [prompt tokens][故事 tokens][eot]; *_meta.json: [[prompt_len, total_len], ...]
  prompt = sft_format.format_prompt(指令), 推理时 generate_run.py --instruction 用的是同一个模板
计分
  每条样本从位置 0 开始放(右侧补 0, 和预训练一致); 位置 t 的目标是位置 t+1 的 token;
  prompt 部分和补零部分的目标置 -100, 不计分, 只有故事和结束符计分
评估
  val_loss: SFT 验证集(500 条)上回答部分的 loss, 用来选 sft_best.pth
  base_TS / base_GB: 预训练固定验证窗口上的 TinyStories / 书本 loss。SFT 前后应基本不变,
  明显上升说明 SFT 在破坏基座的语言能力(灾难性遗忘)

PowerShell 用法:
  & $py sft_run.py --preset v16k --base_ckpt ../models/final_bal/model_best.pth
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from AdamW import AdamW
from TransformerLM import TransformerLM
from function import get_learning_rate, clip_gradient
from train_run import FixedWindows

IGNORE = -100
PRESETS = {
    'v16k': dict(vocab_size=16000, d_model=640, num_layers=6, num_heads=10, d_ff=1728),
}
LEGACY = dict(vocab_size=10000, d_model=512, num_layers=4, num_heads=16, d_ff=1344)


def load_sft_dataset(bin_path, meta_path):
    """返回 (tokens, samples)；samples = [(seq, prompt_len), ...]。"""
    ids = np.fromfile(bin_path, dtype=np.int32).astype(np.int64)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)['samples']
    samples, pos = [], 0
    for plen, total in meta:
        samples.append((ids[pos: pos + total], plen))
        pos += total
    assert pos == len(ids), 'meta 与 bin 长度不一致'
    return ids, samples


def make_batch(batch_samples, ctx, device):
    """右侧补零; prompt 与补零位置的 target = -100。返回 (inputs, targets)。"""
    n = len(batch_samples)
    x = np.zeros((n, ctx), dtype=np.int64)
    y = np.full((n, ctx), IGNORE, dtype=np.int64)
    for k, (seq, plen) in enumerate(batch_samples):
        seq = seq[:ctx]
        L = len(seq)
        x[k, :L] = seq
        # 位置 t 预测 seq[t+1]; 第一个计分位置是 prompt 最后一个 token(它预测故事第一个 token)
        y[k, plen - 1:L - 1] = seq[plen:L]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def evaluate_sft(model, val_batches):
    """回答部分逐 token 平均 loss(按 token 数加权, 不是按 batch 平均)"""
    model.eval()
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for x, y in val_batches:
            logits = model(x)
            nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                   ignore_index=IGNORE, reduction='sum').item()
            cnt += int((y != IGNORE).sum())
    model.train()
    return nll / cnt


def save_ckpt(model, optimizer, it, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': it,
    }, path)


def prune_ckpts(folder, prefix, keep):
    if keep <= 0:
        return
    files = sorted(glob.glob(os.path.join(folder, f'{prefix}_[0-9]*.pth')))
    for f in files[:-keep]:
        os.remove(f)


def main():
    p = argparse.ArgumentParser()
    # 架构(必须与基座一致)
    p.add_argument('--preset', choices=sorted(PRESETS), default=None)
    p.add_argument('--vocab_size', type=int, default=None)
    p.add_argument('--context_length', type=int, default=256)
    p.add_argument('--d_model', type=int, default=None)
    p.add_argument('--num_layers', type=int, default=None)
    p.add_argument('--num_heads', type=int, default=None)
    p.add_argument('--d_ff', type=int, default=None)
    p.add_argument('--dropout', type=float, default=0.1)
    # SFT 超参
    p.add_argument('--base_ckpt', default='../models/final_bal/model_best.pth')
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--epochs', type=float, default=2, help='--max_iters 为 0 时按 epoch 数算步数')
    p.add_argument('--max_iters', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=16, help='每步样本条数')
    p.add_argument('--eval_every', type=int, default=125)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--keep_last', type=int, default=2)
    p.add_argument('--log_every', type=int, default=25)
    p.add_argument('--run_name', default='sft')
    p.add_argument('--train_bin', default='../data/sft_ts_train.bin')
    p.add_argument('--train_meta', default='../data/sft_ts_train_meta.json')
    p.add_argument('--val_bin', default='../data/sft_ts_val.bin')
    p.add_argument('--val_meta', default='../data/sft_ts_val_meta.json')
    p.add_argument('--base_val_windows', default='../data/mixed_bal_val_windows.npz',
                   help='预训练固定验证窗口, 用来监控基座能力; 传空字符串关闭')
    p.add_argument('--save_dir', default='../models/sft_bal')
    p.add_argument('--log_file', default='../models/sft_bal/sft_log.txt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    base = PRESETS[args.preset] if args.preset else LEGACY
    for k, v in base.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
    log_fp = open(args.log_file, 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + '\n')
        log_fp.flush()

    t0 = time.time()
    _, train_samples = load_sft_dataset(args.train_bin, args.train_meta)
    _, val_samples = load_sft_dataset(args.val_bin, args.val_meta)
    total_steps = args.max_iters or math.ceil(len(train_samples) * args.epochs / args.batch_size)
    epochs = total_steps * args.batch_size / len(train_samples)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    min_lr = args.learning_rate * args.min_lr_ratio

    log('=' * 62)
    log(f'run={args.run_name}  sft_train={len(train_samples):,} 条  sft_val={len(val_samples)} 条')
    log(f'base={args.base_ckpt}')
    log(f'steps={total_steps} ({epochs:.2f} epoch)  bs={args.batch_size} 条/步  '
        f'lr {args.learning_rate:.1e}->{min_lr:.1e}  warmup={warmup}')
    log(f'dropout={args.dropout}  weight_decay={args.weight_decay} (norms excluded)  beta2={args.beta2}')
    log('=' * 62)

    model = TransformerLM(vocab_size=args.vocab_size, context_length=args.context_length,
                          d_model=args.d_model, num_layers=args.num_layers,
                          num_heads=args.num_heads, d_ff=args.d_ff, dropout=args.dropout,
                          device=args.device)
    ckpt = torch.load(args.base_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    log(f'base loaded, iteration={ckpt.get("iteration")}  '
        f'params={sum(q.numel() for q in model.parameters()) / 1e6:.2f}M')

    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    optimizer = AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                       {'params': no_decay, 'weight_decay': 0.0}],
                      lr=args.learning_rate, betas=(0.9, args.beta2))

    val_batches = [make_batch(val_samples[i:i + args.batch_size], args.context_length, args.device)
                   for i in range(0, len(val_samples), args.batch_size)]
    windows = (FixedWindows(args.base_val_windows, args.device, ts_weight=0.6)
               if args.base_val_windows else None)

    def run_eval(it):
        vl = evaluate_sft(model, val_batches)
        line = f'[eval] step {it}  val_loss {vl:.4f}'
        extra = None
        if windows:
            extra = windows.evaluate(model, args.batch_size)
            line += f'  base_TS {extra["ts"]:.4f}  base_GB {extra["gb"]:.4f}'
        return vl, line, extra

    vl0, line0, base0 = run_eval(0)
    log(line0 + '   (SFT 前的基座)')

    best_val, last = float('inf'), None
    order, cursor = rng.permutation(len(train_samples)), 0
    model.train()
    for it in range(1, total_steps + 1):
        lr = float(get_learning_rate(it, args.learning_rate, min_lr, warmup, total_steps))
        for group in optimizer.param_groups:
            group['lr'] = lr

        if cursor + args.batch_size > len(order):
            order, cursor = rng.permutation(len(train_samples)), 0
        chunk = [train_samples[j] for j in order[cursor:cursor + args.batch_size]]
        cursor += args.batch_size

        x, y = make_batch(chunk, args.context_length, args.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                               ignore_index=IGNORE)
        loss.backward()
        clip_gradient(model.parameters(), 1.0)
        optimizer.step()

        if it == 1 or it % args.log_every == 0:
            log(f'iter {it}/{total_steps}  loss {loss.item():.4f}  lr {lr:.2e}  '
                f'elapsed {time.time() - t0:.1f}s')

        if it % args.eval_every == 0 or it == total_steps:
            vl, line, last = run_eval(it)
            if vl < best_val:
                best_val = vl
                save_ckpt(model, optimizer, it, os.path.join(args.save_dir, 'sft_best.pth'))
                line += '  <- best'
            log(line)

        if it % args.save_every == 0:
            path = os.path.join(args.save_dir, f'{args.run_name}_{it:06d}.pth')
            save_ckpt(model, optimizer, it, path)
            prune_ckpts(args.save_dir, args.run_name, args.keep_last)
            log(f'[save] {path}')

    final_path = os.path.join(args.save_dir, f'{args.run_name}_final.pth')
    save_ckpt(model, optimizer, total_steps, final_path)
    log('=' * 62)
    log(f'DONE steps={total_steps}  SFT val: {vl0:.4f} -> best {best_val:.4f}')
    if windows:
        log(f'基座能力 base_TS: {base0["ts"]:.4f} -> {last["ts"]:.4f}   '
            f'base_GB: {base0["gb"]:.4f} -> {last["gb"]:.4f}')
    log(f'best  -> {os.path.join(args.save_dir, "sft_best.pth")}')
    log(f'final -> {final_path}')
    log(f'total {time.time() - t0:.1f}s')
    log('=' * 62)
    log_fp.close()


if __name__ == '__main__':
    main()
