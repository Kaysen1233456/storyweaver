"""训练入口：单卡友好 + 自动日志 + 周期评估 + rolling & best checkpoint。

PowerShell 用法示例：
  & $py train_run.py --batch_size 16 --context_length 256 --max_iters 0
  # --max_iters 0 表示自动按 1 epoch 计算步数

关键设计：
- 步数自动按 len(tokens)/(bs*ctx*accum) 算 1 epoch，不用手填
- warmup 与 min_lr 按比例自动推导（默认 4% / 0.1x）
- 评估两种模式:
  * 不传 --val_windows: 旧模式, 在 --val_data 上随机抽 eval_iters 个 batch
  * 传 --val_windows: 固定窗口模式, 每次评估同一批窗口, 分别报告 TinyStories(TS)、
    每本书和书的平均(GB), best 按 balanced = ts_weight*TS + (1-ts_weight)*GB 选
- 存档：<run>_<step>.pth 滚动保留最近 N 个 + model_best.pth；--no_save 时全部不存(搜索用)
- 结束时输出 RESULT 行, 供 bayes_search.py 读取
- 日志同时打印和落盘，watch_loss.py 直接可读
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from AdamW import AdamW
from TransformerLM import TransformerLM
from function import (data_loader, cross_entropy as loss_fn,
                      get_learning_rate, clip_gradient)

LN2 = math.log(2.0)


def load_tokens(path):
    return np.fromfile(path, dtype=np.int32).astype(np.int64)


def evaluate(model, val_data, bs, ctx, device, iters):
    """旧模式: 随机窗口, 返回平均 loss"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = data_loader(val_data, bs, ctx, device)
            total += loss_fn(model(x), y).item()
    model.train()
    return total / iters


class FixedWindows:
    """固定验证窗口: 每次评估同一批文本, 按组(TinyStories / 每本书)汇总"""

    def __init__(self, path, device, ts_weight):
        d = np.load(path)
        self.x = torch.from_numpy(d['x'].astype(np.int64)).to(device)
        self.group = d['group']
        self.nbytes = d['nbytes'].astype(np.float64)
        self.names = [str(n) for n in d['names']]
        self.ts_weight = ts_weight

    def __len__(self):
        return len(self.x)

    def evaluate(self, model, bs):
        model.eval()
        nll = np.zeros(len(self.x))
        with torch.no_grad():
            for i in range(0, len(self.x), bs):
                w = self.x[i:i + bs]
                logits = model(w[:, :-1])
                tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      w[:, 1:].reshape(-1), reduction='none')
                nll[i:i + bs] = tok.view(len(w), -1).sum(dim=1).cpu().numpy()
        model.train()
        n_tok = self.x.shape[1] - 1
        per_loss, per_bpb = {}, {}
        for gi, name in enumerate(self.names):
            m = self.group == gi
            per_loss[name] = nll[m].sum() / (m.sum() * n_tok)
            per_bpb[name] = nll[m].sum() / LN2 / self.nbytes[m].sum()
        books = self.names[1:]
        ts = per_loss[self.names[0]]
        gb = float(np.mean([per_loss[b] for b in books]))
        return {
            'balanced': self.ts_weight * ts + (1 - self.ts_weight) * gb,
            'ts': ts, 'gb': gb,
            'bpb_ts': per_bpb[self.names[0]],
            'bpb_gb': float(np.mean([per_bpb[b] for b in books])),
            'books': {b: per_loss[b] for b in books},
        }


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
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--grad_accum', type=int, default=1,
                   help='梯度累积: 每步跑 N 个 micro-batch 再更新, 等效 batch = batch_size x N')
    p.add_argument('--context_length', type=int, default=256)
    p.add_argument('--vocab_size', type=int, default=10000)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=16)
    p.add_argument('--d_ff', type=int, default=1408)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup_ratio', type=float, default=0.04)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--wd_exclude_norm', action='store_true',
                   help='只对矩阵参数做 weight decay, 归一化层的 1 维系数不衰减')
    p.add_argument('--max_iters', type=int, default=0, help='0 = 自动按 1 epoch')
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--eval_iters', type=int, default=20, help='旧随机评估模式的 batch 数')
    p.add_argument('--val_windows', default=None, help='固定验证窗口 .npz(传入后启用均衡评估)')
    p.add_argument('--ts_weight', type=float, default=0.6, help='balanced 指标里 TinyStories 的权重')
    p.add_argument('--save_every', type=int, default=1500)
    p.add_argument('--keep_last', type=int, default=3)
    p.add_argument('--no_save', action='store_true', help='不存任何 checkpoint(搜索用)')
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--bytes_per_token', type=float, default=4.105,
                   help='旧评估模式下用于 bits-per-byte 的 bytes/token')
    p.add_argument('--run_name', default='model')
    p.add_argument('--train_data', default='../data/TinyStoriesV2-GPT4-train_small_tokens.bin')
    p.add_argument('--val_data', default='../data/TinyStoriesV2-GPT4-valid_tokens.bin')
    p.add_argument('--save_dir', default='../models')
    p.add_argument('--log_file', default='../models/train_log.txt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', default=None, help='从 checkpoint 继续，例如 ../models/model.pth')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)

    log_fp = open(args.log_file, 'a', encoding='utf-8') if args.log_file else None

    def log(msg):
        print(msg, flush=True)
        if log_fp:
            log_fp.write(msg + '\n')
            log_fp.flush()

    t0 = time.time()
    train_data = load_tokens(args.train_data)
    windows = FixedWindows(args.val_windows, args.device, args.ts_weight) if args.val_windows else None
    val_data = None if windows else load_tokens(args.val_data)

    tokens_per_step = args.batch_size * args.context_length * args.grad_accum
    total_steps = args.max_iters or math.ceil(len(train_data) / tokens_per_step)
    epochs = total_steps * tokens_per_step / len(train_data)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    min_lr = args.learning_rate * args.min_lr_ratio

    log('=' * 62)
    val_desc = f'{len(windows)} fixed windows' if windows else f'{len(val_data):,} tok'
    log(f'run={args.run_name}  train={len(train_data):,} tok  val={val_desc}')
    log(f'bs={args.batch_size} x accum {args.grad_accum} ctx={args.context_length} '
        f'-> {tokens_per_step} tok/step')
    log(f'steps={total_steps} ({epochs:.2f} epoch)  lr {args.learning_rate:.1e}->{min_lr:.1e}  '
        f'warmup={warmup}')
    log(f'dropout={args.dropout}  weight_decay={args.weight_decay}'
        f'{" (norms excluded)" if args.wd_exclude_norm else ""}  beta2={args.beta2}')
    save_desc = 'off' if args.no_save else f'every {args.save_every} (keep {args.keep_last})'
    log(f'eval every {args.eval_every}  save {save_desc}')
    log('=' * 62)

    model = TransformerLM(vocab_size=args.vocab_size, context_length=args.context_length,
                          d_model=args.d_model, num_layers=args.num_layers,
                          num_heads=args.num_heads, d_ff=args.d_ff, dropout=args.dropout,
                          device=args.device)
    n_params = sum(q.numel() for q in model.parameters())
    log(f'model params={n_params / 1e6:.2f}M  device={args.device}')
    if args.wd_exclude_norm:
        decay = [q for q in model.parameters() if q.dim() >= 2]
        no_decay = [q for q in model.parameters() if q.dim() < 2]
        param_groups = [{'params': decay, 'weight_decay': args.weight_decay},
                        {'params': no_decay, 'weight_decay': 0.0}]
    else:
        param_groups = model.parameters()
    optimizer = AdamW(param_groups, lr=args.learning_rate,
                      betas=(0.9, args.beta2), weight_decay=args.weight_decay)

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        start_iter = int(ckpt.get('iteration', 0))
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as exc:
                log(f'[resume] optimizer state 未恢复（{exc}），按新优化器继续')
        log(f'[resume] 从 {args.resume} 恢复，起始 step={start_iter}')

    def run_eval(it):
        """返回 (用于选 best 的分数, 日志行列表, 结果 dict)"""
        if windows:
            r = windows.evaluate(model, args.batch_size)
            lines = [f'[eval] step {it}  val_loss {r["balanced"]:.4f}  TS {r["ts"]:.4f}  '
                     f'GB {r["gb"]:.4f}  bpb_ts {r["bpb_ts"]:.4f}  bpb_gb {r["bpb_gb"]:.4f}',
                     '       books: ' + ' | '.join(f'{b} {v:.3f}' for b, v in r['books'].items())]
            return r['balanced'], lines, r
        vl = evaluate(model, val_data, args.batch_size, args.context_length,
                      args.device, args.eval_iters)
        bpb = vl / LN2 / args.bytes_per_token
        return vl, [f'[eval] step {it}  val_loss {vl:.4f}  ppl {math.exp(vl):.2f}  bpb {bpb:.4f}'], \
            {'balanced': vl, 'ts': vl, 'gb': float('nan')}

    best_val, last = float('inf'), None
    status = 'ok'
    model.train()
    for it in range(start_iter + 1, total_steps + 1):
        lr = float(get_learning_rate(it, args.learning_rate, min_lr, warmup, total_steps))
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = data_loader(train_data, args.batch_size, args.context_length, args.device)
            loss = loss_fn(model(x), y) / args.grad_accum
            loss.backward()
            step_loss += loss.item()
        if not math.isfinite(step_loss):
            log(f'[diverged] step {it}  loss={step_loss}')
            status = 'diverged'
            break
        clip_gradient(model.parameters(), 1.0)
        optimizer.step()

        if it == 1 or it % args.log_every == 0:
            log(f'iter {it}/{total_steps}  loss {step_loss:.4f}  lr {lr:.2e}  '
                f'elapsed {time.time() - t0:.1f}s')

        if it % args.eval_every == 0 or it == total_steps:
            score, lines, last = run_eval(it)
            if score < best_val:
                best_val = score
                if not args.no_save:
                    save_ckpt(model, optimizer, it, os.path.join(args.save_dir, 'model_best.pth'))
                lines[0] += '  <- best'
            for ln in lines:
                log(ln)

        if not args.no_save and it % args.save_every == 0:
            path = os.path.join(args.save_dir, f'{args.run_name}_{it:06d}.pth')
            save_ckpt(model, optimizer, it, path)
            prune_ckpts(args.save_dir, args.run_name, args.keep_last)
            log(f'[save] {path}')

    log('=' * 62)
    if status == 'ok':
        if last is None:
            # resume 到已完成步数时循环体一次都没跑, 补一次最终评估再汇总
            _, lines, last = run_eval(total_steps)
            for ln in lines:
                log(ln)
        if windows:
            # 固定窗口评估是确定的: 最后一步的评估就是最终结果, 不需要重复算
            final = last
            log(f'DONE steps={total_steps}  final_val={final["balanced"]:.4f}  '
                f'TS={final["ts"]:.4f}  GB={final["gb"]:.4f}')
            log(f'best_val={best_val:.4f}')
        else:
            final_vl = evaluate(model, val_data, args.batch_size, args.context_length,
                                args.device, args.eval_iters)
            final = {'balanced': final_vl, 'ts': final_vl, 'gb': float('nan')}
            log(f'DONE steps={total_steps}  final_val={final_vl:.4f}  '
                f'ppl={math.exp(final_vl):.2f}  bpb={final_vl / LN2 / args.bytes_per_token:.4f}')
            log(f'best_val={best_val:.4f}  bpb={best_val / LN2 / args.bytes_per_token:.4f}')
        if not args.no_save:
            final_path = os.path.join(args.save_dir, f'{args.run_name}_final.pth')
            save_ckpt(model, optimizer, total_steps, final_path)
            log(f'weights -> {final_path}')
            log(f'best    -> {os.path.join(args.save_dir, "model_best.pth")}')
        log(f'total {time.time() - t0:.1f}s')
        log(f'RESULT status=ok balanced={final["balanced"]:.6f} ts={final["ts"]:.6f} '
            f'gb={final["gb"]:.6f} best={best_val:.6f} seconds={time.time() - t0:.0f}')
    else:
        log(f'RESULT status=diverged balanced=nan ts=nan gb=nan best=nan '
            f'seconds={time.time() - t0:.0f}')
    log('=' * 62)
    if log_fp:
        log_fp.close()


if __name__ == '__main__':
    main()
