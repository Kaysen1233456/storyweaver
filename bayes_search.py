"""高斯过程贝叶斯搜索: 在等 token 预算下搜 learning rate x 等效 batch size。

搜索空间
  lr      : 2e-4 ~ 2e-3, 对数尺度
  eff bs  : 16 / 32 / 64  (micro-batch 固定 16, 梯度累积 1 / 2 / 4, 显存不变)
公平性
  每组训练完全相同的 token 数(默认 8,192,000), 步数 = 预算 / (bs * ctx), 各自一条完整 cosine
  所有组同一随机种子、同一套固定验证窗口, 目标 = 最后一次评估的均衡指标(越低越好)
流程
  前 3 组用之前网格搜索的位置做起点, 之后每组:
  高斯过程(Matern 2.5 + 噪声项)拟合 "超参 -> 指标", 在候选网格上算 Expected Improvement, 取最大
  训练发散的组记为 "当前最差 + 0.5", 让模型学到那片区域不可取
可续跑
  结果写在 --out_dir/results.json; 中断后重跑同一条命令, 已完成的组会跳过

用法
  python bayes_search.py                 # 正式搜索(8 组)
  python bayes_search.py --dry_run       # 用模拟目标函数演示拟合与挑选, 不训练
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# 点很少时 GP 超参容易贴边界, 属正常现象; 屏蔽警告, 避免 stderr 输出被 PowerShell 当成报错
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# lr 下限取 2e-4: 前两轮搜索里 3e-4 始终差于 6e-4, 更小的 lr 基本不可能最优,
# 范围太宽会让 EI 把试验浪费在没有希望的角落
LR_MIN, LR_MAX = 2e-4, 2e-3
BS_CHOICES = (16, 32, 64)
MICRO_BS = 16
INIT_POINTS = [(3e-4, 32), (6e-4, 32), (6e-4, 16)]   # 来自之前网格搜索的位置
RESULT_RE = re.compile(r'RESULT status=(\w+) balanced=(\S+) ts=(\S+) gb=(\S+)')

# 所有组共用的训练参数(除 lr / bs 外与最终训练一致, dropout 固定 0)
FIXED_ARGS = [
    '--vocab_size', '16000', '--d_model', '640', '--num_layers', '6',
    '--num_heads', '10', '--d_ff', '1728', '--context_length', '256',
    '--dropout', '0', '--weight_decay', '0.1', '--wd_exclude_norm', '--beta2', '0.95',
    '--train_data', '../data/mixed_bal_train_tokens.bin',
    '--val_windows', '../data/mixed_bal_val_windows.npz',
    '--no_save', '--seed', '0', '--device', 'cuda:0',
]


# ---------- 编码: 超参 <-> [0,1]^2 ----------
def encode(lr, bs):
    return [(math.log10(lr) - math.log10(LR_MIN)) / (math.log10(LR_MAX) - math.log10(LR_MIN)),
            (math.log2(bs) - 4) / 2]


def candidates(n_lr=161):
    lrs = np.logspace(math.log10(LR_MIN), math.log10(LR_MAX), n_lr)
    return [(float(lr), bs) for bs in BS_CHOICES for lr in lrs]


# ---------- 高斯过程 + EI ----------
def fit_gp(X, y):
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=[0.3, 0.5], length_scale_bounds=(0.05, 5.0), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=8, random_state=0)
    gp.fit(np.array(X), np.array(y))
    return gp


def expected_improvement(gp, Xc, best, xi=0.01):
    mu, sd = gp.predict(np.array(Xc), return_std=True)
    sd = np.maximum(sd, 1e-9)
    imp = best - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z), mu, sd


def objective_values(trials):
    """把发散的组换成 '当前最差 + 0.5', 返回 (X, y)"""
    finite = [t['balanced'] for t in trials if t['status'] == 'ok']
    worst = max(finite) if finite else 10.0
    X = [encode(t['lr'], t['bs']) for t in trials]
    y = [t['balanced'] if t['status'] == 'ok' else worst + 0.5 for t in trials]
    return X, y


def suggest(trials):
    X, y = objective_values(trials)
    gp = fit_gp(X, y)
    tried = {(round(math.log10(t['lr']), 2), t['bs']) for t in trials}
    cands = [c for c in candidates() if (round(math.log10(c[0]), 2), c[1]) not in tried]
    ei, mu, sd = expected_improvement(gp, [encode(*c) for c in cands], min(y))
    i = int(np.argmax(ei))
    return cands[i], {'ei': float(ei[i]), 'pred_mean': float(mu[i]), 'pred_std': float(sd[i])}, gp


# ---------- 单组训练 ----------
def run_trial(idx, lr, bs, args):
    accum = bs // MICRO_BS
    steps = int(args.token_budget // (bs * 256))
    eval_every = max(1, steps // 4)
    name = f'bo{idx:02d}_lr{lr:.2e}_bs{bs}'
    cmd = [sys.executable, 'train_run.py', *FIXED_ARGS,
           '--batch_size', str(MICRO_BS), '--grad_accum', str(accum),
           '--learning_rate', f'{lr:.6g}', '--max_iters', str(steps),
           '--eval_every', str(eval_every), '--log_every', str(max(10, steps // 20)),
           '--run_name', name, '--save_dir', args.out_dir, '--log_file', args.log_file]
    t0 = time.time()
    proc = subprocess.run(cmd)
    result = None
    if os.path.exists(args.log_file):
        lines = open(args.log_file, encoding='utf-8', errors='replace').read().splitlines()
        start = max((i for i, ln in enumerate(lines) if ln.startswith(f'run={name}')), default=None)
        if start is not None:
            for ln in lines[start:]:
                m = RESULT_RE.search(ln)
                if m:
                    result = m
    if result is None:
        raise RuntimeError(f'第 {idx} 组没有输出 RESULT(进程返回码 {proc.returncode}), '
                           f'搜索已停止。查看日志 {args.log_file} 末尾的报错。')
    status = result.group(1)
    return {
        'id': idx, 'name': name, 'lr': lr, 'bs': bs, 'accum': accum, 'steps': steps,
        'status': status,
        'balanced': float(result.group(2)) if status == 'ok' else None,
        'ts': float(result.group(3)) if status == 'ok' else None,
        'gb': float(result.group(4)) if status == 'ok' else None,
        'seconds': round(time.time() - t0),
    }


def fake_trial(idx, lr, bs, args, rng):
    """dry-run 用的模拟目标: 最优点在 lr≈7e-4、bs=32 附近, 加少量噪声; lr 太大时 '发散'"""
    if lr > 1.6e-3 and bs == 16:
        status, val = 'diverged', None
    else:
        u = math.log10(lr) - math.log10(7e-4)
        val = 2.0 + 0.9 * u * u + 0.04 * abs(math.log2(bs) - 5) + rng.normal(0, 0.005)
        status = 'ok'
    steps = int(args.token_budget // (bs * 256))
    return {'id': idx, 'name': f'dry{idx:02d}', 'lr': lr, 'bs': bs, 'accum': bs // MICRO_BS,
            'steps': steps, 'status': status, 'balanced': val, 'ts': val, 'gb': val, 'seconds': 0}


# ---------- 汇总 ----------
def print_table(trials):
    print('\n' + '=' * 72)
    print(f'{"#":>3} {"来源":<5}{"lr":>10}{"bs":>5}{"steps":>7}{"状态":>9}{"均衡":>9}{"TS":>8}{"GB":>8}')
    best = min((t for t in trials if t['status'] == 'ok'), key=lambda t: t['balanced'], default=None)
    for t in trials:
        v = lambda k: f'{t[k]:.4f}' if t[k] is not None else '   -'
        mark = '  <- best' if best is not None and t is best else ''
        print(f'{t["id"]:>3} {t["source"]:<5}{t["lr"]:>10.2e}{t["bs"]:>5}{t["steps"]:>7}'
              f'{t["status"]:>9}{v("balanced"):>9}{v("ts"):>8}{v("gb"):>8}{mark}')
    print('=' * 72)
    return best


def print_surface(gp):
    lrs = [2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3]
    print('高斯过程对整个搜索空间的预测(均衡指标 均值±标准差):')
    print(f'{"lr":>10}' + ''.join(f'{"bs=" + str(b):>16}' for b in BS_CHOICES))
    for lr in lrs:
        mu, sd = gp.predict(np.array([encode(lr, b) for b in BS_CHOICES]), return_std=True)
        print(f'{lr:>10.1e}' + ''.join(f'{m:>10.4f}±{s:<5.3f}' for m, s in zip(mu, sd)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_trials', type=int, default=8)
    p.add_argument('--token_budget', type=float, default=8_192_000)
    p.add_argument('--out_dir', default='../models/search_bo')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.log_file = os.path.join(args.out_dir, 'search_log.txt')
    res_path = os.path.join(args.out_dir, 'results_dryrun.json' if args.dry_run else 'results.json')
    trials = json.load(open(res_path, encoding='utf-8')) if os.path.exists(res_path) and not args.dry_run else []
    if trials:
        print(f'[resume] 已完成 {len(trials)} 组, 从第 {len(trials) + 1} 组继续')
    rng = np.random.default_rng(0)

    while len(trials) < args.n_trials:
        idx = len(trials) + 1
        if idx <= len(INIT_POINTS):
            (lr, bs), source, info, gp = INIT_POINTS[idx - 1], 'init', None, None
        else:
            (lr, bs), info, gp = suggest(trials)
            source = 'bo'
        steps = int(args.token_budget // (bs * 256))
        print(f'\n>>> 第 {idx}/{args.n_trials} 组  lr={lr:.2e}  bs={bs}  steps={steps}  ({source})')
        if info:
            print(f'    挑选依据: 预测 {info["pred_mean"]:.4f}±{info["pred_std"]:.4f}  EI={info["ei"]:.2e}')
        t = fake_trial(idx, lr, bs, args, rng) if args.dry_run else run_trial(idx, lr, bs, args)
        t['source'] = source
        if info:
            t['suggest_info'] = info
        trials.append(t)
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(trials, f, ensure_ascii=False, indent=2)
        print_table(trials)

    best = print_table(trials)
    X, y = objective_values(trials)
    print_surface(fit_gp(X, y))
    if best:
        print(f'\n最优组合: lr={best["lr"]:.2e}  bs={best["bs"]}  '
              f'均衡={best["balanced"]:.4f}  TS={best["ts"]:.4f}  GB={best["gb"]:.4f}')
    print(f'结果文件: {res_path}')
    print('SEARCH_ALL_DONE')


if __name__ == '__main__':
    main()
