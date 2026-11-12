"""终端里直观查看训练结果。

用法：
    python watch_loss.py                        # 看一次
    python watch_loss.py --watch 5              # 每 5 秒刷新
    python watch_loss.py ../models/other_log.txt

日志里有多次训练时(以 run= 行分界), 只显示最新一次的曲线;
底部列出日志里所有已结束训练的 RESULT(搜索时就是每组的结果)。
"""
import argparse
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ITER = re.compile(r'iter (\d+)/(\d+)\s+loss ([\d.]+|nan|inf)\s+lr ([\d.eE+-]+)\s+elapsed ([\d.]+)s')
EVAL = re.compile(r'\[eval\] step (\d+)\s+val_loss ([\d.]+)(?:.*?TS ([\d.]+)\s+GB ([\d.]+))?')
RUN = re.compile(r'^run=(\S+)')
RESULT = re.compile(r'RESULT status=(\w+) balanced=(\S+) ts=(\S+) gb=(\S+)')
BLOCKS = '▁▂▃▄▅▆▇█'


def parse(path):
    """返回 [(run_name, iters, evals, result)]，按日志顺序"""
    runs = []
    if not os.path.exists(path):
        return runs
    cur = None

    def ensure():
        nonlocal cur
        if cur is None:          # 兼容没有 run= 头的旧日志
            cur = {'name': '(unnamed)', 'iters': [], 'evals': [], 'result': None,
                   'books': '', 'done': False}
            runs.append(cur)
        return cur

    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = RUN.search(line)
            if m:
                cur = {'name': m.group(1), 'iters': [], 'evals': [], 'result': None,
                       'books': '', 'done': False}
                runs.append(cur)
                continue
            m = ITER.search(line)
            if m:
                ensure()['iters'].append((int(m.group(1)), int(m.group(2)), float(m.group(3)),
                                          float(m.group(4)), float(m.group(5))))
                continue
            m = EVAL.search(line)
            if m:
                ensure()['evals'].append((int(m.group(1)), float(m.group(2)),
                                          float(m.group(3)) if m.group(3) else None,
                                          float(m.group(4)) if m.group(4) else None,
                                          '<- best' in line))
                continue
            if cur is None:
                continue
            if line.strip().startswith('books:'):
                cur['books'] = line.strip()[len('books:'):].strip()
            elif line.startswith('DONE steps=') or line.startswith('[diverged]'):
                cur['done'] = True
            else:
                m = RESULT.search(line)
                if m:
                    cur['result'] = (m.group(1), m.group(2), m.group(3), m.group(4))
                    cur['done'] = True
    return runs


def spark(vals, width=58):
    if not vals:
        return ''
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return BLOCKS[0] * len(vals)
    return ''.join(BLOCKS[min(7, int((hi - v) / (hi - lo) * 7))] for v in vals)


def render(runs, log_path):
    print('=' * 62)
    print(' LLM from Scratch · 训练监控')
    print(f' 日志 {log_path}')
    print('=' * 62)
    if not runs or not runs[-1]['iters']:
        print(' 还没有训练记录（训练还没开始写日志，或路径不对）。')
        print('=' * 62)
        return

    run = runs[-1]
    rows = run['iters']
    it, total, loss, lr, el = rows[-1]
    pct = it / total * 100 if total else 0
    filled = int(30 * pct / 100)
    spd = el / it if it else 0
    state = '已结束' if run['done'] else '进行中'

    print(f' 当前训练 {run["name"]}  ({state}，日志中第 {len(runs)} 次训练)')
    print(f' 进度   {it}/{total}  [{"#" * filled}{"." * (30 - filled)}] {pct:5.1f}%')
    print(f' loss   {rows[0][2]:.4f} -> {loss:.4f}')
    print(f' lr     {lr:.2e}    速度 {spd:.3f} s/step   elapsed {el:.1f}s')
    if not run['done'] and it < total and spd:
        print(f' 预计   本次训练剩余约 {(total - it) * spd / 60:.1f} 分钟')
    print('-' * 62)
    print(' 训练 loss 曲线（柱越高 = loss 越小）')
    print(' ' + spark([r[2] for r in rows]))

    if run['evals']:
        print('-' * 62)
        print(' 验证（val = 均衡指标；TS = TinyStories；GB = 11 本书平均）')
        for step, val, ts, gb, best in run['evals'][-6:]:
            extra = f'   TS {ts:.4f}   GB {gb:.4f}' if ts is not None else ''
            print(f'   step {step:>6}   val {val:.4f}{extra}{"   <- best" if best else ""}')
        if run['books']:
            print(' 最近一次每本书 loss:')
            parts = run['books'].split(' | ')
            for i in range(0, len(parts), 3):
                print('   ' + ' | '.join(parts[i:i + 3]))

    done = [r for r in runs if r['result']]
    if done:
        print('-' * 62)
        print(' 日志中已结束的训练')
        for r in done:
            st, bal, ts, gb = r['result']
            print(f'   {r["name"]:<28} {st:<9} val {bal[:7]:<8} TS {ts[:6]:<7} GB {gb[:6]}')
    print('=' * 62)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log', nargs='?', default='../models/train_log.txt')
    p.add_argument('--watch', type=float, default=0, help='每 N 秒自动刷新，Ctrl+C 退出')
    args = p.parse_args()

    if args.watch:
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                render(parse(args.log), args.log)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print('\n已退出监控。')
    else:
        render(parse(args.log), args.log)


if __name__ == '__main__':
    main()
