"""用 TinyStories 构建 SFT 数据(指令按故事内容提取, 基座没见过的故事)。

回答源
  完整 TinyStoriesV2-GPT4-train.txt 中 train_small 之后的部分(train_small 恰好是它开头的 100MiB,
  基座只在 train_small 上训练过, 所以这里取到的故事基座一条都没见过)
指令
  从故事开头提取主角(如 "a little girl named Lily" -> 形容词 little / 身份 girl / 名字 Lily),
  套用内容匹配的模板; 提取不到或按 generic_ratio 随机抽中时, 用不带具体要求的通用指令
样本
  sft_format.format_prompt(指令) + 故事 + <|endoftext|>; 超过 256 token 的整条丢弃(不截断, 保证故事有结尾)
产出
  ../data/sft_ts_{train,val}.bin + _meta.json  (sft_run.py 读取)
  ../data/sft_ts_sharegpt.json                  (人工查看用, ShareGPT 格式)
"""
import argparse
import json
import os
import random
from array import array

import regex as re

from encode_corpus_fast import build_encoder
from sft_format import format_prompt

SPECIAL = '<|endoftext|>'
FULL = '../data/TinyStoriesV2-GPT4-train.txt'
SMALL = '../data/TinyStoriesV2-GPT4-train_small.txt'
VOCAB = '../data/mixed_v16k_vocab.json'
MERGES = '../data/mixed_v16k_merges.txt'
CTX = 256

# "a little girl named Lily" / "a big dog called Max" / "a boy named Tim"
HERO = re.compile(
    r'\b(?:a|an)\s+(?:(?P<adj>[a-z]+)\s+)?(?P<role>[a-z]+)\s+(?:named|called)\s+(?P<name>[A-Z][a-z]{1,11})\b')
ADJ_STOP = {'very', 'old', 'new', 'other', 'certain'}

TEMPLATES_FULL = [
    'Write a story about a {adj} {role} named {name}.',
    'Tell me a story about a {adj} {role} called {name}.',
    'Write a short story for kids about {name}, a {adj} {role}.',
    'Can you write a bedtime story about a {adj} {role} named {name}?',
]
TEMPLATES_NAME_ROLE = [
    'Write a story about a {role} named {name}.',
    'Tell me a short story about {name} the {role}.',
    'Write a children\'s story about a {role} called {name}.',
]
TEMPLATES_GENERIC = [
    'Tell me a story.',
    'Write a short story for children.',
    'Tell me a bedtime story.',
    'Write a simple story for a young child.',
    'Can you tell me a short story?',
]


def make_instruction(story, rng, generic_ratio):
    m = HERO.search(story[:400])
    if m is None or rng.random() < generic_ratio:
        return rng.choice(TEMPLATES_GENERIC), 'generic'
    adj, role, name = m.group('adj'), m.group('role'), m.group('name')
    if adj and adj not in ADJ_STOP:
        return rng.choice(TEMPLATES_FULL).format(adj=adj, role=role, name=name), 'full'
    return rng.choice(TEMPLATES_NAME_ROLE).format(role=role, name=name), 'name_role'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--n_val', type=int, default=500)
    p.add_argument('--generic_ratio', type=float, default=0.2)
    p.add_argument('--read_mb', type=int, default=40, help='从 train_small 之后读取多少 MB')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_stem', default='../data/sft_ts')
    args = p.parse_args()
    rng = random.Random(args.seed)

    offset = os.path.getsize(SMALL)
    with open(FULL, 'rb') as f:
        f.seek(offset)
        raw = f.read(args.read_mb * 1024 * 1024).decode('utf-8', errors='ignore')
    parts = raw.split(SPECIAL)
    # 首段是被 seek 切开的半个故事, 末段可能被读取上限截断, 都丢掉
    stories = [s.strip() for s in parts[1:-1] if s.strip()]
    print(f'读取 train_small 之后 {args.read_mb}MB: {len(stories):,} 个完整故事')

    encode_word, eot_id, pat, _ = build_encoder(VOCAB, MERGES)
    patc = re.compile(pat)

    def enc(s):
        return [t for m in patc.finditer(s) for t in encode_word(m.group().encode('utf-8'))]

    need = args.n_train + args.n_val
    rows, kinds, dropped = [], {}, 0
    for story in stories:
        inst, kind = make_instruction(story, rng, args.generic_ratio)
        pids = enc(format_prompt(inst))
        rids = enc(story)
        if len(pids) + len(rids) + 1 > CTX:
            dropped += 1
            continue
        rows.append((inst, story, pids, rids))
        kinds[kind] = kinds.get(kind, 0) + 1
        if len(rows) >= need:
            break
    if len(rows) < need:
        raise SystemExit(f'只凑到 {len(rows)} 条, 少于需要的 {need} 条; 调大 --read_mb')
    print(f'样本 {len(rows):,} 条(因超过 {CTX} token 丢弃 {dropped:,} 条), 指令类型: {kinds}')

    rng.shuffle(rows)
    splits = {'val': rows[:args.n_val], 'train': rows[args.n_val:]}
    for name, part in splits.items():
        ids, meta = array('i'), []
        for _, _, pids, rids in part:
            ids.extend(pids)
            ids.extend(rids)
            ids.append(eot_id)
            meta.append([len(pids), len(pids) + len(rids) + 1])
        with open(f'{args.out_stem}_{name}.bin', 'wb') as f:
            ids.tofile(f)
        with open(f'{args.out_stem}_{name}_meta.json', 'w', encoding='utf-8') as f:
            json.dump({'samples': meta}, f)
        lens = [m[1] for m in meta]
        print(f'{name}: {len(part):,} 条, {len(ids):,} tokens, '
              f'样本长度 均值 {sum(lens) / len(lens):.0f} 最大 {max(lens)}')

    sharegpt = [{'conversations': [{'from': 'human', 'value': inst},
                                   {'from': 'gpt', 'value': story}]}
                for inst, story, _, _ in splits['train']]
    with open(f'{args.out_stem}_sharegpt.json', 'w', encoding='utf-8') as f:
        json.dump(sharegpt, f, ensure_ascii=False, indent=1)
    print(f'DONE -> {args.out_stem}_*')


if __name__ == '__main__':
    main()
