"""把 Gutenberg 原始文本做成 SFT 数据集（ShareGPT 格式 + BPE token 二进制）。

流程:
1. 逐篇读取 ../data/sft_raw/pg*.txt
2. 清洗: 去目录/插图标记/多余空白; 按 Gutenberg 约定切故事
   (优先按标题分隔, 无法识别标题的书按段落块切)
3. 每段截到 max_story_tokens(默认约 220 token 对应字符)以内
4. 合成指令: 从故事首句提取主角名/形容词, 套 5-10 种模板随机生成
5. 输出:
   - ../data/sft_sharegpt.json   (人可读, ShareGPT 格式, 供检查)
   - ../data/sft_train.bin       (int32: prompt_ids + response_ids + eot, 前缀长度另存)
   - ../data/sft_val.bin         (验证集, 同格式)
   - ../data/sft_meta.json       (prompt_len 数组与样本统计)

.bin 布局: 每条样本 = [prompt tokens...][response tokens...][eot]
           sft_run.py 用 prompt_len 区分 mask 区域 (-100 只在 prompt 段)。
"""
import argparse
import glob
import json
import os
import random
from array import array

import regex as re

from encode_corpus_fast import build_encoder

SPECIAL = '<|endoftext|>'
RAW_DIR = '../data/sft_raw'
OUT_DIR = '../data'

# 排版归一化: 与 build_mixed_corpus.py 保持一致(弯引号/连字/em dash -> ASCII)
CHAR_MAP = {
    '\u201c': '"', '\u201d': '"',
    '\u2018': "'", '\u2019': "'",
    '\u2014': ' -- ', '\u2013': ' - ',
    '\u2026': '...',
    '\ufb01': 'fi', '\ufb02': 'fl',
    '\u00a0': ' ', '\t': ' ',
}


def normalize(text):
    for k, v in CHAR_MAP.items():
        text = text.replace(k, v)
    return text

# ---------- 清洗 ----------

DROP_PATS = [
    re.compile(r'^\s*(?:[IVXLC]+\.\s*)?[A-Z][A-Z \'\",;:!?\-—]+$'),   # 全大写标题行/目录行
    re.compile(r'^\s*\[Illustration.*\]?\s*$', re.IGNORECASE),
    re.compile(r'^\s*Contents\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?:Chapter|CHAPTER)\s+[IVXLC0-9]+.*$'),
    re.compile(r'^\s*\*{2,}.*$'),
    re.compile(r'^\s*[-_—=]{3,}\s*$'),
]

# Gutenberg 故事集常见的标题分隔样式, 如:
# [THE FROG KING] / THE FROG KING. / Grimm-style 全大写行后跟空行
STORY_HEAD = re.compile(r'^\s*[\[\(]?([A-Z][A-Z\'\",;:!\?—\-\.\s]{3,60})[\]\)]?\s*$')

ADJ_POOL = ['brave', 'kind', 'clever', 'curious', 'little', 'happy',
            'gentle', 'playful', 'shy', 'adventurous']
ROLE_POOL = ['girl', 'boy', 'child', 'prince', 'princess', 'farmer',
             'woodcutter', 'king', 'queen', 'animal', 'dog', 'cat', 'bird']


def clean_text(text):
    """与 build_mixed_corpus.clean_book 对齐: 行拼流、删排版噪声, 不保留硬换行"""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if any(p.search(s) for p in DROP_PATS):
            continue
        lines.append(s)   # '--' 保持原样: 与 mixed_corpus 一致(CHAR_MAP 已把 em dash 归一成 ' -- ')
    return ' '.join(lines)


def split_stories(text):
    """按全大写标题行切块; 块太少的书退化为按空行段落聚合。"""
    blocks, cur = [], []
    for ln in text.splitlines():
        if STORY_HEAD.match(ln) and cur:
            blocks.append('\n'.join(cur).strip())
            cur = []
        cur.append(ln)
    if cur:
        blocks.append('\n'.join(cur).strip())
    return [b for b in blocks if len(b) > 400]  # 太短的丢弃


SENT_END = re.compile(r'(?<=[.!?”"])?\s+(?=[A-Z])')


def split_sentences(story):
    """把故事块切成句子列表，用于从句界开始拼样本。"""
    return [s.strip() for s in re.split(r'(?<=[.!?…”])\s+', story) if s.strip()]


def chunk_story(story, max_chars):
    """长故事切成 max_chars 左右的块，从句号处开始与结束，不产生半句开头。"""
    sents = split_sentences(story)
    chunks, cur = [], ''
    for s in sents:
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + ' ' + s).strip()
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 200]


# ---------- 指令合成 ----------

NAME_RE = re.compile(r'\b(?:named|called)\s+([A-Z][a-z]{1,11})\b')

TEMPLATES = [
    'Write a short story about a {adj} {role} named {name}.',
    'Write a story about a {adj} {role}.',
    'Tell me a fairy tale about a {role} named {name}.',
    'Tell me a fairy tale.',
    'Write a bedtime story for children.',
    'Write a short story for kids about a {adj} {role}.',
    'Please write a children\'s story about a {role} named {name}.',
]


def make_instruction(story, rng):
    head = story[:300]
    m = NAME_RE.search(head)
    name = m.group(1) if m else rng.choice(['Lily', 'Tom', 'Anna', 'Jack', 'Tim', 'Mia'])
    adj = rng.choice(ADJ_POOL)
    role = rng.choice(ROLE_POOL)
    tpl = rng.choice(TEMPLATES)
    try:
        return tpl.format(adj=adj, role=role, name=name)
    except KeyError:
        return tpl


# ---------- 主流程 ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_story_chars', type=int, default=1000,
                   help='单条回答的最大字符数(约 220 token)')
    p.add_argument('--max_samples', type=int, default=6000)
    p.add_argument('--val_ratio', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--raw_dir', default=RAW_DIR)
    p.add_argument('--out_dir', default=OUT_DIR)
    p.add_argument('--vocab', default='../data/TinyStoriesV2-GPT4-train_small_vocab.json')
    p.add_argument('--merges', default='../data/TinyStoriesV2-GPT4-train_small_merges.txt')
    args = p.parse_args()

    rng = random.Random(args.seed)
    encode_word, eot_id, pat, n_merges = build_encoder(args.vocab, args.merges)
    pat = re.compile(pat)
    print(f'vocab loaded: {n_merges} merges, eot_id={eot_id}')

    files = sorted(glob.glob(os.path.join(args.raw_dir, 'pg*.txt')))
    manifest_path = os.path.join(args.raw_dir, 'manifest.json')
    grades = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            grades = {v['file']: v.get('grade', 'B') for v in json.load(f).values()}
    # 只用 A 级（儿童童话/寓言），B 级（成人文学）不进 SFT
    files = [f for f in files if grades.get(os.path.basename(f), 'B') == 'A']
    print(f'raw books (grade A): {len(files)}')

    samples = []   # (instruction, story)
    for path in files:
        with open(path, encoding='utf-8') as f:
            text = f.read()
        text = normalize(clean_text(text))
        stories = split_stories(text)
        for st in stories:
            for chunk in chunk_story(st, args.max_story_chars):
                if len(chunk) < 200:
                    continue
                samples.append((make_instruction(chunk, rng), chunk))

    rng.shuffle(samples)
    samples = samples[: args.max_samples]
    print(f'samples: {len(samples)}')

    # ShareGPT 人可读输出
    sharegpt = [{'conversations': [
        {'from': 'human', 'value': inst},
        {'from': 'gpt', 'value': story},
    ]} for inst, story in samples]
    sg_path = os.path.join(args.out_dir, 'sft_sharegpt.json')
    with open(sg_path, 'w', encoding='utf-8') as f:
        json.dump(sharegpt, f, ensure_ascii=False, indent=1)
    print(f'sharegpt -> {sg_path}')

    # BPE 编码: 每条 = prompt tokens + response tokens + eot
    def encode(s):
        return [tid for m in pat.finditer(s)
                for tid in encode_word(m.group().encode('utf-8'))]

    n_val = max(1, int(len(samples) * args.val_ratio))
    val, train = samples[:n_val], samples[n_val:]

    def trim_to_budget(story, budget_tokens):
        """把故事裁到 budget_tokens 个 token 以内, 且尽量在句子边界结束。"""
        if len(encode(story)) <= budget_tokens:
            return story
        est_chars = int(budget_tokens * 4.105)
        cut = story[:est_chars + 300]
        sents = [s.strip() for s in re.split(r'(?<=[.!?…”])\s+', cut) if s.strip()]
        out = ''
        for s in sents:
            cand = (out + ' ' + s).strip()
            if len(encode(cand)) > budget_tokens:
                break
            out = cand
        if out:
            return out
        # 单句都超预算: 按字符硬裁再兜底重裁
        for keep in (est_chars, est_chars // 2, est_chars // 4):
            cand = story[:keep]
            if len(encode(cand)) <= budget_tokens:
                return cand
        return story[:est_chars // 4]

    def dump(rows, stem):
        ids, sample_meta = array('i'), []
        for inst, story in rows:
            pids = encode(inst)
            budget = 254 - len(pids)
            story2 = trim_to_budget(story, budget)
            rids = encode(story2)
            assert len(pids) + len(rids) + 1 <= 255
            ids.extend(pids)
            ids.extend(rids)
            ids.append(eot_id)
            sample_meta.append([len(pids), len(pids) + len(rids) + 1])
        bin_path = os.path.join(args.out_dir, stem + '.bin')
        meta_path = os.path.join(args.out_dir, stem + '_meta.json')
        with open(bin_path, 'wb') as f:
            ids.tofile(f)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'samples': sample_meta}, f)
        print(f'{stem}: {len(rows)} samples, {len(ids)} tokens -> {bin_path}')
        return len(ids)

    dump(train, 'sft_train')
    dump(val, 'sft_val')

    avg_p = sum(len(encode(i)) for i, _ in samples[:200]) / 200
    print(f'avg prompt tokens(前200条): {avg_p:.1f}')
    print('DONE')


if __name__ == '__main__':
    main()
