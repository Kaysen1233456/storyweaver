"""构建混合预训练语料: TinyStories + Gutenberg 英文小说/童话。

产出: ../data/mixed_train.txt  —— 故事之间用 <|endoftext|> 分隔(与原格式一致)
       ../data/mixed_stats.txt —— 混合统计

处理:
1. Gutenberg: 归一化排版(弯引号/连字/em dash), 英文占比过滤, 按标题/段落切故事,
   每块保留完整句界, 输出为一段段独立"故事"
2. TinyStories: 直接用现有 train_small.txt(已是 <|endoftext|> 分隔), 原样并入
3. 混合比例: 打乱顺序后输出
"""
import argparse
import glob
import json
import os
import random
import re

import regex as re2  # noqa: F401 (保留与 tokenizer 一致的依赖)

SPECIAL = '<|endoftext|>'
RAW_DIR = '../data/sft_raw'
TS_PATH = '../data/TinyStoriesV2-GPT4-train_small.txt'

# A 级(儿童文学) + B 级英文小说(剔除 19037 工程论文)
USE_BOOKS = {
    'pg2591.txt', 'pg1597.txt', 'pg21.txt', 'pg19033.txt', 'pg113.txt', 'pg514.txt',
    'pg2554.txt', 'pg29000.txt', 'pg30127.txt', 'pg13415.txt', 'pg6897.txt',
}

# 排版归一化(向 TinyStories 风格靠拢: 纯 ASCII)
CHAR_MAP = {
    '\u201c': '"', '\u201d': '"',   # “ ” -> "
    '\u2018': "'", '\u2019': "'",   # ‘ ’ -> '
    '\u2014': ' -- ', '\u2013': ' - ',  # em/en dash
    '\u2026': '...',                # ellipsis
    '\ufb01': 'fi', '\ufb02': 'fl',  # ligatures
    '\u00a0': ' ', '\t': ' ',
}


def normalize(text):
    for k, v in CHAR_MAP.items():
        text = text.replace(k, v)
    return text


WORD_RE = re2.compile(r'\b[A-Za-z]+\b')


def english_ratio(text):
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    return len(words) / max(1, len(text.split()))


def clean_book(text):
    # 去掉 Gutenberg 残留的目录行/插图标记/Produced by 行
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.match(r'^\s*(\[Illustration|Produced by|Title:|Author:|Release Date|Language:|Character set)', s, re.IGNORECASE):
            continue
        if re.match(r'^\s*\*{2,}.*$', s):
            continue
        if re.match(r'^\s*(?:[IVXLC]+\.)?\s*[A-Z][A-Z\'\",;:!?\-\u2014\s]{3,}$', s):   # 全大写标题/目录行
            continue
        lines.append(s)
    # 行间空行全部已去: 直接拼成流, 行间用空格(处理硬换行)
    return ' '.join(lines)


def split_chunks(text, max_chars, min_chars=300):
    """全书按句界切成 max_chars 左右的块"""
    sents = [s.strip() for s in re.split(r'(?<=[.!?"])\s+', text) if s.strip()]
    chunks, cur = [], ''
    for s in sents:
        if len(cur) + len(s) + 1 > max_chars and cur:
            if len(cur) >= min_chars:
                chunks.append(cur)
            cur = s
        else:
            cur = (cur + ' ' + s).strip()
    if cur and len(cur) >= min_chars:
        chunks.append(cur)
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_para_chars', type=int, default=2000)
    p.add_argument('--gb_repeat', type=int, default=4,
                   help='Gutenberg 块复制份数(过采样): 4 份 ≈ 占比 19%')
    p.add_argument('--holdout', type=int, default=300,
                   help='Gutenberg 留出 N 块作验证集(不进训练集)')
    p.add_argument('--out', default='../data/mixed_train.txt')
    args = p.parse_args()

    rng = random.Random(args.seed)

    # --- Gutenberg 部分: 整书清成流式文本再按句切块 ---
    gb_paras = []
    files = sorted(glob.glob(os.path.join(RAW_DIR, 'pg*.txt')))
    used = [f for f in files if os.path.basename(f) in USE_BOOKS]
    print(f'gutenberg books: {len(used)}')
    for f in used:
        with open(f, encoding='utf-8') as fh:
            text = normalize(clean_book(fh.read()))
        chunks = split_chunks(text, args.max_para_chars)
        kept = 0
        for chunk in chunks:
            if english_ratio(chunk) < 0.7:
                continue
            gb_paras.append(chunk)
            kept += 1
        print(f'  {os.path.basename(f)}: {kept} chunks, {sum(len(c) for c in chunks)} chars')
    print(f'gutenberg total chunks: {len(gb_paras)}, chars: {sum(len(x) for x in gb_paras)}')

    # 留出验证块(不进训练集)
    rng.shuffle(gb_paras)
    gb_val = gb_paras[: args.holdout]
    gb_train = gb_paras[args.holdout:]
    print(f'holdout: {len(gb_val)} chunks -> valid')
    # 过采样: Gutenberg 训练块复制 gb_repeat 份
    gb_oversampled = gb_train * args.gb_repeat
    print(f'gutenberg train chunks: {len(gb_train)} x{args.gb_repeat} = {len(gb_oversampled)}')

    # --- TinyStories 部分 ---
    with open(TS_PATH, encoding='utf-8') as f:
        ts = f.read()
    ts_stories = [s.strip() for s in ts.split(SPECIAL) if s.strip()]
    print(f'tinystories: {len(ts_stories)} stories, chars: {sum(len(s) for s in ts_stories)}')

    # --- 合并: Gutenberg 段落视作独立"故事", 与 TinyStories 混洗 ---
    all_items = ts_stories + gb_oversampled
    rng.shuffle(all_items)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(f'{SPECIAL}\n'.join(all_items) + f'\n{SPECIAL}\n')
    total = sum(len(s) for s in all_items)
    # 混合验证集: TinyStories valid 原样 + Gutenberg holdout
    with open('../data/TinyStoriesV2-GPT4-valid.txt', encoding='utf-8') as f:
        ts_val = f.read()
    ts_val_stories = [s.strip() for s in ts_val.split(SPECIAL) if s.strip()]
    val_items = ts_val_stories + gb_val
    rng.shuffle(val_items)
    with open('../data/mixed_valid.txt', 'w', encoding='utf-8') as f:
        f.write(f'{SPECIAL}\n'.join(val_items) + f'\n{SPECIAL}\n')
    print(f'mixed_valid -> {len(val_items)} items')

    with open('../data/mixed_stats.txt', 'w', encoding='utf-8') as f:
        f.write(f'total_items={len(all_items)}\ntotal_chars={total}\n'
                f'ts_stories={len(ts_stories)}\ngb_train_chunks={len(gb_train)}x{args.gb_repeat}\n')
    print(f'DONE -> {args.out}: {len(all_items)} items, {total} chars')
    gb_chars = sum(len(x) for x in gb_oversampled)
    print(f'gutenberg 过采样后占比: {100*gb_chars/total:.1f}% (chars)')


if __name__ == '__main__':
    main()
