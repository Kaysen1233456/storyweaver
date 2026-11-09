"""构建均衡混合语料 + 固定验证窗口(16K 词表)。

配比:
  - TinyStories: 全部 train_small 故事, 不重复, 占约 60%
  - 文学(11 本书): 占约 40%; 书之间份额 ∝ sqrt(书的训练字符数), 每本书最多重复 MAX_REPEAT 遍,
    超出上限的份额按权重分给其他书(water-filling)
验证:
  - 每本书留出 8% 的块(至少 3 块), 不进训练集
  - 固定窗口: TinyStories 512 个 + 每本书最多 32 个, 每个窗口 257 token(输入 256 + 右移目标)
  - 记录每个窗口目标 token 的字节数, 可直接算 bpb

产出:
  ../data/mixed_bal_train.txt / mixed_bal_train_tokens.bin
  ../data/mixed_bal_val_windows.npz   (x, group, nbytes, names)
  ../data/mixed_bal_stats.json
"""
import argparse
import glob
import json
import math
import os
import random

import numpy as np
import regex as re

import build_mixed_corpus as bmc
from encode_corpus_fast import build_encoder, encode_file
from tokenizer import Tokenizer

SPECIAL = '<|endoftext|>'
RAW_DIR = '../data/sft_raw'
TS_TRAIN = '../data/TinyStoriesV2-GPT4-train_small.txt'
TS_VALID = '../data/TinyStoriesV2-GPT4-valid.txt'
VOCAB = '../data/mixed_v16k_vocab.json'
MERGES = '../data/mixed_v16k_merges.txt'

BOOK_NAMES = {
    'pg2591.txt': 'Grimm', 'pg1597.txt': 'Andersen', 'pg21.txt': 'Aesop',
    'pg19033.txt': 'Alice', 'pg113.txt': 'SecretGarden', 'pg514.txt': 'LittleWomen',
    'pg2554.txt': 'CrimePunishment', 'pg29000.txt': 'Trollope', 'pg30127.txt': 'Pickwick',
    'pg13415.txt': 'Chekhov', 'pg6897.txt': 'Marryat',
}


def load_books(max_chars, holdout_ratio, min_holdout, seed):
    books = {}
    for f in sorted(glob.glob(os.path.join(RAW_DIR, 'pg*.txt'))):
        base = os.path.basename(f)
        if base not in bmc.USE_BOOKS:
            continue
        text = bmc.normalize(bmc.clean_book(open(f, encoding='utf-8').read()))
        chunks = [c for c in bmc.split_chunks(text, max_chars) if bmc.english_ratio(c) >= 0.7]
        random.Random(f'{seed}-{base}').shuffle(chunks)
        n_hold = max(min_holdout, round(len(chunks) * holdout_ratio))
        books[BOOK_NAMES[base]] = {'val': chunks[:n_hold], 'train': chunks[n_hold:]}
    return books


def allocate(train_chars, lit_total, max_repeat):
    """water-filling: 份额 ∝ sqrt(chars), 重复次数 <= max_repeat。返回每本书的重复倍数(可为小数)。"""
    weight = {b: math.sqrt(c) for b, c in train_chars.items()}
    capped, repeat = set(), {}
    while True:
        remaining = lit_total - sum(max_repeat * train_chars[b] for b in capped)
        free = [b for b in train_chars if b not in capped]
        wsum = sum(weight[b] for b in free)
        newly = []
        for b in free:
            r = remaining * weight[b] / wsum / train_chars[b]
            repeat[b] = r
            if r > max_repeat:
                newly.append(b)
        if not newly:
            break
        capped.update(newly)
    for b in capped:
        repeat[b] = float(max_repeat)
    return repeat


def expand(chunks, r, rng):
    """把块列表扩成 r 倍: floor(r) 份完整复制 + 小数部分随机抽块"""
    whole = int(r)
    out = chunks * whole
    k = round((r - whole) * len(chunks))
    out += rng.sample(chunks, k)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ts_ratio', type=float, default=0.6)
    p.add_argument('--max_repeat', type=float, default=20)
    p.add_argument('--holdout_ratio', type=float, default=0.08)
    p.add_argument('--min_holdout', type=int, default=3)
    p.add_argument('--chunk_chars', type=int, default=2000)
    p.add_argument('--ts_val_windows', type=int, default=512)
    p.add_argument('--book_val_windows', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_stem', default='../data/mixed_bal')
    args = p.parse_args()
    rng = random.Random(args.seed)

    # ---------- 数据 ----------
    books = load_books(args.chunk_chars, args.holdout_ratio, args.min_holdout, args.seed)
    ts_stories = [s.strip() for s in open(TS_TRAIN, encoding='utf-8').read().split(SPECIAL) if s.strip()]
    ts_chars = sum(len(s) for s in ts_stories)
    train_chars = {b: sum(len(c) for c in d['train']) for b, d in books.items()}
    lit_total = ts_chars * (1 - args.ts_ratio) / args.ts_ratio
    repeat = allocate(train_chars, lit_total, args.max_repeat)

    items = list(ts_stories)
    stats = {'ts_chars': ts_chars, 'books': {}}
    for b, d in books.items():
        expanded = expand(d['train'], repeat[b], rng)
        items += expanded
        stats['books'][b] = {
            'train_chunks': len(d['train']), 'val_chunks': len(d['val']),
            'train_chars': train_chars[b], 'repeat': round(repeat[b], 2),
            'used_chars': sum(len(c) for c in expanded),
        }
    total = sum(len(s) for s in items)
    stats['total_chars'] = total

    print(f'TinyStories: {ts_chars:,} chars ({100 * ts_chars / total:.1f}%)')
    print(f'{"book":<16}{"train_chars":>12}{"repeat":>8}{"share":>8}')
    for b, s in sorted(stats['books'].items(), key=lambda kv: -kv[1]['used_chars']):
        s['share'] = round(100 * s['used_chars'] / total, 2)
        print(f'{b:<16}{s["train_chars"]:>12,}{s["repeat"]:>8.2f}{s["share"]:>7.2f}%')
    lit_used = sum(s['used_chars'] for s in stats['books'].values())
    print(f'literature total: {lit_used:,} chars ({100 * lit_used / total:.1f}%)')

    rng.shuffle(items)
    train_txt = args.out_stem + '_train.txt'
    with open(train_txt, 'w', encoding='utf-8') as f:
        f.write(f'{SPECIAL}\n'.join(items) + f'\n{SPECIAL}\n')

    # ---------- 编码训练集 ----------
    encode_word, eot_id, pat, n_merges = build_encoder(VOCAB, MERGES)
    print(f'vocab: {n_merges} merges, eot_id={eot_id}')
    n_tok = encode_file(train_txt, args.out_stem + '_train_tokens', encode_word, eot_id, pat,
                        report_every=40000)
    stats['train_tokens'] = n_tok

    # ---------- 固定验证窗口 ----------
    patc = re.compile(pat)

    def enc(s):
        return [t for m in patc.finditer(s) for t in encode_word(m.group().encode('utf-8'))]

    tok = Tokenizer.from_files(vocab_filepath=VOCAB, merges_filepath=MERGES,
                               special_tokens=[SPECIAL], split_token=' $~<split_token>~$ ')
    id_bytes = np.zeros(max(tok.token2id.values()) + 1, dtype=np.int64)
    for k, v in tok.token2id.items():
        id_bytes[v] = len(k)

    def windows_from(texts, limit):
        stream = []
        for s in texts:
            stream += enc(s) + [eot_id]
            if len(stream) >= 257 * limit:
                break
        n = min(limit, len(stream) // 257)
        return [stream[i * 257:(i + 1) * 257] for i in range(n)]

    names = ['TinyStories'] + sorted(books)
    xs, groups = [], []
    ts_val = [s.strip() for s in open(TS_VALID, encoding='utf-8').read().split(SPECIAL) if s.strip()]
    for w in windows_from(ts_val, args.ts_val_windows):
        xs.append(w); groups.append(0)
    for gi, b in enumerate(names[1:], start=1):
        ws = windows_from(books[b]['val'], args.book_val_windows)
        stats['books'][b]['val_windows'] = len(ws)
        for w in ws:
            xs.append(w); groups.append(gi)
    x = np.array(xs, dtype=np.int32)
    g = np.array(groups, dtype=np.int32)
    nbytes = id_bytes[x[:, 1:]].sum(axis=1)
    np.savez(args.out_stem + '_val_windows.npz', x=x, group=g, nbytes=nbytes, names=np.array(names))
    stats['val_windows'] = {n: int((g == i).sum()) for i, n in enumerate(names)}
    print('val windows:', stats['val_windows'])

    with open(args.out_stem + '_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'DONE train_tokens={n_tok:,}  -> {args.out_stem}_*')


if __name__ == '__main__':
    main()
