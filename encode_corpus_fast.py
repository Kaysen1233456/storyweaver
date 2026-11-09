"""流式编码 TinyStories 语料（运行副本专用，不改动原版实现）。

相对原版 tokenizer_data.py 的三处必要改动：
1. 不用 encode_parallel / multiprocessing.Pool —— Windows spawn 会递归派生子进程；
2. 把 merges 列表换成 rank 字典，避免 str_encode 里 O(9755) 的 list.index 线性查找；
3. 对重复 pre-token 做缓存，并同时输出文本 tokens.txt（兼容原版 train.py）
   与 int32 二进制 tokens.bin（供 train_run.py 用 np.fromfile 秒读）。

词表必须统一使用训练语料训出的那一套，valid 集也要用它编码。
"""
import argparse
import time
from array import array

import regex as re
from tokenizer import Tokenizer

SPECIAL = '<|endoftext|>'
SPLIT_TOKEN = ' $~<split_token>~$ '

DEFAULT_VOCAB = '../data/TinyStoriesV2-GPT4-train_small_vocab.json'
DEFAULT_MERGES = '../data/TinyStoriesV2-GPT4-train_small_merges.txt'

TASKS = {
    'valid': ('../data/TinyStoriesV2-GPT4-valid.txt',
              '../data/TinyStoriesV2-GPT4-valid_tokens'),
    'train': ('../data/TinyStoriesV2-GPT4-train_small.txt',
              '../data/TinyStoriesV2-GPT4-train_small_tokens'),
}


def build_encoder(vocab_path, merges_path):
    tok = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=[SPECIAL],
        split_token=SPLIT_TOKEN,
    )
    rank = {merged: i for i, merged in enumerate(tok.merges)}
    unk_id = tok.token2id.get(b'[UNK]', 0)
    eot_id = tok.token2id[SPECIAL.encode('utf-8')]
    pat = tok.PAT
    cache = {}

    def encode_word(word_bytes):
        cached = cache.get(word_bytes)
        if cached is not None:
            return cached
        parts = [bytes([b]) for b in word_bytes]
        while len(parts) > 1:
            best_i, best_rank = -1, len(rank) + 1
            for i in range(len(parts) - 1):
                r = rank.get(parts[i] + parts[i + 1])
                if r is not None and r < best_rank:
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            parts[best_i] = parts[best_i] + parts[best_i + 1]
            parts.pop(best_i + 1)
        result = tuple(tok.token2id.get(p, unk_id) for p in parts)
        cache[word_bytes] = result
        return result

    return encode_word, eot_id, pat, len(rank)


def encode_file(input_path, out_stem, encode_word, eot_id, pat, report_every=10000):
    started = time.time()
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    parts = data.split(SPECIAL)
    ids = array('i')
    text_buf = []

    with open(out_stem + '.txt', 'w', encoding='utf-8') as ftxt, \
            open(out_stem + '.bin', 'wb') as fbin:
        for idx, part in enumerate(parts):
            for match in re.finditer(pat, part):
                for tid in encode_word(match.group().encode('utf-8')):
                    ids.append(tid)
                    text_buf.append(str(tid))
                    if len(text_buf) >= 200000:
                        ftxt.write('\n'.join(text_buf) + '\n')
                        text_buf.clear()
            if idx < len(parts) - 1:
                ids.append(eot_id)
                text_buf.append(str(eot_id))
            if report_every and idx % report_every == 0:
                print(f'  [{out_stem}] {idx}/{len(parts)} stories, {len(ids)} tokens, '
                      f'{time.time() - started:.1f}s', flush=True)
        if text_buf:
            ftxt.write('\n'.join(text_buf) + '\n')
        ids.tofile(fbin)

    print(f'DONE {out_stem}: {len(ids)} tokens, '
          f'{time.time() - started:.1f}s -> .txt / .bin', flush=True)
    return len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['valid', 'train', 'both'], default='both')
    parser.add_argument('--vocab', default=DEFAULT_VOCAB)
    parser.add_argument('--merges', default=DEFAULT_MERGES)
    parser.add_argument('--input', default=None, help='自定义语料路径（给全量 train.txt 用）')
    parser.add_argument('--out_stem', default=None, help='输出前缀，不含 .txt/.bin')
    args = parser.parse_args()

    encode_word, eot_id, pat, n_merges = build_encoder(args.vocab, args.merges)
    print(f'vocab loaded: {n_merges} merges, eot_id={eot_id}', flush=True)

    if args.input:
        out_stem = args.out_stem or args.input.rsplit('.txt', 1)[0] + '_tokens'
        encode_file(args.input, out_stem, encode_word, eot_id, pat)
        return

    keys = ['valid', 'train'] if args.only == 'both' else [args.only]
    for key in keys:
        input_path, out_stem = TASKS[key]
        encode_file(input_path, out_stem, encode_word, eot_id, pat)


if __name__ == '__main__':
    main()
