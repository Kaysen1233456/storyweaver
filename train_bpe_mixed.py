"""在混合语料(TinyStories + Gutenberg 过采样)上训练 BPE 词表。

用法:
    python train_bpe_mixed.py                      # 默认 10000, 输出 mixed_vocab.json
    python train_bpe_mixed.py --vocab_size 16000 --tag v16k
        -> ../data/mixed_v16k_vocab.json / mixed_v16k_merges.txt
"""
import argparse
import json
import os
import time

import psutil
from train_bpe import train_bpe

p = argparse.ArgumentParser()
p.add_argument('--vocab_size', type=int, default=10000)
p.add_argument('--input', default='../data/mixed_train.txt')
p.add_argument('--tag', default='', help='输出文件名后缀, 例如 v16k')
args = p.parse_args()

stem = f'../data/mixed_{args.tag}_' if args.tag else '../data/mixed_'
vocab_path = stem + 'vocab.json'
merges_path = stem + 'merges.txt'
split_token = ' $~<split_token>~$ '

process = psutil.Process(os.getpid())
start_time = time.time()
start_memory = process.memory_info().rss / 1024 / 1024

vocab, merges = train_bpe(args.input, args.vocab_size, ['<|endoftext|>', '[UNK]'])

print(f"Training time: {(time.time() - start_time) / 3600:.4f} hours")
print(f"Memory used: {process.memory_info().rss / 1024 / 1024 - start_memory:.2f} MB")

vocab_str = {k: v.decode('utf-8', errors='replace') for k, v in vocab.items()}
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab_str, f, ensure_ascii=False, indent=4)

with open(merges_path, 'w', encoding='utf-8') as f:
    for pair in merges:
        token1 = pair[0].decode('utf-8', errors='replace')
        token2 = pair[1].decode('utf-8', errors='replace')
        f.write(f"{token1}{split_token}{token2}\n")

print(f'vocab  -> {vocab_path}  ({len(vocab)} tokens)')
print(f'merges -> {merges_path}')
