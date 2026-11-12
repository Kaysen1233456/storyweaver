"""在混合语料(TinyStories + Gutenberg 过采样)上训练 BPE 词表。

用法:
    python train_bpe_mixed.py                      # 默认 10000, 输出 mixed_vocab.json
    python train_bpe_mixed.py --vocab_size 16000 --tag v16k
        -> ../data/mixed_v16k_vocab.json / mixed_v16k_merges.txt
"""
import argparse
import os
import time

import psutil
from train_bpe import save_tokenizer, train_bpe

p = argparse.ArgumentParser()
p.add_argument('--vocab_size', type=int, default=10000)
p.add_argument('--input', default='../data/mixed_train.txt')
p.add_argument('--tag', default='', help='输出文件名后缀, 例如 v16k')
args = p.parse_args()

process = psutil.Process(os.getpid())
start_time = time.time()
start_memory = process.memory_info().rss / 1024 / 1024

vocab, merges = train_bpe(args.input, args.vocab_size, ['<|endoftext|>', '[UNK]'])

print(f"Training time: {(time.time() - start_time) / 3600:.4f} hours")
print(f"Memory used: {process.memory_info().rss / 1024 / 1024 - start_memory:.2f} MB")

tag = f'mixed_{args.tag}' if args.tag else 'mixed'
vocab_path, merges_path = save_tokenizer(vocab, merges, tag)

print(f'vocab  -> {vocab_path}  ({len(vocab)} tokens)')
print(f'merges -> {merges_path}')
