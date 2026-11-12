import json
import os
from pathlib import Path

import regex as re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def get_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def initialize_vocab(special_tokens: list[str]):
    initial_vocab = {i : bytes([i]) for i in range(256)}

    for index, special_token in enumerate(special_tokens):
        initial_vocab[index + 256] = special_token.encode("utf-8")

    return initial_vocab

def pre_tokenization(text, special_tokens):
    token_counts = defaultdict(int)

    # 一次按所有特殊 token 切开(split 丢弃分隔符, 与原逻辑一致)。
    # 旧实现逐个 special_token 顺序 split, 后一个 split 会在原始 text 上重切,
    # 覆盖前一个的切分边界; 文本不含任何特殊 token 时 parts 未定义会 NameError。
    if special_tokens:
        special_pat = re.compile('|'.join(re.escape(s) for s in special_tokens))
        parts = special_pat.split(text)
    else:
        parts = [text]

    def count_token(cur_parts):
        cur_token_counts = defaultdict(int)
        for part in cur_parts:
            for match in re.finditer(PAT, part):
                token = match.group().encode('utf-8')
                token_tuple = tuple([bytes([b]) for b in token])
                cur_token_counts[token_tuple] += 1
        return cur_token_counts

    def aggregate_token_counts(token_counts_list):
        for cur_counts in token_counts_list:
            for token, count in cur_counts.items():
                token_counts[token] += count
          
    chunk_size = 1024 * 24
    
    all_parts = []

    index = 0
    while index < len(parts):
        cur_parts = []
        cur_num = 0
        while cur_num < chunk_size and index < len(parts):
            cur_parts.append(parts[index])
            cur_num += len(parts[index])
            index += 1

        all_parts.append(cur_parts)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(count_token, all_parts), total=len(all_parts), desc="Counting tokens"))
        aggregate_token_counts(results)

    return token_counts

def merge(token_counts: dict[tuple[bytes], int]):
    # Count frequency of all adjacent pairs
    pair_counts = defaultdict(int)
    pair_index = defaultdict(set)
    for token_bytes, count in token_counts.items():
        if len(token_bytes) < 2:
            continue

        for i in range(len(token_bytes) - 1):
            pair = (token_bytes[i], token_bytes[i + 1])
            pair_counts[pair] += count
            pair_index[pair].add(token_bytes)

    if not pair_counts:
        return token_counts, None
    
    # Find the most frequent pair
    max_count = max(pair_counts.values())
    most_frequent_pairs = [pair for pair, count in pair_counts.items() if count == max_count]
    most_frequent_pair = max(most_frequent_pairs)

    # Merge the most frequent pair in affected tokens
    new_token_counts = defaultdict(int)
    affected_tokens = pair_index[most_frequent_pair]
    
    for token_bytes, count in token_counts.items():
        if token_bytes not in pair_index[most_frequent_pair]:
            new_token_counts[token_bytes] += count

    # Process affected tokens
    for token_bytes in affected_tokens:
        count = token_counts[token_bytes]
        token_list = list(token_bytes)
        new_token = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == most_frequent_pair:
                merged = token_list[i] + token_list[i + 1]
                new_token.append(merged)
                i += 2
            else:
                new_token.append(token_list[i])
                i += 1
        
        new_token_counts[tuple(new_token)] += count
        
    return dict(new_token_counts), most_frequent_pair

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = initialize_vocab(special_tokens)
    merges = []

    text = get_file(input_path)

    token_counts = pre_tokenization(text, special_tokens)

    cur_merged_pair = '1'
    with tqdm(total=vocab_size, desc="BPE Merges") as pbar:
        while len(vocab) < vocab_size and cur_merged_pair:
            token_counts, cur_merged_pair = merge(token_counts)
            if cur_merged_pair:
                vocab[len(vocab)] = cur_merged_pair[0] + cur_merged_pair[1]
                merges.append(cur_merged_pair)
                pbar.update(1)

    return vocab, merges


def save_tokenizer(vocab, merges, stem):
    """Save byte tokens losslessly to data/<stem>_{vocab,merges}.*."""
    if not stem or Path(stem).name != stem:
        raise ValueError('stem must be a plain filename without directories')
    vocab_path = DATA_DIR / f'{stem}_vocab.json'
    merges_path = DATA_DIR / f'{stem}_merges.txt'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'byte-hex-v1',
        'vocab': {str(k): v.hex() for k, v in vocab.items()},
    }
    with vocab_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    with merges_path.open('w', encoding='ascii', newline='\n') as f:
        f.write('# byte-hex-v1\n')
        for token1, token2 in merges:
            f.write(f'{token1.hex()}\t{token2.hex()}\n')
    return vocab_path, merges_path
