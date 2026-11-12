import json
from typing import Iterable
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = [pair1 + pair2 for pair1, pair2 in merges]
        # merge 结果字节串 -> 合并优先级(rank)；str_encode 用它做 O(1) 查找，
        # 避免在 self.merges 列表上线性 index()
        self.merge_rank = {}
        for rank, merged in enumerate(self.merges):
            self.merge_rank.setdefault(merged, rank)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        for special_token in self.special_tokens:
            special_token = special_token.encode('utf-8')
            if special_token not in self.vocab.values():
                self.vocab[max(self.vocab, default=-1) + 1] = special_token

        self.token2id = {v:k for k, v in self.vocab.items()}
        self.special_token_ids = {
            token: self.token2id[token.encode('utf-8')]
            for token in self.special_tokens
        }
        escaped = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
        self.special_pattern = re.compile('(' + '|'.join(escaped) + ')') if escaped else None
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, split_token=' '):
        vocab = cls._vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath, split_token)
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        if text == '':
            return []

        token_ids = []
        parts = self.special_pattern.split(text) if self.special_pattern else [text]
        for part in parts:
            if not part:
                continue
            special_id = self.special_token_ids.get(part)
            if special_id is not None:
                token_ids.append(special_id)
                continue
            for match in re.finditer(self.PAT, part):
                token_ids.extend(self.str_encode(match.group()))
        return token_ids

    def encode_parallel(self, text: str):
        """Backward-compatible alias; rank lookup makes normal encoding fast enough."""
        return self.encode(text)

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            for token_ids in self.encode(text):
                yield token_ids
    
    def decode(self, ids: list[int]):
        text_bytes = b''.join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode('utf-8', errors='replace')

    @staticmethod
    def _vocab(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('format') == 'byte-hex-v1':
            return {int(k): bytes.fromhex(v) for k, v in data['vocab'].items()}
        # Legacy format: retained for published checkpoints, but not binary-safe
        return {int(k): v.encode('utf-8') for k, v in data.items()}
    
    @staticmethod
    def _load_merges(filepath, split_token):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        if lines and lines[0] == '# byte-hex-v1':
            return [tuple(bytes.fromhex(part) for part in line.split('\t', 1))
                    for line in lines[1:] if line]

        merges = []
        for line in lines:
            if line:
                parts = line.split(split_token, 1)
                if len(parts) < 2:
                    continue
                token1 = parts[0].encode('utf-8')
                token2 = ('\n' if parts[1] == '' else parts[1]).encode('utf-8')
                merges.append((token1, token2))
        return merges

    # encode a single string split by regex
    def str_encode(self, text):
        tokens = [bytes([b]) for b in text.encode('utf-8')]

        # Apply merges in order
        while len(tokens) > 1:
            min_merge_index = len(self.merges)
            merge_pos = -1
            merge_pair = None
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i+1]
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < min_merge_index:
                    merge_pair = pair
                    min_merge_index = rank
                    merge_pos = i

            if merge_pair is None:
                break
            tokens[merge_pos] = merge_pair
            tokens.pop(merge_pos+1)

        return [self.token2id[token if token in self.token2id else '[UNK]'.encode('utf-8')] for token in tokens]

    def contains_special_token(self, text):
        return self.special_pattern is not None and self.special_pattern.search(text) is not None
