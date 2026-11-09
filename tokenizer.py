import json
from typing import Iterable
import regex as re
from tqdm import tqdm
from tools import parallelize_with_multiprocessing

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = [pair1 + pair2 for pair1, pair2 in merges]
        self.special_tokens = special_tokens
        # self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

        if special_tokens:
            for special_token in special_tokens:
                special_token = special_token.encode('utf-8')
                if special_token not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token

        self.token2id = {v:k for k, v in self.vocab.items()}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, split_token=' '):
        vocab = cls._vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath, split_token)
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        if text =='':
            return []

        token_ids = []

        if not self.special_tokens or not self.contains_special_token(text):
            for match in re.finditer(self.PAT, text):
                token_ids.extend(self.str_encode(match.group()))
            return token_ids

        for special_token in self.special_tokens:
            if special_token in text:
                parts = text.split(special_token)
                for part in parts:
                    if part:
                        token_ids.extend(self.encode(part))
                    token_ids.append(self.token2id[special_token.encode('utf-8')])

        return token_ids[:-1]

    def encode_parallel(self, text: str):
        if text =='':
            return []

        token_ids = []

        if not self.special_tokens or not self.contains_special_token(text):
            for match in re.finditer(self.PAT, text):
                token_ids.extend(self.str_encode(match.group()))
            return token_ids
        
        for special_token in self.special_tokens:
            if special_token in text:
                parts = text.split(special_token)
                parts = [part.strip() for part in parts if part]

                results = parallelize_with_multiprocessing(parts, self.encode, 64)
                
                for result in results:
                    token_ids.extend(result)
                    token_ids.append(self.token2id[special_token.encode('utf-8')])

        return token_ids[:-1]

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            for token_ids in self.encode(text):
                yield token_ids
    
    def decode(self, ids: list[int]):
        text_bytes = b''
        for token_id in ids:
            text_bytes += self.vocab[token_id]
        return text_bytes.decode('utf-8', errors='replace')

    @staticmethod
    def _vocab(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return {int(k) : v.encode('utf-8') for k, v in vocab.items()}
    
    @staticmethod
    def _load_merges(filepath, split_token):
        merges = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line[:-1] # remove '\n' in the right
                if line:
                    if len(line.split(split_token)) < 2:
                        continue
                    token1 = line.split(split_token)[0].encode('utf-8')
                    if line.split(split_token)[1] == '':
                        token2 = '\n'.encode('utf-8')
                    else:
                        token2 = line.split(split_token)[1].encode('utf-8')
                    merges.append((token1, token2))

        return merges

    # encode a single string split by regex
    def str_encode(self, text):
        tokens = [bytes([b]) for b in text.encode('utf-8')]

        # Apply merges in order
        while True:
            min_merge_index = 1e8
            merge_pos = -1
            merge_pair = None
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i+1]
                if pair in self.merges:
                    merge_index = self.merges.index(pair)
                    if merge_index < min_merge_index:
                        merge_pair = pair
                        min_merge_index = merge_index
                        merge_pos = i
                        
            if not merge_pair:
                break
            tokens[merge_pos] = merge_pair
            tokens.pop(merge_pos+1)

        return [self.token2id[token if token in self.token2id else '[UNK]'.encode('utf-8')] for token in tokens]

    def contains_special_token(self, text):
        for special_token in self.special_tokens:
            if special_token in text:
                return True
        return False
