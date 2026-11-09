"""新旧模型对比: 在固定文本上按 每字节 loss 比较(两套词表不同, 只能按字节比)
- Gutenberg holdout 300 块(每块取前 256 token)
- TinyStories valid 前 300 个故事
"""
import glob
import math
import os
import random

import regex as re
import torch
import torch.nn.functional as F

import build_mixed_corpus as bmc
from encode_corpus_fast import build_encoder
from tokenizer import Tokenizer
from TransformerLM import TransformerLM

device = 'cuda:0'
LN2 = math.log(2.0)

# ---------- 固定评估文本 ----------
rng = random.Random(42)
gb = []
for f in sorted(glob.glob('../data/sft_raw/pg*.txt')):
    if os.path.basename(f) not in bmc.USE_BOOKS:
        continue
    text = bmc.normalize(bmc.clean_book(open(f, encoding='utf-8').read()))
    for c in bmc.split_chunks(text, 2000):
        if bmc.english_ratio(c) >= 0.7:
            gb.append(c)
rng.shuffle(gb)
gb_hold = gb[:300]

ts_all = open('../data/TinyStoriesV2-GPT4-valid.txt', encoding='utf-8').read().split('<|endoftext|>')
ts_eval = [s.strip() for s in ts_all if len(s.strip()) > 200][:300]

MODELS = [
    ('23M 旧架构(10K词表, 3 epoch)', '../models/model_best.pth',
     '../data/mixed_vocab.json', '../data/mixed_merges.txt',
     dict(vocab_size=10000, d_model=512, num_layers=4, num_heads=16, d_ff=1344)),
    ('50M 旧语料(16K词表, 1 epoch)', '../models/v16k/model_best.pth',
     '../data/mixed_v16k_vocab.json', '../data/mixed_v16k_merges.txt',
     dict(vocab_size=16000, d_model=640, num_layers=6, num_heads=10, d_ff=1728)),
    ('50M 均衡语料(16K词表, dropout)', '../models/final_bal/model_best.pth',
     '../data/mixed_v16k_vocab.json', '../data/mixed_v16k_merges.txt',
     dict(vocab_size=16000, d_model=640, num_layers=6, num_heads=10, d_ff=1728)),
]


def eval_texts(model, enc, id_bytes, texts):
    """返回 (每字节 nats, 每 token nats)"""
    tot_nats, tot_bytes, tot_tok = 0.0, 0, 0
    for s in texts:
        ids = enc(s)[:257]
        if len(ids) < 20:
            continue
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        with torch.no_grad():
            nll = F.cross_entropy(model(x)[0], y[0], reduction='sum').item()
        tot_nats += nll
        tot_bytes += sum(id_bytes[t] for t in ids[1:])
        tot_tok += len(ids) - 1
    return tot_nats / tot_bytes, tot_nats / tot_tok


results = {}
for name, ckpt, vocab, merges, arch in MODELS:
    ew, eot, pat, _ = build_encoder(vocab, merges)
    patc = re.compile(pat)
    enc = lambda s, ew=ew, patc=patc: [t for m in patc.finditer(s) for t in ew(m.group().encode('utf-8'))]
    tok = Tokenizer.from_files(vocab_filepath=vocab, merges_filepath=merges,
                               special_tokens=['<|endoftext|>'],
                               split_token=' $~<split_token>~$ ')
    id_bytes = {v: len(k) for k, v in tok.token2id.items()}

    model = TransformerLM(context_length=256, device=device, **arch)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()

    gb_b, gb_t = eval_texts(model, enc, id_bytes, gb_hold)
    ts_b, ts_t = eval_texts(model, enc, id_bytes, ts_eval)
    results[name] = (gb_b, ts_b)
    print(f'{name}')
    print(f'   Gutenberg : 每字节 {gb_b:.4f} nats  (bpb {gb_b / LN2:.4f})   每token {gb_t:.3f}')
    print(f'   TinyStory : 每字节 {ts_b:.4f} nats  (bpb {ts_b / LN2:.4f})   每token {ts_t:.3f}')
    del model
    torch.cuda.empty_cache()

(o_gb, o_ts), (n_gb, n_ts) = results.values()
print()
print(f'Gutenberg 每字节 loss 变化: {100 * (n_gb - o_gb) / o_gb:+.1f}%')
print(f'TinyStory 每字节 loss 变化: {100 * (n_ts - o_ts) / o_ts:+.1f}%')
