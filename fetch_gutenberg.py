"""从 Project Gutenberg 下载公版童话/短篇故事集，剥离版权头尾，按故事切分保存。

产出: ../data/sft_raw/<ebook_id>_<slug>.txt  —— 每个文件是一个完整故事（或合理长度的段落块）
所有文本均为公有领域（Project Gutenberg 公版书）。

用法:
    python fetch_gutenberg.py            # 下载全部
    python fetch_gutenberg.py --limit 5 # 只试下载前 5 本，先验证
"""
import argparse
import json
import os
import re
import time
import urllib.request

OUT_DIR = '../data/sft_raw'
MIRROR = 'https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt'

# 精选公版文本（ebook id -> 分级、标题）
# 分级: A=可用于儿童故事 SFT；B=只用于混合预训练或人工复核
CATALOG = {
    2591:  ("A", "Grimms' Fairy Tales"),
    19033: ("A", "Alice's Adventures in Wonderland (adapted edition)"),
    514:   ("A", "Little Women"),
    1597:  ("A", "Andersen's Fairy Tales (H.B. Paull translation)"),
    21:    ("A", "Aesop's Fables (Townsend translation)"),
    113:   ("A", "The Secret Garden"),
    2554:  ("B", "Crime and Punishment"),
    13415: ("B", "The Tales of Chekhov, Volume 3"),
    6897:  ("B", "The Little Savage"),
    30127: ("B", "Tales from Dickens"),
    29000: ("B", "The Macdermots of Ballycloran"),
    # 已下载但不用于当前英文语料：非英文、非故事或领域不匹配
    22015: ("B", "As Minas de Salomao (Portuguese)"),
    22016: ("B", "Vie de Madame Elisabeth de France (French)"),
    17325: ("B", "History of Egypt, Chaldea, Syria, Babylonia, and Assyria"),
    9066:  ("B", "German-language text"),
    14774: ("B", "Finnish-language poetry collection"),
    1998:  ("B", "Thus Spake Zarathustra"),
    19037: ("B", "ASCE Transactions, Paper No. 1158"),
}

HEADER_PAT = re.compile(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE | re.DOTALL)
FOOTER_PAT = re.compile(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*', re.IGNORECASE | re.DOTALL)


def fetch(eid, timeout=60):
    url = MIRROR.format(eid=eid)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (sft-data-build)'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', errors='replace')


def strip_boilerplate(text):
    m = HEADER_PAT.search(text)
    if m:
        text = text[m.end():]
    m = FOOTER_PAT.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def slugify(desc):
    s = re.sub(r'[^\w\s-]', '', desc.lower())
    return re.sub(r'[\s_-]+', '_', s).strip('_')[:40]


def looks_like_storybook(text, min_chars=4000):
    """粗筛：太短的（目录页/损坏页）不要"""
    return len(text) >= min_chars


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0, help='只下载前 N 本（0=全部）')
    p.add_argument('--out_dir', default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)

    items = [(k, v) for k, v in CATALOG.items()][: args.limit or None]
    ok = fail = 0
    for eid, (grade, desc) in items:
        out_path = os.path.join(args.out_dir, f'pg{eid}.txt')
        if os.path.exists(out_path):
            with open(out_path, encoding='utf-8', errors='replace') as f:
                chars = len(f.read())
            manifest[str(eid)] = {
                'grade': grade, 'desc': desc, 'chars': chars, 'file': f'pg{eid}.txt'
            }
            print(f'[skip] pg{eid} 已存在（刷新 manifest）')
            ok += 1
            continue
        try:
            raw = fetch(eid)
            text = strip_boilerplate(raw)
            if not looks_like_storybook(text):
                print(f'[warn] pg{eid} 内容太短({len(text)}字)，跳过 -> {desc}')
                fail += 1
                continue
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text)
            manifest[str(eid)] = {'grade': grade, 'desc': desc, 'chars': len(text), 'file': f'pg{eid}.txt'}
            ok += 1
            print(f'[ok]   pg{eid} {len(text):>7} chars  {desc}')
        except Exception as exc:
            fail += 1
            print(f'[fail] pg{eid} {desc}: {exc}')
        time.sleep(1.5)  # 礼貌爬取，别打服务器

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'\nDONE ok={ok} fail={fail} -> {args.out_dir}')


if __name__ == '__main__':
    main()
