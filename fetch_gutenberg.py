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

# 精选公版童话/短篇书集（ebook id -> 描述）
# 分级: A=儿童童话/寓言(风格匹配), 只用 A 级构建 SFT 样本
CATALOG = {
    2591:  ("A", "Grimm's Fairy Tales"),
    19033: ("A", "Grimm's Fairy Tales (另一版本)"),
    514:   ("A", "Fairy Tales (澳大利亚传说)"),
    1597:  ("A", "Andersen's Fairy Tales (H.B. Paull 译本)"),
    22015: ("A", "Andersen's Fairy Tales (另一版本)"),
    22016: ("A", "Andersen 后续集"),
    21:    ("A", "Aesop's Fables (Townsend 译本)"),
    113:   ("A", "The Secret Garden (儿童文学)"),
    17325: ("A", "The Happy Prince and Other Tales (Oscar Wilde)"),
    9066:  ("A", "A House of Pomegranates (Oscar Wilde)"),
    14774: ("A", "Mother Goose 童谣与故事"),
    1998:  ("A", "Aesop's Fables (另一个译本)"),
    2554:  ("B", "The Wife, and Other Stories (Chekhov)"),
    13415: ("B", "Short Story Collection (多作者)"),
    6897:  ("B", "Short Story Collection Vol."),
    19037: ("B", "短篇集"),
    30127: ("B", "短篇集 Vol."),
    29000: ("B", "短篇集"),
}

HEADER_PAT = re.compile(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', re.IGNORECASE | re.DOTALL)
FOOTER_PAT = re.compile(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*', re.IGNORECASE | re.DOTALL)
TITLE_LINE = re.compile(r'^\s*(?:THE\s+)?(?:[A-Z][\w\'’-]+\s+){0,4}[A-Z][\w\'’-]+\s*$')


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
            print(f'[skip] pg{eid} 已存在')
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
