"""运行副本的采样入口：词表、上下文长度、device 全部与 train_run.py 对齐。

用法:
  # 续写模式(预训练基座): 直接把 --prompt 当故事开头
  python generate_run.py --preset v16k --save_path ../models/final_bal/model_best.pth --prompt "Once upon a time,"
  # 指令模式(SFT 模型): 自动套用 sft_format 的对话模板, 只打印故事部分
  python generate_run.py --preset v16k --save_path ../models/sft_bal/sft_best.pth --instruction "Write a story about a brave little dog."

--preset v16k 一次性设好 50M 架构和 16K 词表; 显式传入的参数会覆盖 preset。
"""
import argparse

import torch
from TransformerLM import TransformerLM
from tokenizer import Tokenizer
from function import nucleus_sampling_decoder
from sft_format import format_prompt

SPECIAL = '<|endoftext|>'
SPLIT_TOKEN = ' $~<split_token>~$ '

PRESETS = {
    'v16k': dict(vocab_size=16000, d_model=640, num_layers=6, num_heads=10, d_ff=1728,
                 vocab='../data/mixed_v16k_vocab.json', merges='../data/mixed_v16k_merges.txt'),
}
LEGACY = dict(vocab_size=10000, d_model=512, num_layers=4, num_heads=16, d_ff=1408,
              vocab='../data/TinyStoriesV2-GPT4-train_small_vocab.json',
              merges='../data/TinyStoriesV2-GPT4-train_small_merges.txt')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=sorted(PRESETS), default=None)
    p.add_argument('--vocab_size', type=int, default=None)
    p.add_argument('--context_length', type=int, default=256)
    p.add_argument('--d_model', type=int, default=None)
    p.add_argument('--num_layers', type=int, default=None)
    p.add_argument('--num_heads', type=int, default=None)
    p.add_argument('--d_ff', type=int, default=None)
    p.add_argument('--save_path', default='../models/model.pth')
    p.add_argument('--vocab', default=None)
    p.add_argument('--merges', default=None)
    p.add_argument('--prompt', default='Once upon a time,', help='续写模式的开头')
    p.add_argument('--instruction', default=None, help='指令模式: 套用 SFT 对话模板')
    p.add_argument('--n_samples', type=int, default=1)
    p.add_argument('--max_length', type=int, default=250, help='含 prompt 的总 token 数, 不超过 context_length')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    base = PRESETS[args.preset] if args.preset else LEGACY
    for k, v in base.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.max_length = min(args.max_length, args.context_length)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=[SPECIAL],
        split_token=SPLIT_TOKEN,
    )
    model = TransformerLM(vocab_size=args.vocab_size, context_length=args.context_length,
                          d_model=args.d_model, num_layers=args.num_layers,
                          num_heads=args.num_heads, d_ff=args.d_ff, device=args.device)
    ckpt = torch.load(args.save_path, map_location=args.device, weights_only=True)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[load] full checkpoint, iteration={ckpt.get('iteration')}")
    else:
        model.load_state_dict(ckpt)
        print('[load] weights-only')

    prompt = format_prompt(args.instruction) if args.instruction else args.prompt
    print('--- instruction:' if args.instruction else '--- prompt:',
          args.instruction or args.prompt)
    for i in range(args.n_samples):
        text = nucleus_sampling_decoder(
            model, tokenizer, prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            p=args.top_p,
            device=args.device,
        )
        if args.instruction:
            # 只保留模型写的故事部分
            text = text.split('Story:', 1)[-1].strip()
        if args.n_samples > 1:
            print(f'\n===== sample {i + 1}/{args.n_samples} =====')
        print(text)


if __name__ == '__main__':
    main()
