import argparse
import numpy as np
import torch
import torch.nn as nn
from AdamW import AdamW
from TransformerLM import TransformerLM
from tokenizer import Tokenizer
from function import data_loader, cross_entropy as loss_fn, nucleus_sampling_decoder, load_checkpoint

tokenizer = Tokenizer.from_files(
    vocab_filepath="../data/TinyStoriesV2-GPT4-valid_vocab.json",
    merges_filepath="../data/TinyStoriesV2-GPT4-valid_merges.txt",
    special_tokens=['<|endoftext|>'],
    split_token=' $~<split_token>~$ '
)


def train_loop():
    parser = argparse.ArgumentParser(description="Train a model with configurable hyperparameters.")
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--train_data', type=str, default='../data/TinyStoriesV2-GPT4-valid_tokens.txt')
    parser.add_argument('--val_data', type=str, default='../data/TinyStoriesV2-GPT4-valid_tokens.txt')

    parser.add_argument('--log_file', type=str, default='../data/training_log.txt')

    parser.add_argument('--save_path', type=str, default='../models/model.pth')
    parser.add_argument('--temperature', type=int, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:5', help='Device to use (cpu or cuda:0)')
    args = parser.parse_args()


    # Initialize model and optimizer
    model = TransformerLM(vocab_size=args.vocab_size, context_length=args.context_length, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, device=args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    load_checkpoint(args.save_path, model, optimizer)

    print(nucleus_sampling_decoder(model, tokenizer, 'Once upon a time,', temperature=args.temperature, device=args.device))
    
if __name__ == "__main__":
    train_loop()