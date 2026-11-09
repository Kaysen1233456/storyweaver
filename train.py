import argparse
import numpy as np
import torch
import torch.nn as nn
from AdamW import AdamW
from TransformerLM import TransformerLM
from tokenizer import Tokenizer
from function import data_loader, cross_entropy as loss_fn, save_checkpoint
import logging
import time
from tqdm import tqdm

def setup_logging(log_file):
    """
    Set up logging to file and console.
    
    Args:
        log_file: str, path to the log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def train_loop():
    parser = argparse.ArgumentParser(description="Train a model with configurable hyperparameters.")
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)

    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--train_data', type=str, default='../data/TinyStoriesV2-GPT4-train_tokens.txt')
    parser.add_argument('--val_data', type=str, default='../data/TinyStoriesV2-GPT4-valid_tokens.txt')

    parser.add_argument('--log_file', type=str, default='../data/training_log.txt')

    parser.add_argument('--save_path', type=str, default='../models/model.pth')
    parser.add_argument('--device', type=str, default='cuda:7', help='Device to use (cpu or cuda:0)')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)
    start_time = time.time()
    logging.info("\n\n\n****************Training started****************")
    logging.info("Hyperparameters:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    with open(args.train_data, 'r', encoding='utf-8') as f:
        data = f.read()
    train_data = data.rstrip().split('\n')
    train_data = np.array([int(item) for item in train_data])

    logging.info(f"Loaded training data: {len(train_data)} tokens")

    with open(args.val_data, 'r', encoding='utf-8') as f:
        data = f.read()
    val_data = data.rstrip().split('\n')
    val_data = np.array([int(item) for item in val_data] )

    logging.info(f"Loaded validation data: {len(val_data)} tokens")

    # Initialize model and optimizer
    model = TransformerLM(vocab_size=args.vocab_size, context_length=args.context_length, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, device=args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    logging.info("Model and optimizer initialized")

    iteration = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = len(train_data) // args.batch_size

        for batch_idx in tqdm(range(0, len(train_data) - args.context_length, args.batch_size), desc=f"Training Epoch : {epoch}"):
            iteration += 1

            inputs, targets = data_loader(train_data, args.batch_size, args.context_length, args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if iteration % 10000 == 0:
                logging.info(f"Epoch {epoch+1}/{args.epochs}, Iteration {iteration}/{num_batches}, Train Loss: {total_train_loss/iteration:.4f}")
                save_checkpoint(model, optimizer, iteration, args.save_path)

        avg_train_loss = total_train_loss / num_batches

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            num_val_batches = len(val_data) // args.batch_size
            for batch_idx in tqdm(range(0, len(val_data) - args.context_length, args.batch_size), desc=f"Evaluating Epoch : {epoch}"):
                val_inputs, val_targets = data_loader(val_data, args.batch_size, args.context_length, args.device)
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_targets)
                total_val_loss += val_loss.item()
            
        avg_val_loss = total_val_loss / num_val_batches

        # Log performance
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, iteration, args.save_path)

if __name__ == "__main__":
    train_loop()