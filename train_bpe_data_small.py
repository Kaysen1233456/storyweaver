import os
import time

import psutil

from train_bpe import save_tokenizer, train_bpe

input_path = '../data/TinyStoriesV2-GPT4-train_small.txt'

# Measure time and memory
process = psutil.Process(os.getpid())
start_time = time.time()
start_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB


vocab, merges = train_bpe(input_path, 10000, ['<|endoftext|>', '[UNK]'])

end_time = time.time()
end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
time_hours = (end_time - start_time) / 3600  # Convert seconds to hours
memory_used_mb = end_memory - start_memory

print(f"Training time: {time_hours:.4f} hours")
print(f"Memory used: {memory_used_mb:.2f} MB")

vocab_path, merges_path = save_tokenizer(vocab, merges, 'TinyStoriesV2-GPT4-train_small')
print(f'vocab  -> {vocab_path}')
print(f'merges -> {merges_path}')
