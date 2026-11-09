from train_bpe import train_bpe
import json
import psutil
import os
import time

input_path = '../data/TinyStoriesV2-GPT4-train.txt'

vocab_path = input_path.replace('.txt', '_vocab.json')
merges_path = input_path.replace('.txt', '_merges.txt')
split_token = ' $~<split_token>~$ '

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

vocab_str = {k : v.decode('utf-8', errors='replace') for k, v in vocab.items()}
with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=4)

with open(merges_path, 'w', encoding='utf-8') as f:
    for pair in merges:
        token1 = pair[0].decode('utf-8', errors='replace')
        token2 = pair[1].decode('utf-8', errors='replace')
        f.write(f"{token1}{split_token}{token2}\n")