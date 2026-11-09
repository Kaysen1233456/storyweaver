from tokenizer import Tokenizer

tokenizer = Tokenizer.from_files(
    vocab_filepath="../data/TinyStoriesV2-GPT4-train_vocab.json",
    merges_filepath="../data/TinyStoriesV2-GPT4-train_merges.txt",
    special_tokens=['<|endoftext|>'],
    split_token=' $~<split_token>~$ '
)

with open('../data/TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8') as f:
    data = f.read()

def encode_data():
    tokens = tokenizer.encode_parallel(data)
    with open('../data/TinyStoriesV2-GPT4-train_tokens.txt', 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(str(token) + '\n')

encode_data()