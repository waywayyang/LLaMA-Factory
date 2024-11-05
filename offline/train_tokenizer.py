import argparse
import os
from llamafactory.model.way.tokenizer import WayTokenizerFast
from tokenizers import ByteLevelBPETokenizer
def read_files_in_stream(file_paths):
    # 生成器函数，逐行读取文件
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
def main(directory_path, vocab_size, min_frequency, save_path):
    tokenizer = ByteLevelBPETokenizer()
    # 获取目录中的所有 .txt 文件
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
    tokenizer.train_new_from_iterator(iterator=read_files_in_stream(files), vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    print("train tokenizer finish")
    tokenizer = WayTokenizerFast(tokenizer_object=tokenizer)
    print("start save tokenize")
    tokenizer.save_pretrained(save_path)
    print("end save tokenize")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer and save it.")
    parser.add_argument('--directory_path', type=str, required=True, help='Path to the directory containing .txt files for training the tokenizer.')
    parser.add_argument('--vocab_size', type=int, default=50000, help='The size of the vocabulary.')
    parser.add_argument('--min_frequency', type=int, default=2, help='The minimum frequency of words.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained tokenizer.')

    args = parser.parse_args()
    main(args.directory_path, args.vocab_size, args.min_frequency, args.save_path)
