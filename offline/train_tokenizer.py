import argparse
import os
from transformers import ByteLevelBPETokenizer, PreTrainedTokenizerFast

def main(directory_path, vocab_size, min_frequency, save_path):
    tokenizer = ByteLevelBPETokenizer()
    
    # 获取目录中的所有 .txt 文件
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer and save it.")
    parser.add_argument('--directory_path', type=str, required=True, help='Path to the directory containing .txt files for training the tokenizer.')
    parser.add_argument('--vocab_size', type=int, default=50000, help='The size of the vocabulary.')
    parser.add_argument('--min_frequency', type=int, default=2, help='The minimum frequency of words.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained tokenizer.')

    args = parser.parse_args()
    main(args.directory_path, args.vocab_size, args.min_frequency, args.save_path)
