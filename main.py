import json
import argparse
import re
from bpe import BPE


def load_multilingual_data(data_files):
    all_data = []
    for file in data_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data


def train_tokenizer(data_files, output_file, vocab_size):
    data = load_multilingual_data(data_files)

    tokenizer = BPE()
    tokenizer.train(data, vocab_size=vocab_size)
    tokenizer.save_vocab(output_file)
    tokenizer.print_vocab_sample()
    return tokenizer


def parse_token_ids(token_ids_str):
    # Remove brackets, split by comma or space, and convert to integers
    cleaned = re.sub(r"[\[\]]", "", token_ids_str)
    return [int(id.strip()) for id in re.split(r"[,\s]+", cleaned) if id.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode", type=str)
    parser.add_argument(
        "--decode",
        type=str,
        help="Token IDs to decode in the format '[id1, id2, ...]'",
    )
    parser.add_argument("--tokenize", type=str)
    parser.add_argument(
        "--train",
        type=str,
        nargs="+" 
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Output vocabulary file",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=40000
    )
    parser.add_argument(
        "--print-full-vocab", action="store_true"
    )
    args = parser.parse_args()

    if args.train:
        if len(args.train) != 5:
            print(
                "Please provide exactly 5 training data files, one for each language."
            )
            return
        tokenizer = train_tokenizer(args.train, args.output, args.vocab_size)
    else:
        tokenizer = BPE(args.output)
        tokenizer.print_vocab_sample()

    if args.print_full_vocab:
        tokenizer.print_full_vocab()

    if args.encode:
        token_ids = tokenizer.encode(args.encode)
        print(f"Encoded '{args.encode}' to: {token_ids}")

    if args.decode:
        try:
            token_ids = parse_token_ids(args.decode)
            print(f"Attempting to decode token IDs: {token_ids}")
            decoded_string = tokenizer.decode(token_ids)
            print(f"Decoded {token_ids} to: '{decoded_string}'")
        except ValueError as e:
            print(f"Error parsing token IDs: {e}")
            print(
                "Please provide token IDs in the format: --decode '[id1, id2, ...]'"
            )

    if args.tokenize:
        tokens = tokenizer.tokenize(args.tokenize)
        print(f"Tokenized '{args.tokenize}' to: {tokens}")


if __name__ == "__main__":
    main()
