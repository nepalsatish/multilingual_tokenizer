import json
import argparse
import re
from bpe import BPE


def loadMultilingualData(data_files):
    all_data = []
    for file in data_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data


def trainTokenizer(data_files, output_file, vocab_size):
    data = loadMultilingualData(data_files)

    tokenizer = BPE()
    tokenizer.train(data, vocab_size=vocab_size)
    tokenizer.save_vocab(output_file)
    tokenizer.print_vocab_sample()
    return tokenizer


def parseTokenIds(token_ids_str):
    cleaned = re.sub(r"[\[\]]", "", token_ids_str)
    return [int(id.strip()) for id in re.split(r"[,\s]+", cleaned) if id.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode", type=str)
    parser.add_argument(
        "--decode",
        type=str,
        help="tokens to decode in the format '[token1, token2, ...]'",
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
        tokenizer = trainTokenizer(args.train, args.output, args.vocab_size)
    else:
        tokenizer = BPE(args.output)
        # tokenizer.printVocabSample()

    # if args.printFullVocab:
    #     tokenizer.printFullVocab()

    if args.encode:
        token_ids = tokenizer.encode(args.encode)
        print(f"Encoded '{args.encode}' to: {token_ids}")

    if args.decode:
        try:
            token_ids = parseTokenIds(args.decode)
            decoded_string = tokenizer.decode(token_ids)
            print(f"Decoded to: '{decoded_string}'")
        except ValueError as e:
            print(
                "Please provide tokens in the format: --decode '[token1, token2, ...]'"
            )

    if args.tokenize:
        tokens = tokenizer.tokenize(args.tokenize)
        print(f"Tokenized to: {tokens}")


if __name__ == "__main__":
    main()
