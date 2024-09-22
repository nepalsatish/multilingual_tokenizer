import json
import re
from collections import defaultdict


class BPE:
    def __init__(self, vocab_file=None):
        self.vocab = []
        self.vocabDict = {}
        self.reverseVocabDict = {}
        self.unknownToken = "[UNK]"
        self.spaceToken = "[SPACE]"
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self.initialize_base_vocab()

    def initialize_base_vocab(self):
        # Initialize with all ASCII letters, digits, space, and common punctuation
        self.vocab = [chr(i) for i in range(32, 127)] + [
            self.unknownToken,
            self.spaceToken,
        ]
        self.vocabDict = self.build_vocabDict(self.vocab)
        self.reverseVocabDict = {v: k for k, v in self.vocabDict.items()}

    def load_vocab(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        if self.unknownToken not in self.vocab:
            self.vocab.append(self.unknownToken)
        if self.spaceToken not in self.vocab:
            self.vocab.append(self.spaceToken)
        self.vocabDict = self.build_vocabDict(self.vocab)
        self.reverseVocabDict = {v: k for k, v in self.vocabDict.items()}
        print(f"Loaded vocabulary with {len(self.vocab)} tokens.")

    def build_vocabDict(self, vocab):
        return {token: idx for idx, token in enumerate(vocab)}

    def encode(self, text):
        tokens = self.tokenize(text)
        encoded = [
            self.vocabDict.get(token, self.vocabDict[self.unknownToken])
            for token in tokens
        ]
        print(f"Encoded '{text}' to: {encoded}")
        return encoded

    def decode(self, token_ids):
        decoded = []
        for token_id in token_ids:
            if token_id in self.reverseVocabDict:
                token = self.reverseVocabDict[token_id]
                if token == self.spaceToken:
                    decoded.append(" ")
                else:
                    decoded.append(token)
            else:
                decoded.append(self.unknownToken)
        return "".join(decoded)

    def tokenize(self, text):
        tokens = []
        for char in text:
            if char == " ":
                tokens.append(self.spaceToken)
            elif char in self.vocabDict:
                tokens.append(char)
            else:
                tokens.append(self.unknownToken)
        return tokens

    def train(self, data, vocab_size):
        # For this version, we're not actually training, just ensuring our base vocabulary
        # includes all necessary characters
        char_freq = defaultdict(int)
        for item in data:
            for text in [item["input"], item["output"]]:
                for char in text:
                    char_freq[char] += 1

        # Add all characters seen in the data to the vocabulary
        for char in char_freq:
            if char not in self.vocabDict and char != " ":
                self.vocab.append(char)

        # Ensure we don't exceed the vocab_size
        if len(self.vocab) > vocab_size:
            self.vocab = self.vocab[: vocab_size - 2] + [
                self.unknownToken,
                self.spaceToken,
            ]

        self.vocabDict = self.build_vocabDict(self.vocab)
        self.reverseVocabDict = {v: k for k, v in self.vocabDict.items()}
        print(f"Updated vocabulary with {len(self.vocab)} tokens.")

    def save_vocab(self, vocab_file):
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)
        print(f"Saved vocabulary to {vocab_file}")

    def print_vocab_sample(self, n=10):
        print(f"Sample of first {n} vocabulary items:")
        for i, token in enumerate(self.vocab[:n]):
            print(f"{i}: {token}")

    def print_full_vocab(self):
        print("Full vocabulary:")
        for i, token in enumerate(self.vocab):
            print(f"{i}: {token}")
