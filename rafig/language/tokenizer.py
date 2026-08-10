"""A lightweight offline tokenizer for RAFIQ."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(slots=True)
class Tokenizer:
    """Simple character-aware tokenizer with a small vocabulary."""

    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    vocab: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._initialize_vocab()

    def _initialize_vocab(self) -> None:
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        for char in [" ", "\n", "\t"]:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,;:!?()[]{}<>+-=*/%&|@#'\"":
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        for codepoint in range(0x0600, 0x06FF + 1):
            char = chr(codepoint)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        for codepoint in range(0x0750, 0x077F + 1):
            char = chr(codepoint)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        for codepoint in range(0x08A0, 0x08FF + 1):
            char = chr(codepoint)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        for char in "٣٤٥٦٧٨٩":
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        if not normalized:
            return [self.bos_token]

        tokens: List[str] = []
        for char in normalized:
            if char in self.vocab:
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
        return tokens

    def tokenize_words(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        if not normalized:
            return [self.bos_token]
        return re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def decode(self, tokens: List[int]) -> str:
        inverse = {value: key for key, value in self.vocab.items()}
        chars = []
        for token_id in tokens:
            token = inverse.get(token_id, self.unk_token)
            if token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            chars.append(token)
        return "".join(chars)

    def save_vocabulary(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocabulary(cls, path: str | Path) -> "Tokenizer":
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        tokenizer = cls(vocab=vocab)
        return tokenizer

    def diagnostics(self) -> Dict[str, object]:
        return {
            "vocabulary_size": self.vocab_size,
            "special_tokens": [self.pad_token, self.unk_token, self.bos_token, self.eos_token],
            "normalized_example": self.normalize("  أنا عايز أتعلم بايثون  "),
        }
