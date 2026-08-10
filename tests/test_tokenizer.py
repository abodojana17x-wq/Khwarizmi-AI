import tempfile
import unittest
from pathlib import Path

from rafig.language.tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def test_encode_decode_roundtrip_for_arabic(self) -> None:
        tokenizer = Tokenizer()
        text = "أنا عايز أتعلم بايثون"
        encoded = tokenizer.encode(text)
        self.assertGreater(len(encoded), 0)
        self.assertEqual(tokenizer.decode(encoded), text)

    def test_encode_decode_roundtrip_for_python(self) -> None:
        tokenizer = Tokenizer()
        text = "def hello(name): return f'Hello {name}'"
        encoded = tokenizer.encode(text)
        self.assertGreater(len(encoded), 0)
        self.assertEqual(tokenizer.decode(encoded), text)

    def test_unknown_tokens_are_replaced(self) -> None:
        tokenizer = Tokenizer()
        encoded = tokenizer.encode("🚀")
        self.assertEqual(tokenizer.decode(encoded), tokenizer.unk_token)

    def test_save_and_load_vocabulary(self) -> None:
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vocab.json"
            tokenizer.save_vocabulary(path)
            loaded = Tokenizer.load_vocabulary(path)
            self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)

    def test_tokenizer_reports_diagnostics(self) -> None:
        tokenizer = Tokenizer()
        diagnostics = tokenizer.diagnostics()
        self.assertIn("vocabulary_size", diagnostics)
        self.assertIn("special_tokens", diagnostics)


if __name__ == "__main__":
    unittest.main()
