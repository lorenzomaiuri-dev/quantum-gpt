from functools import cached_property
from transformers import AutoTokenizer


class BaseTokenizer:
    """Interface for all tokenizers."""

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def to_dict(self):
        return self.__dict__

    def from_dict(self, data):
        # Handle cases where keys are stored as strings in JSON but need to be ints
        if "dec" in data:
            data["dec"] = {int(k): v for k, v in data["dec"].items()}
        if "enc" in data:
            data["enc"] = {k: int(v) for k, v in data["enc"].items()}
        self.__dict__.update(data)

    def load_tokenizer_from_dict(data):
        """Function to recreate the correct tokenizer from the saved dictionary."""
        if "hf_model" in data:
            return HFTokenizerWrapper(data["hf_model"])

        # Otherwise, it's a legacy tokenizer.
        t_type = data.get("class", "CharTokenizer")

        # Map class names to actual classes
        from src import tokenizer as tk_module

        cls = getattr(tk_module, t_type, CharTokenizer)

        instance = cls()
        instance.from_dict(data)
        return instance


class LegacyTokenizer(BaseTokenizer):
    """Base for custom character/n-gram tokenizers."""

    def __init__(self, data=None):
        if data:
            self.keys = self._build_keys(data)
            self.enc = {ch: i for i, ch in enumerate(self.keys)}
            self.dec = {i: ch for i, ch in enumerate(self.keys)}

    def _build_keys(self, data):
        raise NotImplementedError

    @cached_property
    def vocab_size(self):
        return len(self.enc)

    def encode(self, text):
        return [
            self.enc[text[i : i + self._step]]
            for i in range(0, len(text), self._step)
            if text[i : i + self._step] in self.enc
        ]

    def decode(self, ids):
        return "".join([self.dec[i] for i in ids])


class CharTokenizer(LegacyTokenizer):
    _step = 1

    def _build_keys(self, data):
        return sorted(list(set(data)))


class BiCharTokenizer(LegacyTokenizer):
    _step = 2

    def _build_keys(self, data):
        # Pairs + single chars to ensure fallback
        pairs = set([data[i : i + 2] for i in range(0, len(data) - 1, 2)])
        chars = set(data)
        return sorted(list(pairs | chars))


class HFTokenizerWrapper(BaseTokenizer):
    """Wrapper for Hugging Face tokenizers (e.g., 'gpt2', 'bert-base-uncased')."""

    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def to_dict(self):
        return {"hf_model": self.tokenizer.name_or_path}
