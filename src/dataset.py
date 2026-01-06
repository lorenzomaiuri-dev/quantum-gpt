import os
import torch
from datasets import load_dataset as load_hf_dataset
from src.tokenizer import CharTokenizer, BiCharTokenizer, HFTokenizerWrapper


class InputDataset:
    def __init__(self, config, file_path_or_repo, dictionary_path=None):
        self.config = config
        self.device = config.device
        self.block_size = config.block_size

        # 1. Setup Tokenizer
        self.tokenizer = self._setup_tokenizer(
            config, dictionary_path, file_path_or_repo
        )

        # 2. Load Data (Local or HF)
        raw_text = self._load_raw_data(file_path_or_repo)

        # 3. Tokenize and Prepare Tensors
        print(f"Tokenizing dataset (Vocab size: {self.tokenizer.vocab_size})...")
        full_data = torch.tensor(self.tokenizer.encode(raw_text), dtype=torch.long)

        # Split 90/10
        n = int(0.9 * len(full_data))
        self.train_data = full_data[:n]
        self.val_data = full_data[n:]

    def _setup_tokenizer(self, config, dict_path, data_sample):
        t_type = config.tokenizer_class

        # Case A: Hugging Face Tokenizer
        if t_type.startswith("hf-") or t_type in ["gpt2", "roberta-base"]:
            model_name = t_type.replace("hf-", "")
            return HFTokenizerWrapper(model_name)

        # Case B: Legacy Tokenizers
        tokenizers_map = {
            "CharTokenizer": CharTokenizer,
            "BiCharTokenizer": BiCharTokenizer,
        }
        cls = tokenizers_map.get(t_type, CharTokenizer)

        tokenizer = cls()
        if dict_path and os.path.exists(dict_path):
            import json

            with open(dict_path, "r") as f:
                tokenizer.from_dict(json.load(f))
        else:
            # We need to build the vocab from text
            text_sample = self._load_raw_data(data_sample)
            tokenizer = cls(text_sample)
        return tokenizer

    def _load_raw_data(self, path):
        """Loads text from local file or HF Hub with robust column detection."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        elif os.path.exists(f"data/{path}"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print(f"File '{path}' not found. Attempting to load from HF Hub...")
            # We use 'train' split by default
            ds = load_hf_dataset(path, split="train")

            # 1. Look for common text column names
            column_names = ds.column_names
            target_column = None
            candidates = ["text", "Text", "content", "body", "document"]

            for candidate in candidates:
                if candidate in column_names:
                    target_column = candidate
                    break

            # 2. Fallback: if no common name is found, pick the first column that contains strings
            if not target_column:
                for col in column_names:
                    # Check the first row to see if it's a string
                    if isinstance(ds[0][col], str):
                        target_column = col
                        break

            if not target_column:
                raise ValueError(
                    f"Could not find a text column in dataset '{path}'. Available columns: {column_names}"
                )

            print(f"Found text in column: '{target_column}'")

            # Join all rows into one large string
            return "\n".join(ds[target_column])

    def get_batch(self, split, batch_size):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
