import torch


class CharTokenizer:
    """
    A simple character-level tokenizer.
    It builds a vocabulary from the input text and handles
    string-to-integer (encode) and integer-to-string (decode) conversions.
    """

    def __init__(self, text):
        # Find all unique characters in the text to build the vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create mappings: string-to-index (stoi) and index-to-string (itos)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, strings):
        """Converts a string into a list of integers."""
        return [self.stoi[c] for c in strings]

    def decode(self, integers):
        """Converts a list of integers back into a string."""
        return "".join([self.itos[i] for i in integers])


class InputDataset:
    """
    Handles loading the text file, tokenizing the data,
    and generating batches for training/validation.
    """

    def __init__(self, file_path, block_size, device):
        # Read the entire text file
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Initialize the tokenizer and convert the whole text to a tensor
        self.tokenizer = CharTokenizer(self.text)
        self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)

        self.block_size = block_size
        self.device = device

        # Create a Train/Validation split (90% training, 10% validation)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split, batch_size):
        """
        Generates a small batch of inputs x and targets y.
        """
        data = self.train_data if split == "train" else self.val_data

        # Select random starting indices for the batch
        # We subtract block_size to ensure we don't go out of bounds
        ix = torch.randint(len(data) - self.block_size, (batch_size,))

        # Stack the rows to create the batch tensors
        # x: the context (0 to block_size)
        # y: the targets (1 to block_size + 1), shifted by one for next-token prediction
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        return x.to(self.device), y.to(self.device)
