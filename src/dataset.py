import torch
import json
from functools import cached_property

# Simple integer tokenizer. By overloading the __init__() mehtod you can
# set the self.keys property to whatever length of characters or whatever
# list of (non repeating) words you prefer
class BaseTokenizer:
    def __init__(self):
        if hasattr(self, "keys"):
            self.enc = self.encoder()
            self.dec = self.decoder()
    
    def to_dict(self):
        return self.__dict__
    
    def from_dict(self, data):
        data['dec'] = {int(k) : v for k,v in data['dec'].items()}
        data['enc'] = {k : int(v) for k,v in data['enc'].items()}
        self.__dict__.update(data)

    def encode(self, strings):
        """Converts a string into a list of integers."""
        return [self.enc[c] for c in strings]

    def decode(self, integers):
        """Converts a list of integers back into a string."""
        return "".join([self.dec[i] for i in integers])

    def decoder(self):
        return {i: ch for i, ch in enumerate(self.keys)}

    def encoder(self):
        return {ch: i for i, ch in enumerate(self.keys)}
    
    @cached_property
    def vocab_size(self):
        return max(len(self.dec), len(self.enc))

class CharTokenizer(BaseTokenizer):
    """
    A simple character-level tokenizer.
    It builds a vocabulary from the input text and handles
    string-to-integer (encode) and integer-to-string (decode) conversions.
    """
    def __init__(self, data=''):
        if data:
            self.keys = sorted(list(set(data)))
            super().__init__()


class BiCharTokenizer(BaseTokenizer):
    def __init__(self, data=''):
        if data:
            char2 = list(set([data[i:i+2] for i in range(0, len(data), 2)]))    # two character keys
            self.keys = char2 + list(set(data))
            self.keys.sort()
            super().__init__()

class TriCharTokenizer(BaseTokenizer):
    def __init__(self, data=''):
        if data:
            char2 = list(set([data[i:i+2] for i in range(0, len(data), 2)]))    # two character keys
            char3 = list(set([data[i:i+3] for i in range(0, len(data), 3)]))    # two character keys
            self.keys = char2 + char3 + list(set(data))
            self.keys.sort()
            super().__init__()

# NO à, è, é, ì, ò, ù :(
class ASCIITokenizer(BaseTokenizer):
    def __init__(self, data=''):
        if data:
            self.keys = [chr(i) for i in range(123)]
            super().__init__()


# class BiASCIITokenizer(BaseTokenizer):
#     def __init__(self, data=''):
#         from itertools import product
#         if data:
#             chars = [chr(i) for i in range(123)]
#             self.keys = chars + list(product(chars, repeat=2))
#             super().__init__()

# class TriASCIITokenizer(BaseTokenizer):
#     def __init__(self, data=''):
#         from itertools import product
#         if data:
#             chars = [chr(i) for i in range(123)]
#             self.keys = chars + list(product(chars, repeat=2)) + list(product(chars, repeat=3))
#             self.keys.sort()
#             super().__init__()

class InputDataset:
    """
    Handles loading the text file, tokenizing the data,
    and generating batches for training/validation.
    """

    def __init__(self, file_path, block_size, device, dictionary_path='', tokenizer : str = "CharTokenizer"):
        if dictionary_path:  # generate
            # Initialize the tokenizer and convert the whole text to a tensor
            self.tokenizer = BaseTokenizer()    # uses BaseTokenizer because data is loaded and not computed
            with open(dictionary_path, "r", encoding="utf-8") as f:
                self.tokenizer.from_dict(json.load(f))
        else:           # train
            # Read the entire text file
            with open(file_path, "r", encoding="utf-8") as f:
                self.text = f.read()
            # Initialize the tokenizer and convert the whole text to a tensor
            self.tokenizer = globals().get(tokenizer)(self.text)
            print(self.tokenizer.enc)
            self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)
            # Create a Train/Validation split (90% training, 10% validation)
            n = int(0.9 * len(self.data))
            self.train_data = self.data[:n]
            self.val_data = self.data[n:]
            
            self.block_size = block_size
            self.device = device

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
