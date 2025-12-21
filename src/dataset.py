import torch

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class ShakespeareDataset:
    def __init__(self, file_path, block_size, device):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        self.tokenizer = CharTokenizer(self.text)
        self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)
        self.block_size = block_size
        self.device = device
        
        # Split
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split, batch_size):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)