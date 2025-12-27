import torch
import torch.nn as nn
from torch.nn import functional as F
from src.quantum_layers import QuantumLayerAdapter

class Head(nn.Module):
    """ 
    One head of self-attention. 
    This is where the model calculates the relationship between tokens.
    """
    def __init__(self, config, head_size):
        super().__init__()
        
        # Initialize Query, Key, and Value projections.
        # This is the critical part: we dynamically choose between a Classical Linear layer
        # or a Quantum Circuit based on the configuration.
        self.key = self._get_layer(config, config.n_embd, head_size)
        self.query = self._get_layer(config, config.n_embd, head_size)
        self.value = self._get_layer(config, config.n_embd, head_size)
        
        # 'tril' is a lower-triangular matrix used for masking.
        # It ensures the model cannot "see the future" (causal masking).
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def _get_layer(self, config, n_in, n_out):
        """Helper method to instantiate either a Quantum or Classical layer."""
        if config.use_quantum:
            # Inject the VQC (Variational Quantum Circuit) adapter here
            return QuantumLayerAdapter(n_in, n_out, config.n_qubits, config.n_qlayers, config.q_device)
        # Standard classical linear projection
        return nn.Linear(n_in, n_out, bias=False)

    def forward(self, x):
        # B = Batch size, T = Time/Sequence length, C = Channel/Embedding dimension
        B, T, C = x.shape
        
        # Project input x to Key, Query, and Value spaces
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # Scaled dot-product attention: (Q @ K^T) / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        # Apply the causal mask (set future positions to -infinity so softmax makes them 0)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # Normalize to get probabilities
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ 
    Multiple heads of self-attention in parallel. 
    Allows the model to attend to different types of information simultaneously.
    """
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Concatenate the outputs of all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Final linear projection back to the embedding dimension
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 
    A simple linear layer followed by a non-linearity. 
    Processed independently for each token (position-wise).
    """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), # Expand dimension by 4
            nn.ReLU(),                                   # Non-linearity
            nn.Linear(4 * config.n_embd, config.n_embd), # Project back down
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ 
    A standard Transformer block: communication followed by computation. 
    Communication = Multi-Head Attention
    Computation = FeedForward Network
    """
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Apply Attention with Residual Connection and LayerNorm
        x = x + self.sa(self.ln1(x))
        # Apply FeedForward with Residual Connection and LayerNorm
        x = x + self.ffwd(self.ln2(x))
        return x

class QuantumGPT(nn.Module):
    """
    The main model assembly.
    Combines embeddings, transformer blocks, and the final output head.
    """
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language Model Head (projects to vocabulary size to predict next token)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get token embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        # Get position embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device)) 
        
        # Combine them
        x = tok_emb + pos_emb 
        x = self.blocks(x) # Apply transformer blocks
        x = self.ln_f(x)   # Apply final normalization
        logits = self.lm_head(x) # Get predictions (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten the logits and targets to calculate Cross Entropy Loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressive generation loop.
        Predicts the next token based on the context, one by one.
        """
        for _ in range(max_new_tokens):
            # Crop the context to the last 'block_size' tokens
            idx_cond = idx[:, -self.config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step (the prediction for the next token)
            logits = logits[:, -1, :] 
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) 
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) 
            
        return idx