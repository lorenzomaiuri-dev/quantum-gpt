from .base import GPTConfig as BaseConfig
from dataclasses import dataclass


@dataclass
class GPTConfig(BaseConfig):
    batch_size: int = 8
    block_size: int = 16
    max_iters: int = 1000
    eval_interval: int = 100
    learning_rate: float = 1e-3
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    use_quantum: bool = False
