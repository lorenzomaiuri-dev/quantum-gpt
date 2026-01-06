from .base import GPTConfig as BaseConfig
from dataclasses import dataclass


@dataclass
class GPTConfig(BaseConfig):
    batch_size: int = 8
    block_size: int = 8
    max_iters: int = 500
    eval_interval: int = 100
    learning_rate: float = 3e-3
    eval_iters: int = 50
    n_embd: int = 8
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.0
    use_quantum: bool = True
