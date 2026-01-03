from .base import GPTConfig as BaseConfig
from dataclasses import dataclass


@dataclass
class GPTConfig(BaseConfig):
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 20000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
