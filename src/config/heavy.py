from .base import GPTConfig as BaseConfig
from dataclasses import dataclass


@dataclass
class GPTConfig(BaseConfig):
    batch_size: int = 64
    block_size: int = 128
    max_iters: int = 20000
    eval_interval: int = 150
    learning_rate: float = 3e-4
    n_embd: int = 128
    n_head: int = 32
    n_layer: int = 9
    dropout: float = 0.15
    use_quantum: bool = False
