from .base import GPTConfig as BaseConfig
from dataclasses import dataclass


@dataclass
class GPTConfig(BaseConfig):
    # Simply inherit everything from BaseConfig
    pass
