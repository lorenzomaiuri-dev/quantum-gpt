from dataclasses import dataclass

@dataclass
class GPTConfig:
    batch_size: int = 8
    block_size: int = 8
    max_iters: int = 1000
    eval_interval: int = 100
    learning_rate: float = 1e-3
    device: str = 'cpu'
    eval_iters: int = 50
    n_embd: int = 8
    n_head: int = 2
    n_layer: int = 1
    dropout: float = 0.0
    
    # Quantum config
    use_quantum: bool = True
    n_qlayers: int = 2
    q_device: str = 'default.qubit'
    
    @property
    def n_qubits(self):
        return self.n_embd // self.n_head