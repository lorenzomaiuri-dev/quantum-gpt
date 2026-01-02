import torch
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    from main import ForceCpu
    tokenizer_class: str = "BiCharTokenizer"

    batch_size: int = (
        16  # Number of independent sequences processed in parallel per training step
    )
    block_size: int = (
        32  # Maximum context length: the number of tokens the model can look back at
    )
    max_iters: int = 5000  # Total number of training iterations (steps)
    eval_interval: int = 100  # How often (in iterations) to run the evaluation loop
    learning_rate: float = (
        1e-3  # The step size used by the optimizer for weight updates
    )
    device: str = (
        "cuda" if torch.cuda.is_available() and not ForceCpu.FORCE_CPU else "cpu"
    )  # Hardware used for computation (GPU or CPU)
    eval_iters: int = 200  # Number of batches to average over when estimating the loss during evaluation
    n_embd: int = 64  # Total dimension of the token embeddings (hidden size)
    n_head: int = 4  # Number of attention heads in the Multi-Head Attention mechanism
    n_layer: int = 4  # Number of Transformer blocks (layers) in the model
    dropout: float = (
        0.0  # Probability of dropping neurons during training to prevent overfitting
    )

    # Quantum config
    use_quantum: bool = False  # Toggle to enable or disable quantum-enhanced layers
    n_qlayers: int = 2  # Depth (number of layers) of the variational quantum circuit
    q_device: str = (
        "default.qubit"  # The backend simulator or quantum hardware used for execution
    )

    def __init__(self):
        self.n_qubits = self.calc_n_qubits()

    # @property
    def calc_n_qubits(self):
        # Calculates the number of qubits, typically equal to the dimension of a single attention head
        return self.n_embd // self.n_head

    def to_dict(self):
        out = asdict(self)
        # out["n_qubits"] = self.n_qubits
        return out
