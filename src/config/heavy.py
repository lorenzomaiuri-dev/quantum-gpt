import torch
from dataclasses import dataclass, asdict

@dataclass
class GPTConfig:
    tokenizer_class: str = "BiCharTokenizer"

    batch_size: int = (
        64  # Number of independent sequences processed in parallel per training step
    )
    block_size: int = (
        128  # Maximum context length: the number of tokens the model can look back at
    )
    max_iters: int = 20000  # Total number of training iterations (steps)
    eval_interval: int = 150  # How often (in iterations) to run the evaluation loop
    learning_rate: float = (
        3e-4  # The step size used by the optimizer for weight updates
    )
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Hardware used for computation (GPU or CPU)
    eval_iters: int = 200  # Number of batches to average over when estimating the loss during evaluation
    n_embd: int = 128  # Total dimension of the token embeddings (hidden size)
    n_head: int = 32  # Number of attention heads in the Multi-Head Attention mechanism
    n_layer: int = 9  # Number of Transformer blocks (layers) in the model
    dropout: float = (
        0.15  # Probability of dropping neurons during training to prevent overfitting
    )

    # Quantum config
    use_quantum: bool = False  # Toggle to enable or disable quantum-enhanced layers
    n_qlayers: int = 2  # Depth (number of layers) of the variational quantum circuit
    q_device: str = (
        "default.qubit"  # The backend simulator or quantum hardware used for execution
    )

    @property
    def n_qubits(self):
        # Calculates the number of qubits, typically equal to the dimension of a single attention head
        return self.n_embd // self.n_head

    def to_dict(self):
        out = asdict(self)
        out["n_qubits"] = self.n_qubits
        return out
