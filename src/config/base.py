import torch
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    # --- Default Training Parameters ---
    tokenizer_class: str = "BiCharTokenizer"
    batch_size: int = (
        32  # Number of independent sequences processed in parallel per training step
    )
    block_size: int = (
        64  # Maximum context length: the number of tokens the model can look back at
    )
    max_iters: int = 5000  # Total number of training iterations (steps)
    eval_interval: int = 100  # How often (in iterations) to run the evaluation loop
    learning_rate: float = (
        1e-3  # The step size used by the optimizer for weight updates
    )
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Hardware used for computation (GPU or CPU)
    eval_iters: int = 200  # Number of batches to average over when estimating the loss during evaluation
    n_embd: int = 32  # Total dimension of the token embeddings (hidden size)
    n_head: int = 8  # Number of attention heads in the Multi-Head Attention mechanism
    n_layer: int = 4  # Number of Transformer blocks (layers) in the model
    dropout: float = (
        0.05  # Probability of dropping neurons during training to prevent overfitting
    )

    # --- Default Quantum Parameters ---
    use_quantum: bool = False  # Toggle to enable or disable quantum-enhanced layers
    n_qlayers: int = 2  # Depth (number of layers) of the variational quantum circuit
    q_device: str = (
        "default.qubit"  # The backend simulator or quantum hardware used for execution
    )

    @property
    def n_qubits(self) -> int:
        """Calculates qubits based on embedding dimension and number of heads."""
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        """Serializes the config to a dictionary, including computed properties."""
        out = asdict(self)
        out["n_qubits"] = self.n_qubits  # Ensure property is included in JSON
        return out
