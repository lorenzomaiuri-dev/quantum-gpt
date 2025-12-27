import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayerAdapter(nn.Module):
    """
    A Hybrid Quantum-Classical Layer.

    This module solves the 'dimensionality mismatch' problem. Transformer embeddings
    are usually large (e.g., 64), but simulating many qubits is slow.

    Strategy:
    Compress input (Classical Linear) -> n_qubits
    Process info (Quantum Circuit) -> n_qubits
    Output result (Reshaped to time-sequence)
    """

    def __init__(self, n_in, n_out, n_qubits, n_qlayers, q_device):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_qubits = n_qubits

        # Initialize the Quantum Device (e.g., 'default.qubit' or 'lightning.qubit')
        self.dev = qml.device(q_device, wires=n_qubits)

        # TODO: PENNYLANE OPTIMIZATION FOR AUTOMATIC PARAMETERS
        # Define the Quantum Circuit (The "QNode")
        # Interface='torch' allows backpropagation through the circuit
        @qml.qnode(self.dev, interface="torch")
        def _circuit(inputs, weights):
            # Encoding: Map classical data to quantum state (rotations)
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

            # Variational Layer: Entanglement and Rotations (The "Trainable" part)
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Measurement: Extract information (Expectation value of Pauli Z)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = _circuit

        # Define shape of trainable weights for the quantum circuit
        # (Layers, Qubits, 3 parameters per qubit per layer)
        self.weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}

        # Classical Adapter (Bottleneck)
        # Projects high-dimensional embedding down to the number of qubits
        self.adapter = nn.Linear(n_in, n_qubits)

        # Wrap the QNode as a Torch Layer so it acts like a standard nn.Module
        self.qlayer = qml.qnn.TorchLayer(self.circuit, self.weight_shapes)

    def forward(self, x):
        # Input shape: (Batch, Time, Embedding_Dim)
        B, T, C = x.shape

        # Flatten Batch and Time to process everything in parallel (or as one large batch)
        x_flat = x.view(-1, C)

        # Compression: Reduce dimension to fit the number of qubits
        x_reduced = self.adapter(x_flat)

        # Scaling: AngleEmbedding expects values roughly between -pi and pi.
        # Tanh squashes data to [-1, 1], multiplying by pi scales to [-pi, pi].
        x_scaled = torch.tanh(x_reduced) * torch.pi

        # Quantum Processing
        out_flat = self.qlayer(x_scaled)

        # Restore original shape for the Transformer
        # This assumes the quantum output dim equals n_out.
        # If n_out > n_qubits, an output linear adapter would be needed here.
        return out_flat.view(B, T, self.n_out)
