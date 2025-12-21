import torch
import torch.nn as nn
import pennylane as qml

class QuantumLayerAdapter(nn.Module):
    def __init__(self, n_in, n_out, n_qubits, n_qlayers, q_device):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_qubits = n_qubits
        
        # Device
        self.dev = qml.device(q_device, wires=n_qubits)
        
        # Create Quantum Circuit
        @qml.qnode(self.dev, interface="torch")
        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.circuit = _circuit
        self.weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        
        # Classic Adapter
        self.adapter = nn.Linear(n_in, n_qubits)
        
        # Torch Quantum Layer
        self.qlayer = qml.qnn.TorchLayer(self.circuit, self.weight_shapes)

    def forward(self, x):
        # x shape: (B, T, n_embd)
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # Adapter and scaling for AngleEmbedding
        x_reduced = self.adapter(x_flat)
        x_scaled = torch.tanh(x_reduced) * torch.pi
        
        # Quantum Layer
        out_flat = self.qlayer(x_scaled)
        
        # Restore shape
        # If n_out is different from n_qubits, another linear layer is needed
        return out_flat.view(B, T, self.n_out)