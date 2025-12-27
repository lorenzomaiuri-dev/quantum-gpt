# Quantum GPT (Hybrid QNN-NanoGPT)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![PennyLane](https://img.shields.io/badge/PennyLane-0.30%2B-yellow)
![MIT](https://img.shields.io/badge/license-MIT-green)

A hybrid Quantum-Classical implementation of a Generative Pre-trained Transformer (GPT).
This project adapts Andrej Karpathy's `nanoGPT` architecture by replacing classical linear layers in the Self-Attention mechanism with **Variational Quantum Circuits (VQC)** using PennyLane.

## 🚀 Scientific Concept

In a standard Transformer, the Attention Head projects input tokens into Query, Key, and Value spaces using linear matrices ($W_Q, W_K, W_V$).

In this **Quantum-Hybrid architecture**, we replace these dense layers with a parameterized quantum evolution:

$$
x \xrightarrow{\text{Adapter}} z \in \mathbb{R}^n \xrightarrow{R(\phi)} |\psi(z)\rangle \xrightarrow{U(\theta)_{\text{entangle}}} \langle Z \rangle \to y
$$

**Where:**

*   **Adapter**: A classical bottleneck layer compressing high-dimensional embeddings to $n$ qubits.
*   $R(\phi)$: **Angle embedding** encoding classical data into quantum states.
*   $U(\theta)$: A sequence of trainable entangling layers (**Strongly Entangling Layers**).
*   $\langle Z \rangle$: **Expectation value measurement** returning the projected vector.

### Why?
This architecture allows us to study if the high-dimensional **Hilbert space** and **quantum interference** can capture semantic relationships more efficiently (parameter-wise) than classical linear algebra, despite the constraints of current NISQ simulation.

This allows exploring the expressivity of quantum circuits within a sequence modeling task.

    Note: We employ a Quantum Bottleneck architecture. High-dimensional classical embeddings are projected down to a lower-dimensional quantum latent space via a trainable adapter, processed by the VQC, and projected back. This maintains computational feasibility while exploiting quantum interference.


## 📂 Project Structure

```text
quantum-transformer/
├── checkpoints/                # Saved models
├── data/                       # Input text data
├── src/                        # Source code
│   ├── config.py               # Hyperparameters & flags
│   ├── dataset.py              # Tokenizer & Dataloader
│   ├── model.py                # Transformer Architecture
│   └── quantum_layers.py       # PennyLane Circuits & Hybrid Layers
├── main.py                     # Entry point (Train/Generate)
└── requirements.txt            # Dependencies
```

## 🛠️ Installation
Clone the repository:
```bash
git clone https://github.com/lorenzomaiuri-dev/quantum-gpt.git
cd quantum-transformer
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚡ Usage
### Training

To train the model on the Shakespeare dataset (included in data/):
```bash
python main.py --mode train
```
Note: Quantum simulation is CPU-intensive. The default configuration uses a "Quantum Bottleneck" (4-8 qubits) to keep training times feasible on consumer hardware.

### Generation

To generate text using the trained checkpoint:
```bash
python main.py --mode generate
```

## ⚙️ Configuration

You can modify hyperparameters in src/config.py:
```Python
# Quantum Settings
USE_QUANTUM = True      # Set False to use standard Linear Layers
N_QUBITS = 4            # Number of qubits per head
N_QLAYERS = 2           # Depth of the quantum circuit
```

## 🧠 Architecture Details

Embedding Dimension: 8 (scaled down for simulation speed)\
Heads: 2\
Qubits per Head: 4

## 📊 Preliminary Results (Coming Soon)
Comparison between Classical (64 params) vs Hybrid Quantum (4 qubits) attention heads:
- [ ] Loss Convergence: Comparing training stability.
- [ ] Parameter Efficiency: Can quantum circuits learn with fewer parameters?
- [ ] Runtime Analysis: Quantifying the overhead of quantum simulation.

## 🙏 Acknowledgements
Andrej Karpathy for the original nanoGPT and Video Lecture.\
Xanadu for the PennyLane library used for quantum machine learning.
