import os
import torch
import argparse
from src.config.default import GPTConfig
from src.dataset import InputDataset
from src.model import QuantumGPT
from tqdm import tqdm

# Set seed
torch.manual_seed(1337)
torch.cuda.empty_cache()

def train(config, model_name, dataset_name):
    print(f"--- Starting Training ---")
    print(f"Device: {config.device} | Quantum: {config.use_quantum}")
    dataset_path = os.path.join("data", f"{dataset_name}")
    dataset = InputDataset(dataset_path, config.block_size, config.device)    

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    dataset = InputDataset(dataset_path, config.block_size, config.device)
    vocab_size = dataset.tokenizer.vocab_size

    print(f"Dataset: {dataset_path}")            

    # Create Model
    model = QuantumGPT(config, vocab_size)
    m = model.to(config.device)
    print(f"Model Parameters: {sum(p.numel() for p in m.parameters())/1e3:.2f}K")

    # Setup Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = dataset.get_batch(split, config.batch_size)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Training Loop
    best_val_loss = float('inf')
    pbar = tqdm(range(config.max_iters), desc="Training Progress")
    
    for iter in pbar:
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            pbar.set_description(f"Step {iter} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

        xb, yb = dataset.get_batch('train', config.batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    save_path = os.path.join("checkpoints", f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model saved to {save_path}")

def generate(config, model_name, dataset_name, max_new_tokens=500):
    print(f"--- Starting Generation ---")
    
    dataset_path = os.path.join("data", f"{dataset_name}")
    dataset = InputDataset(dataset_path, config.block_size, config.device)
    vocab_size = dataset.tokenizer.vocab_size

    model = QuantumGPT(config, vocab_size)
    load_path = os.path.join("checkpoints", f"{model_name}.pth")
    
    if not os.path.exists(load_path):
        print(f"Error: Model not found at {load_path}. Train it first!")
        return

    model.load_state_dict(torch.load(load_path, map_location=config.device))
    m = model.to(config.device)
    m.eval()

    # Generate
    start_str = "\n"
    context_tokens = dataset.tokenizer.encode(start_str)
    context = torch.tensor([context_tokens], dtype=torch.long, device=config.device)
    
    print(f"Generating {max_new_tokens} tokens using {model_name}...")
    out_tokens = m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    decoded_text = dataset.tokenizer.decode(out_tokens)
    
    print("\n--- GENERATED TEXT ---")
    print(decoded_text)
    print("----------------------")

    output_file = os.path.join("outputs", f"{model_name}_generated.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(decoded_text)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Transformer Runner")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate', 'full'], 
                        help="Choose 'train', 'generate' or 'full'")    
    parser.add_argument('--name', type=str, default='quantum_gpt', 
                        help="Name of the model (used for checkpoint filename)")
    parser.add_argument('--dataset', type=str, default='input.txt', 
                        help="Name to the dataset text file")
    
    args = parser.parse_args()
    
    cfg = GPTConfig()

    if args.mode == 'train':
        train(cfg, args.name, args.dataset)
    elif args.mode == 'generate':
        generate(cfg, args.name, args.dataset)
    elif args.mode == 'full':
        train(cfg, args.name, args.dataset)
        generate(cfg, args.name, args.dataset)