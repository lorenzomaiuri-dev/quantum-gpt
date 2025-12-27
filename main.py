import os
import torch
import argparse
import importlib
import json
import time
from datetime import datetime
from dataclasses import asdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.dataset import InputDataset
from src.model import QuantumGPT

# Set seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

def setup_run_dir(model_name):
    """Creates a unique directory for the current training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("experiments", f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def train(config, model_name, dataset_name):
    run_dir = setup_run_dir(model_name)
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=run_dir)
    
    print(f"\n--- Starting Training ---")
    print(f"Run Directory: {run_dir}")
    print(f"To view logs, run: tensorboard --logdir=experiments")
    
    # Save Configuration immediately
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=4)

    # Dataset setup
    dataset_path = os.path.join("data", f"{dataset_name}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    dataset = InputDataset(dataset_path, config.block_size, config.device)
    vocab_size = dataset.tokenizer.vocab_size

    # Create Model
    model = QuantumGPT(config, vocab_size)
    m = model.to(config.device)
    
    # Log the model graph to TensorBoard
    try:
        dummy_xb, _ = dataset.get_batch('train', 1)
        writer.add_graph(m, dummy_xb)
    except Exception as e:
        print(f"Could not log model graph: {e}")
    
    # Save detailed model architecture to a text file
    with open(os.path.join(run_dir, "model_structure.txt"), "w") as f:
        f.write(str(m))
        total_params = sum(p.numel() for p in m.parameters())
        f.write(f"\n\nTotal Parameters: {total_params:,}\n")
        f.write(f"Quantum Enabled: {config.use_quantum}\n")

    print(f"Model Parameters: {total_params/1e3:.2f}K")

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
            out[split] = losses.mean().item()
        model.train()
        return out

    # Training Loop tracking
    best_val_loss = float('inf')
    history = []
    start_time = time.time()

    pbar = tqdm(range(config.max_iters), desc="Training Progress")
    
    for iter in pbar:
        # Evaluation step
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', losses['train'], iter)
            writer.add_scalar('Loss/val', losses['val'], iter)
            
            # Log to JSON history
            history.append({
                "step": iter, 
                "train_loss": round(losses['train'], 4), 
                "val_loss": round(losses['val'], 4)
            })
            
            pbar.set_description(f"Step {iter} | Val Loss: {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        # Training step
        xb, yb = dataset.get_batch('train', config.batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    end_time = time.time()
    total_duration = end_time - start_time

    # Log Final Hparams and Metric to TensorBoard for easy comparison
    writer.add_hparams(
        asdict(config), 
        {"hparam/best_val_loss": best_val_loss, "hparam/train_time": total_duration}
    )

    # Save Final Metrics and History to JSON
    metrics = {
        "model_name": model_name,
        "dataset": dataset_name,
        "total_training_time_seconds": round(total_duration, 2),
        "best_val_loss": round(best_val_loss, 4),
        "history": history
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save final checkpoint and close writer
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))
    writer.close()
    
    print(f"\nTraining completed in {total_duration:.2f}s.")
    return run_dir

def generate(config, run_dir, dataset_name, max_new_tokens=500):
    print(f"\n--- Starting Generation ---")
    dataset_path = os.path.join("data", f"{dataset_name}")
    dataset = InputDataset(dataset_path, config.block_size, config.device)
    
    model = QuantumGPT(config, dataset.tokenizer.vocab_size)
    load_path = os.path.join(run_dir, "best_model.pth")
    
    if not os.path.exists(load_path):
        load_path = os.path.join(run_dir, "final_model.pth")

    model.load_state_dict(torch.load(load_path, map_location=config.device))
    m = model.to(config.device)
    m.eval()

    context = torch.tensor([dataset.tokenizer.encode("\n")], dtype=torch.long, device=config.device)
    out_tokens = m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    decoded_text = dataset.tokenizer.decode(out_tokens)
    
    with open(os.path.join(run_dir, "generated_sample.txt"), "w", encoding="utf-8") as f:
        f.write(decoded_text)
    
    print("\n--- GENERATED TEXT ---")
    print(decoded_text)
    print(f"\nOutput saved to {run_dir}/generated_sample.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Transformer Runner")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate', 'full'])    
    parser.add_argument('--name', type=str, default='quantum_gpt')
    parser.add_argument('--dataset', type=str, default='input.txt')
    parser.add_argument('--config', type=str, default='default')
    parser.add_argument('--run_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Config loading
    try:
        config_module = importlib.import_module(f"src.config.{args.config}")
        GPTConfig = getattr(config_module, "GPTConfig")
        cfg = GPTConfig()
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)

    if args.mode == 'train':
        train(cfg, args.name, args.dataset)
    elif args.mode == 'generate':
        if not args.run_dir:
            print("Provide --run_dir")
        else:
            generate(cfg, args.run_dir, args.dataset)
    elif args.mode == 'full':
        run_folder = train(cfg, args.name, args.dataset)
        generate(cfg, run_folder, args.dataset)