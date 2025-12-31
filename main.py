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

from concurrent.futures import ProcessPoolExecutor

def setup_run_dir(model_name):
	"""Creates a unique directory for the current training run."""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join("experiments", f"{model_name}_{timestamp}")
	os.makedirs(run_dir, exist_ok=True)
	os.makedirs(os.path.join(run_dir, 'generated_samples'), exist_ok=True)
	return run_dir


def train(config, model_name, dataset_name):
	run_dir = setup_run_dir(model_name)

	# Initialize TensorBoard Writer
	writer = SummaryWriter(log_dir=run_dir)

	print("\n--- Starting Training ---")
	print(f"Run Directory: {run_dir}")
	print("To view logs, run: tensorboard --logdir=experiments")

	# Save Configuration immediately
	config_dict = config.to_dict()
	with open(os.path.join(run_dir, "config.json"), "w") as f:
		json.dump(config_dict, f, indent=4)

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
		dummy_xb, _ = dataset.get_batch("train", 1)
		writer.add_graph(m, dummy_xb)
	except Exception as e:
		print(f"Could not log model graph: {e}")

	# Save detailed model architecture to a text file
	with open(os.path.join(run_dir, "model_structure.txt"), "w") as f:
		f.write(str(m))
		total_params = sum(p.numel() for p in m.parameters())
		f.write(f"\n\nTotal Parameters: {total_params:,}\n")
		f.write(f"Quantum Enabled: {config.use_quantum}\n")

	print(f"Model Parameters: {total_params / 1e3:.2f}K")

	# Setup Training
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

	@torch.no_grad()
	def estimate_loss():
		out = {}
		model.eval()
		for split in ["train", "val"]:
			losses = torch.zeros(config.eval_iters)
			for k in range(config.eval_iters):
				X, Y = dataset.get_batch(split, config.batch_size)
				_, loss = model(X, Y)
				losses[k] = loss.item()
			out[split] = losses.mean().item()
		model.train()
		return out

	# Training Loop tracking
	best_val_loss = float("inf")
	history = []
	start_time = time.time()

	pbar = tqdm(range(config.max_iters), desc="Training Progress")

	for iter in pbar:
		# Evaluation step
		if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
			losses = estimate_loss()

			# Log to TensorBoard
			writer.add_scalar("Loss/train", losses["train"], iter)
			writer.add_scalar("Loss/val", losses["val"], iter)

			# Log to JSON history
			history.append(
				{
					"step": iter,
					"train_loss": round(losses["train"], 4),
					"val_loss": round(losses["val"], 4),
				}
			)

			pbar.set_description(f"Step {iter} | Val Loss: {losses['val']:.4f}")

			# Save best model
			if losses["val"] < best_val_loss:
				best_val_loss = losses["val"]
				torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))  # nosec B614

		# Training step
		xb, yb = dataset.get_batch("train", config.batch_size)
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
		{"hparam/best_val_loss": best_val_loss, "hparam/train_time": total_duration},
	)

	# Save Final Metrics and History to JSON
	metrics = {
		"model_name": model_name,
		"dataset": dataset_name,
		"total_training_time_seconds": round(total_duration, 2),
		"best_val_loss": round(best_val_loss, 4),
		"history": history,
	}
	with open(os.path.join(run_dir, "metrics.json"), "w") as f:
		json.dump(metrics, f, indent=4)

	# Save final checkpoint and close writer
	torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))  # nosec B614
	writer.close()

	print(f"\nTraining completed in {total_duration:.2f}s.")
	return run_dir


def generate(config, run_dir, dataset_name, max_new_tokens=500, print_in_place=False, hint='', seeds=''):
	print("\n--- Starting Generation ---")
	dataset_path = os.path.join("data", f"{dataset_name}")
	dataset = InputDataset(dataset_path, config.block_size, config.device)

	model = QuantumGPT(config, dataset.tokenizer.vocab_size)
	load_path = os.path.join(run_dir, "best_model.pth")

	if not os.path.exists(load_path):
		load_path = os.path.join(run_dir, "final_model.pth")
		print('## using final_model')
	else:
		print('## using best_model')
		
	model.load_state_dict(
		torch.load(load_path, map_location=config.device, weights_only=True)  # nosec B614
	)
	m = model.to(config.device)
	m.eval()

	# hint += '\n'
	context = torch.tensor(
		[dataset.tokenizer.encode(hint)], dtype=torch.long, device=config.device
	)
	def do_generate(seed):
		if print_in_place:
			print(hint, end='')
		torch.manual_seed(int(seed))
		if torch.cuda.is_available():
			torch.cuda.manual_seed(int(seed))
		out_tokens = m.generate(context, max_new_tokens=max_new_tokens, print_in_place=print_in_place, decode_function=dataset.tokenizer.decode)[0].tolist()
		decoded_text = dataset.tokenizer.decode(out_tokens)
		return decoded_text
	
	res = []
	# with ProcessPoolExecutor() as executor:
	for i in range(len(seeds)):
		# res = list(executor.map(do_generate, seeds))
		res.append(do_generate(seeds[i]))
		if print_in_place:
			print('\n----------------------------------------------\n')

	if not print_in_place:
		print(f"\n--- GENERATED TEXT{"S" if len(seeds) > 1 else ""} ---")
	for i in range(len(res)):
		if not print_in_place:
			print('-- ', seeds[i], ' --\n', res[i])
		os.makedirs(os.path.join(run_dir, 'generated_samples'), exist_ok=True)
		dir = os.path.join(run_dir, "generated_samples", hint+str(seeds[i])+'.txt')
		with open(
			dir , "w", encoding="utf-8"
		) as f:
			f.write(res[i])
		if not print_in_place:
			print('\n')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Quantum Transformer Runner")
	parser.add_argument(
		"--mode", type=str, required=True, choices=["train", "generate"]
	)
	parser.add_argument("--name", type=str, default="quantum_gpt")
	parser.add_argument("--dataset", type=str, default="input.txt")
	parser.add_argument("--config", type=str, default="default")
	parser.add_argument("--run_dir", type=str, default=None)
	parser.add_argument("--tokens", type=str, default="500")
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--seeds", type=str, default=None)
	parser.add_argument("--generations", type=int, default=-1)
	parser.add_argument("--print_in_place", action='store_true')		# just funy
	parser.add_argument("--hint", type=str, default="", nargs='+')

	args = parser.parse_args()
			

	if args.mode == "train":
		try:
			config_module = importlib.import_module(f"src.config.{args.config}")
			GPTConfig = getattr(config_module, "GPTConfig")
			cfg = GPTConfig()
		except Exception as e:
			print(f"Error loading config: {e}")
			exit(1)
		train(cfg, args.name, args.dataset)

	elif args.mode == "generate":
		args.hint = ' '.join(args.hint)

		if args.hint:
			print(f"## hint: {args.hint}")

		# setup seeds list for multi generation
		seeds = args.seeds.split(",") if args.seeds else []
		if len(seeds) == 0 and args.generations == -1:
			seeds.append(args.seed)
		if args.generations < 0:
			args.generations *= -1
		if len(seeds) < args.generations:
			from random import randint
			for _ in range(args.generations - len(seeds)):
				seeds.append(str(randint(0, 0x7FFFFFFF)))
		if len(seeds) != 1:
			print("seeds:", seeds)
			# if args.print_in_place:
			# 	print("WARNING: Cannot run and print in place concurrent generations")
			# 	args.print_in_place = False
		
		args.run_dir = "experiments/" + args.run_dir

		try:
			config_module = importlib.import_module(f"src.config.{args.config}")
			GPTConfig = getattr(config_module, "GPTConfig")
			cfg = GPTConfig()
			# Loads the config from the run trace
			# with open(f'{args.run_dir}/config.json', 'r') as f:
			# 	for key, prop in json.load(f).items():
			# 		setattr(cfg, key, prop)
		except Exception as e:
			print(f"Error loading config: {e}")
			exit(1)
		if not args.run_dir:
			print("Provide --run_dir")
			
		else:
			generate(cfg, args.run_dir, args.dataset, int(args.tokens), print_in_place=args.print_in_place, hint=args.hint, seeds=seeds)
