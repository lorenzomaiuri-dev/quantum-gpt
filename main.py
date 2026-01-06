import argparse
import importlib
import secrets
import os
import json
import torch
import shutil
from src.training import Trainer
from src.inference import Generator


def main():
    parser = argparse.ArgumentParser(description="Quantum GPT Runner")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "generate", "full"]
    )
    parser.add_argument("--name", type=str, default="quantum_gpt")
    parser.add_argument("--dataset", type=str, default="input.txt")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--print_in_place", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--hint", type=str, default="", nargs="+")

    args = parser.parse_args()
    args.hint = " ".join(args.hint)

    # 1. Load Configuration
    try:
        config_module = importlib.import_module(f"src.config.{args.config}")
        cfg = config_module.GPTConfig()
        if args.force_cpu:
            cfg.device = "cpu"
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. Training Mode
    if args.mode in ["train", "full"]:
        trainer = Trainer(cfg, args.name, args.dataset)
        try:
            args.run_dir = trainer.train()
        except BaseException as e:
            # Handle failure: move partial logs to experiments_failed
            failed_dir = os.path.join(
                "experiments_failed", os.path.basename(trainer.run_dir)
            )
            print(f"Training failed. Moving logs to {failed_dir}")
            shutil.move(trainer.run_dir, failed_dir)
            raise e

    # 3. Generation Mode
    if args.mode in ["generate", "full"]:
        if not args.run_dir:
            print("Error: --run_dir is required for generation mode.")
            return

        # Reload config from the specific run directory to ensure consistency
        with open(os.path.join(args.run_dir, "config.json"), "r") as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                # Check if the attribute exists and if it is a property (read-only)
                is_property = isinstance(getattr(type(cfg), key, None), property)

                if not is_property:
                    setattr(cfg, key, value)

        if args.force_cpu:
            cfg.device = "cpu"

        # Setup seeds
        seeds = args.seeds.split(",") if args.seeds else []
        if not seeds:
            if args.generations > 1:
                seeds = [str(args.seed)] + [
                    str(secrets.randbelow(1000000)) for _ in range(args.generations - 1)
                ]
            else:
                seeds = [args.seed]

        # Run Inference
        gen = Generator(cfg, args.run_dir, args.dataset)
        results = gen.generate(
            hint=args.hint,
            seeds=seeds,
            max_new_tokens=args.tokens,
            print_in_place=args.print_in_place,
        )

        if not args.print_in_place:
            for i, res in enumerate(results):
                print(f"\n--- Result (Seed {seeds[i]}) ---\n{res}\n")


if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(1337)
    main()
