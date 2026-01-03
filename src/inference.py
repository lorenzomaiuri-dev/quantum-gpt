import os
import torch
from src.dataset import InputDataset
from src.model import QuantumGPT


class Generator:
    def __init__(self, config, run_dir, dataset_name):
        self.config = config
        self.run_dir = run_dir

        # Load Dataset and Tokenizer from run artifacts
        self.dataset = InputDataset(
            os.path.join("data", dataset_name),
            config.block_size,
            config.device,
            os.path.join(run_dir, "dictionary.json"),
            config.tokenizer_class,
        )

        # Model Loading
        self.model = QuantumGPT(config, self.dataset.tokenizer.vocab_size)

        # Priority: Best Model -> Final Model
        load_path = os.path.join(run_dir, "best_model.pth")
        if not os.path.exists(load_path):
            load_path = os.path.join(run_dir, "final_model.pth")
            print("## Using final_model.pth")
        else:
            print("## Using best_model.pth")

        self.model.load_state_dict(
            torch.load(load_path, map_location=config.device, weights_only=True)  # nosec B614
        )
        self.model.to(config.device)
        self.model.eval()

    def generate(self, hint="", seeds=None, max_new_tokens=500, print_in_place=False):
        print("\n--- Starting Generation ---")
        if not seeds:
            seeds = [1337]

        if not hint:
            hint = "\n"

        context = torch.tensor(
            [self.dataset.tokenizer.encode(hint)],
            dtype=torch.long,
            device=self.config.device,
        )

        results = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(int(seed))

            if print_in_place:
                print(f"\n-- Seed {seed} --\n{hint}", end="")

            out_tokens = self.model.generate(
                context,
                max_new_tokens=max_new_tokens,
                print_in_place=print_in_place,
                decode_function=self.dataset.tokenizer.decode,
            )[0].tolist()

            decoded_text = self.dataset.tokenizer.decode(out_tokens)
            results.append(decoded_text)

            self._save_to_file(decoded_text, hint, seed)

            if print_in_place:
                print("\n----------------------------------------------")

        return results

    def _save_to_file(self, text, hint, seed):
        save_dir = os.path.join(self.run_dir, "generated_samples")
        os.makedirs(save_dir, exist_ok=True)
        # Clean hint for filename
        safe_hint = "".join([c if c.isalnum() else "_" for c in hint[:20]])
        file_path = os.path.join(save_dir, f"h_{safe_hint}_s_{seed}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
