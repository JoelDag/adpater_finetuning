import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Remove internal_xlora_classifier tensors from adapter safetensors.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing adapter_* subdirectories.")
    parser.add_argument("--adapters", default="adapter_1,adapter_2", help="Comma-separated adapter subdirs.")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    adapters = [name.strip() for name in args.adapters.split(",") if name.strip()]

    for sub in adapters:
        f = ckpt / sub / "adapter_model.safetensors"
        sd = load_file(f)
        keys_to_drop = [k for k in sd if k.startswith("internal_xlora_classifier.")]
        if keys_to_drop:
            print(f"{f} -> removing {len(keys_to_drop)} classifier tensors")
            for k in keys_to_drop:
                del sd[k]
            save_file(sd, f)
            print("Saved cleaned file")
        else:
            print(f"{f} -> no classifier tensors to remove")


if __name__ == "__main__":
    main()
