import argparse

from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Load a base model with an xLoRA adapter.")
    parser.add_argument("--base-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter-path", required=True)
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.base_id)
    cfg.use_cache = False

    base = AutoModelForCausalLM.from_pretrained(
        args.base_id, config=cfg, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter_path, local_files_only=True)
    model.eval()
    print("Model loaded successfully with XLORA adapters.")


if __name__ == "__main__":
    main()
