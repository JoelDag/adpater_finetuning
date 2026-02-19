import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_adapters(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        adapters = json.load(f)
    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("--adapters-json must be a non-empty JSON object")
    return adapters


def main():
    parser = argparse.ArgumentParser(description="Run quick text generation with multiple xLoRA adapters.")
    parser.add_argument("--base-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapters-json", required=True, help="JSON mapping of adapter_name -> adapter_path")
    parser.add_argument("--prompt", default="Translate to Swahili: Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    adapter_paths = load_adapters(args.adapters_json)
    adapter_names = list(adapter_paths.keys())

    tokenizer = AutoTokenizer.from_pretrained(args.base_id)
    cfg = AutoConfig.from_pretrained(args.base_id)
    cfg.use_cache = False

    base = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    first_name = adapter_names[0]
    model = PeftModel.from_pretrained(
        base,
        adapter_paths[first_name],
        adapter_name=first_name,
        local_files_only=True,
    )
    for adapter_name in adapter_names[1:]:
        model.load_adapter(
            adapter_paths[adapter_name],
            adapter_name=adapter_name,
            local_files_only=True,
        )

    print("Loaded adapters:", list(model.peft_config.keys()))

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    for adapter_name in adapter_names:
        print(f"\nUsing {adapter_name}")
        model.set_adapter(adapter_name)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output with {adapter_name}: {output_text}")


if __name__ == "__main__":
    main()
