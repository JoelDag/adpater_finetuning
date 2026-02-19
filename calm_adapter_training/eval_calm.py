import argparse

import torch
from transformers import AutoTokenizer
from model.calm import CALM, CALMConfig  # adjust this import path if needed

def main():
    parser = argparse.ArgumentParser(description="Run inference for a trained CALM model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--anchor-model", default="google/gemma-7b")
    parser.add_argument("--aug-model", default="google/gemma-2b")
    parser.add_argument("--num-connections", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--prompt", default="The universe is")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = CALMConfig(
        anchor_model=args.anchor_model,
        aug_model=args.aug_model,
        num_connections=args.num_connections,
        num_heads=args.num_heads,
    )

    model = CALM.from_pretrained(args.model_path, config=config)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.anchor_model)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nOutput:\n{output_text}")


if __name__ == "__main__":
    main()
