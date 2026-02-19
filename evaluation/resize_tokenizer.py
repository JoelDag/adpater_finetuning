import argparse

from transformers import PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser(description="Resize a tokenizer to a target vocabulary size.")
    parser.add_argument("--input-tokenizer", required=True, help="Path or model id of the source tokenizer.")
    parser.add_argument("--target-size", type=int, required=True, help="Target vocabulary size.")
    parser.add_argument("--output-tokenizer", required=True, help="Output path for resized tokenizer.")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.input_tokenizer)
    if len(tokenizer) > args.target_size:
        raise ValueError(f"Tokenizer already has size {len(tokenizer)} > target {args.target_size}")

    missing = args.target_size - len(tokenizer)
    tokenizer.add_tokens([f"<extra_token_{i}>" for i in range(missing)])
    tokenizer.save_pretrained(args.output_tokenizer)
    print(f"Saved tokenizer with {len(tokenizer)} tokens to {args.output_tokenizer}")


if __name__ == "__main__":
    main()
