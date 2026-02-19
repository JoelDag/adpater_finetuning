import argparse

from safetensors.torch import safe_open


def main():
    parser = argparse.ArgumentParser(description="Inspect xLoRA classifier tensor keys.")
    parser.add_argument("--classifier-path", required=True)
    args = parser.parse_args()

    with safe_open(args.classifier_path, framework="pt") as f:
        keys = list(f.keys())

    inner_keys = [k for k in keys if k.startswith("inner.")]
    print(f"Found {len(inner_keys)} classifier layer keys:")
    for key in sorted(inner_keys):
        print(" ", key)


if __name__ == "__main__":
    main()
