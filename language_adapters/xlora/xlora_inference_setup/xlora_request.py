import argparse
import json

import requests

def query_xlora(prompt, max_tokens=500, url="http://localhost:1234/v1/completions", model="default"):
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=60
    )

    if response.ok:
        print(response.json()["choices"][0]["text"])
    else:
        print(response.status_code, response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a completion request to an xLoRA server.")
    parser.add_argument("--prompt", default="How old is the universe?")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--url", default="http://localhost:1234/v1/completions")
    parser.add_argument("--model", default="default")
    args = parser.parse_args()

    query_xlora(args.prompt, max_tokens=args.max_tokens, url=args.url, model=args.model)
