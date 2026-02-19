# xLoRA Utilities

This directory contains training/evaluation utilities for xLoRA experiments.

## Adapter Inputs

xLoRA training scripts now require an explicit adapter mapping JSON:

```json
{
  "adapter_1": "/path/to/adapter/checkpoint_a",
  "adapter_2": "/path/to/adapter/checkpoint_b"
}
```

Use `adapters.example.json` as a template and pass it with:

- `train_xlora_adapter.py --adapters_json <path>`
- `train_xlora_adapter_enhanced.py --adapters_json <path>`

## Artifact Policy

Large checkpoints and optimizer states are intentionally not tracked in git.
Store them externally and point the scripts to those paths via CLI arguments or environment variables.
