# Adapter Fine-Tuning for Multilingual LLMs

This repository contains end-to-end code for:
- multilingual data preparation and sampling,
- tokenizer extension and tokenization pipelines,
- parameter-efficient adapter training (LoRA / xLoRA),
- LM Harness-based evaluation and experiment logging.

The codebase is organized to support reproducible training workflows for large language models, with scripts for both local and cluster environments.

## Repository Structure

- `data_preparation/`: dataset download and preprocessing utilities
- `data_sampler/`: FineWeb2 sampling and dataset sizing tools
- `training/`: baseline training scripts
- `language_adapters/`: adapter-focused training, tokenizer extension, and analysis
- `language_adapters/xlora/`: xLoRA training and inference utilities
- `evaluation/`: LM Harness evaluation scripts and result export tools
- `calm_adapter_training/`: experimental CALM adapter approach
- `docs/`: setup notes and supporting documentation

## Quickstart

1. Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Prepare data:

```bash
python data_preparation/download_datasets.py --output_dir ./data/raw
python data_preparation/preprocess_data.py --input_dir ./data/raw --output_dir ./data/processed
```

3. (Optional) Sample FineWeb2:

```bash
bash data_sampler/run_fineweb2_sampler.sh
```

4. Train an adapter:

```bash
python language_adapters/train_language_adapter.py --help
```

5. Evaluate checkpoints:

```bash
python evaluation/lm_harness_single.py --help
python evaluation/lm_harness_single_model.py --help
```

## Reproducibility Notes

- Training/evaluation scripts set deterministic seeds where applicable.
- Checkpoint artifacts and logs are intentionally excluded from version control.
- Many run scripts are starter templates and should be adapted via CLI args or environment variables for your infrastructure.

## Artifact Policy

This repository tracks source code, configs, and lightweight metadata only. Large model checkpoints, optimizer states, and run logs are excluded via `.gitignore` and should be stored in artifact storage (for example: W&B artifacts, object storage, or external model registries).
