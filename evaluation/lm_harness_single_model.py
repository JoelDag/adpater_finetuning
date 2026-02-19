import argparse
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cpu_limits() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--eval_tasks", required=True, help="Comma-separated list")
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--limit", default="50")
    parser.add_argument("--wandb_project", default="eval_project")
    parser.add_argument("--wandb_group", default="eval_group")
    parser.add_argument("--run_name", default="eval_run")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_cpu_limits()

    logdir = Path(args.base_dir) / "logs"
    eval_output_base = Path(args.base_dir) / "lm_eval"
    logdir.mkdir(parents=True, exist_ok=True)
    eval_output_base.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(logdir))

    wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.run_name, resume="allow")
    run_id = wandb.run.id

    summary_results = {}
    eval_tasks = [task.strip() for task in args.eval_tasks.split(",") if task.strip()]
    step = 0
    step_dir = eval_output_base / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating base model '{args.model_name}' on {len(eval_tasks)} tasks.")

    for task in eval_tasks:
        task_output_dir = step_dir / task
        task_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n-- Task: {task} --")
        run_env = os.environ.copy()
        run_env["WANDB_PROJECT"] = args.wandb_project
        run_env["WANDB_RUN_ID"] = run_id
        run_env["WANDB_RESUME"] = "allow"

        try:
            subprocess.run(
                [
                    "lm_eval",
                    "--model",
                    "hf",
                    "--model_args",
                    f"pretrained={args.model_name},tokenizer={args.tokenizer_name}",
                    "--tasks",
                    task,
                    "--batch_size",
                    args.batch_size,
                    "--limit",
                    args.limit,
                    "--output_path",
                    str(task_output_dir),
                    "--log_samples",
                ],
                env=run_env,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"[ERROR] Evaluation failed for task '{task}'. Skipping.")
            traceback.print_exc()
            continue

        result_files = list(task_output_dir.glob("results_*.json"))
        if not result_files:
            print(f"[WARNING] No result file found for task '{task}' at step {step}.")
            continue

        try:
            with open(result_files[0], "r", encoding="utf-8") as f:
                results = json.load(f)["results"]

            log_data = {}
            for task_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        tag = f"{task_name}/{metric_name.replace(',', '_')}"
                        log_data[tag] = value
                        tb_writer.add_scalar(tag, value, global_step=step)
                        if metric_name == "acc,none":
                            summary_results[task_name] = value
            wandb.log(log_data, step=step)
        except Exception as exc:
            print(f"[ERROR] Failed to process results for task '{task}': {exc}")
            traceback.print_exc()

    tb_writer.close()
    wandb.finish()

    print("\n=== Summary of Accuracies ===")
    df = pd.DataFrame(list(summary_results.items()), columns=["Task", "Accuracy"]).sort_values(by="Task")
    print(df.to_markdown(index=False))

    with open(eval_output_base / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
