import argparse
from typing import List

import pandas as pd
import wandb


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export selected W&B metrics to Excel.")
    parser.add_argument(
        "--run-path",
        required=True,
        help="W&B run path in the format entity/project/run_id.",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Comma-separated metric names, for example: task_a/acc_none,task_b/acc_none",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="Optional comma-separated steps to keep. Leave empty to keep all steps.",
    )
    parser.add_argument(
        "--output",
        default="wandb_metrics.xlsx",
        help="Output Excel file path.",
    )
    args = parser.parse_args()

    metric_cols = parse_csv(args.metrics)
    target_steps = [int(step) for step in parse_csv(args.steps)] if args.steps else []

    api = wandb.Api()
    run = api.run(args.run_path)
    history = run.history()

    columns_to_keep = metric_cols + ["_step"]
    missing = [column for column in columns_to_keep if column not in history.columns]
    if missing:
        raise ValueError(f"Missing columns in W&B history: {missing}")

    df = history[columns_to_keep].copy()
    if target_steps:
        df = df[df["_step"].isin(target_steps)]

    df["row_sum"] = df[metric_cols].sum(axis=1, numeric_only=True)
    print(df)
    df.to_excel(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
