"""
Combine an offline-phase log.pkl and an online-phase log.pkl into a single log.pkl.

Usage:
    uv run python src/scripts/combine_logs.py \
        --offline_log exp/my_offline_run/log.pkl \
        --online_log  exp/my_online_run/log.pkl \
        --output_dir  exp/combined \
        --online_step_offset 500000  # set to 0 if online log already starts at 500k

The script:
  1. Takes all train/eval rows from the offline log (steps 0..offline_end)
  2. Takes all train/eval rows from the online log, adding --online_step_offset to each step
  3. Deduplicates eval rows at the seam (keeps offline version of step=500k)
  4. Writes combined log.pkl, train.csv, eval.csv, flags.json into --output_dir
"""

import argparse
import copy
import csv
import json
import os
import pickle
from datetime import datetime


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def shift_steps(rows: list, offset: int) -> list:
    if offset == 0:
        return rows
    shifted = []
    for row in rows:
        r = copy.deepcopy(row)
        r["step"] = r["step"] + offset
        shifted.append(r)
    return shifted


def write_csv(rows: list, path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def combine(offline_log: str, online_log: str, output_dir: str, online_step_offset: int) -> None:
    offline = load_pkl(offline_log)
    online = load_pkl(online_log)

    offline_train = offline["train"]
    offline_eval = offline["eval"]

    online_train = shift_steps(online["train"], online_step_offset)
    online_eval = shift_steps(online["eval"], online_step_offset)

    # Find the boundary step (last offline step)
    offline_last_step = max(r["step"] for r in offline_train) if offline_train else 0

    # Remove any online train/eval rows at or before the offline boundary to avoid overlap
    online_train = [r for r in online_train if r["step"] > offline_last_step]
    # For eval, keep offline version of the seam step
    online_eval_seam_step = min((r["step"] for r in online_eval), default=None)
    offline_eval_steps = {r["step"] for r in offline_eval}
    online_eval_deduped = [r for r in online_eval if r["step"] not in offline_eval_steps]

    combined_train = offline_train + online_train
    combined_eval = offline_eval + online_eval_deduped

    # Sort by step
    combined_train.sort(key=lambda r: r["step"])
    combined_eval.sort(key=lambda r: r["step"])

    # Use offline config as base, patch in online_training_steps if present
    config = copy.deepcopy(offline["config"])
    if "online_training_steps" in online.get("config", {}):
        config["online_training_steps"] = online["config"]["online_training_steps"]

    os.makedirs(output_dir, exist_ok=True)

    data = {
        "train": combined_train,
        "train_hash": hash(json.dumps(combined_train, sort_keys=True, default=str)),
        "eval": combined_eval,
        "eval_hash": hash(json.dumps(combined_eval, sort_keys=True, default=str)),
        "config": config,
        "config_hash": hash(json.dumps(config, sort_keys=True, default=str)),
        "time": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    with open(os.path.join(output_dir, "log.pkl"), "wb") as f:
        pickle.dump(data, f)

    write_csv(combined_train, os.path.join(output_dir, "train.csv"))
    write_csv(combined_eval, os.path.join(output_dir, "eval.csv"))

    with open(os.path.join(output_dir, "flags.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"Combined log written to {output_dir}")
    print(f"  Train rows: {len(offline_train)} offline + {len(online_train)} online = {len(combined_train)}")
    print(f"  Eval rows:  {len(offline_eval)} offline + {len(online_eval_deduped)} online = {len(combined_eval)}")
    print(f"  Eval step range: {combined_eval[0]['step']} -> {combined_eval[-1]['step']}")
    print(f"  Final eval success_rate: {combined_eval[-1].get('eval/success_rate', 'N/A')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_log", required=True, help="Path to offline log.pkl")
    parser.add_argument("--online_log", required=True, help="Path to online log.pkl")
    parser.add_argument("--output_dir", required=True, help="Directory to write combined logs")
    parser.add_argument(
        "--online_step_offset",
        type=int,
        default=500000,
        help="Add this offset to all step values in the online log. "
             "Set to 0 if online log already starts at step 500000. "
             "Default: 500000 (assumes online log steps start from 0).",
    )
    args = parser.parse_args()
    combine(args.offline_log, args.online_log, args.output_dir, args.online_step_offset)


if __name__ == "__main__":
    main()
