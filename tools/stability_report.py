"""
Quantify stability of every run under Download/.

For each eval.csv we report:
    final         : eval/success_rate at the very last logged step
    last_step     : the step that "final" was logged at
    late_mean     : mean of last K eval points
    late_std      : std  of last K eval points
    late_min      : min  of last K eval points  <-- predicts autograder
    late_max      : max  of last K eval points
    cv            : late_std / max(late_mean, eps)  (coefficient of variation)
    n_evals_in_window : how many eval rows fall in the autograder window
                        (offline+online - 2*eval_interval, offline+online]

K (last_k) defaults to 3 (~60k steps with eval_interval=20k). Tweak if you
want a longer / shorter window.

The script also reports cross-seed stability when it can detect groups of runs
that share (algo, env, alpha, inv_temp, expectile, od, w) but differ only in
the seed prefix `sdN_`.

Usage (from repo root):
    python tools\stability_report.py
    python tools\stability_report.py --root Download --last_k 3
    python tools\stability_report.py --csv stability.csv
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev


SEED_RE = re.compile(r"^sd(\d+)_(\d+_\d+)_(.*)$")


def parse_eval_csv(path: Path):
    """Return list[(step:int, value:float)] sorted by step. Drops duplicates,
    keeping the last occurrence (which is the online-phase value when offline
    and online both write at the same step like 500000)."""
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append((int(r["step"]), float(r["eval/success_rate"])))
            except (KeyError, ValueError):
                continue
    by_step = {}
    for s, v in rows:
        by_step[s] = v  # last write wins
    return sorted(by_step.items())


def load_flags(eval_path: Path):
    flags_path = eval_path.with_name("flags.json")
    if not flags_path.exists():
        return {}
    try:
        return json.loads(flags_path.read_text())
    except Exception:
        return {}


def stab_metrics(rows, last_k, flags):
    if not rows:
        return None
    steps = [s for s, _ in rows]
    vals  = [v for _, v in rows]

    final = vals[-1]
    last_step = steps[-1]

    last_window = vals[-last_k:]
    late_mean = mean(last_window)
    late_std  = pstdev(last_window) if len(last_window) > 1 else 0.0
    late_min  = min(last_window)
    late_max  = max(last_window)
    cv        = late_std / max(late_mean, 1e-9)

    # Autograder window:
    #   (offline + online - 2*eval_interval, offline + online]
    n_in_window = None
    if flags:
        offline = flags.get("offline_training_steps")
        online  = flags.get("online_training_steps")
        ei      = flags.get("eval_interval")
        if offline is not None and online is not None and ei is not None:
            lo = offline + online - 2 * ei
            hi = offline + online
            n_in_window = sum(1 for s in steps if lo < s <= hi)

    return {
        "final": final,
        "last_step": last_step,
        "late_mean": late_mean,
        "late_std": late_std,
        "late_min": late_min,
        "late_max": late_max,
        "cv": cv,
        "n_in_window": n_in_window,
        "n_evals": len(rows),
    }


def family_key(folder_name: str, flags: dict) -> str:
    """A stable key for "this is the same recipe, different seed". Strip the
    seed prefix and timestamp and squash any algo-specific HP suffixes into a
    canonical string."""
    m = SEED_RE.match(folder_name)
    if m:
        rest = m.group(3)
    else:
        rest = folder_name
    # rest is like "qsm_antsoccer-...-v0_a100.0_i100.0_online_offline"
    return rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Download")
    ap.add_argument("--last_k", type=int, default=3)
    ap.add_argument("--csv", default=None,
                    help="Optional path to write the per-run table as CSV.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root {root} does not exist.")

    rows = []
    for eval_csv in sorted(root.rglob("eval.csv")):
        flags = load_flags(eval_csv)
        rec = parse_eval_csv(eval_csv)
        m = stab_metrics(rec, args.last_k, flags)
        if m is None:
            continue
        rows.append({
            "group":   eval_csv.parent.parent.name,
            "run":     eval_csv.parent.name,
            "agent":   flags.get("agent", "?"),
            "env":     flags.get("env_name", "?"),
            "alpha":   flags.get("agent_kwargs", {}).get("alpha"),
            "i":       flags.get("agent_kwargs", {}).get("inv_temp"),
            "exp":     flags.get("agent_kwargs", {}).get("expectile"),
            "od":      flags.get("offline_data"),
            "w":       flags.get("wsrl_steps"),
            **m,
        })

    if not rows:
        raise SystemExit(f"No eval.csv files found under {root}.")

    # --- per-run table sorted by late_min (autograder predictor) ---
    rows.sort(key=lambda r: (-(r["late_min"] or 0), -(r["late_mean"] or 0)))

    cols = [
        ("group",     12),
        ("agent",      6),
        ("alpha",      7),
        ("i",          6),
        ("exp",        6),
        ("od",         8),
        ("w",          7),
        ("final",      6),
        ("late_mean",  9),
        ("late_min",   8),
        ("late_max",   8),
        ("late_std",   8),
        ("cv",         5),
        ("last_step",  9),
        ("n_in_window", 4),
        ("run",       60),
    ]
    print(f"\nPer-run stability (sorted by late_min, last_k={args.last_k}):\n")
    header = "  ".join(f"{c:>{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c, w in cols:
            v = r.get(c)
            if v is None:
                cell = "-"
            elif isinstance(v, float):
                cell = f"{v:.3f}"
            else:
                cell = str(v)
            cells.append(f"{cell:>{w}}"[: w])
        print("  ".join(cells))

    # --- cross-seed grouping ---
    families = {}
    for r in rows:
        key = family_key(r["run"], {"agent_kwargs": {}})
        families.setdefault((r["group"], key), []).append(r)
    multi = {k: v for k, v in families.items() if len(v) > 1}

    if multi:
        print("\nCross-seed stability (groups with >=2 runs sharing the same recipe):\n")
        for (group, key), runs in sorted(multi.items(), key=lambda kv: -mean(r["late_mean"] for r in kv[1])):
            seeds = [SEED_RE.match(r["run"]).group(1) for r in runs if SEED_RE.match(r["run"])]
            late_means = [r["late_mean"] for r in runs]
            late_mins  = [r["late_min"]  for r in runs]
            print(f"  [{group}] {key}")
            print(f"     seeds={seeds}  n={len(runs)}")
            print(f"     late_mean across seeds: mean={mean(late_means):.3f}  std={pstdev(late_means) if len(late_means)>1 else 0:.3f}")
            print(f"     late_min  across seeds: mean={mean(late_mins):.3f}   std={pstdev(late_mins)  if len(late_mins) >1 else 0:.3f}")
    else:
        print("\nCross-seed stability: no recipe has >=2 seeds yet.")
        print("  (Launch the same HPs with seeds 1, 2, 3 to populate this section.)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[c for c, _ in cols])
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c, _ in cols})
        print(f"\nWrote per-run table to {args.csv}")


if __name__ == "__main__":
    main()
