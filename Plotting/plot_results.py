"""
plot_results.py — Publication-quality plots for the offline-to-online RL project.

Generates per section:
  1. HP sweep plots  : eval success rate vs steps for each HP value (per env + overlaid)
  2. Best-run plots  : "BEST" download runs vs. submission run (per env)
  3. Global plot     : accumulates submission runs across all processed sections

Usage:
    python tools/plot_results.py --section s2_dsrl
    python tools/plot_results.py --section s2_dsrl,s2_ifql
    python tools/plot_results.py --section all

Run from the repo root:
    cd /path/to/RL_Project_Offiline
    python tools/plot_results.py --section s2_dsrl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Map Download section name → relative path to its Submission exp dir
SUBMISSION_MAP = {
    "s2_dsrl":    "Submissions/S2_IFQL_DSRL/exp/s2_dsrl",
    "s2_ifql":    "Submissions/S2_IFQL_DSRL/exp/s2_ifql",
    "s2_offline": "Submissions/S2_FQL_OFFLINE_WRSL/exp/s2_offline",
    "s2_wsrl":    "Submissions/S2_FQL_OFFLINE_WRSL/exp/s2_wsrl",
    "s2_qsm":     "Submissions/S2_QSM/exp/s2_qsm",
}

SECTION_LABELS = {
    "s2_dsrl":    "DSRL",
    "s2_ifql":    "IFQL",
    "s2_offline": "FQL + Offline Data",
    "s2_wsrl":    "FQL + WSRL",
    "s2_qsm":     "QSM",
}

# agent_kwargs key → LaTeX-ish display label
HP_LABELS = {
    "noise_scale":  r"$\sigma_z$",
    "alpha":        r"$\alpha$",
    "expectile":    r"$\tau$",
    "inv_temp":     r"$\eta$",
    "wsrl_steps":   "WSRL steps",
    "offline_data": "Offline data",
}

ENV_DISPLAY = {
    "cube-double-play-singletask-task1-v0":         "Cube Double Play",
    "antsoccer-arena-navigate-singletask-task1-v0": "Antsoccer Arena",
}

ENV_SHORT = {
    "cube-double-play-singletask-task1-v0":         "cube-double",
    "antsoccer-arena-navigate-singletask-task1-v0": "antsoccer",
}

# Colors for the global multi-section plot
GLOBAL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# Per-section HP sweep color palette
SWEEP_CMAP = "viridis"

OFFLINE_BOUNDARY = 500_000

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

RC = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "legend.fontsize":    9,
    "lines.linewidth":    2.0,
    "figure.dpi":         120,
}


def apply_style():
    plt.rcParams.update(RC)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_eval_csv(path: Path):
    """Return (steps: list[int], vals: list[float]) sorted by step, deduped (last write wins)."""
    by_step = {}
    try:
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    by_step[int(row["step"])] = float(row["eval/success_rate"])
                except (KeyError, ValueError):
                    pass
    except FileNotFoundError:
        return [], []
    pairs = sorted(by_step.items())
    if not pairs:
        return [], []
    s, v = zip(*pairs)
    return list(s), list(v)


def load_flags(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def is_run_dir(d: Path) -> bool:
    return d.is_dir() and (d / "eval.csv").exists() and (d / "flags.json").exists()


def env_short(env_name: str) -> str:
    if "antsoccer" in env_name:
        return "antsoccer"
    if "cube-double" in env_name:
        return "cube-double"
    return env_name


# ---------------------------------------------------------------------------
# RunInfo
# ---------------------------------------------------------------------------

class RunInfo:
    def __init__(self, path: Path, flags: dict, steps: list, vals: list):
        self.path = path
        self.flags = flags
        self.steps = steps
        self.vals = vals
        self.is_best = path.name.lower().startswith("best")
        self.env_name = flags.get("env_name", "")
        self.env_short = env_short(self.env_name)
        self.env_display = ENV_DISPLAY.get(self.env_name, self.env_name)
        self.agent = flags.get("agent", "?")
        self.agent_kwargs = flags.get("agent_kwargs", {})
        self.offline_steps = int(flags.get("offline_training_steps", OFFLINE_BOUNDARY))
        self.seed = flags.get("seed", None)

    def hp(self, key: str):
        return self.agent_kwargs.get(key)

    def seed_label(self) -> str:
        if self.seed is not None:
            return f"seed={self.seed}"
        return ""

    def __repr__(self):
        return f"RunInfo({self.path.name}, env={self.env_short}, seed={self.seed}, best={self.is_best})"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(section_dir: Path) -> list:
    """
    Finds all run dirs under section_dir.
    Handles both:
      - Flat:   section_dir/<run_name>/eval.csv
      - Nested: section_dir/<env_subdir>/eval.csv
    """
    runs = []
    if not section_dir.exists():
        return runs
    for child in sorted(section_dir.iterdir()):
        if not child.is_dir():
            continue
        if is_run_dir(child):
            flags = load_flags(child / "flags.json")
            steps, vals = load_eval_csv(child / "eval.csv")
            if steps:
                runs.append(RunInfo(child, flags, steps, vals))
    return runs


def find_varying_hps(runs: list) -> list:
    """Return agent_kwargs keys whose values differ across runs."""
    all_keys = set()
    for r in runs:
        all_keys.update(r.agent_kwargs)
    varying = []
    for k in sorted(all_keys):
        values = {r.agent_kwargs.get(k) for r in runs}
        if len(values) > 1:
            varying.append(k)
    return varying


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def new_ax(title: str, figsize=(8, 5)):
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Eval Success Rate")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.set_ylim(-0.02, 1.05)
    return fig, ax


def add_phase_line(ax, offline_steps: int):
    ax.axvline(
        x=offline_steps, color="dimgray", linestyle=":",
        linewidth=1.4, alpha=0.8, label=f"Online start ({offline_steps//1000}k)",
        zorder=1,
    )
    ax.axvspan(0, offline_steps, alpha=0.04, color="steelblue", zorder=0)
    ax.axvspan(offline_steps, ax.get_xlim()[1] if ax.get_xlim()[1] > offline_steps else offline_steps * 1.5,
               alpha=0.04, color="darkorange", zorder=0)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# HP sweep plots
# ---------------------------------------------------------------------------

def plot_hp_sweep_per_env(runs: list, hp_key: str, section_label: str, out_dir: Path):
    """One plot per environment, lines = different HP values."""
    hp_label = HP_LABELS.get(hp_key, hp_key)
    by_env = defaultdict(list)
    for r in runs:
        if r.hp(hp_key) is not None:
            by_env[r.env_short].append(r)

    for env_key, env_runs in by_env.items():
        hp_values = sorted(set(r.hp(hp_key) for r in env_runs))
        colors = matplotlib.colormaps[SWEEP_CMAP](np.linspace(0.15, 0.85, max(len(hp_values), 1)))
        color_map = dict(zip(hp_values, colors))

        env_disp = env_runs[0].env_display
        title = f"{section_label} — {env_disp}\n{hp_label} Sweep"
        fig, ax = new_ax(title)

        add_phase_line(ax, env_runs[0].offline_steps)

        for r in sorted(env_runs, key=lambda x: x.hp(hp_key)):
            v = r.hp(hp_key)
            seed_str = f", {r.seed_label()}" if r.seed_label() else ""
            best_str = " ★" if r.is_best else ""
            ax.plot(r.steps, r.vals, color=color_map[v],
                    label=f"{hp_label} = {v}{seed_str}{best_str}", linewidth=2, alpha=0.9)

        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), loc="upper left", framealpha=0.9)

        save(fig, out_dir / f"hp_sweep_{hp_key}_{env_key}.pdf")


def plot_hp_sweep_overlaid(runs: list, hp_key: str, section_label: str, out_dir: Path):
    """Both environments on one plot, distinguished by linestyle."""
    hp_label = HP_LABELS.get(hp_key, hp_key)
    valid = [r for r in runs if r.hp(hp_key) is not None]
    if not valid:
        return

    hp_values = sorted(set(r.hp(hp_key) for r in valid))
    colors = matplotlib.colormaps[SWEEP_CMAP](np.linspace(0.15, 0.85, max(len(hp_values), 1)))
    color_map = dict(zip(hp_values, colors))

    env_styles = {"antsoccer": "-", "cube-double": "--"}
    env_markers = {"antsoccer": "Antsoccer", "cube-double": "Cube Double"}

    title = f"{section_label} — {hp_label} Sweep\n(Antsoccer ─   Cube Double ╌ ╌)"
    fig, ax = new_ax(title)
    add_phase_line(ax, valid[0].offline_steps)

    for r in sorted(valid, key=lambda x: (x.env_short, x.hp(hp_key))):
        v = r.hp(hp_key)
        ls = env_styles.get(r.env_short, "-")
        env_name = env_markers.get(r.env_short, r.env_short)
        seed_str = f", {r.seed_label()}" if r.seed_label() else ""
        best_str = " ★" if r.is_best else ""
        ax.plot(r.steps, r.vals, color=color_map[v], linestyle=ls,
                label=f"{hp_label}={v}{seed_str}{best_str} ({env_name})", linewidth=2, alpha=0.85)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="upper left", framealpha=0.9, ncol=2)

    save(fig, out_dir / f"hp_sweep_{hp_key}_both_envs.pdf")


# ---------------------------------------------------------------------------
# Multi-seed plotting helper
# ---------------------------------------------------------------------------

def _plot_seeds_on_ax(ax, runs: list, label: str, color: str):
    """
    Plot runs on ax with multi-seed awareness.
    - 1 run  → single line
    - 2+ runs → mean solid line + shaded std band, individual runs as faint dashes
    """
    if not runs:
        return
    if len(runs) == 1:
        ax.plot(runs[0].steps, runs[0].vals, color=color, linewidth=2,
                alpha=0.9, label=label)
        return

    # Align all runs to a common step grid (union, forward-fill missing)
    all_steps = sorted(set(s for r in runs for s in r.steps))
    interp_vals = []
    for r in runs:
        step_to_val = dict(zip(r.steps, r.vals))
        last = 0.0
        row = []
        for s in all_steps:
            if s in step_to_val:
                last = step_to_val[s]
            row.append(last)
        interp_vals.append(row)

    arr = np.array(interp_vals)           # shape (n_seeds, n_steps)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    ax.plot(all_steps, mean, color=color, linewidth=2, alpha=0.9, label=label)
    ax.fill_between(all_steps, mean - std, mean + std, color=color, alpha=0.15)
    for r in runs:
        ax.plot(r.steps, r.vals, color=color, linewidth=0.8, alpha=0.35, linestyle="--")


# ---------------------------------------------------------------------------
# Best vs Submission comparison plots
# ---------------------------------------------------------------------------

def plot_best_vs_submission(
    best_runs: list,
    submission_runs: list,
    section_label: str,
    out_dir: Path,
    global_axes: dict = None,  # {"antsoccer": ax, "cube-double": ax}
    global_color: str = None,
):
    """
    Per-env plot: best runs + submission runs labeled only by seed and HP values.
    Optionally draws onto global_axes for the Part III plot.
    """
    all_runs = best_runs + submission_runs

    by_env = defaultdict(list)
    for r in all_runs:
        by_env[r.env_short].append(r)

    # Assign distinct colors across all runs in this plot
    colors = matplotlib.colormaps["tab10"](np.linspace(0, 0.9, max(len(all_runs), 1)))

    for env_key, env_runs in by_env.items():
        env_disp = env_runs[0].env_display
        title = f"{section_label} — {env_disp}"
        fig, ax = new_ax(title)

        offline_steps = env_runs[0].offline_steps
        add_phase_line(ax, offline_steps)

        varying_hps = find_varying_hps(env_runs)
        for i, r in enumerate(env_runs):
            parts = [f"{HP_LABELS.get(k, k)}={r.hp(k)}" for k in varying_hps if r.hp(k) is not None]
            if r.seed_label():
                parts.append(r.seed_label())
            label = ", ".join(parts) if parts else r.path.name
            ax.plot(
                r.steps, r.vals,
                color=colors[i], linewidth=2, alpha=0.9,
                label=label,
            )

        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), loc="upper left", framealpha=0.9)

        save(fig, out_dir / f"best_vs_submission_{env_key}.pdf")

        # --- Add to global Part III plot ---
        if global_axes and global_color and env_key in global_axes:
            g_ax = global_axes[env_key]
            # Prefer submission runs; fall back to best download
            pool = [r for r in (submission_runs if submission_runs else best_runs)
                    if r.env_short == env_key]
            _plot_seeds_on_ax(g_ax, pool, label=section_label, color=global_color)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_section(
    section: str,
    out_root: Path,
    global_axes: dict = None,
    global_color: str = None,
):
    section_label = SECTION_LABELS.get(section, section)
    section_dir = REPO_ROOT / "Download" / section
    out_dir = out_root / section

    print(f"\n{'─'*60}")
    print(f"  Section : {section_label}  ({section})")
    print(f"  Input   : {section_dir.relative_to(REPO_ROOT)}")
    print(f"  Output  : {out_dir.relative_to(REPO_ROOT)}")
    print(f"{'─'*60}")

    all_runs = discover_runs(section_dir)
    if not all_runs:
        print("  [WARN] No runs found — skipping.")
        return

    best_runs = [r for r in all_runs if r.is_best]
    sweep_runs = [r for r in all_runs if not r.is_best]

    print(f"  Runs found  : {len(all_runs)}  (best={len(best_runs)}, sweep={len(sweep_runs)})")

    # ---- HP sweep plots (only for non-BEST runs) ----
    # Include BEST runs in sweep too so all HP values are shown
    sweep_for_plot = all_runs  # show everything in HP sweep
    varying_hps = find_varying_hps(sweep_for_plot)
    print(f"  Varying HPs : {varying_hps}")

    for hp_key in varying_hps:
        plot_hp_sweep_per_env(sweep_for_plot, hp_key, section_label, out_dir)
        plot_hp_sweep_overlaid(sweep_for_plot, hp_key, section_label, out_dir)

    # ---- Best vs Submission ----
    sub_rel = SUBMISSION_MAP.get(section)
    submission_runs = []
    if sub_rel:
        sub_dir = REPO_ROOT / sub_rel
        submission_runs = discover_runs(sub_dir)
        print(f"  Submission  : {len(submission_runs)} runs from {sub_dir.relative_to(REPO_ROOT)}")

    if best_runs or submission_runs:
        plot_best_vs_submission(
            best_runs, submission_runs, section_label, out_dir,
            global_axes=global_axes, global_color=global_color,
        )
    else:
        print("  [INFO] No BEST runs or submission runs found; skipping best-vs-submission plot.")


def build_global_axes():
    """Create the global Part III comparison figure with two subplots."""
    apply_style()
    fig, (ax_ant, ax_cube) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, title in [
        (ax_ant,  "Antsoccer Arena — All Methods"),
        (ax_cube, "Cube Double Play — All Methods"),
    ]:
        ax.set_title(title, pad=10)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Eval Success Rate")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
        ax.set_ylim(-0.02, 1.05)
        ax.axvline(OFFLINE_BOUNDARY, color="dimgray", linestyle=":",
                   linewidth=1.4, alpha=0.8, label="Online start (500k)", zorder=1)
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    return fig, {"antsoccer": ax_ant, "cube-double": ax_cube}


def finalise_global(fig, axes: dict, out_root: Path):
    for env_key, ax in axes.items():
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), loc="upper left", framealpha=0.9)

    path = out_root / "global_comparison.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Global plot → {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot offline-to-online RL results")
    ap.add_argument(
        "--section", default="s2_dsrl",
        help="Comma-separated section(s) to process, or 'all'. "
             "E.g.: s2_dsrl  or  s2_dsrl,s2_ifql  or  all",
    )
    ap.add_argument(
        "--output_dir", default="Plotting",
        help="Root output directory (relative to repo root or absolute)",
    )
    args = ap.parse_args()

    out_root = (
        Path(args.output_dir) if Path(args.output_dir).is_absolute()
        else REPO_ROOT / args.output_dir
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if args.section.strip().lower() == "all":
        sections = list(SUBMISSION_MAP.keys())
    else:
        sections = [s.strip() for s in args.section.split(",")]

    print(f"\nRepo root : {REPO_ROOT}")
    print(f"Output    : {out_root.relative_to(REPO_ROOT)}")
    print(f"Sections  : {sections}")

    global_fig, global_axes = build_global_axes()

    for i, section in enumerate(sections):
        color = GLOBAL_COLORS[i % len(GLOBAL_COLORS)]
        process_section(section, out_root, global_axes=global_axes, global_color=color)

    finalise_global(global_fig, global_axes, out_root)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
