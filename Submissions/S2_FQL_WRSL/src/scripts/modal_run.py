import os
import time
import argparse
from pathlib import Path

import modal

from scripts.train_offline_online import main, setup_arguments


APP_NAME = "offline-to-online-project"
NETRC_CANDIDATE_PATHS = [
    Path("~/.netrc").expanduser(),   # Unix / macOS
    Path("~/_netrc").expanduser(),   # Windows (wandb commonly writes here)
]
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/exp"
DEFAULT_GPU = "A10G"
DEFAULT_CPU = 4.0
DEFAULT_MEMORY = 16384  # MB
volume = modal.Volume.from_name("offline-to-online-project-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""

    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# Build a container image with the project's dependencies using uv.
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").uv_sync()
# Download OGBench datasets.
image = image.run_commands("python -c \"import ogbench;ogbench.download_datasets(['cube-single-play-v0', 'cube-double-play-v0','antsoccer-arena-navigate-v0'])\"")
# Copy netrc for wandb logging.
for netrc_path in NETRC_CANDIDATE_PATHS:
    if netrc_path.is_file():
        image = image.add_local_file(
            netrc_path,
            remote_path="/root/.netrc",
            copy=True,
        )
        break
# Copy the current directory.
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
}

# Forward W&B credentials from the local environment into the container.
# Modal does not auto-forward local env vars, so we explicitly pass them as a Secret.
_wandb_secret_env = {
    k: v
    for k, v in {
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
    }.items()
    if v
}
secrets = (
    [modal.Secret.from_dict(_wandb_secret_env)] if _wandb_secret_env else []
)


@app.function(volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 12, env=env, secrets=secrets, image=image, gpu=DEFAULT_GPU, cpu=DEFAULT_CPU, memory=DEFAULT_MEMORY)
def offline_to_online_modal_remote(*args: str) -> None:
    args = setup_arguments(args)
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(
            job_specs=args.job_specs,
            njobs=args.njobs,
            entrypoint_module="scripts.train_offline_online",
        )

    else:
        # Run a single job
        main(args)
    volume.commit()
