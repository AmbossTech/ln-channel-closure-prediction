import os
from pathlib import Path

import git
from omegaconf import OmegaConf

from ln_gossip_benchmark.utils.hydra import decode_path, get_device

try:
    PROJECT_ROOT = Path(
        git.Repo(Path.cwd(), search_parent_directories=True).working_dir
    )
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["WANDB_CACHE_DIR"] = str(PROJECT_ROOT / ".wandb_cache")

if not OmegaConf.has_resolver("path"):
    OmegaConf.register_new_resolver("path", lambda path: decode_path(path))
if not OmegaConf.has_resolver("device"):
    OmegaConf.register_new_resolver("device", lambda device: get_device(device))
if not OmegaConf.has_resolver("to_str"):
    OmegaConf.register_new_resolver("to_str", lambda path: str(path))
