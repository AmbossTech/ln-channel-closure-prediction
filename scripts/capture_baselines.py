import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = PROJECT_ROOT / "baselines.json"

CONFIGS = [
    ("random_uniform", "random", ["model.strategy=uniform"]),
    ("random_majority", "random", ["model.strategy=majority"]),
    ("random_stratified", "random", ["model.strategy=stratified"]),
    ("mlp", "mlp", ["model.num_epochs=1"]),
    ("temporal_gnn", "temporal_gnn", ["model.num_epochs=1"]),
    ("snapshot_gnn", "snapshot_gnn", ["model.snapshot_epochs=5"]),
    ("mlp_spectral", "edge_predictor_spectral", ["model.num_epochs=1"]),
    ("xgboost", "xgboost", []),
    ("lightgbm", "lightgbm", []),
]

COMMON_OVERRIDES = [
    "model.seed=0",
    "data.warm_start=false",
    "data.exclude_first_day=false",
    "wandb.entity=null",
]

BOOTSTRAP = r"""
import json, os, runpy
import wandb
_orig_init = wandb.init
def _patched_init(*a, **kw):
    run = _orig_init(*a, **kw)
    _orig_log = run.log
    out = os.environ.get("METRICS_OUT")
    def _log(data, step=None, commit=None):
        if out and isinstance(data, dict):
            try:
                payload = {
                    k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else str(v))
                    for k, v in data.items()
                }
                with open(out, "a") as f:
                    f.write(json.dumps({"step": step, "data": payload}) + "\n")
            except Exception as e:
                with open(out, "a") as f:
                    f.write(json.dumps({"err": str(e)}) + "\n")
        return _orig_log(data, step=step, commit=commit)
    run.log = _log
    return run
wandb.init = _patched_init
runpy.run_path(os.environ["TRAIN_SCRIPT"], run_name="__main__")
"""


def run_one(name: str, model: str, extra: list) -> dict:
    metrics_path = f"/tmp/baseline_capture_new/{name}_metrics.jsonl"
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).unlink(missing_ok=True)
    python = os.environ.get("BASELINE_PYTHON", sys.executable)
    cmd = [
        python,
        "-c",
        BOOTSTRAP,
        f"model={model}",
        *COMMON_OVERRIDES,
        *extra,
        f"hydra.run.dir=/tmp/baseline_capture_new/{name}",
    ]
    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "METRICS_OUT": metrics_path,
        "TRAIN_SCRIPT": str(PROJECT_ROOT / "scripts" / "train.py"),
        "PYTHONPATH": str(PROJECT_ROOT / "src")
        + ":"
        + os.environ.get("PYTHONPATH", ""),
    }
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=1800
    )
    elapsed = time.time() - t0
    output = proc.stdout + "\n" + proc.stderr

    metrics = {}
    if Path(metrics_path).exists():
        with open(metrics_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                data = entry.get("data", {})
                for k, v in data.items():
                    if k.startswith("test") and "confusion_matrix" not in k:
                        metrics[k] = v
    metrics = metrics or None

    result = {
        "name": name,
        "model": model,
        "extra": extra,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics,
    }
    if proc.returncode != 0 or metrics is None:
        result["stderr_tail"] = "\n".join(output.splitlines()[-40:])
        print(
            f"FAILED ({proc.returncode}, {elapsed:.0f}s):\n{result['stderr_tail']}",
            flush=True,
        )
    else:
        print(
            f"OK ({elapsed:.0f}s): macro_f1={metrics.get('test/macro avg/f1-score'):.4f}",
            flush=True,
        )
    return result


def main():
    only = sys.argv[1:]
    results = {}
    for name, model, extra in CONFIGS:
        if only and name not in only:
            continue
        results[name] = run_one(name, model, extra)
        OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
