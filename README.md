# ln-gossip-benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2605.12759-b31b1b.svg)](https://arxiv.org/abs/2605.12759)

Code and benchmarking pipeline for the paper *Predicting Channel Closures in
the Lightning Network with Machine Learning* (Antonelli et al., 2026,
[arXiv:2605.12759](https://arxiv.org/abs/2605.12759)).

The task is **temporal link classification**: at each timestamp $t$, for every
currently-open channel, predict whether it will remain open, close
cooperatively (mutual), or be force-closed within the next $\Delta_t$ days,
using only public LN gossip data up to $t$.

## Installation

```bash
uv sync
```

> After `uv sync`, install PyG's C-extension wheels (replace `+cpu` with
> `+cu121` etc. on GPU):
>
> ```bash
> uv pip install --no-deps \
>   torch-scatter torch-cluster torch-sparse torch-spline-conv \
>   --find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html
> ```

## Dataset

The gossip dataset (LN events, 2022-06-09 → 2024-10-14, ~293MB CSV) is
hosted publicly on HuggingFace Datasets at
[`Amboss-Tech/ln-gossip`](https://huggingface.co/datasets/Amboss-Tech/ln-gossip)
and the pipeline downloads it on first run — no authentication required.

The processed tensors are cached at
`~/.cache/ln-gossip-benchmark/tgbl-ln_processed.pkl` so subsequent runs
skip the parse step. Override the cache location with `LN_GOSSIP_CACHE_DIR`.

## Quick start

```bash
uv run python scripts/train.py \
  model=mlp \
  data.warm_start=true \
  model.seed=0
```

## Reproducing the paper

All paper results use the warm-start configuration, $\Delta_t = 180$ days
(180-day forward closure window) and 3 seeds (0, 1, 2). For a single seed,
swap `model.seed=0,1,2` with `model.seed=0`.

The MLP and TGN headline configs are baked into `conf/model/mlp.yaml` and
`conf/model/temporal_gnn.yaml` respectively (architecture flags are *not*
specified on the command line — that proved error-prone earlier).

### Main results (Table I)

```bash
# Random baselines
uv run python scripts/train.py \
  model=random model.strategy=uniform    data.warm_start=true model.seed=0,1,2 -m
uv run python scripts/train.py \
  model=random model.strategy=majority   data.warm_start=true model.seed=0,1,2 -m
uv run python scripts/train.py \
  model=random model.strategy=stratified data.warm_start=true model.seed=0,1,2 -m

# Static (snapshot) GNN
uv run python scripts/train.py \
  model=snapshot_gnn data.warm_start=true model.seed=0,1,2 -m

# TGN
uv run python scripts/train.py \
  model=temporal_gnn data.warm_start=true model.seed=0,1,2 -m

# MLP + Spectral
uv run python scripts/train.py \
  model=edge_predictor_spectral data.warm_start=true model.seed=0,1,2 -m

# Tabular baselines
uv run python scripts/train.py \
  model=xgboost  data.warm_start=true model.seed=0,1,2 -m
uv run python scripts/train.py \
  model=lightgbm data.warm_start=true model.seed=0,1,2 -m

# MLP (best, paper headline — 0.38)
uv run python scripts/train.py \
  model=mlp data.warm_start=true model.seed=0,1,2 -m
```

### Feature ablation (Table II)

```bash
# Time only
uv run python scripts/train.py \
  model=mlp model.use_edge_features=false model.use_node_history=false \
  data.warm_start=true model.seed=0,1,2 -m
# + Edge feat.
uv run python scripts/train.py \
  model=mlp model.use_node_history=false \
  data.warm_start=true model.seed=0,1,2 -m
# + Node feat. (== best MLP)
uv run python scripts/train.py \
  model=mlp data.warm_start=true model.seed=0,1,2 -m
# Edge + Time + GNN
uv run python scripts/train.py \
  model=temporal_gnn model.use_node_history=false \
  data.warm_start=true model.seed=0,1,2 -m
# All + GNN (== best TGN)
uv run python scripts/train.py \
  model=temporal_gnn data.warm_start=true model.seed=0,1,2 -m
```

### Layer ablation (Figure 5)

Vary `model.mlp_num_hidden_layers` ∈ {0, 1, 2} for `model=mlp` and
`model=temporal_gnn`.

### Prediction-window ablation (Figure 6)

```bash
for dt in 30 90 180 365; do
  uv run python scripts/train.py \
    model=mlp \
    model.ground_truth_oracle.time_window_days=$dt \
    data.warm_start=true model.seed=0,1,2 -m
done
```

### Class-imbalance ablation (Table III)

Vary `model.criterion.weight.data` (e.g. `[1,5,5]`, `[1,1,1]`, `[1,3,6]`, ...)
and/or downsampling ratio `r` per the paper.

### SHAP feature importance (Figure 4 right)

```bash
uv run python scripts/train.py \
  model=mlp data.warm_start=true model.seed=0 \
  +model.save_model_path=checkpoints/best_mlp.pt

uv run python scripts/feature_importance.py \
  +checkpoint_path=checkpoints/best_mlp.pt
```

## Logging

Weights & Biases is installed but **disabled by default** (we set
`WANDB_MODE=disabled` in the example commands). To send metrics to wandb
instead:

```bash
wandb login
unset WANDB_MODE     # or `export WANDB_MODE=online`
```

## License

MIT.

## Citation

```bibtex
@misc{antonelli2026predicting,
  title={Predicting Channel Closures in the Lightning Network with Machine Learning},
  author={Simone Antonelli and Vincent Davis and Harrison Rush and Anthony Potdevin and Jesse Shrader and Vikash Singh and Emanuele Rossi},
  year={2026},
  eprint={2605.12759},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.12759}
}
```
