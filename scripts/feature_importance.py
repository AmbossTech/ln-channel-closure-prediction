import logging
import sys
from pathlib import Path

# Allow `from _shap_utils import ...` when this file is invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import matplotlib
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import torch
import wandb
from _shap_utils import WrappedMLP, compute_mean_abs_shap, feature_names_for_pipeline

from ln_gossip_benchmark import PROJECT_ROOT
from ln_gossip_benchmark.data import apply_warm_start_mask, build_train_end_snapshot
from ln_gossip_benchmark.modules import TemporalModel
from ln_gossip_benchmark.utils.seed import set_random_seed

log = logging.getLogger(__name__)


def _normalize_edge_features(data, train_mask):
    data.msg = torch.log1p(data.msg)
    train_msg = data.msg[train_mask]
    feat_min = train_msg.min(dim=0, keepdim=True)[0]
    feat_max = train_msg.max(dim=0, keepdim=True)[0]
    feat_range = feat_max - feat_min
    feat_range = torch.where(feat_range > 1e-8, feat_range, torch.ones_like(feat_range))
    data.msg = (data.msg - feat_min) / feat_range


def _flatten_features(data, model, neighbor_loader, edge_original_indices, device):
    open_src = data.src[edge_original_indices].to(device)
    open_dst = data.dst[edge_original_indices].to(device)
    edge_messages = data.msg[edge_original_indices].to(device)
    edge_timestamps = data.t[edge_original_indices].to(device)

    src_nf = torch.log1p(model.get_node_features(open_src, neighbor_loader))
    dst_nf = torch.log1p(model.get_node_features(open_dst, neighbor_loader))

    current_t = edge_timestamps.max()
    edge_age = (current_t - edge_timestamps).float()
    src_recency = (current_t - model.last_update[open_src]).float()
    dst_recency = (current_t - model.last_update[open_dst]).float()

    flat = torch.cat(
        [
            edge_messages,
            src_nf,
            dst_nf,
            edge_age.unsqueeze(-1),
            src_recency.unsqueeze(-1),
            dst_recency.unsqueeze(-1),
        ],
        dim=-1,
    )
    return flat, edge_messages.size(-1), src_nf.size(-1)


def _save_outputs(df: pd.DataFrame, csv_path: Path, plot_path: Path, top_k: int = 15):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"Saved CSV to {csv_path}")
    log.info(f"\n{df.head(20).to_string(index=False)}")

    df_top = df.head(top_k).iloc[::-1]
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times"],
            "text.usetex": False,
        }
    )
    fig_w = 252 / 72.27
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    ax.barh(
        df_top["feature"],
        df_top["mean_abs_shap"],
        color="#7B68AE",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=8)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    log.info(f"Saved plot to {plot_path}")


@hydra.main(
    version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="default"
)
def main(cfg: omegaconf.DictConfig):
    set_random_seed(cfg.model.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    run = hydra.utils.call(cfg.wandb)
    wandb.config.update(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    )
    device = cfg.device

    dataset = hydra.utils.instantiate(cfg.data.dataset)
    num_classes = dataset.edge_label.unique().int().max().item() + 1
    data = dataset.get_TemporalData().to(device)

    first_day_mask = None
    if cfg.data.warm_start:
        first_day_mask = apply_warm_start_mask(data, dataset)

    data = data.to(device)
    data.global_id = torch.arange(len(data.src), device=device)
    data.raw_t = data.t.clone()
    data.t = data.t - data.t.min()
    log.info("Normalising edge features...")
    _normalize_edge_features(data, dataset.train_mask)

    train_loader = hydra.utils.instantiate(
        cfg.data.loader, data=data[dataset.train_mask]
    )

    log.info("Building end-of-training snapshot...")
    warm_start_snapshot = None
    if first_day_mask is not None:
        warm_start_snapshot = build_train_end_snapshot(
            hydra.utils.instantiate(
                cfg.data.loader, data=data[first_day_mask.to(device)]
            ),
            cfg.model.neighbor_loader,
            num_nodes=data.num_nodes,
            num_classes=num_classes,
            device=device,
        )
    train_end_snapshot = build_train_end_snapshot(
        train_loader,
        cfg.model.neighbor_loader,
        num_nodes=data.num_nodes,
        num_classes=num_classes,
        device=device,
        initial_snapshot=warm_start_snapshot,
    )

    neighbor_loader = hydra.utils.instantiate(
        cfg.model.neighbor_loader, num_nodes=data.num_nodes, device=device
    )
    neighbor_loader.load_state_dict(train_end_snapshot["neighbor_loader_state"])

    edge_features_dim = data.msg.size(-1)
    enrich_node_features = cfg.model.enrich_node_features
    node_dim = num_classes + 4 if enrich_node_features else num_classes
    predictor = hydra.utils.instantiate(
        cfg.model.predictor,
        edge_dim=edge_features_dim,
        node_dim=node_dim,
        out_channels=num_classes,
    ).to(device)
    model = TemporalModel(
        predictor=predictor,
        num_nodes=data.num_nodes,
        num_classes=num_classes,
        enrich_node_features=enrich_node_features,
    ).to(device)

    ckpt_path = cfg.model.load_model_path
    if ckpt_path is None:
        raise ValueError(
            "feature_importance requires a checkpoint. Pass it via "
            "+model.load_model_path=path/to/best_mlp.pt"
        )
    log.info(f"Loading model checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.feature_storage.copy_(train_end_snapshot["feature_storage"])
    model.last_update.copy_(train_end_snapshot["last_update"])

    log.info("Extracting features for end-of-training open edges...")
    _, _, edge_original_indices = neighbor_loader()
    if edge_original_indices.numel() == 0:
        raise RuntimeError("No open edges at end of training")
    flat, edge_dim, ns = _flatten_features(
        data,
        model,
        neighbor_loader,
        edge_original_indices,
        device,
    )
    log.info(f"Flat feature tensor shape: {flat.shape}")

    wrapped = WrappedMLP(predictor=model.predictor, edge_dim=edge_dim, node_dim=ns).to(
        device
    )
    wrapped.eval()

    log.info("Computing SHAP values...")
    mean_abs = compute_mean_abs_shap(
        wrapped,
        flat,
        n_background=100,
        n_sample=2000,
        seed=0,
    )

    feature_names = feature_names_for_pipeline(enrich_node_features)
    assert len(feature_names) == len(mean_abs), (
        f"Feature count mismatch: {len(feature_names)} names vs {len(mean_abs)} importances"
    )
    df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)

    _save_outputs(
        df,
        csv_path=PROJECT_ROOT / "results" / "feature_importance_mlp_shap.csv",
        plot_path=PROJECT_ROOT / "paper" / "figures" / "feature_importance.pdf",
    )

    run.finish()


if __name__ == "__main__":
    main()
